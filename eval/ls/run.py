"""Unified LangSmith executor. Only --dry-run is free; model calls need --paid."""

# ANCHOR: verified cases → one SDK evaluation → durable checkpoints; rejudge uses
# stored runs and stable UUIDs, preserving original or explicitly updated references.

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from eval.ls.artifacts import Artifacts, summarize_records
from eval.ls.audit import AuditedModel, audited
from eval.ls.dataset import (
    DATASET_NAME,
    DEFAULT_CSV,
    DEFAULT_LABELS,
    SPLIT_VERIFIED,
    build_examples,
    labels_index_snapshot,
)
from eval.ls.evaluators import CODE_EVALUATORS, summarize
from eval.ls.groundedness import make_groundedness_evaluator
from eval.ls.judge import make_citations_evaluator, make_correctness_evaluator
from eval.ls.manifest import make_manifest
from eval.ls.policy import case_type, is_oos
from eval.ls.target import make_target

LABELS_SNAPSHOT = "build-j7ceu4r4"
JUDGE_KEYS = {
    "correctness": ("correctness", "forbidden_claim"),
    "citations": ("citations", "citations_complete"),
    "groundedness": ("groundedness",),
}
ALL_KEYS = (
    tuple(e.__name__ for e in CODE_EVALUATORS)
    + tuple(k for keys in JUDGE_KEYS.values() for k in keys)
    + ("premise_corrected",)
)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def judge_evaluators() -> list:
    """Bounded audited models via the existing no-retry judge factory."""
    from config.settings import settings
    from src.infra.llm_factory import get_citation_judge_llm, get_judge_llm

    def general():
        llm = get_judge_llm(max_tokens=16000, temperature=0.0)
        # Same bounded transport policy as the project's citation judge.
        llm.root_client = llm.root_client.with_options(max_retries=0, timeout=300)
        llm.client = llm.root_client.chat.completions
        return AuditedModel(llm, settings.JUDGE_MODEL_NAME)

    return [
        make_correctness_evaluator(general()),
        make_citations_evaluator(
            AuditedModel(
                get_citation_judge_llm(temperature=0.0),
                settings.CITATION_JUDGE_MODEL_NAME,
            )
        ),
        make_groundedness_evaluator(general()),
    ]


def with_current_references(evaluators: list, current: dict[str, dict]) -> list:
    """Historical helper; new execution joins references by UUID below."""

    def wrap(evaluator):
        def evaluate(inputs, outputs, reference_outputs):
            return evaluator(inputs, outputs, current[inputs["question"].strip()])

        evaluate.__name__ = evaluator.__name__
        return evaluate

    return [wrap(e) for e in evaluators]


def with_frozen_references(evaluator, references):
    """Bind to immutable references, independent of edited question text."""
    references = copy.deepcopy(references)

    def evaluate(run, example):
        reference = references.get(str(example.id))
        if reference is None:
            raise ValueError(f"Missing frozen reference for {example.id}")
        return evaluator(run.inputs or {}, run.outputs or {}, reference.outputs or {})

    evaluate.__name__ = evaluator.__name__
    return evaluate


def select_examples(examples, cases, limit):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if cases:
        requested = set(cases)
        known = {(e.metadata or {}).get("case_id") for e in examples}
        if requested - known:
            raise ValueError(f"Unknown case IDs: {sorted(requested - known)}")
        examples = [
            e for e in examples if (e.metadata or {}).get("case_id") in requested
        ]
    return examples[:limit] if limit is not None else examples


def _applicable(key, reference):
    if key == "oos_handled":
        return is_oos(reference)
    if key == "premise_corrected":
        return case_type(reference) == "false_premise"
    if key == "forbidden":
        return bool(reference.get("must_not_contain"))
    if key == "forbidden_claim":
        return not is_oos(reference) and bool(reference.get("forbidden_claims"))
    return not is_oos(reference)


def _scores(raw, reference, *, skip_judge, only_judges=False):
    result = {s.key: {"score": s.score, "comment": s.comment or ""} for s in raw}
    judge_keys = {k for keys in JUDGE_KEYS.values() for k in keys} | {
        "premise_corrected"
    }
    for key in ALL_KEYS:
        result.setdefault(
            key,
            {
                "score": None,
                "comment": (
                    "judge_skipped"
                    if skip_judge and key in judge_keys
                    else "evaluation_missing"
                ),
            },
        )
        result[key]["applicable"] = _applicable(key, reference)
        if only_judges and key in {e.__name__ for e in CODE_EVALUATORS}:
            result[key] = {
                "score": None,
                "comment": "not_rejudged",
                "applicable": False,
            }
    return result


def consume_results(
    results,
    sink,
    examples,
    usages,
    *,
    skip_judge=False,
    checkpoints=None,
    cancelled=None,
    reuse_pipeline=False,
    require_parent_context=False,
):
    """Persist each completed SDK row before advancing its iterator."""
    records = []
    references = {str(e.id): e for e in examples}

    def summary():
        value = summarize_records(records, assigned=len(examples))
        completed = {r["example_id"] for r in records}
        pending_scores = [
            {
                "scores": _scores(
                    [],
                    e.outputs or {},
                    skip_judge=skip_judge,
                    only_judges=reuse_pipeline,
                )
            }
            for eid, e in references.items()
            if eid not in completed
        ]
        value["metrics"] = summarize_records(
            records + pending_scores, assigned=len(examples)
        )["metrics"]
        value["mrr"] = summarize([r["scores"] for r in records])["mrr"]
        value["valid"] = not (
            require_parent_context
            and len(records) == len(examples)
            and not value["parent_context"]["parent_replaced_total"]
        )
        value["validation_errors"] = (
            []
            if value["valid"]
            else ["parent_replaced_total == 0 while parent context is enabled"]
        )
        if reuse_pipeline:
            value["reused_pipeline_cost"] = value.pop("pipeline_cost")
            value["total_cost_usd"] = value["judge_cost"]["cost_usd"]
            value["cost_incomplete"] = (
                value["judge_cost"]["incomplete"] or value["pending"] > 0
            )
        return value

    try:
        for row in results:
            actual, example = row["run"], row["example"]
            reference = references[str(example.id)]
            record = {
                "case_id": (reference.metadata or {}).get("case_id"),
                "example_id": str(example.id),
                "langsmith_run_id": str(actual.id),
                "outputs": {
                    **(checkpoints or {}).get(str(example.id), {}),
                    **(actual.outputs or {}),
                },
                "error": actual.error,
                "scores": _scores(
                    row["evaluation_results"]["results"],
                    reference.outputs or {},
                    skip_judge=skip_judge,
                    only_judges=reuse_pipeline,
                ),
                "judge_usage": usages.get(str(actual.id), []),
            }
            sink.append("results", record)
            records.append(record)
            sink.finish(summary())
    finally:
        if cancelled is not None:
            cancelled.set()
        sink.finish(summary())
    value = summary()
    if not value["valid"]:
        raise RuntimeError(value["validation_errors"][0])
    return value


def _wrapped_evaluators(
    examples, sink, usages, *, skip_judge, cancelled=None, only_judges=False
):
    references = {str(e.id): e for e in examples}
    evaluators = ([] if only_judges else list(CODE_EVALUATORS)) + (
        [] if skip_judge else judge_evaluators()
    )
    wrapped = []
    for evaluator in evaluators:
        audit = audited(
            evaluator,
            JUDGE_KEYS.get(evaluator.__name__, (evaluator.__name__,)),
            sink,
            usages,
            cancelled,
        )

        def evaluate(run, example, audit=audit):
            return audit(run, references[str(example.id)])

        evaluate.__name__ = evaluator.__name__
        wrapped.append(evaluate)
    return wrapped


def _new_id(prefix):
    return (
        f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    )


def run_experiment(
    split=SPLIT_VERIFIED,
    prefix="unified",
    skip_judge=False,
    limit=None,
    cases=None,
    *,
    dry_run=False,
    output_dir=None,
    csv_path=DEFAULT_CSV,
    labels_path=DEFAULT_LABELS,
    labels_snapshot=LABELS_SNAPSHOT,
) -> str:
    if split != SPLIT_VERIFIED:
        raise ValueError(
            "Normative runner requires --split verified; use eval.ls.align for alignment splits"
        )
    local = [SimpleNamespace(**e) for e in build_examples(csv_path, labels_path)]
    selected = select_examples(local, cases, limit)
    if not selected:
        raise ValueError("No verified cases selected")
    run_id = _new_id("eval")
    manifest = make_manifest(
        selected,
        run_id=run_id,
        split=split,
        dry_run=dry_run,
        skip_judge=skip_judge,
        git_sha=_git_sha(),
        labels_snapshot=labels_snapshot,
        labels_path=labels_path,
    )
    if dry_run:
        sink = Artifacts(
            Path(output_dir) if output_dir else Path("eval/runs") / run_id, manifest
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "assigned": len(selected),
                    "case_ids": [e.metadata["case_id"] for e in selected],
                    "case_counts": manifest["case_counts"],
                    "unlabelled_case_ids": manifest["unlabelled_case_ids"],
                    "snapshot_compatible": manifest["index_snapshot"]
                    == labels_snapshot
                    == labels_index_snapshot(labels_path),
                    "artifacts": str(sink.directory),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return "dry-run"
    declared = labels_index_snapshot(labels_path)
    if declared != labels_snapshot:
        raise ValueError(
            f"{labels_path}: labels_manifest declares {declared!r}, expected "
            f"{labels_snapshot!r}; labels must be built on the indexed snapshot"
        )
    if (
        manifest["index_snapshot"] != labels_snapshot
        or not manifest["snapshot_report_sha256"]
    ):
        raise ValueError(
            "Index/labels snapshot mismatch or missing snapshot report; configure CHROMA_DB_PATH and matching --labels/--labels-snapshot before paid execution"
        )
    from langsmith import Client

    client = Client()
    remote = {
        str(e.id): e
        for e in client.list_examples(dataset_name=DATASET_NAME, splits=[split])
    }
    examples = []
    for e in selected:
        old = remote.get(str(e.id))
        if old is None or old.inputs != e.inputs or old.outputs != e.outputs:
            raise ValueError(
                f"LangSmith dataset is stale for {e.metadata['case_id']}; review and upsert explicitly before running"
            )
        examples.append(
            old.model_copy(
                update={
                    "metadata": e.metadata,
                    "inputs": {
                        **e.inputs,
                        "case_id": e.metadata["case_id"],
                        "example_id": str(e.id),
                    },
                }
            )
        )
    manifest = make_manifest(
        examples,
        run_id=run_id,
        split=split,
        dry_run=False,
        skip_judge=skip_judge,
        git_sha=_git_sha(),
        labels_snapshot=labels_snapshot,
        labels_path=labels_path,
    )
    sink = Artifacts(
        Path(output_dir) if output_dir else Path("eval/runs") / run_id, manifest
    )
    usages = {}
    checkpoints = {}
    cancelled = threading.Event()
    # Initial summary survives setup/provider errors before the SDK yields a row.
    consume_results([], sink, examples, usages, skip_judge=skip_judge)

    def checkpoint(row):
        identity = {k: row["inputs"].get(k) for k in ("case_id", "example_id")}
        sink.append("outputs", {**identity, **row})
        checkpoints[identity["example_id"]] = row.get("outputs") or {
            k: v for k, v in row.items() if k != "inputs"
        }

    evaluators = _wrapped_evaluators(
        examples, sink, usages, skip_judge=skip_judge, cancelled=cancelled
    )
    target = make_target(run_id=run_id, checkpoint=checkpoint)

    def guarded_target(inputs):
        if cancelled.is_set():
            raise InterruptedError("Evaluation cancelled")
        return target(inputs)

    results = client.evaluate(
        guarded_target,
        data=examples,
        evaluators=evaluators,
        experiment_prefix=prefix,
        max_concurrency=1,
        blocking=False,
        error_handling="log",
        metadata={
            "run_id": run_id,
            "git_sha": manifest["git_sha"],
            "index_snapshot": manifest["index_snapshot"],
            "effort": manifest["settings"]["OPENROUTER_REASONING_EFFORT"],
            "simple_effort": manifest["settings"]["SIMPLE_REASONING_EFFORT"],
        },
    )
    summary = consume_results(
        results,
        sink,
        examples,
        usages,
        skip_judge=skip_judge,
        checkpoints=checkpoints,
        cancelled=cancelled,
        require_parent_context=manifest["v7"]["PARENT_CONTEXT_ENABLE"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"experiment: {results.experiment_name}\nurl: {results.url}\nartifacts: {sink.directory}"
    )
    return results.experiment_name


def rejudge(experiment, *, output_dir=None, current_references=False):
    """Judge stored runs against pinned references; never build a RAG target."""
    from langsmith import Client

    client = Client()
    project = client.read_project(project_name=experiment)
    version = (project.metadata or {}).get("dataset_version")
    if not version:
        raise ValueError(
            "Experiment has no frozen dataset_version; cannot safely rejudge"
        )
    runs = list(client.list_runs(project_name=experiment, is_root=True))
    ids = {str(r.reference_example_id) for r in runs}
    original = {
        str(e.id): e
        for e in client.list_examples(
            dataset_id=project.reference_dataset_id, as_of=version
        )
    }
    if ids - original.keys():
        raise ValueError("Original reference missing for a stored run")
    if len(ids) != len(runs):
        raise ValueError(
            "Repeated examples require a per-repetition rejudge selection; refusing ambiguous accounting"
        )
    references = original
    if current_references:
        references = {
            str(e.id): e
            for e in client.list_examples(dataset_id=project.reference_dataset_id)
        }
        if ids - references.keys():
            raise ValueError("Current reference missing for a stored run")
    examples = [references[eid] for eid in sorted(ids)]
    run_id = _new_id("rejudge")
    manifest = make_manifest(
        examples,
        run_id=run_id,
        split=SPLIT_VERIFIED,
        dry_run=False,
        skip_judge=False,
        git_sha=_git_sha(),
        labels_snapshot=LABELS_SNAPSHOT,
        labels_path=DEFAULT_LABELS,
    )
    manifest.update(
        {
            "mode": "rejudge",
            "source_experiment": experiment,
            "source_dataset_version": version,
            "reference_mode": "current" if current_references else "original",
            "source_run_ids": [str(r.id) for r in runs],
            "source_index_snapshot": (project.metadata or {}).get("index_snapshot"),
            "pipeline_cost_reused": True,
            "index_snapshot": (project.metadata or {}).get("index_snapshot"),
            "pipeline_settings_used": False,
            "reference_csv_hashes": sorted(
                {(e.metadata or {}).get("csv_sha256", "unknown") for e in examples}
            ),
            "reference_labels_hashes": sorted(
                {(e.metadata or {}).get("labels_sha256", "unknown") for e in examples}
            ),
        }
    )
    sink = Artifacts(
        Path(output_dir) if output_dir else Path("eval/runs") / run_id, manifest
    )
    usages = {}
    cancelled = threading.Event()
    consume_results([], sink, examples, usages, reuse_pipeline=True)
    results = client.evaluate(
        experiment,
        evaluators=_wrapped_evaluators(
            examples,
            sink,
            usages,
            skip_judge=False,
            cancelled=cancelled,
            only_judges=True,
        ),
        max_concurrency=1,
        blocking=False,
    )
    summary = consume_results(
        results, sink, examples, usages, reuse_pipeline=True, cancelled=cancelled
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return run_id


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Unified verified evaluation (paid except --dry-run)"
    )
    parser.add_argument("--split", default=SPLIT_VERIFIED)
    parser.add_argument("--prefix", default="unified")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--paid", action="store_true", help="required for any run that calls models"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cases", help="comma-separated IDs, applied before --limit")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--labels-snapshot", default=LABELS_SNAPSHOT)
    parser.add_argument("--rejudge", metavar="EXPERIMENT")
    parser.add_argument(
        "--current-references",
        action="store_true",
        help="explicitly use current references, joined by UUID",
    )
    args = parser.parse_args(argv)
    if args.paid == args.dry_run:
        parser.error("choose exactly one of --dry-run (free) or --paid")
    if args.rejudge:
        if args.dry_run:
            parser.error(
                "--dry-run selects local cases; it cannot inspect a remote experiment"
            )
        return rejudge(
            args.rejudge,
            output_dir=args.output_dir,
            current_references=args.current_references,
        )
    if args.current_references:
        parser.error("--current-references requires --rejudge")
    return run_experiment(
        args.split,
        args.prefix,
        args.skip_judge,
        args.limit,
        args.cases.split(",") if args.cases else None,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        csv_path=args.csv,
        labels_path=args.labels,
        labels_snapshot=args.labels_snapshot,
    )


if __name__ == "__main__":
    main()
