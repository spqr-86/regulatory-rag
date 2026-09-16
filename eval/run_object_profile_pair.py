"""Paired v1/v2 object profile run: one mode per process (spec object-profile §5.2).

# ANCHOR: paired object profile run
# Role: run the department Q&A stack over eval/data/object_profile_pair_expectations.yaml
#   in one mode (v1 or v2, picked by DEPARTMENT_QA_MODE), logging every call in full for
#   later comparison against the hand-written expectations.
# Input: expectations file (n, unit_id, question); DepartmentStack from
#   build_department_stack(recorder=...).
# Output: eval/runs/<out>/<mode>/q<N>.json per question (prompt, raw model output incl.
#   schema retries, search calls with raw hits, evidence ids in prompt order,
#   DepartmentResponse) and
#   eval/runs/<out>/<mode>/config.json (run configuration, no secrets).
#
# The Chroma store and BM25 index are process-wide singletons taken from settings,
# so each mode runs in its own process with its own CHROMA_DB_PATH/collection.
# Every call is logged in full: prompt, raw model output (all attempts), search calls,
# ids of the evidence given to the prompt, DepartmentResponse; plus the run configuration.
# Runs recorded before Issue #47 store raw search hits under `evidence_passed`.

Usage::

    RUN=eval/runs/object_profile_pair_2026-09-DD
    DEPARTMENT_QA_MODE=v1 CHROMA_DB_PATH=./chroma_db_dept CHROMA_COLLECTION_NAME=department_demo \\
      CORPUS_MANIFEST_PATH=corpus/manifest.yaml SOURCE_DOCS_PATH=./source_docs_dept \\
      .venv/bin/python eval/run_object_profile_pair.py --out $RUN
    DEPARTMENT_QA_MODE=v2 CHROMA_DB_PATH=./chroma_db_dept_v2 CHROMA_COLLECTION_NAME=department_demo_v2 \\
      CORPUS_MANIFEST_PATH=corpus/manifest.yaml SOURCE_DOCS_PATH=./source_docs_dept \\
      .venv/bin/python eval/run_object_profile_pair.py --out $RUN

Paid: 9 calls per mode (+ at most 9 schema retries). Budget agreed 15.09: < $0.05 for 18 calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

EXPECTATIONS = REPO_ROOT / "eval" / "data" / "object_profile_pair_expectations.yaml"


PAID_QUESTION_COUNT = 9  # spec §5.2: 9 questions × 2 modes = 18 calls, budget < $0.05


DEFAULT_ALLOWED_MODELS = frozenset({"gpt-4o-mini", "openai/gpt-4o-mini"})


def check_paid_run(
    settings,
    questions: list[dict],
    mode_dir: Path,
    allowed_models=DEFAULT_ALLOWED_MODELS,
    expected_count: int = PAID_QUESTION_COUNT,
) -> None:
    """Refuse a non-dry paid run outside the agreed budget (spec §5.2, final-review #2).

    Raises ``ValueError`` before any paid call when the model, temperature or
    provider don't match the agreed run, the expectations file doesn't yield
    exactly 9 questions, or ``mode_dir`` already holds ``q*.json`` results from
    a prior run.
    """
    errors: list[str] = []
    if settings.SIMPLE_MODEL_NAME not in allowed_models:
        errors.append(
            f"model must be one of {sorted(allowed_models)} (default gpt-4o-mini, "
            f"spec §5.2), got {settings.SIMPLE_MODEL_NAME!r}"
        )
    if settings.TEMPERATURE != 0:
        errors.append(
            f"temperature must be 0 (spec §5.2), got {settings.TEMPERATURE!r}"
        )
    if settings.SIMPLE_LLM_PROVIDER not in {"openai", "openrouter"}:
        errors.append(
            "provider must be openai or openrouter (spec §5.2), "
            f"got {settings.SIMPLE_LLM_PROVIDER!r}"
        )
    if len(questions) != expected_count:
        errors.append(
            f"expectations must yield exactly {expected_count} questions "
            f"(spec §5.2 default {PAID_QUESTION_COUNT}), got {len(questions)}"
        )
    mode_dir = Path(mode_dir)
    if mode_dir.exists():
        existing = sorted(p.name for p in mode_dir.glob("q*.json"))
        if existing:
            errors.append(
                f"{mode_dir} already has results ({', '.join(existing)}); "
                "refusing to re-spend into an existing run"
            )
    if errors:
        raise ValueError("; ".join(errors))


def load_questions(path: Path) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [
        {"n": q["n"], "unit_id": q.get("unit_id"), "question": q["question"]}
        for q in data.get("questions") or []
    ]


def select_questions(questions: list[dict], only: str) -> list[dict]:
    """Keep questions whose ``n`` is listed in ``only`` ("1,2"), in file order."""
    wanted = {int(x) for x in only.split(",") if x.strip()}
    unknown = wanted - {q["n"] for q in questions}
    if unknown:
        raise ValueError(f"unknown question numbers: {sorted(unknown)}")
    return [q for q in questions if q["n"] in wanted]


_PROMPT_EVIDENCE_ID = re.compile(r"^\[((?:ext|int|obj)_\w+)\]", re.MULTILINE)


def prompt_evidence_ids(prompt: str) -> list[str]:
    """Ids of the evidence blocks in the rendered prompt, in prompt order (spec §5.2)."""
    return _PROMPT_EVIDENCE_ID.findall(prompt)


def run_mode(
    stack,
    questions: list[dict],
    out_dir: Path,
    raw_log: list[dict],
    verifier_log: list[dict] | None = None,
) -> list[dict]:
    from src.department_qa.service import answer_question

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for q in questions:
        prompts: list[str] = []
        verify_prompts: list[str] = []
        raw_log.clear()
        if verifier_log is not None:
            verifier_log.clear()

        def model_fn(prompt: str, _prompts=prompts):
            _prompts.append(prompt)
            return stack.model_fn(prompt)

        stack_verifier = getattr(stack, "verifier_fn", None)

        def verifier_fn(prompt: str, _prompts=verify_prompts):
            _prompts.append(prompt)
            return stack_verifier(prompt)

        profile = stack.config.profiles.get(q["unit_id"]) if q["unit_id"] else None
        passed: list[dict] = []

        def search_fn(query, filters=None, top_k=8, _passed=passed):
            hits = stack.search_fn(query, filters=filters, top_k=top_k)
            _passed.append({"filters": filters, "hits": hits})
            return hits

        response = answer_question(
            q["question"],
            q["unit_id"],
            search_fn,
            model_fn,
            snapshot_id=stack.manifest.snapshot_id,
            profile=profile,
            prompt_version=stack.config.prompt_version,
            verifier_fn=verifier_fn if stack_verifier is not None else None,
        )
        record = {
            "n": q["n"],
            "mode": stack.config.mode,
            "unit_id": q["unit_id"],
            "question": q["question"],
            "prompt": prompts[0] if prompts else "",
            "prompt_sha256": (
                hashlib.sha256(prompts[0].encode()).hexdigest() if prompts else None
            ),
            "raw_model_output": list(raw_log),
            "verify_prompt": verify_prompts[0] if verify_prompts else "",
            "verify_prompt_sha256": (
                hashlib.sha256(verify_prompts[0].encode()).hexdigest()
                if verify_prompts
                else None
            ),
            "raw_verifier_output": list(verifier_log or []),
            "search_calls": passed,
            "evidence_ids": prompt_evidence_ids(prompts[0]) if prompts else [],
            "profile_sha256": profile.content_sha256 if profile else None,
            "response": json.loads(response.model_dump_json()),
        }
        (out_dir / f"q{q['n']}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expectations", type=Path, default=EXPECTATIONS)
    parser.add_argument(
        "--only",
        help="comma-separated question numbers for a partial paid run, e.g. 1,2",
    )
    parser.add_argument(
        "--model",
        help="explicitly allow this SIMPLE_MODEL_NAME for the paid run (agreed per run)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="configuration only, no model calls"
    )
    args = parser.parse_args()

    from config.settings import settings
    from src.department_qa.service import TOP_K_PER_LEVEL
    from src.department_qa.wiring import build_department_stack

    raw_log: list[dict] = []
    verifier_log: list[dict] = []
    stack = build_department_stack(
        recorder=raw_log.append, verifier_recorder=verifier_log.append
    )
    questions = load_questions(args.expectations)
    if args.only:
        questions = select_questions(questions, args.only)
    mode_dir = args.out / stack.config.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        try:
            check_paid_run(
                settings,
                questions,
                mode_dir,
                allowed_models=({args.model} if args.model else DEFAULT_ALLOWED_MODELS),
                expected_count=(len(questions) if args.only else PAID_QUESTION_COUNT),
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
    config = {
        "mode": stack.config.mode,
        "model": settings.SIMPLE_MODEL_NAME,
        "provider": settings.SIMPLE_LLM_PROVIDER,
        "temperature": settings.TEMPERATURE,
        "top_k_per_level": TOP_K_PER_LEVEL,
        "fusion": "rrf(vector, bm25)",
        "chroma_db_path": settings.CHROMA_DB_PATH,
        "collection": settings.CHROMA_COLLECTION_NAME,
        "chunks": sum(1 for _ in stack.store.iter_all_documents()),
        "prompt_version": stack.config.prompt_version,
        "verify_prompt_version": "v1",
        "verification_enabled": True,
        "expectations_sha256": hashlib.sha256(
            args.expectations.read_bytes()
        ).hexdigest(),
        "profiles": {u: p.content_sha256 for u, p in stack.config.profiles.items()},
        "questions": len(questions),
    }
    (mode_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    usage = {"input_tokens": 0, "output_tokens": 0}
    for r in run_mode(stack, questions, mode_dir, raw_log, verifier_log):
        print(r["n"], r["response"]["status"], r["response"]["reason_codes"])
        for call in r["raw_model_output"] + r["raw_verifier_output"]:
            for key in usage:
                usage[key] += (call.get("usage") or {}).get(key) or 0
    print("usage", json.dumps(usage))
    return 0


if __name__ == "__main__":
    sys.exit(main())
