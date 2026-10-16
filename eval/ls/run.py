"""Run a LangSmith experiment: v7 graph over the golden dataset, scored by code evaluators
and the LLM judges (correctness, citations).

CLI:
    python -m eval.ls.run --split verified --prefix v7-low [--skip-judge] [--limit N]
    python -m eval.ls.run --rejudge <experiment>   # judges only, on stored answers

Paid: every example goes through the real graph (LLM + embeddings), and the judges
call their models. --rejudge pays only for the judges. Experiments run before 15.10
stored no context: their citations score None ("no_context").
"""

# ANCHOR: Run/rejudge LangSmith experiments serially; metadata records shared and
# effective simple effort so low vs disabled experiments are reproducible.

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from eval.ls.dataset import DATASET_NAME, SPLIT_VERIFIED
from eval.ls.evaluators import CODE_EVALUATORS, summarize, summarize_cost
from eval.ls.judge import make_citations_evaluator, make_correctness_evaluator
from eval.ls.target import make_target


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def judge_evaluators() -> list:
    """Both LLM judges with their production models (paid on every call)."""
    from src.infra.llm_factory import get_citation_judge_llm, get_judge_llm

    return [
        # Same settings as the golden judge in run_v7_eval.
        make_correctness_evaluator(get_judge_llm(temperature=0.0, seed=12345)),
        make_citations_evaluator(get_citation_judge_llm(temperature=0.0)),
    ]


def with_current_references(evaluators: list, current: dict[str, dict]) -> list:
    """Evaluators scored against the current references, keyed by question.

    LangSmith pins an existing experiment to the dataset version it ran on, so a
    rejudge after editing references silently used the old ones (16.10).
    """

    def wrap(evaluator):
        def evaluator_with_current_reference(inputs, outputs, reference_outputs):
            reference = current[inputs["question"].strip()]
            return evaluator(inputs, outputs, reference)

        evaluator_with_current_reference.__name__ = getattr(
            evaluator, "__name__", "evaluator"
        )
        return evaluator_with_current_reference

    return [wrap(e) for e in evaluators]


def _print_scores(results) -> None:
    rows = [
        {
            r.key: {"score": r.score, "comment": r.comment}
            for r in row["evaluation_results"]["results"]
        }
        for row in results
    ]
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


def rejudge(experiment: str) -> None:
    """Run the judges on the answers of an existing experiment; no generation."""
    from dotenv import load_dotenv
    from langsmith import Client

    load_dotenv()
    client = Client()
    current = {
        e.inputs["question"].strip(): e.outputs
        for e in client.list_examples(dataset_name=DATASET_NAME)
    }
    results = client.evaluate(
        experiment,
        evaluators=with_current_references(judge_evaluators(), current),
        max_concurrency=1,
    )
    _print_scores(results)
    print(f"experiment: {experiment}")


def run_experiment(split: str, prefix: str, skip_judge: bool, limit: int | None) -> str:
    """Run the experiment, print the summary and return the experiment name."""
    from dotenv import load_dotenv
    from langsmith import Client

    from config.settings import settings

    load_dotenv()
    client = Client()
    examples = list(
        client.list_examples(dataset_name=DATASET_NAME, splits=[split], limit=limit)
    )
    evaluators = list(CODE_EVALUATORS)
    if not skip_judge:
        evaluators += judge_evaluators()
    results = client.evaluate(
        make_target(),
        data=examples,
        evaluators=evaluators,
        experiment_prefix=prefix,
        max_concurrency=1,
        error_handling="log",
        metadata={
            "effort": settings.OPENROUTER_REASONING_EFFORT,
            "simple_effort": (
                settings.SIMPLE_REASONING_EFFORT
                if settings.SIMPLE_REASONING_EFFORT is not None
                else settings.OPENROUTER_REASONING_EFFORT
            ),
            "git_sha": _git_sha(),
            # Chunk ids in the labels are tied to the index snapshot.
            "index_snapshot": Path(settings.CHROMA_DB_PATH).resolve().name,
        },
    )

    _print_scores(results)
    # A failed example has no outputs.
    outputs = [row["run"].outputs or {} for row in results]
    print(json.dumps({"cost": summarize_cost(outputs)}, ensure_ascii=False, indent=2))
    # Failed examples are scored None by every evaluator and drop out of the rates.
    errors = [
        row["example"].metadata.get("case_id") for row in results if row["run"].error
    ]
    print(f"errors: {len(errors)} {errors}")
    print(f"experiment: {results.experiment_name}")
    print(f"url: {results.url}")
    return results.experiment_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LangSmith experiment on v7")
    parser.add_argument("--split", default=SPLIT_VERIFIED)
    parser.add_argument("--prefix", help="experiment name prefix (required to run)")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--rejudge",
        metavar="EXPERIMENT",
        help="run only the judges on an existing experiment's answers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="first N examples, for a smoke run; OOS come first, use 6+ to reach in-scope",
    )
    args = parser.parse_args()
    if args.rejudge:
        rejudge(args.rejudge)
        return
    if not args.prefix:
        parser.error("--prefix is required unless --rejudge is given")
    run_experiment(args.split, args.prefix, args.skip_judge, args.limit)


if __name__ == "__main__":
    main()
