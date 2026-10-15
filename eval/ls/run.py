"""Run a LangSmith experiment: v7 graph over the golden dataset, scored by code evaluators.

CLI:
    python -m eval.ls.run --split verified --prefix v7-low --skip-judge [--limit N]

Paid: every example goes through the real graph (LLM + embeddings).
"""

from __future__ import annotations

import argparse
import json
import subprocess

from eval.ls.dataset import DATASET_NAME, SPLIT_VERIFIED
from eval.ls.evaluators import CODE_EVALUATORS, summarize, summarize_cost
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


def run_experiment(split: str, prefix: str, skip_judge: bool, limit: int | None) -> str:
    """Run the experiment, print the summary and return the experiment name."""
    if not skip_judge:
        raise NotImplementedError("the correctness judge is not wired in yet")

    from dotenv import load_dotenv
    from langsmith import Client

    from config.settings import settings

    load_dotenv()
    client = Client()
    examples = list(
        client.list_examples(dataset_name=DATASET_NAME, splits=[split], limit=limit)
    )
    results = client.evaluate(
        make_target(),
        data=examples,
        evaluators=CODE_EVALUATORS,
        experiment_prefix=prefix,
        max_concurrency=1,
        error_handling="log",
        metadata={
            "effort": settings.OPENROUTER_REASONING_EFFORT,
            "git_sha": _git_sha(),
        },
    )

    rows = [
        {
            r.key: {"score": r.score, "comment": r.comment}
            for r in row["evaluation_results"]["results"]
        }
        for row in results
    ]
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))
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
    parser.add_argument("--prefix", required=True, help="experiment name prefix")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="first N examples, for a smoke run; OOS come first, use 6+ to reach in-scope",
    )
    args = parser.parse_args()
    run_experiment(args.split, args.prefix, args.skip_judge, args.limit)


if __name__ == "__main__":
    main()
