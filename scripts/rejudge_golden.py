"""Rejudge saved answers against the reviewed generation key, without rerunning RAG.

Usage:
    .venv/bin/python scripts/rejudge_golden.py \
      benchmarks/eval_v7_deepseek_low_2026-09-23.jsonl \
      --output benchmarks/rejudge_reviewed_2026-09-30.json
    .venv/bin/python scripts/rejudge_golden.py INPUT --dry-run
"""

# ANCHOR: Select only verified in-scope references, preserve the old answer text,
# and checkpoint each judge result. Input: saved eval JSON + current CSV key.
# Output: separate JSON report with dataset hash, eligible denominator, and scores.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from eval.run_v7_eval import (
    DATASET_PATH,
    correctness_prompt_sha256,
    evaluate_correctness,
    load_dataset,
)


def selected_cases(
    saved: list[dict[str, Any]], key: list[dict[str, str]]
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    if len(saved) != len(key):
        raise ValueError(f"Saved run has {len(saved)} rows, current key has {len(key)}")
    selected = []
    for case, old in zip(key, saved, strict=True):
        if old.get("case_id") and old["case_id"] != case["case_id"]:
            raise ValueError(f"Saved case_id disagrees with key at {case['case_id']}")
        if case["reference_status"] != "verified" or case["oos_type"]:
            continue
        if old.get("question") != case["question"]:
            raise ValueError(
                f"Question changed for {case['case_id']}; cannot align saved answer"
            )
        if old.get("answer") and "error" not in old:
            selected.append((case, old))
    return selected


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    key = load_dataset(DATASET_PATH)
    selected = selected_cases(source["results"], key)
    dataset_sha256 = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()
    print(
        f"Eligible saved answers: {len(selected)}/{sum(not row['oos_type'] for row in key)} in-scope"
    )
    print(f"Dataset SHA-256: {dataset_sha256}")
    if args.dry_run:
        print("Cases:", ", ".join(case["case_id"] for case, _ in selected))
        return
    if args.output is None:
        parser.error("--output is required unless --dry-run is used")

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": source.get("run_id"),
        "source_path": str(args.input),
        "dataset_sha256": dataset_sha256,
        "correctness_prompt_sha256": correctness_prompt_sha256(),
        "judge_provider": settings.JUDGE_LLM_PROVIDER,
        "judge_model": settings.JUDGE_MODEL_NAME,
        "judge_temperature": 0.0,
        "judge_seed": 12345,
        "eligible_cases": len(selected),
        "in_scope_cases": sum(not row["oos_type"] for row in key),
        "results": [],
    }
    if args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            report.get("dataset_sha256") != dataset_sha256
            or report.get("source_run_id") != source.get("run_id")
            or report.get("correctness_prompt_sha256") != correctness_prompt_sha256()
            or report.get("judge_model") != settings.JUDGE_MODEL_NAME
        ):
            raise ValueError("Existing report belongs to another key or saved run")

    done = {row["case_id"] for row in report["results"]}
    if len(done) < len(selected):
        from src.infra.llm_factory import get_judge_llm

        judge = get_judge_llm(temperature=0.0, seed=12345)
        for case, old in selected:
            if case["case_id"] in done:
                continue
            score = evaluate_correctness(
                case["question"],
                case["ground_truth"],
                old["answer"],
                judge,
                case["forbidden_claims"],
                case["legal_as_of"],
            )
            report["results"].append(
                {
                    "case_id": case["case_id"],
                    "question": case["question"],
                    "old_score": old.get("correctness_score"),
                    **score,
                }
            )
            save_report(args.output, report)
            done.add(case["case_id"])
            print(f"{case['case_id']}: {score['correctness_score']:.1f}/10")

    scores = [r["correctness_score"] for r in report["results"]]
    print(f"Reviewed-key mean: {sum(scores) / len(scores):.2f}/10 (n={len(scores)})")
    print(
        "This is a partial, reused-answer score; it is not a new end-to-end baseline."
    )


if __name__ == "__main__":
    main()
