"""Append-only run artifacts and explicit quality/reliability denominators."""

# ANCHOR: frozen manifest + per-case checkpoints → local, inspectable run summary.

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from eval.pricing import cost_for_usages, percentile

NOT_APPLICABLE = {
    "oos",
    "in_scope",
    "false_premise_scored_by_correctness",
    "no_pattern",
    "no_forbidden_claims",
}


def encode(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)


class Artifacts:
    def __init__(self, directory: Path, manifest: dict):
        self.assessment_id = manifest.get("run_id")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f"Artifact directory must be empty: {self.directory}")
        with (self.directory / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(encode(manifest) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def append(self, name: str, record: dict) -> None:
        if name not in {"results", "outputs", "judges", "assessments"}:
            raise ValueError(f"Unknown artifact stream: {name}")
        with (self.directory / f"{name}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(encode(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self, summary: dict) -> None:
        temporary = self.directory / "summary.json.tmp"
        temporary.write_text(encode(summary) + "\n", encoding="utf-8")
        temporary.replace(self.directory / "summary.json")


def summarize_records(
    records: list[dict], *, assigned: int, include_paths=True
) -> dict:
    metrics = {}
    for row in records:
        for key, result in row.get("scores", {}).items():
            bucket = metrics.setdefault(
                key,
                {
                    "eligible": 0,
                    "scored": 0,
                    "passed": 0,
                    "failed": 0,
                    "unresolved": 0,
                    "not_applicable": 0,
                },
            )
            if (
                result.get("applicable") is False
                or result.get("comment", "") in NOT_APPLICABLE
            ):
                bucket["not_applicable"] += 1
                continue
            bucket["eligible"] += 1
            score = result.get("score")
            if score is None:
                bucket["unresolved"] += 1
            else:
                bucket["scored"] += 1
                bucket["passed" if score == 1 else "failed"] += 1
    for bucket in metrics.values():
        bucket["pass_rate"] = (
            bucket["passed"] / bucket["scored"] if bucket["scored"] else None
        )
        bucket["coverage"] = (
            bucket["scored"] / bucket["eligible"] if bucket["eligible"] else None
        )
    outputs = [r.get("outputs") or {} for r in records]
    failed = sum(
        bool(
            r.get("error")
            or o.get("generation_failed")
            or o.get("technical_failure")
            or not (o.get("answer") or "").strip()
        )
        for r, o in zip(records, outputs)
    )
    costs = [o["cost_usd"] for o in outputs if "cost_usd" in o]
    unknown = sum(
        bool(
            o.get("cost_incomplete")
            or o.get("untracked_cost_components")
            or "cost_usd" not in o
        )
        for o in outputs
    )
    latencies = [o["elapsed_sec"] for o in outputs if "elapsed_sec" in o]
    judge_usage = [u for r in records for u in r.get("judge_usage", [])]
    judge_cost = cost_for_usages(judge_usage)
    judge_incomplete = any(u.get("usage_unknown") for u in judge_usage) or bool(
        judge_cost["unpriced_models"]
    )
    summary = {
        "assigned": assigned,
        "completed": len(records),
        "pending": assigned - len(records),
        "target_failed": failed,
        "target_succeeded": len(records) - failed,
        "metrics": metrics,
        "latency_sec": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        },
        "pipeline_cost": {
            "total_usd": sum(costs),
            "incomplete_examples": unknown,
            "unpriced_models": sorted(
                {m for o in outputs for m in o.get("unpriced_models", [])}
            ),
            "untracked_components": sorted(
                {c for o in outputs for c in o.get("untracked_cost_components", [])}
            ),
        },
        "judge_cost": {**judge_cost, "incomplete": judge_incomplete},
        "total_cost_usd": sum(costs) + judge_cost["cost_usd"],
        "cost_incomplete": bool(unknown or judge_incomplete or len(records) < assigned),
        "path_counts": dict(Counter(o.get("path", "unknown") for o in outputs)),
        "route_reasons": dict(
            Counter(o.get("route_reason") for o in outputs if o.get("route_reason"))
        ),
        "abstain_reasons": dict(
            Counter(o.get("abstain_reason") for o in outputs if o.get("abstain_reason"))
        ),
        "parent_context": {
            f"parent_{key}_total": sum(o.get(f"parent_{key}", 0) for o in outputs)
            for key in (
                "replaced",
                "children_kept",
                "deduplicated",
                "truncated",
                "slots_lost",
            )
        },
        "parent_unavailable": sorted(
            {reason for o in outputs for reason in o.get("parent_unavailable", [])}
        ),
    }
    if include_paths:
        summary["by_path"] = {}
        for path in summary["path_counts"]:
            rows = [
                r
                for r in records
                if (r.get("outputs") or {}).get("path", "unknown") == path
            ]
            block = summarize_records(rows, assigned=len(rows), include_paths=False)
            summary["by_path"][path] = {
                "queries": len(rows),
                "pipeline_cost": block["pipeline_cost"],
                "latency_sec": block["latency_sec"],
            }
    return summary
