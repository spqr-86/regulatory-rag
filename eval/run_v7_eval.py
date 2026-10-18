"""Historical V7 report helpers and a compatibility CLI for eval.ls.run.

Only --dry-run is free. --skip-judge still pays for RAG. --output is retired;
--output-dir selects the new manifest/JSONL/summary artifact directory.
Historical helper functions retain their original metric definitions.
"""

# ANCHOR: CLI → unified executor; historical helpers remain for saved report tools.

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.resources
import json
import re
import sys
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.infra.llm_factory import (  # noqa: E402
    apply_ipv6_patch_for_googleapis,
)

apply_ipv6_patch_for_googleapis()

from eval.pricing import cost_for_usages, percentile  # noqa: E402
from src.v7.config import v7_config  # noqa: E402
from src.v7.runner import run_query as run_with_telemetry  # noqa: E402
from utils.logging import configure_logging  # noqa: E402

_REFUSAL_RE = re.compile(r"\bнет\b|не могу")

configure_logging()

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent.parent / "tests" / "dataset.csv"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent
    / "benchmarks"
    / f"eval_v7_{date.today().isoformat()}.jsonl"
)

# False-sufficiency: "simple" path answer with correctness < this threshold → false positive
FALSE_SUFFICIENCY_THRESHOLD = 5.0  # out of 10


# ── Dataset ───────────────────────────────────────────────────────────────────


class CorrectnessPromptVariables(BaseModel):
    question: str
    legal_as_of: str
    ground_truth: str
    forbidden_claims: str
    answer: str


def correctness_prompt_text() -> str:
    return (
        importlib.resources.files("eval")
        .joinpath("prompts/golden_correctness.md")
        .read_text(encoding="utf-8")
    )


def correctness_prompt_sha256() -> str:
    return hashlib.sha256(correctness_prompt_text().encode("utf-8")).hexdigest()


def load_dataset(path: Path) -> list[dict[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "case_id",
            "question",
            "ground_truth",
            "reference_status",
            "legal_as_of",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Golden dataset missing columns: {', '.join(sorted(missing))}"
            )
        seen_ids: set[str] = set()
        for row in reader:
            q = row.get("question", "").strip()
            gt = row.get("ground_truth", "").strip()
            if q and gt:
                case_id = (row.get("case_id") or "").strip()
                status = (row.get("reference_status") or "").strip()
                if not case_id or case_id in seen_ids:
                    raise ValueError(
                        f"Missing or duplicate golden case_id: {case_id!r}"
                    )
                if status not in {
                    "verified",
                    "incorrect",
                    "needs_clarification",
                    "expert_candidate",
                }:
                    raise ValueError(
                        f"Invalid reference_status for {case_id}: {status!r}"
                    )
                seen_ids.add(case_id)
                # Unreviewed expert answers are kept for review, not for paid runs.
                if status == "expert_candidate":
                    continue
                legal_as_of = (row.get("legal_as_of") or "").strip()
                if not legal_as_of:
                    raise ValueError(f"Missing legal_as_of for {case_id}")
                seen_ids.add(case_id)
                rows.append(
                    {
                        "case_id": case_id,
                        "question": q,
                        "ground_truth": gt,
                        "reference_status": status,
                        "reviewed_at": (row.get("reviewed_at") or "").strip(),
                        "legal_as_of": legal_as_of,
                        "corpus_support": (
                            row.get("corpus_support") or "unverified"
                        ).strip(),
                        "forbidden_claims": (row.get("forbidden_claims") or "").strip(),
                        "oos_type": (row.get("oos_type") or "").strip(),
                        "must_not_contain": (row.get("must_not_contain") or "").strip(),
                    }
                )
    return rows


def scorable_in_scope(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only reviewed legal references may contribute to normative correctness."""
    return [
        row
        for row in results
        if row.get("reference_status") == "verified"
        and not row.get("oos_type")
        and row.get("answer")
        and row.get("correctness_score") is not None
        and "error" not in row
        and not row.get("generation_failed")
    ]


# ── Graph runner ──────────────────────────────────────────────────────────────


def new_run_id() -> str:
    """The id every row of one eval run shares (issue #18).

    Sortable by time so runs line up in the journal by themselves, with a short
    random tail because two runs can start inside the same second.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval-{stamp}-{uuid.uuid4().hex[:4]}"


def run_query(
    graph, question: str, writer=None, run_id: str | None = None
) -> dict[str, Any]:
    """Run one question through V7 graph, return structured result.

    Goes through the telemetry runner so an eval run lands in the same table as
    live traffic, separated only by ``source`` (monitoring module 05) — and, among
    eval rows, by ``run_id`` (issue #18).
    """
    start = time.time()
    state, _query_id = run_with_telemetry(
        graph, question, source="eval", writer=writer, run_id=run_id
    )
    elapsed = round(time.time() - start, 2)

    answer = state.get("answer", "")
    final_passages = state.get("final_passages") or []
    retrieval_attempts = state.get("retrieval_attempts") or []

    # Determine path taken: simple or complex
    stages = [a.get("stage", "unknown") for a in retrieval_attempts]
    path = "complex" if "complex" in stages else "simple"

    # Build context string from retrieved passages
    context = "\n\n".join(p.get("text", "") for p in final_passages if p.get("text"))

    # Token usage carried up from the pipeline (src/v7/usage.py). Priced here:
    # the pipeline counts tokens, the runner counts dollars.
    usages = state.get("llm_usage") or []
    priced = cost_for_usages(usages)

    return {
        "generation_failed": state.get("generation_failed", False),
        "generation_error": state.get("generation_error") or {},
        "route_decision": state.get("route_decision", ""),
        "route_reason": state.get("route_reason", ""),
        "abstain_reason": state.get("abstain_reason", ""),
        "technical_failure": state.get("technical_failure", False),
        "usage_unknown": any(u.get("usage_unknown") for u in usages),
        "answer": answer,
        "context": context,
        "path": path,
        "elapsed_sec": elapsed,
        "retrieval_attempts": len(retrieval_attempts),
        "llm_calls": len(usages),
        # Per-call breakdown: the cost figure has to be checkable — which model,
        # which node, how many tokens — not taken on the summary's word.
        "usage": usages,
        **(
            {
                "parent_context_stats": state.get("parent_context_stats", []),
                "parent_truncated": sum(
                    s.get("truncated", 0) for s in state.get("parent_context_stats", [])
                ),
                "parent_slots_lost": sum(
                    s.get("slots_lost", 0)
                    for s in state.get("parent_context_stats", [])
                ),
            }
            if v7_config.PARENT_CONTEXT_ENABLE
            else {}
        ),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
        "reasoning_tokens": sum(u.get("reasoning_tokens", 0) for u in usages),
        "providers": sorted({u["provider"] for u in usages if u.get("provider")}),
        "cost_usd": priced["cost_usd"],
        "unpriced_models": priced["unpriced_models"],
    }


_RECORD_FIELDS = (
    "generation_failed",
    "generation_error",
    "route_decision",
    "route_reason",
    "abstain_reason",
    "technical_failure",
    "usage_unknown",
    "elapsed_sec",
    "retrieval_attempts",
    "llm_calls",
    "usage",
    "parent_context_stats",
    "parent_truncated",
    "parent_slots_lost",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "providers",
    "cost_usd",
    "unpriced_models",
)


def record_fields(run_result: dict[str, Any]) -> dict[str, Any]:
    """Per-question telemetry copied from run_query into the result record.

    One list for both record shapes (with and without judge), so a field added
    to run_query cannot silently miss the summary again.
    """
    return {key: run_result[key] for key in _RECORD_FIELDS if key in run_result}


def summarize_parents(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum replacement-stage diagnostics across all packing passes and questions."""
    stats = [s for row in results for s in row.get("parent_context_stats", [])]
    reasons: dict[str, int] = {}
    for item in stats:
        reason = item.get("unavailable_reason")
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "parent_replaced_total": sum(s.get("replaced", 0) for s in stats),
        "parent_children_total": sum(s.get("children", 0) for s in stats),
        "parent_deduplicated_total": sum(s.get("deduplicated", 0) for s in stats),
        "parent_truncated_total": sum(s.get("truncated", 0) for s in stats),
        "parent_slots_lost_total": sum(s.get("slots_lost", 0) for s in stats),
        "parent_unavailable_reasons": reasons,
    }


def summarize_cost(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Cost and latency summary for a run, split by retrieval path.

    Split is mandatory: complex costs an order of magnitude more than simple
    and runs ~24% of queries, so one mean hides what we actually pay for.
    Latency goes out as p50/p95 — the CrossEncoder adds seconds to a minority
    of queries and a mean smears that.
    """

    def _block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        costs = [r.get("cost_usd", 0.0) for r in rows]
        latencies = [r.get("elapsed_sec", 0.0) for r in rows]
        return {
            "queries": n,
            "total_cost_usd": sum(costs),
            "mean_cost_usd": (sum(costs) / n) if n else 0.0,
            "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in rows),
            "completion_tokens": sum(r.get("completion_tokens", 0) for r in rows),
            "reasoning_tokens": sum(r.get("reasoning_tokens", 0) for r in rows),
            "latency_p50_sec": percentile(latencies, 50),
            "latency_p95_sec": percentile(latencies, 95),
        }

    summary = _block(results)
    summary["usage_unknown_queries"] = sum(
        bool(r.get("usage_unknown")) for r in results
    )
    summary["cost_incomplete"] = bool(summary["usage_unknown_queries"])

    by_path: dict[str, Any] = {}
    for path in sorted({r.get("path", "unknown") for r in results}):
        by_path[path] = _block([r for r in results if r.get("path") == path])
    summary["by_path"] = by_path

    unpriced: list[str] = []
    for r in results:
        for model in r.get("unpriced_models", []) or []:
            if model not in unpriced:
                unpriced.append(model)
    summary["unpriced_models"] = unpriced

    return summary


# ── Correctness judge ─────────────────────────────────────────────────────────


def is_oos_rejection(answer: str) -> bool:
    """Empty answer (domain gate) or a refusal marker near the start of the answer."""
    return not answer or bool(_REFUSAL_RE.search(answer.lower()[:200]))


def oos_rejection_rate(results: list[dict[str, Any]]) -> float:
    """Share of out-of-scope questions the pipeline refused, over all OOS rows.

    Rows with an empty answer stay in the denominator: that is how the domain
    gate refuses. Graph errors (no ``answer`` key) are not refusals.
    """
    oos = [r for r in results if r.get("oos_type") == "out_of_scope" and "answer" in r]
    if not oos:
        return 0.0
    return sum(
        1
        for r in oos
        if not r.get("generation_failed") and is_oos_rejection(r["answer"])
    ) / len(oos)


def evaluate_correctness(
    question: str,
    ground_truth: str,
    answer: str,
    llm,
    forbidden_claims: str = "",
    legal_as_of: str = "",
) -> dict[str, Any]:
    """Judge answer against a reviewed key and conditional forbidden assertions."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(correctness_prompt_text())
    variables = CorrectnessPromptVariables(
        question=question,
        legal_as_of=legal_as_of,
        ground_truth=ground_truth,
        forbidden_claims=forbidden_claims,
        answer=answer,
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(variables.model_dump())
    raw = response.strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw.removeprefix("```json").removesuffix("```").strip()
    data = json.loads(raw)
    score = float(data["score"])
    if not 0 <= score <= 10:
        raise ValueError(f"Judge score out of range: {score}")
    asserted = data.get("forbidden_claims_asserted", [])
    if not isinstance(asserted, list) or any(not isinstance(x, str) for x in asserted):
        raise ValueError("Judge returned invalid forbidden_claims_asserted")
    return {
        "correctness_score": score,
        "correctness_reasoning": str(data.get("reasoning", "")),
        "forbidden_claims_asserted": asserted,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def run(
    limit: int | None = None,
    output: Path | None = None,
    skip_judge: bool = False,
    *,
    dry_run: bool = False,
    output_dir: Path | None = None,
) -> str:
    """Compatibility entry to eval.ls.run; historical helpers remain importable."""
    from eval.ls.run import run_experiment

    if output is not None:
        raise ValueError(
            "Legacy --output JSON format is retired. Use --output-dir DIR for "
            "manifest.json, results.jsonl and summary.json; historical files stay unchanged."
        )
    print("Deprecated entry point: use python -m eval.ls.run")
    return run_experiment(
        split="verified",
        prefix="unified",
        limit=limit,
        skip_judge=skip_judge,
        dry_run=dry_run,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compatibility CLI for unified evaluation"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, help="retired format; use --output-dir")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--skip-judge", action="store_true", help="skip judges; RAG still incurs cost"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output is not None:
        parser.error("--output format is retired; use --output-dir DIR")
    run(
        limit=args.limit,
        skip_judge=args.skip_judge,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )
