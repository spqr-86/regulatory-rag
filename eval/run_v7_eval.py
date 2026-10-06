"""Evaluation runner for V7 RAG pipeline.

Runs the golden dataset through the V7 graph and measures:
  - faithfulness       — are claims grounded in retrieved context?
  - answer_relevance   — does the answer address the question?
  - correctness        — does the answer match the ground truth? (LLM-judge, 0-10)
  - false_sufficiency_rate — % of simple-path answers that scored badly (< threshold)

Usage:
    cd /home/petr/projects/ai/regulatory-rag
    source .venv/bin/activate
    python eval/run_v7_eval.py
    python eval/run_v7_eval.py --limit 5          # quick smoke test
    python eval/run_v7_eval.py --skip-judge       # pipeline only, no LLM judge (~$0)
    python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
"""

# ANCHOR: Run the golden cases, keep reference-review status with each result,
# and aggregate normative correctness only from verified in-scope references.
# Input: versioned CSV, current RAG graph and judge. Output: JSON report with
# dataset hash, score denominator, per-case telemetry and judge findings.

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
    get_judge_llm,
)

apply_ipv6_patch_for_googleapis()

from config.settings import settings  # noqa: E402
from eval.advanced_generation_metrics import (  # noqa: E402
    evaluate_answer_relevance,
    evaluate_faithfulness,
)
from eval.pricing import cost_for_usages, percentile  # noqa: E402
from src.backends.vector_store import get_vector_store_backend  # noqa: E402
from src.v7.bridge import build_full_v7_runtime  # noqa: E402
from src.v7.graph import build_graph  # noqa: E402
from src.v7.runner import default_writer  # noqa: E402
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
                if status not in {"verified", "incorrect", "needs_clarification"}:
                    raise ValueError(
                        f"Invalid reference_status for {case_id}: {status!r}"
                    )
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
        "answer": answer,
        "context": context,
        "path": path,
        "elapsed_sec": elapsed,
        "retrieval_attempts": len(retrieval_attempts),
        "llm_calls": len(usages),
        # Per-call breakdown: the cost figure has to be checkable — which model,
        # which node, how many tokens — not taken on the summary's word.
        "usage": usages,
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
        "reasoning_tokens": sum(u.get("reasoning_tokens", 0) for u in usages),
        "providers": sorted({u["provider"] for u in usages if u.get("provider")}),
        "cost_usd": priced["cost_usd"],
        "unpriced_models": priced["unpriced_models"],
    }


_RECORD_FIELDS = (
    "elapsed_sec",
    "retrieval_attempts",
    "llm_calls",
    "usage",
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
    return {key: run_result[key] for key in _RECORD_FIELDS}


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
    return sum(1 for r in oos if is_oos_rejection(r["answer"])) / len(oos)


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
    output: Path = DEFAULT_OUTPUT,
    skip_judge: bool = False,
) -> None:
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    if limit:
        dataset = dataset[:limit]
    print(f"  {len(dataset)} questions")

    print("Initializing V7 graph...")
    vector_store = get_vector_store_backend(load_existing=True)
    runtime = build_full_v7_runtime(vector_store)
    graph = build_graph(runtime=runtime).compile()
    telemetry_writer = default_writer()
    # One id for the whole run: it ties the rows in the journal to this file.
    run_id = new_run_id()
    print("  Graph ready.")

    judge_llm = None
    if skip_judge:
        print(
            "  [--skip-judge] LLM judge disabled; generation and embeddings may still incur cost.\n"
        )
    else:
        print("Loading judge LLM...")
        # seed makes OpenAI judging best-effort reproducible (cuts run-to-run noise).
        judge_llm = get_judge_llm(temperature=0.0, seed=12345)
        print("  Judge ready.\n")

    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        case_id = item["case_id"]
        ground_truth = item["ground_truth"]
        oos_type = item.get("oos_type", "")
        reference_status = item["reference_status"]
        print(f"[{i}/{len(dataset)}] {case_id}: {question[:70]}...")

        # Run graph
        try:
            run_result = run_query(
                graph, question, writer=telemetry_writer, run_id=run_id
            )
        except Exception as e:
            print(f"  ERROR running graph: {e}")
            results.append({"case_id": case_id, "question": question, "error": str(e)})
            continue

        answer = run_result["answer"]
        context = run_result["context"]
        path = run_result["path"]

        if not answer:
            print(f"  WARNING: empty answer (path={path})")
            results.append(
                {
                    "case_id": case_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": "",
                    "path": path,
                    "oos_type": oos_type,
                    "reference_status": reference_status,
                    "legal_as_of": item["legal_as_of"],
                    "corpus_support": item["corpus_support"],
                    "error": "empty answer",
                }
            )
            continue

        if skip_judge:
            record = {
                "case_id": case_id,
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "path": path,
                "oos_type": oos_type,
                "reference_status": reference_status,
                "legal_as_of": item["legal_as_of"],
                "corpus_support": item["corpus_support"],
                **record_fields(run_result),
            }
            results.append(record)
            print(
                f"  path={path} | elapsed={run_result['elapsed_sec']:.1f}s | "
                f"${run_result['cost_usd']:.5f}"
            )
            continue

        # Evaluate with LLM judge
        try:
            faithfulness = evaluate_faithfulness(question, context, answer, judge_llm)
        except Exception as e:
            faithfulness = {"faithfulness_score": 0.0, "faithfulness_reasoning": str(e)}

        try:
            relevance = evaluate_answer_relevance(question, answer, judge_llm)
        except Exception as e:
            print(f"  WARNING: relevance eval failed: {e}")
            relevance = {"answer_relevance_score": 0.0}

        if reference_status == "verified" and not oos_type:
            try:
                correctness = evaluate_correctness(
                    question,
                    ground_truth,
                    answer,
                    judge_llm,
                    item["forbidden_claims"],
                    item["legal_as_of"],
                )
            except Exception as e:
                correctness = {
                    "correctness_score": None,
                    "correctness_reasoning": str(e),
                }
        else:
            correctness = {
                "correctness_score": None,
                "correctness_reasoning": "reference not eligible for normative scoring",
            }

        record = {
            "case_id": case_id,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "path": path,
            **record_fields(run_result),
            "oos_type": oos_type,
            "reference_status": reference_status,
            "legal_as_of": item["legal_as_of"],
            "corpus_support": item["corpus_support"],
            "forbidden_claims": item["forbidden_claims"],
            **faithfulness,
            **relevance,
            **correctness,
        }
        results.append(record)

        print(
            f"  path={path} | "
            f"faith={faithfulness.get('faithfulness_score', 0):.2f} | "
            f"rel={relevance.get('answer_relevance_score', 0):.2f} | "
            f"correct={correctness['correctness_score'] if correctness['correctness_score'] is not None else 'unscored'}"
        )

    # Aggregate
    valid = [r for r in results if "error" not in r and r.get("answer")]
    n = len(valid)

    if n == 0:
        print("\nNo valid results to aggregate.")
        return

    avg_elapsed = sum(r.get("elapsed_sec", 0) for r in valid) / n
    complex_rate = sum(1 for r in valid if r.get("path") == "complex") / n
    abstain_count = sum(
        1
        for r in valid
        if not r.get("answer") or r.get("answer", "").startswith("Не могу")
    )

    cost_summary = summarize_cost(valid)

    aggregate: dict[str, Any] = {
        "complex_path_rate": round(complex_rate, 3),
        "mean_elapsed_sec": round(avg_elapsed, 2),
        "answered": n,
        "abstained": abstain_count,
        "cost": cost_summary,
    }

    if not skip_judge:
        # Split: in-scope vs OOS
        in_scope = scorable_in_scope(valid)
        n_in_scope = len(in_scope)
        scored = [r for r in valid if r.get("correctness_score") is not None]

        avg_faith = sum(r.get("faithfulness_score", 0) for r in valid) / n
        avg_rel = sum(r.get("answer_relevance_score", 0) for r in valid) / n
        avg_correct = (
            sum(r["correctness_score"] for r in scored) / len(scored)
            if scored
            else None
        )
        # In-scope only correctness (excludes OOS noise)
        avg_correct_inscope = (
            sum(r["correctness_score"] for r in in_scope) / n_in_scope
            if in_scope
            else None
        )

        simple_path = [r for r in in_scope if r.get("path") == "simple"]
        false_sufficiency_cases = [
            r
            for r in simple_path
            if r["correctness_score"] < FALSE_SUFFICIENCY_THRESHOLD
        ]
        false_sufficiency_rate = (
            len(false_sufficiency_cases) / len(simple_path) if simple_path else 0.0
        )
        aggregate.update(
            {
                "faithfulness": round(avg_faith, 3),
                "answer_relevance": round(avg_rel, 3),
                "correctness_mean": round(avg_correct, 2)
                if avg_correct is not None
                else None,
                "correctness_inscope": round(avg_correct_inscope, 2)
                if avg_correct_inscope is not None
                else None,
                "correctness_scored": n_in_scope,
                "correctness_in_scope_total": sum(
                    1 for r in valid if not r.get("oos_type")
                ),
                "oos_rejection_rate": round(
                    oos_rejection_rate(
                        [r for r in results if r.get("reference_status") == "verified"]
                    ),
                    3,
                ),
                "false_sufficiency_rate": round(false_sufficiency_rate, 3),
            }
        )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "skip_judge": skip_judge,
        "dataset": str(DATASET_PATH),
        "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
        "correctness_prompt_sha256": correctness_prompt_sha256(),
        "judge_provider": settings.JUDGE_LLM_PROVIDER,
        "judge_model": settings.JUDGE_MODEL_NAME,
        "judge_temperature": 0.0,
        "judge_seed": 12345,
        "dataset_size": len(dataset),
        "valid_results": n,
        "aggregate": aggregate,
        "false_sufficiency_threshold": FALSE_SUFFICIENCY_THRESHOLD,
        "results": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 55}")
    print(f"Results ({n}/{len(dataset)} valid)")
    if not skip_judge:
        print(
            f"  Faithfulness:          {aggregate['faithfulness']:.3f}  (target >0.85)"
        )
        print(
            f"  Answer Relevance:      {aggregate['answer_relevance']:.3f}  (target >0.85)"
        )
        if aggregate["correctness_inscope"] is not None:
            print(
                f"  Correctness (reviewed): {aggregate['correctness_inscope']:.1f}/10  (n={n_in_scope})"
            )
        else:
            print("  Correctness (reviewed): n/a (no eligible answers)")
        print(
            f"  Reference coverage:    {n_in_scope}/{aggregate['correctness_in_scope_total']} in-scope answers"
        )
        print(
            f"  OOS rejection rate:    {aggregate['oos_rejection_rate']:.1%}  (target >90%)"
        )
        print(
            f"  False-sufficiency:     {aggregate['false_sufficiency_rate']:.1%}  (target <10%)"
        )
        if false_sufficiency_cases:
            print(f"\nFalse-sufficiency cases ({len(false_sufficiency_cases)}):")
            for r in false_sufficiency_cases:
                print(
                    f"  - [{r.get('correctness_score', 0):.1f}] {r['question'][:60]}..."
                )
    else:
        print("  [skip-judge mode] Quality metrics not computed.")
    print(f"  Complex path rate:     {complex_rate:.1%}")
    print(
        f"  Latency p50 / p95:     {cost_summary['latency_p50_sec']:.1f}s / "
        f"{cost_summary['latency_p95_sec']:.1f}s  (mean {avg_elapsed:.1f}s)"
    )
    print(
        f"  Cost per query:        ${cost_summary['mean_cost_usd']:.5f}  "
        f"(run total ${cost_summary['total_cost_usd']:.4f})"
    )
    for path_name, block in cost_summary["by_path"].items():
        print(
            f"    {path_name:<8} n={block['queries']:<4} "
            f"${block['mean_cost_usd']:.5f}/query | "
            f"p50 {block['latency_p50_sec']:.1f}s / p95 {block['latency_p95_sec']:.1f}s"
        )
    if cost_summary["unpriced_models"]:
        print(
            "  WARNING: no rate card for "
            + ", ".join(cost_summary["unpriced_models"])
            + " — their tokens are priced at $0. Add them to eval/pricing.py."
        )
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate V7 RAG pipeline")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of questions"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path"
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM judge (no faithfulness/correctness scoring). Pipeline runs only. Cost ~$0.",
    )
    args = parser.parse_args()
    run(limit=args.limit, output=args.output, skip_judge=args.skip_judge)
