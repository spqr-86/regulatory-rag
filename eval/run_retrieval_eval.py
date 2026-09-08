"""Retrieval evaluation runner — baseline Hit Rate / MRR for the V7 paths.

Thin layer over eval/retrieval_metrics.evaluate_retrieval_batch: builds a
retrieval function on the engine of the requested path (simple = hybrid
vector+BM25 → RRF; complex = wide fetch + section expand + rerank;
vector = dense only; bm25 = lexical only), runs it over the reviewed ground
truth and prints Hit Rate@{5,10,12}, MRR and per-query retrieval latency
(p50/p95/p99, mean, max).

For the vector/bm25 backbones the router still builds the plan (top_k,
glossary expansion) exactly as in prod, then the single retriever is called
directly — no RRF, no rerank, no LLM. simple already is the hybrid.

The retrieval nodes are called directly rather than through the graph — no
router LLM, no triage, no generation, so a full run costs nothing beyond
embeddings. tests/test_run_retrieval_eval.py pins the parity of that shortcut
against the full graph.

Usage:
    .venv/bin/python eval/run_retrieval_eval.py --path simple
    .venv/bin/python eval/run_retrieval_eval.py --path complex --limit 20
    .venv/bin/python eval/run_retrieval_eval.py --path simple --out benchmarks/r.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.retrieval_metrics import evaluate_retrieval_batch  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GT = PROJECT_ROOT / "eval" / "data" / "retrieval_gt_reviewed.jsonl"
DEFAULT_KS: tuple[int, ...] = (5, 10, 12)
NODE_PATHS = ("simple", "complex")
BACKBONE_PATHS = ("vector", "bm25")
PATHS = NODE_PATHS + BACKBONE_PATHS


# ── Ground truth ──────────────────────────────────────────────────────────


def load_gt(path: Path, limit: int | None = None) -> List[dict]:
    """Load reviewed GT records. Skips records missing question or chunk_id."""
    records: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            question = (raw.get("question") or "").strip()
            chunk_id = (raw.get("chunk_id") or "").strip()
            if not question or not chunk_id:
                continue
            # Held-out разметка отмечает все чанки, которые отвечают на вопрос;
            # синтетическая GT — ровно один. Оба вида читаются одинаково.
            relevant = [
                str(cid).strip()
                for cid in (raw.get("relevant_chunk_ids") or [chunk_id])
                if str(cid).strip()
            ]
            records.append(
                {
                    "question": question,
                    "chunk_id": chunk_id,
                    "relevant_chunk_ids": relevant,
                    "source": raw.get("source") or chunk_id.split("#")[0],
                }
            )
            if limit is not None and len(records) >= limit:
                break
    return records


# ── Retrieval ─────────────────────────────────────────────────────────────


def extract_chunk_ids(passages: Iterable[dict]) -> List[str]:
    """Ordered, deduplicated passage identities of the retrieved passages.

    Identity is ``passage_identity`` — the same ``"{source}#{chunk_id}"`` key the
    GT generator wrote, not the bare index-time ``chunk_id`` (a per-source int,
    which collides across documents).
    """
    from src.v7.nlp_core import passage_identity

    ids: List[str] = []
    seen: set[str] = set()
    for p in passages or []:
        if not p:
            continue
        pid = passage_identity(p)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        ids.append(pid)
    return ids


def to_candidates(passages: Iterable[dict]) -> List[dict]:
    """Retrieved passages as labelling candidates: identity, text, source.

    Same order and same deduplication as ``extract_chunk_ids`` — a human
    labelling one pool while the metric measures another is the failure this
    shared shape exists to prevent.
    """
    from src.v7.nlp_core import passage_identity

    out: List[dict] = []
    seen: set[str] = set()
    for p in passages or []:
        if not p:
            continue
        pid = passage_identity(p)
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(
            {
                "chunk_id": pid,
                "text": p.get("text", "") or "",
                "source": (p.get("metadata") or {}).get("source")
                or p.get("doc_id")
                or pid.split("#")[0],
            }
        )
    return out


def _make_backbone_fn(
    path: str, return_passages: bool
) -> Callable[[str], List[str]] | Callable[[str], List[dict]]:
    """query → [chunk_id] on a single retriever (dense or lexical).

    The router builds the plan so top_k and the glossary expansion match
    production; then vector or BM25 is called on its own — no RRF, no rerank.
    """
    from src.v7.hard_gates import validate_filters
    from src.v7.nlp_core import bm25_search
    from src.v7.nodes import rag_simple as rag_simple_mod
    from src.v7.nodes.router import router

    def _retrieve(query: str) -> List[str] | List[dict]:
        state: dict = {"query": query, "filters": None}
        state.update(router(state))
        if state.get("clarify_message"):  # too short for the router to plan
            return []
        plan = state["plan"]
        active_q = state.get("active_query", query)
        safe_filters = validate_filters(state.get("filters"))
        if path == "vector":
            passages = rag_simple_mod._vector_search(
                query=active_q, filters=safe_filters, top_k=plan["top_k"]
            )
        else:
            passages = bm25_search(
                query=active_q, filters=safe_filters, top_k=plan["top_k"]
            )
        return (
            to_candidates(passages) if return_passages else extract_chunk_ids(passages)
        )

    return _retrieve


def make_retrieval_fn(
    path: str, return_passages: bool = False
) -> Callable[[str], List[str]] | Callable[[str], List[dict]]:
    """Build query → [chunk_id] on the engine of the given path.

    The router node builds the plan so that thresholds, top_k and the glossary
    expansion match production exactly; only the LLM-bearing nodes are skipped.
    With ``return_passages`` the same call returns the candidate dicts (identity
    plus text and source) that held-out labelling needs.
    """
    if path not in PATHS:
        raise ValueError(f"unknown path {path!r}, expected one of {PATHS}")

    if path in BACKBONE_PATHS:
        return _make_backbone_fn(path, return_passages)

    from src.v7.nodes.rag_complex import rag_complex
    from src.v7.nodes.rag_simple import rag_simple
    from src.v7.nodes.router import router

    node = rag_simple if path == "simple" else rag_complex
    stage = path

    def _retrieve(query: str) -> List[str]:
        state: dict = {"query": query, "filters": None}
        state.update(router(state))
        if state.get("clarify_message"):  # too short for the router to plan
            return []
        update = node(state)
        attempts = [
            a for a in (update.get("retrieval_attempts") or []) if a["stage"] == stage
        ]
        if not attempts:
            return []
        passages = attempts[-1]["passages"]
        return (
            to_candidates(passages) if return_passages else extract_chunk_ids(passages)
        )

    return _retrieve


def init_engine() -> None:
    """Wire the real Chroma-backed engine into the retrieval nodes."""
    import os

    from dotenv import load_dotenv

    load_dotenv()
    # LangSmith tracing is read via *_TRACING_V2 and .env turns it on; a full run
    # would burn the monthly trace quota on retrieval calls we never inspect.
    os.environ["LANGSMITH_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

    from src.backends.vector_store import get_vector_store_backend
    from src.v7.bridge import init_v7_pipeline

    # llm_provider=None: no generator, no query expander — retrieval only.
    init_v7_pipeline(get_vector_store_backend(), llm_provider=None)


# ── Metrics ───────────────────────────────────────────────────────────────


def _first_rank(retrieved: Sequence[str], relevant: Sequence[str] | str) -> int | None:
    """Rank of the first retrieved chunk that is relevant, 1-based.

    ``relevant`` is a list: a held-out question can be answered by several
    chunks (a norm repeated across documents), and a hit on any of them is a
    hit. A bare string is accepted so the synthetic GT keeps working.
    """
    wanted = {relevant} if isinstance(relevant, str) else set(relevant)
    for i, cid in enumerate(retrieved, start=1):
        if cid in wanted:
            return i
    return None


def _latency_summary(latencies_ms: Sequence[float]) -> dict:
    """Aggregate per-query latencies into mean and nearest-rank percentiles.

    Retrieval latency is a portfolio-facing prod metric; the summary run only
    times search, no generation. Nearest-rank keeps small-N percentiles honest.
    """
    s = sorted(latencies_ms)
    if not s:
        return {"n": 0}

    def pct(p: float) -> float:
        rank = max(1, math.ceil(p / 100 * len(s)))
        return s[rank - 1]

    return {
        "n": len(s),
        "mean_ms": round(sum(s) / len(s), 1),
        "p50_ms": round(pct(50), 1),
        "p95_ms": round(pct(95), 1),
        "p99_ms": round(pct(99), 1),
        "max_ms": round(s[-1], 1),
    }


def evaluate(
    gt_records: Sequence[dict],
    retrieval_fn: Callable[[str], List[str]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Run retrieval over the GT and aggregate Hit Rate@k, MRR and per-source rates."""
    retrieved_list: List[List[str]] = []
    relevant_list: List[List[str]] = []
    records: List[dict] = []
    errors = 0
    t0 = time.perf_counter()

    latencies_ms: List[float] = []
    for rec in gt_records:
        q0 = time.perf_counter()
        try:
            retrieved = retrieval_fn(rec["question"])
        except Exception as exc:  # a dead query must not kill the run
            retrieved = []
            errors += 1
            print(f"  ! retrieval failed: {rec['question'][:60]}… — {exc}")
        latency_ms = (time.perf_counter() - q0) * 1000
        latencies_ms.append(latency_ms)
        relevant = rec.get("relevant_chunk_ids") or [rec["chunk_id"]]
        rank = _first_rank(retrieved, relevant)
        retrieved_list.append(retrieved)
        relevant_list.append(list(relevant))
        records.append(
            {
                "question": rec["question"],
                "chunk_id": rec["chunk_id"],
                "source": rec["source"],
                "retrieved": retrieved,
                "rank": rank,
                "hit": rank is not None,
                "latency_ms": round(latency_ms, 1),
            }
        )

    metrics: dict[str, float] = {}
    for k in ks:
        if not records:  # evaluate_retrieval_batch returns {} on an empty batch
            metrics[f"hit_rate@{k}"] = 0.0
            continue
        batch = evaluate_retrieval_batch(retrieved_list, relevant_list, k=k)
        metrics[f"hit_rate@{k}"] = float(batch[f"hit_rate@{k}"])
    # MRR over the untruncated lists — it is a ranking metric, not a @k one.
    metrics["mrr"] = (
        float(evaluate_retrieval_batch(retrieved_list, relevant_list, k=1)["mrr"])
        if records
        else 0.0
    )

    per_source: dict[str, dict] = {}
    for rec in records:
        bucket = per_source.setdefault(rec["source"], {"n": 0, "ranks": []})
        bucket["n"] += 1
        bucket["ranks"].append(rec["rank"])
    for src, bucket in per_source.items():
        ranks = bucket.pop("ranks")
        for k in ks:
            bucket[f"hit_rate@{k}"] = sum(
                1 for r in ranks if r is not None and r <= k
            ) / len(ranks)

    return {
        "n": len(records),
        "ks": list(ks),
        "metrics": metrics,
        "per_source": per_source,
        "latency": _latency_summary(latencies_ms),
        "records": records,
        "errors": errors,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }


def format_report(result: dict) -> str:
    lines = [
        f"path: {result.get('path', '?')}   questions: {result['n']}"
        f"   errors: {result['errors']}   {result['elapsed_s']}s",
        "",
    ]
    for name, value in result["metrics"].items():
        lines.append(f"  {name:<14} {value:.3f}")
    lat = result.get("latency") or {}
    if lat.get("n"):
        lines.append("")
        lines.append(
            f"  латентность, мс: p50 {lat['p50_ms']}  p95 {lat['p95_ms']}  "
            f"p99 {lat['p99_ms']}  max {lat['max_ms']}  (mean {lat['mean_ms']})"
        )
    lines.append("")
    lines.append("  по документам:")
    ks = result["ks"]
    head = "  ".join(f"HR@{k}" for k in ks)
    lines.append(f"    {'документ':<32} {'n':>4}  {head}")
    for src, bucket in sorted(result["per_source"].items(), key=lambda kv: -kv[1]["n"]):
        cells = "  ".join(f"{bucket[f'hit_rate@{k}']:.2f}" for k in ks)
        lines.append(f"    {src[:32]:<32} {bucket['n']:>4}  {cells}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", choices=PATHS, default="simple")
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="cutoffs for Hit Rate",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write full result json (default: benchmarks/retrieval_eval_<path>_<date>.json)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gt = load_gt(args.gt, limit=args.limit)
    print(f"GT: {len(gt)} вопросов из {args.gt}")

    init_engine()
    result = evaluate(gt, make_retrieval_fn(args.path), ks=tuple(args.ks))
    result["path"] = args.path
    result["gt_path"] = str(args.gt)
    result["date"] = date.today().isoformat()

    print()
    print(format_report(result))

    out = args.out or (
        PROJECT_ROOT
        / "benchmarks"
        / f"retrieval_eval_{args.path}_{date.today().isoformat()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПолный результат: {out}")


if __name__ == "__main__":
    main()
