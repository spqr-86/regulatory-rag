"""Cost-Per-Sample (CPS) measurement for V7 RAG pipeline.

Runs N questions from dataset_original.csv through V7 graph, captures token
usage via langchain UsageMetadataCallbackHandler, computes avg per-query
input/output tokens and USD cost using Gemini 2.5 Flash pricing.

Pricing source: ai.google.dev/gemini-api/docs/pricing (Gemini 2.5 Flash):
    input  : $0.30 / 1M tokens
    output : $2.50 / 1M tokens (includes thinking tokens)

Usage:
    python scripts/measure_cps.py --n 5
    python scripts/measure_cps.py --n 10 --out benchmarks/cps_2026-05-22.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Gemini 2.5 Flash pricing (USD per 1M tokens), Jan 2026
PRICE_INPUT_PER_M = 0.30
PRICE_OUTPUT_PER_M = 2.50


def run(n: int, dataset_path: Path, out_path: Path | None) -> dict[str, Any]:
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    from src.v7.bridge import init_v7_pipeline
    from src.v7.graph import build_graph
    from src.backends.vector_store import get_vector_store_backend

    print("[init] loading vector store…", flush=True)
    vs = get_vector_store_backend(load_existing=True)
    init_v7_pipeline(vs)
    app = build_graph().compile()

    with dataset_path.open() as f:
        rows = list(csv.DictReader(f))[:n]

    per_query = []
    t0_all = time.time()

    for i, row in enumerate(rows, 1):
        q = row["question"]
        cb = UsageMetadataCallbackHandler()
        t0 = time.time()
        try:
            state = app.invoke({"query": q}, config={"callbacks": [cb]})
            err = None
        except Exception as e:  # noqa: BLE001
            state = {}
            err = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0

        # Aggregate token usage across all models in this query
        total_in = total_out = 0
        per_model = {}
        for model_name, usage in cb.usage_metadata.items():
            in_t = usage.get("input_tokens", 0)
            out_t = usage.get("output_tokens", 0)
            total_in += in_t
            total_out += out_t
            per_model[model_name] = {"input": in_t, "output": out_t}

        cost = (total_in / 1e6) * PRICE_INPUT_PER_M + (
            total_out / 1e6
        ) * PRICE_OUTPUT_PER_M

        rec = {
            "idx": i,
            "question": q[:80],
            "elapsed_sec": round(elapsed, 2),
            "input_tokens": total_in,
            "output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "cost_usd": round(cost, 6),
            "per_model": per_model,
            "path": state.get("path") if isinstance(state, dict) else None,
            "error": err,
        }
        per_query.append(rec)
        print(
            f"[{i}/{n}] {elapsed:5.1f}s  in={total_in:>6}  out={total_out:>5}  "
            f"${cost:.5f}  {'ERR: ' + err if err else ''}",
            flush=True,
        )

    total_elapsed = time.time() - t0_all
    ok = [r for r in per_query if r["error"] is None]
    if not ok:
        print("\nALL queries failed.")
        return {"per_query": per_query, "aggregate": {}}

    n_ok = len(ok)
    avg_in = sum(r["input_tokens"] for r in ok) / n_ok
    avg_out = sum(r["output_tokens"] for r in ok) / n_ok
    avg_cost = sum(r["cost_usd"] for r in ok) / n_ok
    avg_elapsed = sum(r["elapsed_sec"] for r in ok) / n_ok

    aggregate = {
        "n_total": n,
        "n_ok": n_ok,
        "avg_input_tokens": round(avg_in, 1),
        "avg_output_tokens": round(avg_out, 1),
        "avg_total_tokens": round(avg_in + avg_out, 1),
        "avg_cost_usd": round(avg_cost, 6),
        "avg_elapsed_sec": round(avg_elapsed, 2),
        "cost_per_1k_queries_usd": round(avg_cost * 1000, 2),
        "pricing": {
            "model": "gemini-2.5-flash",
            "input_per_1M_usd": PRICE_INPUT_PER_M,
            "output_per_1M_usd": PRICE_OUTPUT_PER_M,
            "note": "output includes thinking tokens",
        },
        "wall_clock_sec": round(total_elapsed, 1),
    }

    print("\n=== AGGREGATE ===")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    result = {"aggregate": aggregate, "per_query": per_query}
    if out_path:
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n[saved] {out_path}")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent.parent / "tests" / "dataset_original.csv",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    run(args.n, args.dataset, args.out)


if __name__ == "__main__":
    main()
