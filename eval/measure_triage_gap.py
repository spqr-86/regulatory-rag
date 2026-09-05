"""Measure the triage gap (issue #13, stage B2) — escalation rate and HR@12.

Runs router → rag_simple → evaluate_triage over the held-out set. No
generation, no complex path and no query expander, so the run makes no LLM
calls at all: the only cost is embeddings.

Two numbers decide stage B2, plus one diagnostic:
  * escalation rate — the share of questions triage sends to rag_complex;
  * Hit Rate@12 over the passage list the triage hands on (NOT the retriever
    output: the task does not touch search, so the retriever's HR cannot move);
  * how many structured gaps were seen and how many closed in place.

The run is done twice — on main and on the branch — and the two JSONs are
compared with --baseline: the escalation rate must drop strictly and no single
question may lose its hit.

Usage:
    .venv/bin/python eval/measure_triage_gap.py --out benchmarks/triage_gap_main.json
    .venv/bin/python eval/measure_triage_gap.py --out benchmarks/triage_gap_b2.json \
        --baseline benchmarks/triage_gap_main.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_retrieval_eval import (  # noqa: E402
    extract_chunk_ids,
    init_engine,
    load_gt,
)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_GT = PROJECT_ROOT / "eval" / "data" / "golden_retrieval_labeled.jsonl"
DEFAULT_K = 12


# ── Pure parts ────────────────────────────────────────────────────────────


def passages_after_triage(update: dict, retrieved: List[dict]) -> List[dict]:
    """The passage list the triage hands on to the next node.

    Sufficient → final_passages (what generation sees). Escalation →
    fallback_passages, the list rag_complex starts from; when the triage saved
    no fallback, what the retriever returned.
    """
    if update.get("sufficient"):
        return update.get("final_passages") or retrieved
    return update.get("fallback_passages") or retrieved


def summarize(records: Sequence[dict], k: int = DEFAULT_K) -> dict:
    """Aggregate per-question records into the three headline numbers."""
    n = len(records)
    if not n:
        return {
            "n": 0,
            "escalation_rate": 0.0,
            f"hit_rate@{k}": 0.0,
            "gaps_seen": 0,
            "gaps_closed": 0,
        }
    gaps_seen = sum(1 for r in records if r.get("gap_seen"))
    return {
        "n": n,
        "escalation_rate": sum(1 for r in records if r["escalated"]) / n,
        f"hit_rate@{k}": sum(1 for r in records if r["hit"]) / n,
        "gaps_seen": gaps_seen,
        "gaps_closed": sum(
            1 for r in records if r.get("gap_seen") and not r.get("gap_open")
        ),
    }


def compare_runs(baseline: dict, new: dict) -> dict:
    """Apply the stage-B2 acceptance rule to two runs.

    Passes only when the escalation rate falls strictly and not a single
    question that used to hit stops hitting. On 43 held-out questions one
    question is 2.3 pp, so "about the same" is a failure, not a wash.
    """
    base_by_q = {r["question"]: r for r in baseline.get("records", [])}
    new_by_q = {r["question"]: r for r in new.get("records", [])}

    missing = sorted(set(base_by_q) ^ set(new_by_q))
    shared = [q for q in base_by_q if q in new_by_q]

    regressed = sorted(
        q for q in shared if base_by_q[q].get("hit") and not new_by_q[q].get("hit")
    )
    recovered = sorted(
        q for q in shared if not base_by_q[q].get("hit") and new_by_q[q].get("hit")
    )

    def _rate(by_q: dict) -> float:
        return (
            sum(1 for r in by_q.values() if r.get("escalated")) / len(by_q)
            if by_q
            else 0.0
        )

    delta = _rate(new_by_q) - _rate(base_by_q)
    return {
        "escalation_delta": delta,
        "regressed": regressed,
        "recovered": recovered,
        "missing": missing,
        "passed": bool(delta < 0 and not regressed and not missing),
    }


# ── Run ───────────────────────────────────────────────────────────────────


def run(gt_records: Sequence[dict], k: int = DEFAULT_K) -> dict:
    """Run router → rag_simple → evaluate_triage over the GT."""
    from src.v7.nodes.evaluate_triage import evaluate_triage, route_after_triage
    from src.v7.nodes.rag_simple import rag_simple
    from src.v7.nodes.router import router

    records: List[dict] = []
    errors = 0
    t0 = time.perf_counter()

    for rec in gt_records:
        question = rec["question"]
        try:
            state: dict = {"query": question, "filters": None}
            state.update(router(state))
            if state.get("clarify_message"):
                records.append(
                    {
                        "question": question,
                        "escalated": False,
                        "hit": False,
                        "gap_seen": False,
                        "gap_open": False,
                        "skipped": "clarify",
                    }
                )
                continue
            state.update(rag_simple(state))
            attempts = [
                a
                for a in (state.get("retrieval_attempts") or [])
                if a["stage"] == "simple"
            ]
            retrieved = attempts[-1]["passages"] if attempts else []

            update = evaluate_triage(state)
            handed_on = passages_after_triage(update, retrieved)
            gap = update.get("triage_gap")

            merged = {**state, **update}
            escalated = route_after_triage(merged) == "rag_complex"

            relevant = set(rec.get("relevant_chunk_ids") or [rec["chunk_id"]])
            hit = bool(set(extract_chunk_ids(handed_on)[:k]) & relevant)

            records.append(
                {
                    "question": question,
                    "source": rec.get("source"),
                    "escalated": escalated,
                    "hit": hit,
                    "n_handed_on": len(handed_on),
                    "gap_seen": gap is not None,
                    "gap_open": bool(gap and gap.get("open")),
                    "gap": gap,
                }
            )
        except Exception as exc:  # a dead query must not kill the run
            errors += 1
            print(f"  ! failed: {question[:60]}… — {exc}")
            records.append(
                {
                    "question": question,
                    "escalated": False,
                    "hit": False,
                    "gap_seen": False,
                    "gap_open": False,
                    "error": str(exc),
                }
            )

    from src.v7.config import v7_config

    result = summarize(records, k=k)
    result.update(
        {
            "k": k,
            "v8_evidence_assess": bool(v7_config.V8_ENABLE_EVIDENCE_ASSESS),
            "records": records,
            "errors": errors,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "date": date.today().isoformat(),
        }
    )
    return result


def format_report(result: dict, verdict: dict | None = None) -> str:
    k = result.get("k", DEFAULT_K)
    lines = [
        f"вопросов: {result['n']}   ошибок: {result['errors']}   {result['elapsed_s']}s",
        "",
        f"  доля эскалаций   {result['escalation_rate']:.3f}",
        f"  hit_rate@{k}      {result[f'hit_rate@{k}']:.3f}",
        f"  пробелов найдено {result['gaps_seen']}, закрыто {result['gaps_closed']}",
    ]
    if result.get("v8_evidence_assess"):
        lines += [
            "",
            "  ВНИМАНИЕ: V8_ENABLE_EVIDENCE_ASSESS=true — триаж идёт через "
            "_evidence_assess,",
            "  а этап B2 живёт в _legacy_triage. Замер к B2 отношения не имеет: "
            "запускать с V7_V8_ENABLE_EVIDENCE_ASSESS=false.",
        ]
    if verdict is not None:
        lines += [
            "",
            f"  Δ доли эскалаций {verdict['escalation_delta']:+.3f}",
            f"  потеряли хит:    {len(verdict['regressed'])} "
            + (f"({', '.join(q[:40] for q in verdict['regressed'])})" if verdict["regressed"] else ""),
            f"  приобрели хит:   {len(verdict['recovered'])}",
        ]
        if verdict["missing"]:
            lines.append(f"  вопросы не совпали: {len(verdict['missing'])}")
        lines.append(f"  критерий B2: {'ПРОЙДЕН' if verdict['passed'] else 'НЕ ПРОЙДЕН'}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="previous run json to compare against (the main-branch run)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gt = load_gt(args.gt, limit=args.limit)
    print(f"GT: {len(gt)} вопросов из {args.gt}")

    init_engine()
    result = run(gt, k=args.k)
    result["gt_path"] = str(args.gt)

    verdict = None
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        verdict = compare_runs(baseline, result)
        result["baseline_path"] = str(args.baseline)
        result["verdict"] = verdict

    print()
    print(format_report(result, verdict))

    out = args.out or (
        PROJECT_ROOT / "benchmarks" / f"triage_gap_{date.today().isoformat()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПолный результат: {out}")


if __name__ == "__main__":
    main()
