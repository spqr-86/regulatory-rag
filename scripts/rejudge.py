"""Re-judge saved eval answers with the current judge (prompt + model from .env).

Reads one or more eval JSONL result files, re-scores correctness on the SAVED
answers (no pipeline re-run), and prints per-file aggregate + per-question deltas
between two files. Cheap apples-to-apples comparison under a single judge.

Usage:
    python scripts/rejudge.py <before.jsonl> <after.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.infra.llm_factory import (  # noqa: E402
    apply_ipv6_patch_for_googleapis,
    get_judge_llm,
)

apply_ipv6_patch_for_googleapis()

from eval.run_v7_eval import evaluate_correctness  # noqa: E402


def load_results(path: str) -> list[dict]:
    return json.load(open(path))["results"]


def rejudge_file(path: str, judge) -> dict[str, dict]:
    out: dict[str, dict] = {}
    rows = load_results(path)
    for i, r in enumerate(rows):
        q = r["question"]
        if r.get("oos_type"):
            out[q] = {"score": None, "oos": True, "path": r.get("path")}
            continue
        res = evaluate_correctness(q, r["ground_truth"], r.get("answer") or "", judge)
        out[q] = {
            "score": res["correctness_score"],
            "oos": False,
            "path": r.get("path"),
            "reasoning": res["correctness_reasoning"],
        }
        print(f"  [{i + 1}/{len(rows)}] {res['correctness_score']:.0f}  {q[:55]}")
    return out


def main() -> None:
    before_path, after_path = sys.argv[1], sys.argv[2]
    judge = get_judge_llm(temperature=0.0, seed=12345)

    print(f"\n=== RE-JUDGE BEFORE: {before_path} ===")
    before = rejudge_file(before_path, judge)
    print(f"\n=== RE-JUDGE AFTER: {after_path} ===")
    after = rejudge_file(after_path, judge)

    def agg(d):
        xs = [v["score"] for v in d.values() if not v["oos"] and v["score"] is not None]
        return sum(xs) / len(xs), len(xs)

    ab, nb = agg(before)
    aa, na = agg(after)
    print("\n" + "=" * 60)
    print("in-scope correctness (single judge, re-judged):")
    print(f"  BEFORE: {ab:.2f}  (n={nb})")
    print(f"  AFTER:  {aa:.2f}  (n={na})")
    print(f"  delta:  {aa - ab:+.2f}")

    print("\n=== per-question changes ===")
    ups = downs = 0
    for q in before:
        b, a = before[q], after.get(q)
        if not a or b["oos"] or b["score"] is None or a["score"] is None:
            continue
        d = a["score"] - b["score"]
        if d > 0:
            ups += 1
        elif d < 0:
            downs += 1
        if d != 0:
            arr = "↑" if d > 0 else "↓"
            print(f"  {b['score']:.0f}->{a['score']:.0f} {arr} {q[:60]}")
    print(f"\nИтого: ↑{ups} / ↓{downs}")


if __name__ == "__main__":
    main()
