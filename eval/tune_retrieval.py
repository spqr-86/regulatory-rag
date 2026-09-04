"""Retrieval parameter tuning — scan a V7Config knob against the reviewed GT.

Step 3 of docs/roadmap.md. The simple path is a one-dimensional scan of
``RRF_K``: the fusion constant that decides how strongly the top of each ranked
list (vector, BM25) outweighs its tail when the two are merged.

``top_k`` is deliberately not scanned. It is a product decision — recall against
the price of context — so instead of a grid the report prints the Hit Rate@k
curve of the winning configuration and leaves the cutoff to a human.

The winner is **not** written into ``V7Config``: the report recommends, a manual
PR after review applies. Retrieval runs LLM-free (no router LLM, no generation),
so a scan costs nothing beyond embeddings.

Usage:
    .venv/bin/python eval/tune_retrieval.py --param RRF_K --values 10 20 40 60 90 120
    .venv/bin/python eval/tune_retrieval.py --limit 10          # smoke run
    .venv/bin/python eval/tune_retrieval.py --out benchmarks/tune_rrf.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Callable, Iterator, List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_retrieval_eval import (  # noqa: E402
    DEFAULT_KS,
    evaluate,
    init_engine,
    load_gt,
    make_retrieval_fn,
)
from src.v7.config import v7_config  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent
# The held-out set, labelled by the judge pair and read by a human — not the
# synthetic GT the baseline runner defaults to. Tuning against questions
# generated from the chunks they point at would tune against the generator.
DEFAULT_GT = PROJECT_ROOT / "eval" / "data" / "golden_retrieval_labeled.jsonl"
DEFAULT_METRIC = "mrr"
DEFAULT_CURVE_MAX_K = 20
# On ~43 held-out questions a single question moves Hit Rate by 0.023; a spread
# narrower than this across the whole scan is sampling noise, not a better value.
NOISE_SPREAD = 0.01
# Around the Cormack et al. default of 60, wide enough to show a flat curve.
DEFAULT_RRF_VALUES: tuple[int, ...] = (5, 10, 20, 40, 60, 90, 120, 200)


# ── Config override ───────────────────────────────────────────────────────


@contextmanager
def override_config(**overrides) -> Iterator[None]:
    """Temporarily set V7Config attributes, restoring them on the way out.

    Restores after an exception too — a scan that dies mid-value must not leave
    the process running on a tuning value nobody chose.
    """
    for name in overrides:
        if not hasattr(v7_config, name):
            raise ValueError(f"unknown V7Config parameter {name!r}")

    previous = {name: getattr(v7_config, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(v7_config, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(v7_config, name, value)


# ── Scan ──────────────────────────────────────────────────────────────────


def scan_param(
    param: str,
    values: Sequence,
    gt_records: Sequence[dict],
    retrieval_fn_factory: Callable[[], Callable[[str], List[str]]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Evaluate the GT once per candidate value of ``param``.

    The retrieval function is rebuilt inside each override so a path that reads
    the config at construction time sees the scanned value, not the default.
    """
    if not values:
        raise ValueError("values must not be empty")

    results: List[dict] = []
    for value in values:
        with override_config(**{param: value}):
            result = evaluate(gt_records, retrieval_fn_factory(), ks=ks)
        results.append(
            {
                "value": value,
                "metrics": result["metrics"],
                "per_source": result["per_source"],
                "records": result["records"],
                "errors": result["errors"],
                "elapsed_s": result["elapsed_s"],
            }
        )

    return {
        "param": param,
        "n": len(gt_records),
        "ks": list(ks),
        "baseline": getattr(v7_config, param),
        "results": results,
    }


def best_value(scan: dict, metric: str = DEFAULT_METRIC):
    """Value with the highest ``metric``; ties go to the first one scanned."""
    best = None
    best_score = None
    for entry in scan["results"]:
        score = entry["metrics"][metric]  # KeyError on an unknown metric is the answer
        if best_score is None or score > best_score:
            best, best_score = entry["value"], score
    return best


def hit_rate_curve(records: Sequence[dict], max_k: int = DEFAULT_CURVE_MAX_K) -> dict:
    """Share of questions whose first relevant chunk is at rank ≤ k, for each k.

    This is what replaces scanning ``top_k``: the curve shows what each extra
    chunk of context buys, and a human picks the cutoff.
    """
    total = len(records)
    ranks = [r.get("rank") for r in records]
    return {
        k: (sum(1 for r in ranks if r is not None and r <= k) / total if total else 0.0)
        for k in range(1, max_k + 1)
    }


# ── Report ────────────────────────────────────────────────────────────────


def format_scan_report(scan: dict) -> str:
    metric = scan.get("metric", DEFAULT_METRIC)
    best = scan.get("best", best_value(scan, metric=metric))
    ks = scan["ks"]

    lines = [
        f"{scan['param']}: скан {len(scan['results'])} значений"
        f" на {scan['n']} вопросах   метрика отбора: {metric}",
        "",
    ]
    head = "  ".join(f"HR@{k}" for k in ks)
    lines.append(f"    {'значение':>10}  {head}  {metric:>6}")
    for entry in scan["results"]:
        cells = "  ".join(f"{entry['metrics'][f'hit_rate@{k}']:.3f}" for k in ks)
        mark = " ←" if entry["value"] == best else ""
        lines.append(
            f"    {str(entry['value']):>10}  {cells}"
            f"  {entry['metrics'][metric]:.3f}{mark}"
        )

    baseline = scan.get("baseline")
    scores = [e["metrics"][metric] for e in scan["results"]]
    spread = max(scores) - min(scores)

    lines.append("")
    lines.append(f"  рекомендация: {scan['param']} = {best} (сейчас в конфиге {baseline})")
    if spread < NOISE_SPREAD:
        lines.append(
            f"  разброс {metric} по всему скану {spread:.4f} < {NOISE_SPREAD} —"
            f" это шум, а не выигрыш: менять {scan['param']} оснований нет"
        )
    lines.append("  в V7Config не применено — только ручной PR после ревью")

    curve = scan.get("curve")
    if curve:
        lines.append("")
        lines.append(f"  кривая Hit Rate@k при {scan['param']} = {best}:")
        for k in sorted(int(k) for k in curve):
            lines.append(f"    k={k:<3} {curve[k] if k in curve else curve[str(k)]:.3f}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", choices=("simple",), default="simple",
                   help="complex is excluded: its held-out GT is pooled from its own output")
    p.add_argument("--param", default="RRF_K")
    p.add_argument("--values", type=int, nargs="+", default=list(DEFAULT_RRF_VALUES))
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--metric", default=DEFAULT_METRIC, help="metric the winner is picked by")
    p.add_argument("--curve-max-k", type=int, default=DEFAULT_CURVE_MAX_K)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    gt = load_gt(args.gt, limit=args.limit)
    print(f"GT: {len(gt)} вопросов из {args.gt}")
    print(f"скан {args.param}: {args.values}\n")

    init_engine()
    scan = scan_param(
        args.param,
        args.values,
        gt,
        lambda: make_retrieval_fn(args.path),
        ks=tuple(args.ks),
    )
    scan["metric"] = args.metric
    scan["best"] = best_value(scan, metric=args.metric)
    scan["path"] = args.path
    scan["gt_path"] = str(args.gt)
    scan["date"] = date.today().isoformat()

    winner = next(e for e in scan["results"] if e["value"] == scan["best"])
    scan["curve"] = hit_rate_curve(winner["records"], max_k=args.curve_max_k)

    print(format_scan_report(scan))

    out = args.out or (
        PROJECT_ROOT
        / "benchmarks"
        / f"tune_{args.param.lower()}_{args.path}_{date.today().isoformat()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПолный результат: {out}")


if __name__ == "__main__":
    main()
