"""Calibrate domain gate threshold.

Loads eval/tests/dataset.csv (or tests/dataset.csv as fallback),
embeds each question, computes cosine similarity to corpus centroid,
and prints statistics to help choose DOMAIN_GATE_THRESHOLD.

Usage:
    python scripts/calibrate_domain_gate.py
    python scripts/calibrate_domain_gate.py --csv path/to/dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.infra.llm_factory import get_embedding_model  # noqa: E402
from src.v7.domain_gate import cosine_similarity, get_corpus_centroid  # noqa: E402


def _find_dataset(override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        return p
    candidates = [
        PROJECT_ROOT / "eval" / "tests" / "dataset.csv",
        PROJECT_ROOT / "tests" / "dataset.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"dataset.csv not found. Tried: {[str(c) for c in candidates]}"
    )


def _load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate domain gate threshold.")
    parser.add_argument("--csv", default=None, help="Path to dataset CSV")
    args = parser.parse_args()

    csv_path = _find_dataset(args.csv)
    print(f"Dataset: {csv_path}")

    rows = _load_rows(csv_path)
    questions = [r["question"] for r in rows]
    is_oos_flags = [bool(r.get("oos_type", "").strip()) for r in rows]

    print(
        f"Total rows: {len(rows)} (domain: {is_oos_flags.count(False)}, OOS: {is_oos_flags.count(True)})"
    )
    print("Computing corpus centroid...")
    centroid = get_corpus_centroid()

    print("Embedding questions...")
    embedding_model = get_embedding_model()
    embeddings = embedding_model.embed_documents(questions)

    # Compute similarities
    sims = []
    for emb in embeddings:
        q = np.array(emb, dtype=np.float32)
        sims.append(cosine_similarity(q, centroid))

    # Print table
    col_w = 60
    print()
    print(f"{'Question':<{col_w}}  {'is_oos':<6}  {'similarity':>10}")
    print("-" * (col_w + 22))
    for row, is_oos, sim in zip(rows, is_oos_flags, sims):
        q_text = (
            row["question"][: col_w - 3] + "..."
            if len(row["question"]) > col_w
            else row["question"]
        )
        label = "OOS" if is_oos else "domain"
        print(f"{q_text:<{col_w}}  {label:<6}  {sim:>10.4f}")

    # Statistics
    domain_sims = [s for s, oos in zip(sims, is_oos_flags) if not oos]
    oos_sims = [s for s, oos in zip(sims, is_oos_flags) if oos]

    print()
    print("=" * 50)
    print("Statistics:")
    if domain_sims:
        print(
            f"  Domain  — min={min(domain_sims):.4f}  mean={np.mean(domain_sims):.4f}"
            f"  max={max(domain_sims):.4f}  n={len(domain_sims)}"
        )
    if oos_sims:
        print(
            f"  OOS     — min={min(oos_sims):.4f}  mean={np.mean(oos_sims):.4f}"
            f"  max={max(oos_sims):.4f}  n={len(oos_sims)}"
        )

    if domain_sims:
        p5 = float(np.percentile(domain_sims, 5))
        print()
        print(f"Recommended DOMAIN_GATE_THRESHOLD (5th percentile of domain): {p5:.4f}")
        print("  Set V7_DOMAIN_GATE_THRESHOLD=<value> in .env to activate.")
        print("  0.0 = disabled (default, backward compatible).")


if __name__ == "__main__":
    main()
