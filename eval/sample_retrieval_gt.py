"""Stratified sample of the synthetic retrieval GT for hand review.

The full GT (13k+ questions) is nobody's review workload, and its per-document
counts follow the order the generator happened to run in, not the corpus. This
module carves out a reproducible subset of 400-600 questions whose per-document
split follows the *corpus* shares, so retrieval metrics measured on it describe
the collection rather than the generator's stopping point.

Two rules shape the draw:

* one question per chunk while chunks last — three questions off one chunk
  measure the same retrieval event three times;
* everything is seeded — a reviewed sample that cannot be rebuilt is a dead end.

Usage::

    python eval/sample_retrieval_gt.py --n 500 --weights corpus
    python eval/sample_retrieval_gt.py --n 500 --weights gt --out eval/data/x.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

DEFAULT_GT = REPO_ROOT / "eval" / "data" / "retrieval_gt.jsonl"
DEFAULT_OUT = REPO_ROOT / "eval" / "data" / "retrieval_gt_sample.jsonl"
DEFAULT_N = 500
PREVIEW_CHARS = 300


def load_records(path: Path) -> list[dict]:
    """Read a GT ``.jsonl`` into a list of records; blank lines are skipped."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def allocate(
    weights: dict[str, int | float],
    n: int,
    available: dict[str, int] | None = None,
) -> dict[str, int]:
    """Split ``n`` seats between sources proportionally to ``weights``.

    Largest-remainder apportionment, capped by ``available``: a source that
    cannot fill its share hands the surplus back to the others instead of
    shrinking the sample. When nothing can absorb the surplus the result is
    simply smaller than ``n`` — every record there is.
    """
    keys = list(dict.fromkeys([*weights, *(available or {})]))
    cap = {k: (available or {}).get(k, n) for k in keys}

    fixed: dict[str, int] = {k: 0 for k in keys}
    active = [k for k in keys if weights.get(k, 0) > 0 and cap[k] > 0]
    remaining = n

    while active and remaining > 0:
        total_weight = sum(weights[k] for k in active)
        exact = {k: remaining * weights[k] / total_weight for k in active}
        quota = {k: int(exact[k]) for k in active}
        # largest remainder gets the odd seats
        leftover = remaining - sum(quota.values())
        for k in sorted(active, key=lambda k: (-(exact[k] - quota[k]), k))[:leftover]:
            quota[k] += 1

        overflow = [k for k in active if quota[k] > cap[k]]
        if not overflow:
            for k in active:
                fixed[k] = quota[k]
            remaining = 0
            break

        for k in overflow:
            fixed[k] = cap[k]
            remaining -= cap[k]
            active.remove(k)

    return fixed


def _draw_from_source(
    records: list[dict], quota: int, rng: random.Random
) -> list[dict]:
    """Take ``quota`` questions off one document, spreading them over chunks.

    Chunks are visited in a shuffled order, one question each, before any chunk
    is asked for a second question.
    """
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_chunk[record.get("chunk_id", "")].append(record)

    chunk_ids = list(by_chunk)
    rng.shuffle(chunk_ids)
    for cid in chunk_ids:
        rng.shuffle(by_chunk[cid])

    drawn: list[dict] = []
    depth = 0
    while len(drawn) < quota:
        took_any = False
        for cid in chunk_ids:
            if len(drawn) >= quota:
                break
            if depth < len(by_chunk[cid]):
                drawn.append(by_chunk[cid][depth])
                took_any = True
        if not took_any:
            break
        depth += 1
    return drawn


def stratified_sample(
    records: list[dict],
    n: int = DEFAULT_N,
    seed: int = 0,
    weights: dict[str, int | float] | None = None,
) -> list[dict]:
    """Draw ``n`` questions, split across documents by ``weights``.

    ``weights`` defaults to the GT's own per-document counts; pass corpus chunk
    counts to correct the generator's skew.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_source[record.get("source", "")].append(record)

    # a weighted source with no GT questions must show up as available=0, or its
    # share is quietly deducted from the sample instead of going to the others
    available = {source: len(rows) for source, rows in by_source.items()}
    for source in weights or {}:
        available.setdefault(source, 0)
    quotas = allocate(weights or available, n, available)

    sample: list[dict] = []
    for source in sorted(by_source):
        quota = quotas.get(source, 0)
        if quota:
            rng = random.Random(f"{seed}:{source}")
            sample.extend(_draw_from_source(by_source[source], quota, rng))
    return sample


def sample_distribution(sample: list[dict]) -> dict[str, int]:
    """Per-document question counts of a drawn sample."""
    return dict(Counter(record.get("source", "") for record in sample))


def corpus_weights() -> dict[str, int]:
    """Non-junk chunk counts per document, straight from the live collection."""
    from eval.generate_retrieval_gt import iter_corpus_chunks, is_junk_chunk

    counts: Counter[str] = Counter()
    for chunk in iter_corpus_chunks():
        if not is_junk_chunk(chunk.get("text", "") or ""):
            counts[(chunk.get("metadata") or {}).get("source", "")] += 1
    return dict(counts)


def write_sample(sample: list[dict], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in sample:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_review_sheet(sample: list[dict], path: Path) -> None:
    """A TSV for the hand pass: fill ``verdict`` with ok / brak, note optional.

    Tab-separated because the questions are full of commas and the file is meant
    to be opened in a spreadsheet and typed into.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(
            ["n", "verdict", "note", "question", "source", "chunk_id", "preview"]
        )
        for i, record in enumerate(sample, start=1):
            preview = (record.get("chunk_preview", "") or "").replace("\n", " ")
            writer.writerow(
                [
                    i,
                    "",
                    "",
                    record.get("question", ""),
                    record.get("source", ""),
                    record.get("chunk_id", ""),
                    preview[:PREVIEW_CHARS],
                ]
            )


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, default=DEFAULT_GT, help="full GT jsonl")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="sample size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--weights",
        default="corpus",
        help="'corpus' (non-junk chunk shares, reads Chroma), 'gt' (GT's own "
        "counts) or a path to a JSON {source: weight}",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--review-sheet",
        type=Path,
        default=None,
        help="also write a TSV for the hand pass (default: <out>.review.tsv)",
    )
    return p.parse_args(argv)


def _resolve_weights(spec: str) -> dict[str, int] | None:
    if spec == "gt":
        return None
    if spec == "corpus":
        return corpus_weights()
    return json.loads(Path(spec).read_text(encoding="utf-8"))


if __name__ == "__main__":
    args = _parse_args()
    records = load_records(args.gt)
    weights = _resolve_weights(args.weights)
    sample = stratified_sample(records, n=args.n, seed=args.seed, weights=weights)

    write_sample(sample, args.out)
    sheet = args.review_sheet or Path(str(args.out) + ".review.tsv")
    write_review_sheet(sample, sheet)

    dist = sample_distribution(sample)
    chunks = len({r.get("chunk_id") for r in sample})
    print(
        f"sample: {len(sample)} questions over {chunks} chunks "
        f"(from {len(records)} GT questions, weights={args.weights}, seed={args.seed})"
    )
    for source, count in sorted(dist.items(), key=lambda kv: -kv[1]):
        share = count / len(sample) * 100 if sample else 0
        print(f"  {source:<28} {count:>4}  {share:5.1f}%")
    if weights:
        missing = sorted(set(weights) - set(dist))
        if missing:
            print(f"  not represented (no GT questions): {', '.join(missing)}")
    print(f"-> {args.out}\n-> {sheet}")
