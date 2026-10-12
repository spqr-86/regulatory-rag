"""Remap retrieval GT labels (source#chunk_id) from one Chroma DB to a rebuilt snapshot."""

# ANCHOR: after reindexing, chunk_id numbers shift, so GT labels point at wrong
# chunks. Each labeled old chunk is mapped by text, per source:
#   exact  - same normalized text in the new DB (unique or nearest by old number);
#   cover  - new chunks whose rare words lie mostly inside the old chunk (split),
#            or the new chunk that contains most of the old chunk's words (merge).
# Non-exact mappings are printed for manual review. No API calls.
# In: old/new chroma.sqlite3, GT jsonl. Out: relabeled GT jsonl + review report.
import argparse
import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

COVER = 0.6  # share of a split new chunk's rare words inside the old chunk
CONTAIN = 0.8  # share of the old chunk's rare words inside one merged new chunk
COMMON = 0.1  # words in more than this share of a source's chunks do not count


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return set(re.findall(r"[а-яёa-z0-9]{4,}", text.lower()))


def load(db):
    """source -> {chunk_id: text}."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute(
        "select s.string_value, c.int_value, d.string_value from embedding_metadata s"
        " join embedding_metadata c on c.id=s.id and c.key='chunk_id'"
        " join embedding_metadata d on d.id=s.id and d.key='chroma:document'"
        " where s.key='source'"
    )
    out = defaultdict(dict)
    for src, num, text in rows:
        out[src][num] = text
    return out


def map_one(num, text, new):
    """Return (new ids, kind) for one old chunk within its source."""
    exact = [n for n, t in new.items() if norm(t) == norm(text)]
    if exact:
        return [min(exact, key=lambda n: abs(n - num))], "exact"
    bags = {n: words(t) for n, t in new.items()}
    df = defaultdict(int)
    for bag in bags.values():
        for w in bag:
            df[w] += 1
    common = {w for w, k in df.items() if k > COMMON * len(new)}
    ow = words(text) - common
    if not ow:
        return [], "no-words"
    split = sorted(
        n
        for n, bag in bags.items()
        if len(bag - common) >= 5
        and len((bag - common) & ow) / len(bag - common) >= COVER
    )
    if split:
        return split, "split"
    best = max(bags, key=lambda n: len(bags[n] & ow))
    if len(bags[best] & ow) / len(ow) >= CONTAIN:
        return [best], "merge"
    return [], "unmapped"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, default=Path("chroma_db/chroma.sqlite3"))
    p.add_argument("--new", type=Path, required=True)
    p.add_argument(
        "--gt", type=Path, default=Path("eval/data/golden_retrieval_labeled.jsonl")
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument(
        "--overrides", type=Path, help="JSON {old id: [new ids]} after manual review"
    )
    a = p.parse_args()

    gt = [
        json.loads(line)
        for line in a.gt.read_text(encoding="utf-8").splitlines()
        if line
    ]
    old, new = load(a.old), load(a.new)
    overrides = (
        json.loads(a.overrides.read_text(encoding="utf-8")) if a.overrides else {}
    )
    labels = sorted({c for r in gt for c in [r["chunk_id"], *r["relevant_chunk_ids"]]})
    mapping, kinds, review = {}, defaultdict(int), []
    for cid in labels:
        src, num = cid.rsplit("#", 1)
        text = old[src].get(int(num))
        if cid in overrides:
            ids, kind = [int(x.rsplit("#", 1)[1]) for x in overrides[cid]], "override"
        elif text is None:
            ids, kind = [], "missing-old"
        else:
            ids, kind = map_one(int(num), text, new[src])
        mapping[cid] = [f"{src}#{n}" for n in ids]
        kinds[kind] += 1
        if kind != "exact":
            review.append((cid, kind, text or "", [(n, new[src][n]) for n in ids]))

    out = []
    for r in gt:
        rel = list(
            dict.fromkeys(x for c in r["relevant_chunk_ids"] for x in mapping[c])
        )
        primary = mapping[r["chunk_id"]]
        out.append(
            {
                **r,
                "chunk_id": (
                    primary[0] if primary else (rel[0] if rel else "<unmapped>")
                ),
                "relevant_chunk_ids": rel or ["<unmapped>"],
            }
        )
    a.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8"
    )

    lines = [
        f"# GT remap {a.old} -> {a.new}",
        "",
        f"labels: {len(labels)}, kinds: {dict(kinds)}",
    ]
    lines.append(
        f"questions with <unmapped>: {sum('<unmapped>' in r['relevant_chunk_ids'] for r in out)}"
    )
    for cid, kind, text, hits in review:
        lines += ["", f"## {cid} — {kind}", "OLD: " + norm(text)[:400]]
        lines += [f"NEW #{n}: " + norm(t)[:400] for n, t in hits]
    a.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[2], lines[3], sep="\n")


if __name__ == "__main__":
    main()
