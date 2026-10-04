"""Golden-90 retrieval regression: production index vs 29н re-chunked (current / explicit_ctx / rows)."""

# ANCHOR: does the new 29н table chunking hurt the other documents? Same 90
# practitioner questions and production `simple` path as eval/run_retrieval_eval.py,
# run on the production index and on scratch copies with 29н chunks replaced.
# Labels on other documents are compared by id. 29н labels point at old chunk ids
# that do not exist after re-chunking: each old chunk is mapped to the new chunks
# whose words lie mostly inside it (printed for manual review). Paid: embeddings only.
import argparse
import json
import re
import sqlite3
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from config.settings import settings
from eval.run_retrieval_eval import evaluate, init_engine, load_gt, make_retrieval_fn
from scripts.hybrid_29n_table_context import SOURCE, build_store, variant_docs
from src.infra.llm_factory import get_embedding_model
from src.v7.bridge import build_v7_runtime

COVER = 0.6  # share of a new chunk's words found in the old labeled chunk
COMMON = 0.1  # words in more than this share of chunks (headings) do not count


def words(text):
    return set(re.findall(r"[а-яёa-z0-9]{4,}", text.lower()))


def old_texts(db_path, ids):
    """Old labeled 29н chunk texts by metadata chunk_id (Chroma row ids are uuids)."""
    con = sqlite3.connect(f"file:{db_path}/chroma.sqlite3?mode=ro", uri=True)
    out = {}
    for cid in ids:
        num = int(cid.split("#")[1])
        row = con.execute(
            "select d.string_value from embedding_metadata s"
            " join embedding_metadata c on c.id=s.id and c.key='chunk_id'"
            " join embedding_metadata d on d.id=s.id and d.key='chroma:document'"
            " where s.key='source' and s.string_value=? and c.int_value=?",
            (SOURCE, num),
        ).fetchone()
        out[cid] = row[0] if row else None
    return out


def map_labels(old, docs):
    """old 29н chunk id -> new chunk ids whose words lie mostly inside the old chunk."""
    bags = [words(d.page_content) for d in docs]
    df = {}
    for bag in bags:
        for w in bag:
            df[w] = df.get(w, 0) + 1
    common = {w for w, n in df.items() if n > COMMON * len(docs)}
    mapping = {}
    for cid, text in old.items():
        ow = words(text or "") - common
        hits = []
        for d, bag in zip(docs, bags):
            nw = bag - common
            if len(nw) >= 5 and len(nw & ow) / len(nw) >= COVER:
                hits.append(f"{SOURCE}#{d.metadata['chunk_id']}")
        mapping[cid] = hits
    return mapping


def relabel(gt, mapping):
    out = []
    for rec in gt:
        rel = []
        for cid in rec["relevant_chunk_ids"]:
            rel += mapping.get(cid, []) if cid.startswith(SOURCE) else [cid]
        out.append({**rec, "relevant_chunk_ids": rel or ["<unmapped>"]})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path, help="saved Docling IR JSON for 29н")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--src-db", type=Path, default=Path(settings.CHROMA_DB_PATH))
    parser.add_argument("--work", type=Path, required=True, help="scratch dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map-only", action="store_true", help="label mapping, no API")
    parser.add_argument("--variants", nargs="+", help="subset of variants to run")
    args = parser.parse_args()

    gt = load_gt(args.gt)
    doc = DoclingDocument.model_validate_json(args.ir.read_text(encoding="utf-8"))
    variants = variant_docs(doc)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}
    old_ids = sorted(
        {c for r in gt for c in r["relevant_chunk_ids"] if c.startswith(SOURCE)}
    )
    old = old_texts(args.src_db, old_ids)
    mappings = {name: map_labels(old, docs) for name, docs in variants.items()}
    for cid in old_ids:
        print(cid, "|", (old[cid] or "<missing>")[:90].replace("\n", " "))
        for name, m in mappings.items():
            print("   ", name, len(m[cid]), m[cid][:8])
    if args.map_only:
        return

    runs = {"production": gt}
    runs.update({name: relabel(gt, mappings[name]) for name in variants})
    report = {"gt": str(args.gt), "cover": COVER, "mappings": mappings, "runs": {}}
    embed = None
    for name, labels in runs.items():
        if name == "production":
            runtime = init_engine()
        else:
            embed = embed or get_embedding_model()
            store = build_store(args.src_db, args.work / name, variants[name], embed)
            runtime = build_v7_runtime(store)
        result = evaluate(labels, make_retrieval_fn("simple", runtime))
        report["runs"][name] = {
            "metrics": result["metrics"],
            "per_source": result["per_source"],
            "errors": result["errors"],
            "records": [
                {k: r[k] for k in ("question", "source", "rank")}
                | {"relevant": lab["relevant_chunk_ids"]}
                for r, lab in zip(result["records"], labels)
            ],
        }
        m = result["metrics"]
        print(name, {k: round(v, 3) for k, v in m.items()}, "errors", result["errors"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
