"""Dense retrieval regression on 29н: current chunks vs explicit table rows (paid embeddings)."""

# ANCHOR: saved 29н IR + golden 29н questions -> hit@k per variant, JSON report.
# Labels: rows/text of the labeled chunks in the stored index, matched by row id
# (table) or word containment (text), so they survive changed chunk boundaries.
# Corpus is 29н only; this isolates chunking, not the production hybrid pipeline.
import argparse
import json
import math
import re
import sqlite3
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from scripts.compare_29n_table_context import (
    explicit_rows,
    merge_page_continuations,
    norm,
)
from src.indexing.file_handler import DocumentProcessor
from src.infra.llm_factory import get_embedding_model

QUESTION_FILES = (
    "eval/data/golden_retrieval_labeled.jsonl",
    "eval/data/golden_retrieval_labeled_ext.jsonl",
)
# Docling table text: "18.1, <col> = value"; explicit units: "Колонка 1: 18.1".
ROW_IN_TABLE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)+|\d+), ")
ROW_IN_UNIT = re.compile(r"^Колонка 1: (\S+)$", re.M)
KS = (1, 3, 5, 12)
PLACEHOLDER = re.compile(r"^Колонка (?!1:)\d+: ", re.M)


def words(text):
    return set(re.findall(r"[а-яёa-z0-9]{4,}", text.lower()))


def stored_chunks(db_path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = con.execute(
        "select id, key, coalesce(string_value, cast(int_value as text))"
        " from embedding_metadata where id in (select id from embedding_metadata"
        " where key='source' and string_value='29н.pdf')"
    )
    chunks = {}
    for emb_id, key, value in rows:
        chunks.setdefault(emb_id, {})[key] = value
    return {
        f"29н.pdf#{m['chunk_id']}": m
        for m in chunks.values()
        if "chunk_id" in m and "chroma:document" in m
    }


def label_of(meta):
    text = meta["chroma:document"]
    if meta.get("element_type") == "table":
        ids = set(ROW_IN_TABLE.findall(text))
        if ids:
            return {"kind": "table", "rows": sorted(ids)}
    return {"kind": "text", "words": sorted(words(text))}


def chunk_rows(text):
    return set(ROW_IN_TABLE.findall(text)) | set(ROW_IN_UNIT.findall(text))


def is_hit(text, labels):
    for label in labels:
        # Fixed row label: row ids repeat across appendices, so the row must also
        # carry its own words.
        if label["kind"] == "row":
            low = text.lower()
            if label["row"] in chunk_rows(text) and all(
                m in low for m in label["must"]
            ):
                return True
            # Docling may lose the row id (6.1/6.2 merged into a row "7"); a long
            # verbatim phrase of the row name still identifies it.
            if label.get("phrase") and norm(label["phrase"]) in norm(text):
                return True
        if label["kind"] == "table" and chunk_rows(text) & set(label["rows"]):
            return True
        if label["kind"] == "text" and label["words"]:
            overlap = len(words(text) & set(label["words"])) / len(label["words"])
            if overlap >= 0.6:
                return True
    return False


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def evaluate(name, texts, questions, embed):
    vectors = embed.embed_documents(texts)
    per_question, hits = [], {k: 0 for k in KS}
    for q in questions:
        scores = sorted(
            ((cosine(q["vector"], v), i) for i, v in enumerate(vectors)), reverse=True
        )
        ranks = [
            r for r, (_, i) in enumerate(scores, 1) if is_hit(texts[i], q["labels"])
        ]
        first = ranks[0] if ranks else None
        for k in KS:
            hits[k] += bool(first and first <= k)
        per_question.append(
            {"question": q["question"], "kind": q["kind"], "rank": first}
        )
    return {
        "chunks": len(texts),
        "hit_at": {k: f"{hits[k]}/{len(questions)}" for k in KS},
        "per_question": per_question,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path, help="saved Docling IR JSON for 29н")
    parser.add_argument(
        "--stored-db", type=Path, default=Path("chroma_db/chroma.sqlite3")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--row-questions",
        type=Path,
        help="JSONL with fixed table rows (question, row, must) instead of golden",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check that every label matches a chunk in each variant; no embeddings",
    )
    args = parser.parse_args()

    import tiktoken

    questions = []
    for line in (
        args.row_questions.read_text(encoding="utf-8").splitlines()
        if args.row_questions
        else []
    ):
        q = json.loads(line)
        label = {"kind": "row", "row": q["row"], "must": q["must"]}
        label["phrase"] = q.get("phrase")
        questions.append(
            {"question": q["question"], "labels": [label], "kind": "table"}
        )
    stored = {} if args.row_questions else stored_chunks(args.stored_db)
    for path in () if args.row_questions else QUESTION_FILES:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            q = json.loads(line)
            ids = [r for r in q["relevant_chunk_ids"] if r.startswith("29н.pdf#")]
            labels = [label_of(stored[r]) for r in ids if r in stored]
            if labels and q["question"] not in {x["question"] for x in questions}:
                kind = (
                    "table" if any(lb["kind"] == "table" for lb in labels) else "text"
                )
                questions.append(
                    {"question": q["question"], "labels": labels, "kind": kind}
                )

    doc = DoclingDocument.model_validate_json(args.ir.read_text(encoding="utf-8"))
    processed = DocumentProcessor()._process_docling_document(doc, "29н.pdf")
    current = [c.page_content for c in processed]
    units = explicit_rows(
        merge_page_continuations(doc), tiktoken.get_encoding("cl100k_base")
    )
    # Table chunks are replaced by row units; text chunks stay identical.
    explicit = [
        c.page_content for c in processed if c.metadata.get("element_type") != "table"
    ] + [u["text"] for u in units]

    # explicit_ctx: same row units, but with the section heading every current chunk
    # gets for embedding, and without placeholder column names.
    section_of = {}
    for c in processed:
        for ref in json.loads(c.metadata.get("doc_item_refs", "[]")):
            section_of.setdefault(ref, c.metadata.get("parent_section", ""))
    # Heading per merged group: row ids repeat across appendices.
    explicit_ctx = [
        c.page_content for c in processed if c.metadata.get("element_type") != "table"
    ]
    for group in merge_page_continuations(doc):
        heading = section_of.get(group[0].self_ref, "")
        explicit_ctx += [
            (heading + "\n" + PLACEHOLDER.sub("", u["text"])).strip()
            for u in explicit_rows([group], tiktoken.get_encoding("cl100k_base"))
        ]
    variants = (
        ("current", current),
        ("explicit", explicit),
        ("explicit_ctx", explicit_ctx),
    )
    if args.dry_run:
        for name, texts in variants:
            missing = [
                q["question"]
                for q in questions
                if not any(is_hit(t, q["labels"]) for t in texts)
            ]
            matches = [sum(is_hit(t, q["labels"]) for t in texts) for q in questions]
            print(name, len(texts), "unmatched:", missing, "matches:", matches)
        return

    embed = get_embedding_model()
    for q, vector in zip(
        questions, embed.embed_documents([q["question"] for q in questions])
    ):
        q["vector"] = vector
    report = {
        "questions": len(questions),
        "table_questions": sum(q["kind"] == "table" for q in questions),
        "variants": {
            name: evaluate(name, texts, questions, embed) for name, texts in variants
        },
        "labels": [
            {"question": q["question"], "labels": q["labels"]} for q in questions
        ],
    }
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, result in report["variants"].items():
        print(name, result["chunks"], result["hit_at"])


if __name__ == "__main__":
    main()
