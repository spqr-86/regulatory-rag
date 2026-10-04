"""Hybrid retrieval regression on 29н rows: current vs explicit_ctx vs production rows in the full corpus (paid embeddings)."""

# ANCHOR: saved 29н IR + fixed-row questions -> rank of the labeled row in the
# production retrieval-only graph (vector + BM25 -> RRF -> CrossEncoder rerank).
# Each variant is a scratch copy of the full Chroma index with 29н chunks replaced;
# the production index is only read. Router is keyword-based: no LLM calls.
import argparse
import json
import shutil
from pathlib import Path

import tiktoken
from docling_core.types.doc import DoclingDocument
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from scripts.compare_29n_table_context import explicit_rows, merge_page_continuations
from scripts.retrieval_29n_table_context import PLACEHOLDER, is_hit
from src.indexing import file_handler
from src.indexing.file_handler import DocumentProcessor
from src.infra.llm_factory import get_embedding_model
from src.v7.bridge import build_v7_runtime
from src.v7.retrieval import retrieve_context

SOURCE = "29н.pdf"
KS = (1, 3, 5, 12)


def chroma_meta(meta):
    return {
        k: v
        for k, v in meta.items()
        if isinstance(v, (str, int, float, bool)) and v is not None
    }


def legacy_chunks(doc):
    """Chunker output before the row layer (production index as of 04.10)."""
    real = file_handler.explicit_table_chunks
    file_handler.explicit_table_chunks = lambda tables, chunks: chunks
    try:
        return DocumentProcessor()._process_docling_document(doc, SOURCE)
    finally:
        file_handler.explicit_table_chunks = real


def variant_docs(doc):
    processed = legacy_chunks(doc)
    current = [Document(c.page_content, metadata=dict(c.metadata)) for c in processed]
    # rows: the production chunker as committed (src/indexing/table_rows.py).
    rows = [
        Document(c.page_content, metadata=dict(c.metadata))
        for c in DocumentProcessor()._process_docling_document(doc, SOURCE)
    ]
    # explicit_ctx: text chunks unchanged, table chunks -> heading + one row unit.
    # Own copies: chunk ids are renumbered per variant below.
    ctx = [
        Document(c.page_content, metadata=dict(c.metadata))
        for c in current
        if c.metadata.get("element_type") != "table"
    ]
    meta_of, section_of = {}, {}
    for c in processed:
        for ref in json.loads(c.metadata.get("doc_item_refs", "[]")):
            meta_of.setdefault(ref, c.metadata)
            section_of.setdefault(ref, c.metadata.get("parent_section", ""))
    encoding = tiktoken.get_encoding("cl100k_base")
    for group in merge_page_continuations(doc):
        ref = group[0].self_ref
        base = {
            k: v
            for k, v in meta_of.get(ref, {}).items()
            if k not in {"bbox", "doc_item_refs"}
        }
        heading = section_of.get(ref, "")
        for u in explicit_rows([group], encoding):
            text = (heading + "\n" + PLACEHOLDER.sub("", u["text"])).strip()
            ctx.append(Document(text, metadata={**base, "element_type": "table"}))
    for docs in (current, ctx, rows):
        for i, d in enumerate(docs):
            d.metadata.update(source=SOURCE, chunk_id=i)
    return {"current": current, "explicit_ctx": ctx, "rows": rows}


def build_store(src_db, dst, docs, embed):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_db, dst)
    store = Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embed,
        persist_directory=str(dst),
        create_collection_if_not_exists=False,
    )
    store._collection.delete(where={"source": SOURCE})
    store.add_documents(
        [Document(d.page_content, metadata=chroma_meta(d.metadata)) for d in docs],
        ids=[f"{SOURCE}#{d.metadata['chunk_id']}" for d in docs],
    )
    return store


def ranks(result, labels):
    """1-based rank of the first labeled chunk: final context, then candidate pool."""
    final = [p.get("text", "") for p in result.final_context]
    pool = []
    for attempt in result.attempts:
        for p in attempt.get("passages") or []:
            if p.get("text", "") not in pool:
                pool.append(p.get("text", ""))

    def first(texts):
        return next((i + 1 for i, t in enumerate(texts) if is_hit(t, labels)), None)

    return first(final), first(pool), len(final), len(pool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path, help="saved Docling IR JSON for 29н")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--src-db", type=Path, default=Path(settings.CHROMA_DB_PATH))
    parser.add_argument("--work", type=Path, required=True, help="scratch dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="count chunks only")
    parser.add_argument("--variants", nargs="+", help="subset of variants to run")
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    doc = DoclingDocument.model_validate_json(args.ir.read_text(encoding="utf-8"))
    variants = variant_docs(doc)
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}
    encoding = tiktoken.get_encoding("cl100k_base")
    for name, docs in variants.items():
        tokens = sum(len(encoding.encode(d.page_content)) for d in docs)
        print(name, "chunks", len(docs), "tokens", tokens)
    if args.dry_run:
        return

    embed = get_embedding_model()
    report = {"questions": len(questions), "variants": {}}
    for name, docs in variants.items():
        store = build_store(args.src_db, args.work / name, docs, embed)
        runtime = build_v7_runtime(store)
        rows = []
        for q in questions:
            label = {"kind": "row", "row": q["row"], "must": q["must"]}
            label["phrase"] = q.get("phrase")
            result = retrieve_context(q["question"], runtime=runtime)
            final_rank, pool_rank, n_final, n_pool = ranks(result, [label])
            rows.append(
                {
                    "question": q["question"],
                    "row": q["row"],
                    "outcome": result.outcome,
                    "route": result.route,
                    "final_rank": final_rank,
                    "pool_rank": pool_rank,
                    "final_size": n_final,
                    "pool_size": n_pool,
                }
            )
        hit = [r["final_rank"] for r in rows]
        report["variants"][name] = {
            "chunks": len(docs),
            "final_hit_at": {k: sum(r is not None and r <= k for r in hit) for k in KS},
            "final_any": sum(r is not None for r in hit),
            "pool_any": sum(r["pool_rank"] is not None for r in rows),
            "rows": rows,
        }
        v = report["variants"][name]
        print(name, v["final_hit_at"], "final", v["final_any"], "pool", v["pool_any"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
