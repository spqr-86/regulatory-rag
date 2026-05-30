"""Cross-reference expansion for v7 passages.

Pure passage logic (no LLM-factory deps). Extracted from bridge.py.

Given the retrieved passages, pulls in chunks linked via in-text references
(пункт N, статья N), numbered-list siblings, same-bbox continuations, and a
BM25 re-search within the same sources. All per-source backend fetches are
cached for the duration of one call so each source is scanned at most once.
"""

from __future__ import annotations

import re

import structlog

from src.v7.nlp_core import bm25_search

logger = structlog.get_logger()

# Patterns for Russian references to clauses/articles in regulations
_REF_PATTERNS = [
    re.compile(r"пункт\w*\s+(\d+)", re.IGNORECASE),
    re.compile(r'подпункт\w*\s+[«"]?([а-яё])[»"]?', re.IGNORECASE),
    re.compile(r"стать\w+\s+(\d+)", re.IGNORECASE),
]

# Subparagraph of a numbered list, e.g. "46. а) ..."
_SUBPARA_RE = re.compile(r"^\s*(\d+)\.\s+[а-яё]\)", re.IGNORECASE)


def _extract_refs(text: str) -> list[str]:
    """Extract clause/article references from chunk text."""
    found: list[str] = []
    for pattern in _REF_PATTERNS:
        for m in pattern.finditer(text):
            ref = m.group(1)
            if ref not in found:
                found.append(ref)
    return found


def expand_cross_references(
    passages: list[dict],
    backend,
    query: str = "",
) -> list[dict]:
    """Fetch chunks linked to the found passages via cross-references.

    Mechanisms:
    1. Explicit refs: searches for "пункт N", "статью N" in passage text and pulls
       chunks from the same source where that number appears.
    2. BM25 re-search: for each unique source in passages runs bm25_search
       on the query and adds chunks from that source (catches reverse references).
    3. Numbered-list expansion: sibling subparagraphs "N. б)", "N. в)" ...
    4. Same-bbox block expansion: siblings sharing source + page_no + bbox.

    Each source's docs are fetched ONCE per call and cached (mechanisms 1/3/4
    reuse the same scan), avoiding O(passages × refs) full-collection scans.
    """
    if not passages:
        return passages

    existing_texts = {p["text"] for p in passages}
    extra: list[dict] = []

    # Per-call cache: source -> list[Document]. Fetch each source at most once.
    _source_docs_cache: dict[str, list] = {}

    def _get_source_docs(source: str) -> list:
        if source not in _source_docs_cache:
            try:
                _source_docs_cache[source] = backend.get_by_filter(
                    where={"source": source},
                    limit=500,
                )
            except Exception as exc:
                logger.warning(
                    "cross_ref source fetch failed", source=source, error=str(exc)
                )
                _source_docs_cache[source] = []
        return _source_docs_cache[source]

    def _passage_from_doc(doc, score: float) -> dict:
        meta = dict(doc.metadata or {})
        passage = {
            "text": doc.page_content,
            "score": score,
            "metadata": meta,
            "cross_ref": True,
        }
        if "chunk_id" in meta:
            passage["chunk_id"] = meta["chunk_id"]
        return passage

    def _add(doc, score: float = 0.35) -> None:
        if doc.page_content and doc.page_content not in existing_texts:
            existing_texts.add(doc.page_content)
            extra.append(_passage_from_doc(doc, score))

    # ── Mechanism 1: explicit refs in passage text ───────────────────────────
    for passage in passages:
        source = passage.get("metadata", {}).get("source", "")
        if not source:
            continue
        refs = _extract_refs(passage.get("text", ""))
        if not refs:
            continue
        docs = _get_source_docs(source)
        for doc in docs:
            if any(ref in doc.page_content for ref in refs):
                _add(doc)

    # ── Mechanism 2: BM25 re-search within same sources ───────────────────────
    if query:
        try:
            unique_sources = {
                p.get("metadata", {}).get("source", "") for p in passages
            } - {""}
            bm25_results = bm25_search(query, top_k=30)
            for r in bm25_results:
                if r.get("metadata", {}).get("source") in unique_sources:
                    text = r.get("text", "")
                    if text and text not in existing_texts:
                        existing_texts.add(text)
                        meta = r.get("metadata", {})
                        p = {
                            "text": text,
                            "score": 0.35,
                            "metadata": meta,
                            "cross_ref": True,
                        }
                        if r.get("chunk_id") is not None:
                            p["chunk_id"] = r["chunk_id"]
                        elif "chunk_id" in meta:
                            p["chunk_id"] = meta["chunk_id"]
                        extra.append(p)
        except Exception:
            pass

    # ── Mechanism 3: numbered-list expansion ─────────────────────────────────
    # If a passage starts with "N. а)" (subparagraph of a numbered list), pull
    # sibling subparagraphs "N. б)", "N. в)" ... from the same source so the
    # full enumeration is available for generation.
    for passage in list(passages):  # iterate original only
        text = passage.get("text", "")
        m = _SUBPARA_RE.match(text)
        if not m:
            continue
        para_num = m.group(1)
        source = passage.get("metadata", {}).get("source", "")
        if not source:
            continue
        for doc in _get_source_docs(source):
            if _SUBPARA_RE.match(doc.page_content) and doc.page_content.startswith(
                para_num + "."
            ):
                _add(doc)

    # ── Mechanism 4: same-bbox block expansion ───────────────────────────────
    # HybridChunker sometimes splits one PDF text block into multiple chunks
    # that share the same source + page_no + bbox. Pull all siblings so a
    # truncated sentence ("связана с ...") gets its continuation.
    # Siblings inherit the parent passage score so they rank alongside it.
    bbox_extra: list[dict] = []

    def _add_bbox(doc, parent_score: float) -> None:
        if doc.page_content and doc.page_content not in existing_texts:
            existing_texts.add(doc.page_content)
            bbox_extra.append(_passage_from_doc(doc, parent_score))

    for passage in list(passages):  # iterate original only
        meta = passage.get("metadata", {})
        source = meta.get("source", "")
        page_no = meta.get("page_no")
        bbox = meta.get("bbox")
        if not (source and page_no is not None and bbox):
            continue
        parent_score = passage.get("score", 0.35)
        for doc in _get_source_docs(source):
            dm = doc.metadata or {}
            if (
                dm.get("page_no") == page_no
                and dm.get("bbox") == bbox
                and dm.get("source") == source
            ):
                _add_bbox(doc, parent_score)

    # Insert bbox siblings right after their parent passage so they stay adjacent
    # in the ranked list and don't get pushed past the [:30] cutoff.
    result: list[dict] = []
    for passage in passages:
        result.append(passage)
        meta = passage.get("metadata", {})
        source, page_no, bbox = (
            meta.get("source"),
            meta.get("page_no"),
            meta.get("bbox"),
        )
        siblings = [
            p
            for p in bbox_extra
            if p["metadata"].get("source") == source
            and p["metadata"].get("page_no") == page_no
            and p["metadata"].get("bbox") == bbox
        ]
        result.extend(siblings)
    # Append remaining extra (mechanisms 1-3) at the end
    result.extend(extra)
    return result
