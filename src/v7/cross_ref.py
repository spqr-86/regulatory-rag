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

# Patterns for Russian references to clauses/articles in regulations.
# Each pattern yields (kind, captured_value).
_REF_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("clause", re.compile(r"(?:пункт\w*|пп\.?)\s+(\d+)", re.IGNORECASE)),
    ("article", re.compile(r"стать\w+\s+(\d+)", re.IGNORECASE)),
    ("subpara", re.compile(r'(?:подпункт\w*|пп\.?)\s+[«"]([а-яё])[»"]', re.IGNORECASE)),
]

# Subparagraph of a numbered list, e.g. "46. а) ..."
_SUBPARA_RE = re.compile(r"^\s*(\d+)\.\s+[а-яё]\)", re.IGNORECASE)

# Caps on how many chunks each mechanism may add.
_CAP_MECH1_PER_PASSAGE = 5
_CAP_MECH1_TOTAL = 15
_CAP_MECH3_TOTAL = 10
_CAP_MECH4_TOTAL = 10


def _extract_refs(text: str) -> list[tuple[str, str]]:
    """Extract typed clause/article/subparagraph references from chunk text.

    Returns a list of (kind, value) tuples, e.g.:
      ("clause", "46"), ("article", "5"), ("subpara", "а")
    """
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for kind, pattern in _REF_PATTERNS:
        for m in pattern.finditer(text):
            ref = (kind, m.group(1))
            if ref not in seen:
                seen.add(ref)
                found.append(ref)
    return found


def _ref_matches_doc(
    kind: str, value: str, content: str, has_clause_context: bool
) -> bool:
    """Return True if (kind, value) structurally matches the document content.

    Uses regex-based structural matching to avoid false positives from bare
    substring search (e.g. "46" matching "2464", "146", years, sums).
    """
    v = re.escape(value)
    if kind == "clause":
        # Match "46. " at line start OR "пункт 46" as a phrase
        return bool(
            re.search(
                rf"(?m)(^\s*{v}\.(?=\s))|(\bпункт\w*\s+{v}\b)",
                content,
                re.IGNORECASE,
            )
        )
    if kind == "article":
        return bool(re.search(rf"\bстать\w+\s+{v}\b", content, re.IGNORECASE))
    if kind == "subpara":
        # Only match as a list marker "а)" at the start of a line AND only when
        # the passage already established a clause context (otherwise every Russian
        # chunk containing the letter would match).
        if not has_clause_context:
            return False
        return bool(re.search(rf"(?m)^\s*{v}\)", content, re.IGNORECASE))
    return False


def expand_cross_references(
    passages: list[dict],
    backend,
    query: str = "",
) -> list[dict]:
    """Fetch chunks linked to the found passages via cross-references.

    Mechanisms:
    1. Explicit refs: searches for "пункт N", "статью N" in passage text and pulls
       chunks from the same source where that number appears (structural regex match).
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
            "doc_id": meta.get("source", "unknown"),
        }
        if "chunk_id" in meta:
            passage["chunk_id"] = meta["chunk_id"]
        return passage

    def _add(doc, score: float = 0.35) -> None:
        if doc.page_content and doc.page_content not in existing_texts:
            existing_texts.add(doc.page_content)
            extra.append(_passage_from_doc(doc, score))

    # ── Mechanism 1: explicit refs in passage text ───────────────────────────
    mech1_total = 0
    for passage in passages:
        if mech1_total >= _CAP_MECH1_TOTAL:
            break
        source = passage.get("metadata", {}).get("source", "")
        if not source:
            continue
        refs = _extract_refs(passage.get("text", ""))
        if not refs:
            continue

        # Determine if the passage establishes a clause context (needed for
        # subpara matching — prevents bare letter "а" from matching everything).
        has_clause_context = any(kind == "clause" for kind, _ in refs)

        docs = _get_source_docs(source)
        added_for_passage = 0
        for doc in docs:
            if added_for_passage >= _CAP_MECH1_PER_PASSAGE:
                break
            if mech1_total >= _CAP_MECH1_TOTAL:
                break
            content = doc.page_content
            if content in existing_texts:
                continue
            matched = any(
                _ref_matches_doc(kind, value, content, has_clause_context)
                for kind, value in refs
            )
            if matched:
                _add(doc)
                added_for_passage += 1
                mech1_total += 1

    logger.debug("cross_ref.mech1_added", count=mech1_total)

    # ── Mechanism 2: BM25 re-search within same sources ───────────────────────
    mech2_total = 0
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
                        mech2_total += 1
        except Exception as exc:
            logger.warning("cross_ref.bm25_research_failed", error=str(exc))

    logger.debug("cross_ref.mech2_added", count=mech2_total)

    # ── Mechanism 3: numbered-list expansion ─────────────────────────────────
    # If a passage starts with "N. а)" (subparagraph of a numbered list), pull
    # sibling subparagraphs "N. б)", "N. в)" ... from the same source so the
    # full enumeration is available for generation.
    mech3_total = 0
    for passage in list(passages):  # iterate original only
        if mech3_total >= _CAP_MECH3_TOTAL:
            break
        text = passage.get("text", "")
        m = _SUBPARA_RE.match(text)
        if not m:
            continue
        para_num = m.group(1)
        source = passage.get("metadata", {}).get("source", "")
        if not source:
            continue
        for doc in _get_source_docs(source):
            if mech3_total >= _CAP_MECH3_TOTAL:
                break
            if _SUBPARA_RE.match(doc.page_content) and doc.page_content.startswith(
                para_num + "."
            ):
                if doc.page_content not in existing_texts:
                    _add(doc)
                    mech3_total += 1

    logger.debug("cross_ref.mech3_added", count=mech3_total)

    # ── Mechanism 4: same-bbox block expansion ───────────────────────────────
    # HybridChunker sometimes splits one PDF text block into multiple chunks
    # that share the same source + page_no + bbox. Pull all siblings so a
    # truncated sentence ("связана с ...") gets its continuation.
    # Siblings inherit the parent passage score so they rank alongside it.
    bbox_extra: list[dict] = []
    mech4_total = 0

    def _add_bbox(doc, parent_score: float) -> None:
        nonlocal mech4_total
        if mech4_total >= _CAP_MECH4_TOTAL:
            return
        if doc.page_content and doc.page_content not in existing_texts:
            existing_texts.add(doc.page_content)
            bbox_extra.append(_passage_from_doc(doc, parent_score))
            mech4_total += 1

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

    logger.debug("cross_ref.mech4_added", count=mech4_total)

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
