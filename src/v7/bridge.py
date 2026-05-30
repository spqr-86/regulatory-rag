"""Bridge: adapt existing ChromaDB vector store for v7 pipeline.

Responsibilities:
1. Wrap ChromaDB's similarity_search_with_score -> v7 dict format
2. Build BM25 corpus from ChromaDB docs
3. Inject search functions into rag_simple / rag_complex nodes
4. Inject FlashRank reranker into rag_complex
5. Inject LLM-backed verify, rewrite, and generate functions
"""

from __future__ import annotations

import re
import threading
from typing import Callable, List, Literal

import structlog
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.backends.vector_store import (
    VectorStoreBackend,
)  # noqa: F401 — used in isinstance checks
from src.infra.llm_factory import get_complex_llm, get_simple_llm  # noqa: F401
from src.infra.parsers import (
    extract_text,
)  # noqa: F401  # used by other make_*_fn
from src.infra.prompt_manager import PromptManager
from src.v7.nlp_core import init_bm25_index
from src.v7.nodes import generate_answer as generate_answer_mod
from src.v7.nodes import llm_verifier as llm_verifier_mod
from src.v7.nodes import rag_complex as rag_complex_mod
from src.v7.nodes import rag_simple as rag_simple_mod
from src.v7.nodes import rewriter as rewriter_mod
from src.v7.nodes import visual_enrichment as visual_enrichment_mod
from src.v7.nodes.llm_verifier import VERIFIER_SYSTEM_PROMPT
from src.v7.nodes.utils import extract_doc_identifiers
from src.v7.state_types import VerificationResult

logger = structlog.get_logger()
_pm = PromptManager()

# Module-level RLock protecting init_v7_from_chroma.
# Concurrent calls serialize to prevent readers observing a half-initialized
# pipeline state (7 set_*_fn injectors + BM25 build are not atomic individually).
_init_lock = threading.RLock()


def make_visual_proof_fn() -> Callable[[str, int, list, str], str]:
    """Create a visual proof function for v7 visual_enrichment node.

    Wraps _visual_proof_impl from agent_tools.
    Signature: fn(source, page_no, bbox, mode) -> str (path or analysis text).
    Returns None if agent_tools is unavailable.
    """
    try:
        from src.agent_tools import _visual_proof_impl
    except ImportError:
        logger.warning(
            "make_visual_proof_fn: src.agent_tools not available, visual enrichment disabled"
        )
        return None  # type: ignore[return-value]

    def _visual_proof(source: str, page_no: int, bbox: list, mode: str = "show") -> str:
        return _visual_proof_impl(source, page_no, bbox, mode)

    return _visual_proof


def make_rerank_fn(
    model_name: str = "ms-marco-MiniLM-L-12-v2",
    cache_dir: str = ".reranker_cache",
) -> Callable[[str, List[dict], int], List[dict]]:
    """Create a FlashRank reranker function for v7 rag_complex.

    Signature: fn(query, passages, top_k) -> passages (reranked, ≤ top_k items).
    Each passage must have a 'text' key.

    Preserves original vector_score so downstream MMR/gates use vector scores
    (FlashRank scores are not calibrated for thresholds — see CLAUDE.md session 22).
    """
    from flashrank import Ranker, RerankRequest

    _ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def _rerank(query: str, passages: List[dict], top_k: int) -> List[dict]:
        if not passages:
            return passages

        # Build indexed map for O(1) lookup after rerank
        text_to_original = {p.get("text", ""): p for p in passages}
        passages_for_ranker = [{"text": p.get("text", "")} for p in passages]
        rerank_request = RerankRequest(query=query, passages=passages_for_ranker)
        results = _ranker.rerank(rerank_request)

        reranked: List[dict] = []
        for result in results[:top_k]:
            original = text_to_original.get(result["text"], {})
            vector_score = original.get("vector_score", original.get("score", 0.0))
            reranked.append(
                {
                    **original,
                    "vector_score": vector_score,
                    "score": round(float(result["score"]), 4),
                }
            )
        return reranked

    return _rerank


def make_crossencoder_rerank_fn(
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    batch_size: int = 32,
) -> Callable[[str, List[dict], int], List[dict]]:
    """CrossEncoder reranker with sigmoid normalization to [0, 1].

    Multilingual (works on Russian). Raw logits (~-5..+7) are squashed via
    sigmoid so triage thresholds calibrated on FlashRank (HARD=0.50, SOFT=0.38)
    keep their meaning.
    """
    import math

    from sentence_transformers import CrossEncoder

    _model = CrossEncoder(model_name)

    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def _rerank(query: str, passages: List[dict], top_k: int) -> List[dict]:
        if not passages:
            return passages

        pairs = [(query, p.get("text", "")) for p in passages]
        raw_scores = _model.predict(pairs, batch_size=batch_size)
        scored = [(p, _sigmoid(float(s))) for p, s in zip(passages, raw_scores)]
        scored.sort(key=lambda item: item[1], reverse=True)

        reranked: List[dict] = []
        for original, score in scored[:top_k]:
            vector_score = original.get("vector_score", original.get("score", 0.0))
            reranked.append(
                {
                    **original,
                    "vector_score": vector_score,
                    "score": round(score, 4),
                }
            )
        return reranked

    return _rerank


def make_vector_search_fn(vector_store) -> Callable[..., List[dict]]:
    """Create a v7-compatible vector search function from ChromaDB store.

    v7 interface: fn(query, filters=None, top_k=12, **kwargs) -> list[dict]
    Each dict has: text, metadata, score.
    """

    def _search(
        query: str,
        filters: dict | None = None,
        top_k: int = 12,
        **kwargs,
    ) -> List[dict]:
        docs_and_scores = vector_store.similarity_search_with_score(query, k=top_k)
        results = []
        for doc, distance in docs_and_scores:
            # ChromaDB returns L2 distance (0..inf). Convert to similarity (0..1).
            similarity = 1.0 / (1.0 + distance)
            results.append(
                {
                    "text": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": round(similarity, 4),
                }
            )
        return results

    return _search


def make_section_fetch_fn(
    vector_store,
    max_section_chunks: int = 50,
    anchor_count: int = 3,
) -> Callable[[List[dict]], List[dict]]:
    """Create a section-aware expander from ChromaDB store.

    Takes the top N anchor passages, extracts parent_section + source from each,
    and fetches all chunks from those sections (up to max_section_chunks per anchor).
    Multiple anchors help when the answer is scattered across sections from
    different sources (e.g. ст.229 ТК + 223н разд.III for one question).
    Returns passages not already in the input list.
    """

    def _fetch_one_section(section: str, source: str) -> List[dict]:
        if not section or not source:
            return []
        try:
            if isinstance(vector_store, VectorStoreBackend):
                docs = vector_store.get_by_filter(
                    {
                        "parent_section": section,
                        "source": source,
                    },
                    limit=max_section_chunks,
                )
                return [
                    {
                        "text": d.page_content,
                        "metadata": dict(d.metadata or {}),
                        "score": 0.0,
                    }
                    for d in docs
                ]
            col = vector_store._collection
            results = col.get(
                where={
                    "$and": [
                        {"parent_section": {"$eq": section}},
                        {"source": {"$eq": source}},
                    ]
                },
                include=["documents", "metadatas"],
                limit=max_section_chunks,
            )
            extra = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                extra.append(
                    {
                        "text": doc,
                        "metadata": dict(meta),
                        "score": 0.0,
                    }
                )
            return extra
        except Exception as exc:
            logger.warning("section_fetch failed: %s", exc)
            return []

    def _fetch_section(passages: List[dict]) -> List[dict]:
        if not passages:
            return []
        seen_sections: set[tuple[str, str]] = set()
        out: List[dict] = []
        for anchor in passages[:anchor_count]:
            meta = anchor.get("metadata", {})
            section = (meta.get("parent_section") or "").strip()
            source = meta.get("source", "")
            key = (section, source)
            if not section or not source or key in seen_sections:
                continue
            seen_sections.add(key)
            out.extend(_fetch_one_section(section, source))
        return out

    return _fetch_section


class _VerifierVerdictSchema(BaseModel):
    """Pydantic schema for structured verifier output (Gemini → typed object).

    Maps 1:1 to ``VerificationResult`` TypedDict; the name is suffixed with
    ``Schema`` to avoid clashing with the ``VerifierVerdict`` Literal alias
    declared in ``state_types``.
    """

    verdict: Literal["sufficient", "rewrite", "escalate"]
    reason: str = ""
    rewrite_hint: str = ""
    missing_aspects: List[str] = Field(default_factory=list)
    confidence: float = 0.0


def make_verify_fn(llm) -> Callable[..., VerificationResult]:
    """Create a v7-compatible verify function backed by Gemini LLM.

    Uses ``llm.with_structured_output(_VerifierVerdictSchema)`` so Gemini
    returns a typed Pydantic object directly — no regex/braces JSON parsing,
    no «could not parse JSON» warnings on malformed responses.
    """
    structured_llm = llm.with_structured_output(_VerifierVerdictSchema)

    def _verify(
        original_query: str, active_query: str, passages: List[dict]
    ) -> VerificationResult:
        passages_text = "\n\n".join(
            f"[{i + 1}] (score={p.get('score', 'N/A')}): {p.get('text', '')}"
            for i, p in enumerate(passages)
        )
        prompt = (
            f"{VERIFIER_SYSTEM_PROMPT}\n\n"
            f"Запрос пользователя: {original_query}\n"
            f"Активный запрос: {active_query}\n\n"
            f"Найденные passages ({len(passages)}):\n{passages_text}"
        )
        try:
            verdict_obj = structured_llm.invoke([HumanMessage(content=prompt)])
            if verdict_obj is None:
                raise ValueError("Structured output returned None")
            return VerificationResult(
                verdict=verdict_obj.verdict,
                reason=verdict_obj.reason,
                rewrite_hint=verdict_obj.rewrite_hint,
                missing_aspects=list(verdict_obj.missing_aspects),
                confidence=float(verdict_obj.confidence),
            )
        except Exception as exc:
            logger.warning("LLM verify failed: %s", exc)
            return VerificationResult(
                verdict="escalate",
                reason=f"LLM parse error: {type(exc).__name__}",
                missing_aspects=[],
                confidence=0.0,
            )

    return _verify


def make_rewrite_fn(llm) -> Callable[..., str]:
    """Create a v7-compatible rewrite function backed by Gemini LLM."""

    def _rewrite(
        original_query: str,
        active_query: str,
        rewrite_hint: str,
        missing_aspects: List[str],
    ) -> str:
        protected_ids = extract_doc_identifiers(original_query)
        aspects_str = ", ".join(missing_aspects) if missing_aspects else ""
        prompt = (
            "Переформулируй поисковый запрос для поиска в базе нормативных документов.\n"
            "Сохрани ВСЕ номера документов (ГОСТ, СП, СНиП и т.д.) из исходного запроса.\n\n"
            f"Исходный запрос: {original_query}\n"
            f"Текущий запрос: {active_query}\n"
            f"Подсказка: {rewrite_hint}\n"
            f"Недостающие аспекты: {aspects_str}\n\n"
            "Верни ТОЛЬКО переформулированный запрос, без пояснений."
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            rewritten = extract_text(response.content).strip()
            if not rewritten:
                raise ValueError("Empty rewrite response")
            # Protect doc identifiers
            for doc_id in protected_ids:
                if doc_id not in rewritten:
                    rewritten = f"{rewritten} [{doc_id}]"
            return rewritten
        except Exception as exc:
            logger.warning("LLM rewrite failed: %s, falling back to stub", exc)
            # Fallback to stub logic
            rewritten = (
                f"{original_query} ({aspects_str})" if aspects_str else original_query
            )
            for doc_id in protected_ids:
                if doc_id not in rewritten:
                    rewritten = f"{rewritten} [{doc_id}]"
            return rewritten

    return _rewrite


def make_expand_fn(llm, n: int = 3) -> Callable[[str, int], List[str]]:
    """Create an LLM-backed query expansion function for V8 multi-query expand.

    Signature: fn(query: str, n: int) -> list[str] — list of alternative queries.
    On failure returns empty list (caller falls back to single-query mode).
    """

    def _expand(query: str, n: int = n) -> List[str]:
        prompt = _pm.render("query_expand", query=query, n=n)
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = extract_text(response.content).strip()
            alternatives = [line.strip() for line in raw.splitlines() if line.strip()]
            return alternatives[:n]
        except Exception as exc:
            logger.warning("LLM expand failed: %s", exc)
            return []

    return _expand


# Patterns for Russian references to clauses/articles in regulations
_REF_PATTERNS = [
    re.compile(r"пункт\w*\s+(\d+)", re.IGNORECASE),
    re.compile(r'подпункт\w*\s+[«"]?([а-яё])[»"]?', re.IGNORECASE),
    re.compile(r"стать\w+\s+(\d+)", re.IGNORECASE),
]


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

    Two mechanisms:
    1. Explicit refs: searches for "пункт N", "статью N" in passage text and pulls
       chunks from the same source where that number appears.
    2. BM25 re-search: for each unique source in passages runs bm25_search
       on the query and adds chunks from that source (catches reverse references,
       e.g. п.60 referencing п.46.в, not the other way round).
    """
    if not passages:
        return passages

    existing_texts = {p["text"] for p in passages}
    extra: list[dict] = []

    def _add(text: str, metadata: dict) -> None:
        if text and text not in existing_texts:
            existing_texts.add(text)
            extra.append(
                {"text": text, "score": 0.35, "metadata": metadata, "cross_ref": True}
            )

    # ── Mechanism 1: explicit refs in passage text ───────────────────────────
    for passage in passages:
        source = passage.get("metadata", {}).get("source", "")
        if not source:
            continue
        refs = _extract_refs(passage.get("text", ""))
        for ref in refs:
            try:
                docs = backend.get_by_filter(
                    where={"source": source},
                    limit=500,
                )
                for doc in docs:
                    if ref in doc.page_content:
                        _add(doc.page_content, doc.metadata)
            except Exception:
                continue

    # ── Mechanism 2: BM25 re-search within same sources ───────────────────────
    if query:
        try:
            from src.v7.nlp_core import bm25_search

            unique_sources = {
                p.get("metadata", {}).get("source", "") for p in passages
            } - {""}
            bm25_results = bm25_search(query, top_k=30)
            for r in bm25_results:
                if r.get("metadata", {}).get("source") in unique_sources:
                    _add(r.get("text", ""), r.get("metadata", {}))
        except Exception:
            pass

    # ── Mechanism 3: numbered-list expansion ─────────────────────────────────
    # If a passage starts with "N. а)" (subparagraph of a numbered list), pull
    # sibling subparagraphs "N. б)", "N. в)" ... from the same source so the
    # full enumeration is available for generation.
    _SUBPARA_RE = re.compile(r"^\s*(\d+)\.\s+[а-яё]\)", re.IGNORECASE)
    for passage in list(passages):  # iterate original only
        text = passage.get("text", "")
        m = _SUBPARA_RE.match(text)
        if not m:
            continue
        para_num = m.group(1)
        source = passage.get("metadata", {}).get("source", "")
        if not source:
            continue
        try:
            docs = backend.get_by_filter(where={"source": source}, limit=500)
            for doc in docs:
                if _SUBPARA_RE.match(doc.page_content) and doc.page_content.startswith(
                    para_num + "."
                ):
                    _add(doc.page_content, doc.metadata)
        except Exception:
            continue

    # ── Mechanism 4: same-bbox block expansion ───────────────────────────────
    # HybridChunker sometimes splits one PDF text block into multiple chunks
    # that share the same source + page_no + bbox. Pull all siblings so a
    # truncated sentence ("связана с ...") gets its continuation.
    # Siblings inherit the parent passage score so they rank alongside it.
    bbox_extra: list[dict] = []

    def _add_bbox(text: str, metadata: dict, parent_score: float) -> None:
        if text and text not in existing_texts:
            existing_texts.add(text)
            bbox_extra.append(
                {
                    "text": text,
                    "score": parent_score,
                    "metadata": metadata,
                    "cross_ref": True,
                }
            )

    for passage in list(passages):  # iterate original only
        meta = passage.get("metadata", {})
        source = meta.get("source", "")
        page_no = meta.get("page_no")
        bbox = meta.get("bbox")
        if not (source and page_no is not None and bbox):
            continue
        parent_score = passage.get("score", 0.35)
        try:
            docs = backend.get_by_filter(where={"source": source}, limit=500)
            for doc in docs:
                dm = doc.metadata
                if (
                    dm.get("page_no") == page_no
                    and dm.get("bbox") == bbox
                    and dm.get("source") == source
                ):
                    _add_bbox(doc.page_content, doc.metadata, parent_score)
        except Exception:
            continue

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


def make_generate_fn(llm, backend=None) -> Callable[[str, str, List[dict]], str]:
    """Create an LLM-backed answer generation function for v7 generate_answer node.

    Signature: fn(query, active_query, passages) -> answer_text.
    Relies on ChatGoogleGenerativeAI's built-in retry (max_retries=3 in
    ``get_gemini_llm``) for transient 5xx / 429 errors. On final failure
    falls back to a stub (concatenated top passages).
    If backend is provided, cross-reference expansion is applied before generation.
    """

    def _call_llm(prompt: str) -> str:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = extract_text(response.content).strip()
        if not answer:
            raise ValueError("Empty generation response")
        return answer

    def _score_label(score: float) -> str:
        if score >= 0.6:
            return "HIGH"
        if score >= 0.4:
            return "MED"
        return "LOW"

    def _short_source(passage: dict) -> str:
        """Extract short document name from metadata.source."""
        raw = passage.get("metadata", {}).get("source", "")
        # "Трудовой кодекс РФ - Система Охрана труда. Премиальная версия.pdf"
        # → "Трудовой кодекс РФ"
        name = raw.split(" - ")[0].replace(".pdf", "").strip()
        return name or "Unknown source"

    def _section_label(passage: dict) -> str:
        """Extract short section label (parent_section / heading_path) for the chunk.

        Helps the LLM and reranker distinguish similar wording across different
        normative sections (e.g. Ст.228 vs Ст.228.1).
        """
        meta = passage.get("metadata", {}) or {}
        raw = (meta.get("parent_section") or meta.get("heading_path") or "").strip()
        if not raw:
            return ""
        # Truncate very long section titles
        return raw[:120] + ("…" if len(raw) > 120 else "")

    def _chunk_header(i: int, passage: dict) -> str:
        score = _score_label(passage.get("score", 0.0))
        src = _short_source(passage)
        section = _section_label(passage)
        if section:
            return f"[{i + 1}] ({score}) [Источник: {src}; Раздел: {section}]"
        return f"[{i + 1}] ({score}) [Источник: {src}]"

    def _generate(query: str, active_query: str, passages: List[dict]) -> str:
        if not passages:
            return ""
        expanded = (
            expand_cross_references(passages, backend, query=query)
            if backend
            else passages
        )
        # final_passages is already capped at 24 upstream (merge_all_passages);
        # cross-reference expansion appends extra passages — allow up to 30 so
        # low-ranked but answer-bearing cross-refs (e.g. п.60 for программа В) are included.
        top_passages = expanded[:30]
        passages_text = "\n\n".join(
            f"{_chunk_header(i, p)}\n{p.get('text', '')}"
            for i, p in enumerate(top_passages)
        )
        prompt = _pm.render(
            "generate_answer",
            query=query,
            context=passages_text,
            passages_count=len(top_passages),
        )
        try:
            return _call_llm(prompt)
        except Exception as exc:
            logger.warning(
                "LLM generate failed after retries: %s, falling back to stub", exc
            )
            return "\n\n".join(p.get("text", "") for p in passages[:10])

    return _generate


def init_v7_pipeline(vector_store, llm_provider: str | None = "gemini") -> None:
    """Initialize V7 pipeline from a vector store (raw Chroma or VectorStoreBackend).

    1. Creates vector search wrapper
    2. Injects it into rag_simple and rag_complex nodes
    3. Builds BM25 index from full corpus
    4. Injects FlashRank reranker into rag_complex
    5. Injects LLM-backed verify, rewrite, and generate functions (if provider available)
    """
    from config.settings import settings

    # Serialize concurrent initialization: the 7 set_*_fn injectors + BM25 build
    # are not individually atomic, so a reader could observe a half-initialized
    # pipeline without this lock.
    with _init_lock:
        search_fn = make_vector_search_fn(vector_store)
        rag_simple_mod.set_vector_search(search_fn)
        rag_complex_mod.set_vector_search(search_fn)

        # Build BM25 corpus. VectorStoreBackend exposes iter_all_documents();
        # legacy raw Chroma exposes .get() — kept for backward compat.
        if isinstance(vector_store, VectorStoreBackend):
            corpus = list(vector_store.iter_all_documents())
        else:
            all_data = vector_store.get(include=["metadatas", "documents"])
            corpus = [
                {"text": doc, "metadata": meta}
                for doc, meta in zip(all_data["documents"], all_data["metadatas"])
            ]
        init_bm25_index(corpus)

        # Inject section-aware expander for complex path
        try:
            section_fetch_fn = make_section_fetch_fn(vector_store)
            rag_complex_mod.set_section_fetch_fn(section_fetch_fn)
            logger.info("v7 section-aware expander injected successfully")
        except Exception as exc:
            logger.warning("Failed to initialize section fetch for v7: %s.", exc)

        # Inject reranker (FlashRank or CrossEncoder+sigmoid)
        try:
            backend = (settings.RERANKER_BACKEND or "flashrank").lower()
            if backend == "crossencoder":
                rerank_fn = make_crossencoder_rerank_fn(
                    model_name=settings.CROSSENCODER_MODEL,
                )
                logger.info(
                    "v7 CrossEncoder reranker injected (model=%s)",
                    settings.CROSSENCODER_MODEL,
                )
            else:
                rerank_fn = make_rerank_fn(
                    model_name=settings.RERANKING_MODEL,
                    cache_dir=settings.FLASHRANK_CACHE_DIR,
                )
                logger.info("v7 FlashRank reranker injected successfully")
            rag_complex_mod.set_rerank_fn(rerank_fn)
            rag_simple_mod.set_reranker(rerank_fn)
        except Exception as exc:
            logger.warning(
                "Failed to initialize reranker for v7: %s. Complex path will skip reranking.",
                exc,
            )

        # Inject LLM-backed verify, rewrite, generate, and expand functions
        if llm_provider:
            try:
                verifier_llm = get_complex_llm(
                    thinking_budget=1024, response_mime_type="application/json"
                )
                llm_verifier_mod.set_verify_fn(make_verify_fn(verifier_llm))

                rewriter_llm = get_complex_llm(thinking_budget=1024)
                rewriter_mod.set_rewrite_fn(make_rewrite_fn(rewriter_llm))

                # Split generators: cheap model for the simple path, full-quality
                # model for the complex path (where verifier/rewriter already paid
                # the latency tax — answer quality dominates CPS savings).
                generator_llm_complex = get_complex_llm(thinking_budget=4096)
                generator_llm_simple = get_simple_llm(thinking_budget=4096)
                xref_backend = (
                    vector_store
                    if isinstance(vector_store, VectorStoreBackend)
                    else None
                )
                generate_answer_mod.set_generate_fns(
                    simple=make_generate_fn(generator_llm_simple, backend=xref_backend),
                    complex_=make_generate_fn(
                        generator_llm_complex, backend=xref_backend
                    ),
                )

                expander_llm = get_simple_llm(thinking_budget=0)
                rag_simple_mod.set_expand_fn(make_expand_fn(expander_llm))

                logger.info(
                    "v7 LLM verifier, rewriter, generator, and expander injected successfully"
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize LLM for v7 verifier/rewriter/generator: %s. "
                    "Using rule-based stubs.",
                    exc,
                )

        # Inject visual proof function for visual_enrichment node
        try:
            visual_proof_fn = make_visual_proof_fn()
            if visual_proof_fn is not None:
                visual_enrichment_mod.set_visual_proof_fn(visual_proof_fn)
                logger.info("v7 visual proof injected successfully")
        except Exception as exc:
            logger.warning("Failed to initialize visual proof for v7: %s.", exc)


# Backward-compat alias — remove after one release cycle.
init_v7_from_chroma = init_v7_pipeline
