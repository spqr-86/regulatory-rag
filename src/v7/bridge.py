"""Bridge: adapt existing ChromaDB vector store for v7 pipeline.

Responsibilities:
1. Wrap ChromaDB's similarity_search_with_score -> v7 dict format
2. Build BM25 corpus from ChromaDB docs
3. Inject search functions into rag_simple / rag_complex nodes
4. Inject FlashRank reranker into rag_complex
5. Inject LLM-backed generate and expand functions
"""

from __future__ import annotations

import threading
import time  # used for generate.timing.crossref / generate.timing.llm logs
from typing import Callable, List

import structlog
from langchain_core.messages import HumanMessage

from src.backends.vector_store import (
    VectorStoreBackend,
)  # noqa: F401 — used in isinstance checks
from src.infra.llm_factory import get_complex_llm, get_simple_llm  # noqa: F401
from src.infra.parsers import (
    extract_text,
)  # noqa: F401  # used by other make_*_fn
from src.infra.prompt_manager import PromptManager
from src.v7.cross_ref import expand_cross_references
from src.v7.hard_gates import sanitize_for_llm
from src.v7.nlp_core import init_bm25_index
from src.v7.usage import LLMUsage, usage_from_response
from src.v7.nodes import generate_answer as generate_answer_mod
from src.v7.nodes import rag_complex as rag_complex_mod
from src.v7.nodes import rag_simple as rag_simple_mod
from src.v7.nodes import visual_enrichment as visual_enrichment_mod

logger = structlog.get_logger()
_pm = PromptManager()

# Module-level RLock protecting init_v7_from_chroma.
# Concurrent calls serialize to prevent readers observing a half-initialized
# pipeline state (7 set_*_fn injectors + BM25 build are not atomic individually).
_init_lock = threading.RLock()


def _doc_to_passage(text: str, meta: dict, score: float = 0.0) -> dict:
    """Build a v7 passage dict, lifting chunk_id from metadata to top level.

    Centralises passage construction from a Chroma/backend result so RRF fusion
    and dedup key on document identity (see nlp_core.passage_identity).
    """
    passage = {
        "text": text,
        "metadata": meta,
        "score": score,
        "doc_id": meta.get("source", "unknown"),
    }
    if "chunk_id" in meta:
        passage["chunk_id"] = meta["chunk_id"]
    return passage


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
            rerank_score = round(float(result["score"]), 4)
            reranked.append(
                {
                    **original,
                    "vector_score": vector_score,
                    "rerank_score": rerank_score,
                    "score": rerank_score,
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

        t0 = time.perf_counter()
        pairs = [(query, p.get("text", "")) for p in passages]
        raw_scores = _model.predict(pairs, batch_size=batch_size)
        t_predict = time.perf_counter() - t0

        scored = [(p, _sigmoid(float(s))) for p, s in zip(passages, raw_scores)]
        scored.sort(key=lambda item: item[1], reverse=True)

        reranked: List[dict] = []
        for original, score in scored[:top_k]:
            vector_score = original.get("vector_score", original.get("score", 0.0))
            rerank_score = round(score, 4)
            reranked.append(
                {
                    **original,
                    "vector_score": vector_score,
                    "rerank_score": rerank_score,
                    "score": rerank_score,
                }
            )
        logger.info(
            "rerank.timing",
            candidates=len(passages),
            top_k=top_k,
            predict_s=round(t_predict, 3),
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
        t0 = time.perf_counter()
        docs_and_scores = vector_store.similarity_search_with_score(
            query, k=top_k, filter=filters or None
        )
        logger.info(
            "vector_search.timing",
            top_k=top_k,
            search_s=round(time.perf_counter() - t0, 3),
        )
        results = []
        for doc, distance in docs_and_scores:
            # ChromaDB returns L2 distance (0..inf). Convert to similarity (0..1).
            similarity = round(1.0 / (1.0 + distance), 4)
            meta = dict(doc.metadata)
            passage = {
                "text": doc.page_content,
                "metadata": meta,
                "score": similarity,
                "vector_score": similarity,
                "doc_id": meta.get("source", "unknown"),
            }
            # Lift chunk_id to top level for RRF fusion / dedup identity.
            if "chunk_id" in meta:
                passage["chunk_id"] = meta["chunk_id"]
            results.append(passage)
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
                # get_by_filter paginates until exhausted — hard cap here.
                return [
                    _doc_to_passage(d.page_content, dict(d.metadata or {}))
                    for d in docs[:max_section_chunks]
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
                extra.append(_doc_to_passage(doc, dict(meta or {})))
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


def model_name_of(llm) -> str:
    """Model id of a chat model, whatever the provider calls the attribute.

    OpenAI/DeepSeek expose ``model_name``, Gemini exposes ``model``. Unknown is
    a label, not an exception: usage accounting must never break a query.
    """
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _zero_usage(model: str, node: str) -> LLMUsage:
    """Usage record for a call that never reached the provider (or failed)."""
    return {
        "model": model,
        "node": node,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def make_expand_fn(llm, n: int = 3) -> Callable[..., tuple]:
    """Create an LLM-backed query expansion function for V8 multi-query expand.

    Signature: fn(query: str, n: int) -> (list[str], LLMUsage) — alternatives
    plus the token usage of the call (roadmap 4a). On failure returns an empty
    list with zero usage (caller falls back to single-query mode).
    """

    model = model_name_of(llm)

    def _expand(query: str, n: int = n) -> tuple:
        prompt = _pm.render("query_expand", query=query, n=n)
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = extract_text(response.content).strip()
            alternatives = [line.strip() for line in raw.splitlines() if line.strip()]
            return alternatives[:n], usage_from_response(response, model, "expand")
        except Exception as exc:
            logger.warning("LLM expand failed: %s", exc)
            return [], _zero_usage(model, "expand")

    return _expand


def make_generate_fn(llm, backend=None) -> Callable[..., tuple]:
    """Create an LLM-backed answer generation function for v7 generate_answer node.

    Signature: fn(query, active_query, passages) -> (answer_text, LLMUsage) —
    the answer plus the token usage of the call (roadmap 4a).
    Relies on ChatGoogleGenerativeAI's built-in retry (max_retries=3 in
    ``get_gemini_llm``) for transient 5xx / 429 errors. On final failure
    falls back to a stub (concatenated top passages).
    If backend is provided, cross-reference expansion is applied before generation.
    """

    model = model_name_of(llm)

    def _call_llm(prompt: str) -> tuple[str, LLMUsage]:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = extract_text(response.content).strip()
        if not answer:
            raise ValueError("Empty generation response")
        return answer, usage_from_response(response, model, "generate")

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

    def _generate(query: str, active_query: str, passages: List[dict]) -> tuple:
        if not passages:
            return "", _zero_usage(model, "generate")
        t0 = time.perf_counter()
        expanded = (
            expand_cross_references(passages, backend, query=query)
            if backend
            else passages
        )
        t_crossref = time.perf_counter() - t0
        # final_passages is already capped at 24 upstream (merge_all_passages);
        # cross-reference expansion appends extra passages — allow up to 30 so
        # low-ranked but answer-bearing cross-refs (e.g. п.60 для программа В) are included.
        top_passages = expanded[:30]
        passages_text = "\n\n".join(
            f"{_chunk_header(i, p)}\n{sanitize_for_llm(p.get('text', ''))}"
            for i, p in enumerate(top_passages)
        )
        prompt_tokens_approx = len(passages_text) // 4
        prompt = _pm.render(
            "generate_answer",
            query=query,
            context=passages_text,
            passages_count=len(top_passages),
        )
        logger.info(
            "generate.timing.crossref",
            crossref_s=round(t_crossref, 3),
            passages_in=len(passages),
            passages_out=len(top_passages),
            prompt_tokens_approx=prompt_tokens_approx,
        )
        t1 = time.perf_counter()
        try:
            result, usage = _call_llm(prompt)
            logger.info(
                "generate.timing.llm",
                llm_s=round(time.perf_counter() - t1, 3),
                completion_tokens=usage["completion_tokens"],
                answer_chars=len(result),
            )
            return result, usage
        except Exception as exc:
            logger.warning(
                "LLM generate failed after retries: %s, falling back to stub", exc
            )
            fallback = "\n\n".join(p.get("text", "") for p in passages[:10])
            return fallback, _zero_usage(model, "generate")

    return _generate


def init_v7_pipeline(vector_store, llm_provider: str | None = "gemini") -> None:
    """Initialize V7 pipeline from a vector store (raw Chroma or VectorStoreBackend).

    1. Creates vector search wrapper
    2. Injects it into rag_simple and rag_complex nodes
    3. Builds BM25 index from full corpus
    4. Injects FlashRank reranker into rag_complex
    5. Injects LLM-backed generate and expand functions (if provider available)
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

        # Inject LLM-backed generate and expand functions
        if llm_provider:
            try:
                # Split generators: cheap model for the simple path, full-quality
                # model for the complex path (answer quality dominates CPS savings).
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

                logger.info("v7 LLM generator and expander injected successfully")
            except Exception as exc:
                logger.warning(
                    "Failed to initialize LLM for v7 generator: %s. "
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
