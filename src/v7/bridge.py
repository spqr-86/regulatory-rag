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
    cache_dir: str = ".flashrank_cache",
) -> Callable[[str, List[dict], int], List[dict]]:
    """Create a FlashRank reranker function for v7 rag_complex.

    Thin adapter over ``langchain_community.document_compressors.FlashrankRerank``:
    converts v7 passage dicts ↔ ``Document`` and preserves the original vector
    similarity score in ``metadata["vector_score"]`` so downstream MMR/gates
    keep using vector scores (FlashRank scores are not calibrated for Russian
    domain — see fix 2026-05-16).

    Signature: fn(query, passages, top_k) -> passages (reranked, ≤ top_k items).
    Each passage must have a 'text' key.
    """
    from flashrank import Ranker
    from langchain_community.document_compressors import FlashrankRerank
    from langchain_core.documents import Document

    ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    def _rerank(query: str, passages: List[dict], top_k: int) -> List[dict]:
        if not passages:
            return passages

        # Freeze vector_score BEFORE rerank so we don't overwrite it with the
        # FlashRank score (see CLAUDE.md "Known Issues" — fix 2026-05-16).
        compressor = FlashrankRerank(client=ranker, top_n=top_k)
        docs: List[Document] = []
        for idx, p in enumerate(passages):
            meta = dict(p.get("metadata", {}))
            meta["_v7_passage_idx"] = idx
            docs.append(Document(page_content=p.get("text", ""), metadata=meta))

        compressed = compressor.compress_documents(documents=docs, query=query)

        reranked: List[dict] = []
        for doc in compressed:
            idx = doc.metadata.get("_v7_passage_idx")
            orig = passages[idx] if isinstance(idx, int) else {}
            vector_score = orig.get("vector_score", orig.get("score", 0.0))
            relevance = float(doc.metadata.get("relevance_score", 0.0))
            new_meta = {
                k: v
                for k, v in doc.metadata.items()
                if k not in ("_v7_passage_idx", "id", "relevance_score")
            }
            reranked.append(
                {
                    **orig,
                    "text": doc.page_content,
                    "metadata": new_meta,
                    "vector_score": vector_score,
                    "score": round(relevance, 4),
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
    max_section_chunks: int = 30,
) -> Callable[[List[dict]], List[dict]]:
    """Create a section-aware expander from ChromaDB store.

    Takes the top anchor passage, extracts parent_section + source from its metadata,
    and fetches all chunks from that section (up to max_section_chunks).
    Returns passages not already in the input list.
    """

    def _fetch_section(passages: List[dict]) -> List[dict]:
        if not passages:
            return []
        anchor_meta = passages[0].get("metadata", {})
        section = (anchor_meta.get("parent_section") or "").strip()
        source = anchor_meta.get("source", "")
        if not section or not source:
            return []
        try:
            # VectorStoreBackend (Protocol) exposes get_by_filter; raw Chroma uses _collection.
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
                        "score": 0.0,  # no vector score for fetched chunks
                    }
                )
            return extra
        except Exception as exc:
            logger.warning("section_fetch failed: %s", exc)
            return []

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


_EXPAND_SYSTEM_PROMPT = (
    "Ты — эксперт по нормативным документам в области охраны труда. "
    "Сгенерируй альтернативные формулировки поискового запроса для поиска в базе нормативных актов. "
    "Используй синонимы, смежные термины и другие углы зрения на тот же вопрос. "
    "Верни ровно N формулировок — по одной на строке, без нумерации и пояснений."
)


def make_expand_fn(llm, n: int = 3) -> Callable[[str, int], List[str]]:
    """Create an LLM-backed query expansion function for V8 multi-query expand.

    Signature: fn(query: str, n: int) -> list[str] — list of alternative queries.
    On failure returns empty list (caller falls back to single-query mode).
    """

    def _expand(query: str, n: int = n) -> List[str]:
        prompt = (
            f"{_EXPAND_SYSTEM_PROMPT}\n\n"
            f"Исходный запрос: {query}\n"
            f"Количество альтернатив: {n}\n\n"
            "Альтернативные формулировки:"
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw = extract_text(response.content).strip()
            alternatives = [line.strip() for line in raw.splitlines() if line.strip()]
            return alternatives[:n]
        except Exception as exc:
            logger.warning("LLM expand failed: %s", exc)
            return []

    return _expand


_GENERATE_SYSTEM_PROMPT = """\
Ты отвечаешь СТРОГО на основе предоставленных фрагментов нормативных документов.
Всё что не написано явно в тексте фрагментов — не используй и не додумывай.

ПРАВИЛА (обязательны):
1. Ссылки: каждое утверждение сопровождай ссылкой [Фрагмент N: Документ, п. X.X].
   Если пункт не указан в тексте фрагмента — пиши [Фрагмент N: Документ, без пункта].
2. Цитирование: ключевые формулировки нормы (требования, сроки, цифры) цитируй дословно \
в кавычках. Остальное можно пересказывать.
3. Три варианта ответа:
   a) Полный — фрагменты покрывают вопрос: дай ответ со ссылками.
   b) Частичный — покрывают часть: ответь на покрытое, затем "Не покрыто: [что именно]."
      Если вопрос не содержит данных для применения нормы (вид работ, категория, должность) — \
дай ответ для наиболее типового сценария, явно пометив: "Допущение: [что предполагается]. \
Для других сценариев ответ может отличаться." Перечисли недостающие параметры.
   c) Нет ответа — фрагменты не содержат нужной информации: \
"В фрагментах ответа нет. Не хватает: [что именно]."
4. Противоречия: если фрагменты противоречат — укажи оба источника и оба варианта. \
Если в тексте фрагментов видны даты документов — укажи их. \
Окончательный выбор оставь пользователю.
5. Терминология: сохраняй термины первоисточника дословно — классификации, категории, \
группы (напр. "3.3", "1Б группа", "класс В1").
6. Фрагменты с низкой релевантностью (помечены LOW) используй только при отсутствии \
фрагментов с высокой или средней релевантностью.

ФОРМАТ ОТВЕТА:
Вывод: [прямой ответ в минимальном объёме — одна фраза для да/нет, список для перечней, \
без преамбул и повторения вопроса]
Обоснование: [детали с обязательными ссылками [Фрагмент N: Документ, п. X.X]]
Первоисточники: [только если источников больше двух — список в конце; иначе пропусти раздел]"""

# Паттерны для русских ссылок на пункты/статьи НПА
_REF_PATTERNS = [
    re.compile(r"пункт\w*\s+(\d+)", re.IGNORECASE),
    re.compile(r'подпункт\w*\s+[«"]?([а-яё])[»"]?', re.IGNORECASE),
    re.compile(r"стать\w+\s+(\d+)", re.IGNORECASE),
]


def _extract_refs(text: str) -> list[str]:
    """Извлечь ссылки на пункты/статьи из текста чанка."""
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
    """Подтянуть чанки, связанные с найденными passages через cross-reference.

    Два механизма:
    1. Явные ссылки: ищет "пункт N", "статью N" в тексте passage и тянет чанки
       из того же source, где встречается этот номер.
    2. BM25-доиск: для каждого уникального source из passages запускает bm25_search
       по запросу и добавляет чанки из этого source (улавливает обратные ссылки,
       когда п.60 ссылается на п.46.в, а не наоборот).
    """
    if not passages:
        return passages

    existing_texts = {p["text"] for p in passages}
    extra: list[dict] = []

    def _add(text: str, metadata: dict) -> None:
        if text and text not in existing_texts:
            existing_texts.add(text)
            extra.append(
                {"text": text, "score": 0.0, "metadata": metadata, "cross_ref": True}
            )

    # ── Mechanism 1: explicit refs in passage text ────────────────────────────
    for passage in passages:
        source = passage.get("metadata", {}).get("source", "")
        if not source:
            continue
        refs = _extract_refs(passage.get("text", ""))
        for ref in refs:
            try:
                docs = backend.get_by_filter(
                    where={"source": source},
                    limit=100,
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

    return passages + extra


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
        """Извлечь короткое название документа из metadata.source."""
        raw = passage.get("metadata", {}).get("source", "")
        # "Трудовой кодекс РФ - Система Охрана труда. Премиальная версия.pdf"
        # → "Трудовой кодекс РФ"
        name = raw.split(" - ")[0].replace(".pdf", "").strip()
        return name or "Неизвестный источник"

    def _generate(query: str, active_query: str, passages: List[dict]) -> str:
        if not passages:
            return ""
        expanded = (
            expand_cross_references(passages, backend, query=query)
            if backend
            else passages
        )
        # final_passages is already capped at 24 upstream (merge_all_passages);
        # re-truncating below that drops answer-bearing passages that ranked low.
        top_passages = expanded[:24]
        passages_text = "\n\n".join(
            f"[{i + 1}] ({_score_label(p.get('score', 0.0))}) [Источник: {_short_source(p)}]\n{p.get('text', '')}"
            for i, p in enumerate(top_passages)
        )
        prompt = (
            f"{_GENERATE_SYSTEM_PROMPT}\n\n"
            f"Вопрос: {query}\n\n"
            f"Фрагменты (отсортированы по релевантности, {len(top_passages)} шт.):\n"
            f"{passages_text}\n\n"
            "Ответ:"
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

    # Inject FlashRank reranker for complex path and V8 evidence assess (simple path)
    try:
        rerank_fn = make_rerank_fn(
            model_name=settings.RERANKING_MODEL,
            cache_dir=settings.FLASHRANK_CACHE_DIR,
        )
        rag_complex_mod.set_rerank_fn(rerank_fn)
        rag_simple_mod.set_reranker(rerank_fn)
        logger.info("v7 FlashRank reranker injected successfully")
    except Exception as exc:
        logger.warning(
            "Failed to initialize FlashRank for v7: %s. Complex path will skip reranking.",
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
                vector_store if isinstance(vector_store, VectorStoreBackend) else None
            )
            generate_answer_mod.set_generate_fns(
                simple=make_generate_fn(generator_llm_simple, backend=xref_backend),
                complex_=make_generate_fn(generator_llm_complex, backend=xref_backend),
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
