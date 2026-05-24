"""Bridge: adapt ChromaDB ГОСТ corpus for ers_rag pipeline (V7 fork).

Fork of src/v7/bridge.py:
- Vector store loaded from ERS_CHROMA_PATH / ERS_COLLECTION_NAME (defaults
  chroma_db_gosts / wta_gosts).
- LLM backend = DeepSeek (deepseek-chat) instead of Gemini llm_factory.

Responsibilities:
1. Wrap ChromaDB's similarity_search_with_score -> v7 dict format
2. Build BM25 corpus from ChromaDB docs
3. Inject search functions into rag_simple / rag_complex nodes
4. Inject BGE-reranker-v2-m3 (ONNX int8) into rag_complex / rag_simple
5. Inject LLM-backed verify, rewrite, and generate functions
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List

import structlog
from langchain_core.messages import HumanMessage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.parsers import extract_text, parse_json_from_response
from src.ers_rag.nlp_core import init_bm25_index
from src.ers_rag.nodes import generate_answer as generate_answer_mod
from src.ers_rag.nodes import llm_verifier as llm_verifier_mod
from src.ers_rag.nodes import rag_complex as rag_complex_mod
from src.ers_rag.nodes import rag_simple as rag_simple_mod
from src.ers_rag.nodes import rewriter as rewriter_mod
from src.ers_rag.nodes import visual_enrichment as visual_enrichment_mod
from src.ers_rag.nodes.llm_verifier import VERIFIER_SYSTEM_PROMPT
from src.ers_rag.nodes.utils import extract_doc_identifiers
from src.ers_rag.state_types import VerificationResult

logger = structlog.get_logger()


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
    model_dir: str = "models/bge-reranker-v2-m3-onnx-int8",
    max_length: int = 512,
) -> Callable[[str, List[dict], int], List[dict]]:
    """Create a BGE-reranker-v2-m3 (ONNX int8) rerank function for ers_rag.

    Multilingual cross-encoder optimised for Russian technical text. Drop-in
    replacement for the previous FlashRank wrapper: same fn(query, passages,
    top_k) signature and same passage-dict contract (preserves 'vector_score',
    writes sigmoid relevance into 'score').
    """
    from src.ers_rag.nodes.bge_reranker import make_bge_rerank_fn

    return make_bge_rerank_fn(model_dir=model_dir, max_length=max_length)


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
        section = anchor_meta.get("parent_section", "")
        source = anchor_meta.get("source", "")
        if not section or not source:
            return []
        try:
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


def make_verify_fn(llm) -> Callable[..., VerificationResult]:
    """Create a v7-compatible verify function backed by Gemini LLM."""

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
            response = llm.invoke([HumanMessage(content=prompt)])
            raw_text = extract_text(response.content)
            data = parse_json_from_response(raw_text)
            if not data or "verdict" not in data:
                raise ValueError("No verdict in LLM response")
            return VerificationResult(
                verdict=data["verdict"],
                reason=data.get("reason", ""),
                rewrite_hint=data.get("rewrite_hint", ""),
                missing_aspects=data.get("missing_aspects", []),
                confidence=float(data.get("confidence", 0.0)),
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


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 503 / RESOURCE_EXHAUSTED / rate limit errors from Gemini."""
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("503", "resource_exhausted", "rate limit", "quota", "overloaded")
    )


def make_generate_fn(llm) -> Callable[[str, str, List[dict]], str]:
    """Create an LLM-backed answer generation function for v7 generate_answer node.

    Signature: fn(query, active_query, passages) -> answer_text.
    Retries up to 3 times on Gemini 503/rate-limit before falling back to stub.
    """

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        reraise=True,
    )
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
        # final_passages is already capped at 24 upstream (merge_all_passages);
        # re-truncating below that drops answer-bearing passages that ranked low.
        top_passages = passages[:24]
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


# --- DeepSeek LLM adapter (langchain-like .invoke() contract) ---------------

ERS_CHROMA_PATH = os.getenv("ERS_CHROMA_PATH", "./chroma_db_gosts")
ERS_COLLECTION_NAME = os.getenv("ERS_COLLECTION_NAME", "wta_gosts")
ERS_DEEPSEEK_MODEL = os.getenv("ERS_DEEPSEEK_MODEL", "deepseek-chat")


@dataclass
class _DeepSeekResponse:
    """Mimic langchain BaseMessage with .content string."""

    content: str


class DeepSeekLLM:
    """Minimal LLM exposing .invoke([HumanMessage]) -> obj with .content.

    Drop-in replacement for langchain Gemini llm used by ers_rag bridge
    (make_verify_fn / make_rewrite_fn / make_generate_fn / make_expand_fn).
    """

    def __init__(
        self,
        model: str = ERS_DEEPSEEK_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> None:
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY env var is not set")
        self._client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._response_format = response_format

    def invoke(self, messages):
        prompt_parts: list[str] = []
        for m in messages:
            content = getattr(m, "content", None)
            if content is None:
                content = str(m)
            prompt_parts.append(content)
        prompt = "\n\n".join(prompt_parts)

        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._response_format is not None:
            kwargs["response_format"] = self._response_format

        response = self._client.chat.completions.create(**kwargs)
        text = response.choices[0].message.content or ""
        return _DeepSeekResponse(content=text)


def _load_ers_vector_store():
    """Load ChromaDB ГОСТ corpus (mirrors src/gosts_pipeline._load_store)."""
    from langchain_chroma import Chroma

    from src.llm_factory import get_embedding_model

    if not os.path.isdir(ERS_CHROMA_PATH):
        raise FileNotFoundError(
            f"ChromaDB ГОСТов не найдена: {ERS_CHROMA_PATH}. "
            "Запустите python index_gosts.py"
        )
    embeddings = get_embedding_model()
    vs = Chroma(
        collection_name=ERS_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=ERS_CHROMA_PATH,
    )
    count = vs._collection.count()
    if count == 0:
        raise ValueError(
            f"ChromaDB ГОСТов пуста (collection={ERS_COLLECTION_NAME}). "
            "Запустите python index_gosts.py"
        )
    logger.info("ers_rag: store loaded chunks=%d", count)
    return vs


def init_ers_rag_from_chroma(vector_store=None, llm_enabled: bool = True) -> None:
    """Initialize ers_rag (V7 fork) pipeline against ГОСТ ChromaDB + DeepSeek.

    1. Loads vector store from ERS_CHROMA_PATH / ERS_COLLECTION_NAME if not given.
    2. Injects vector search into rag_simple / rag_complex nodes.
    3. Builds BM25 index from full corpus.
    4. Injects BGE-reranker-v2-m3 (complex + simple evidence assess).
    5. Injects DeepSeek-backed verify, rewrite, generate, expand functions.
    """
    from config.settings import settings

    if vector_store is None:
        vector_store = _load_ers_vector_store()

    search_fn = make_vector_search_fn(vector_store)
    rag_simple_mod.set_vector_search(search_fn)
    rag_complex_mod.set_vector_search(search_fn)

    # Build BM25 corpus from ChromaDB
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
        logger.info("ers_rag section-aware expander injected successfully")
    except Exception as exc:
        logger.warning("Failed to initialize section fetch for ers_rag: %s.", exc)

    # Inject BGE-reranker-v2-m3 (ONNX int8) for complex path and evidence assess
    # (simple path). Multilingual cross-encoder, strong on Russian ГОСТ/СНиП.
    try:
        model_dir = getattr(
            settings,
            "BGE_RERANKER_DIR",
            "models/bge-reranker-v2-m3-onnx-int8",
        )
        rerank_fn = make_rerank_fn(model_dir=model_dir)
        rag_complex_mod.set_rerank_fn(rerank_fn)
        rag_simple_mod.set_reranker(rerank_fn)
        logger.info("ers_rag BGE reranker injected successfully (dir=%s)", model_dir)
    except Exception as exc:
        logger.warning(
            "Failed to initialize BGE reranker for ers_rag: %s. "
            "Complex path will skip reranking.",
            exc,
        )

    # Inject DeepSeek-backed verify, rewrite, generate, expand functions
    if llm_enabled:
        try:
            verifier_llm = DeepSeekLLM(
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            llm_verifier_mod.set_verify_fn(make_verify_fn(verifier_llm))

            rewriter_llm = DeepSeekLLM(max_tokens=512)
            rewriter_mod.set_rewrite_fn(make_rewrite_fn(rewriter_llm))

            generator_llm = DeepSeekLLM(max_tokens=2048)
            generate_answer_mod.set_generate_fn(make_generate_fn(generator_llm))

            expander_llm = DeepSeekLLM(max_tokens=256)
            rag_simple_mod.set_expand_fn(make_expand_fn(expander_llm))

            logger.info(
                "ers_rag DeepSeek verifier, rewriter, generator, expander injected"
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize DeepSeek for ers_rag: %s. "
                "Using rule-based stubs.",
                exc,
            )

    # Inject visual proof function for visual_enrichment node (no-op if unavailable)
    try:
        visual_proof_fn = make_visual_proof_fn()
        if visual_proof_fn is not None:
            visual_enrichment_mod.set_visual_proof_fn(visual_proof_fn)
            logger.info("ers_rag visual proof injected successfully")
    except Exception as exc:
        logger.warning("Failed to initialize visual proof for ers_rag: %s.", exc)
