"""V7 node: intent_gate — noise/domain classification.

Strategy:
- Noise (chitchat): deterministic regex patterns for Russian/English greetings,
  farewells, meta-questions about the bot. Fast, no LLM.
- Explicit out-of-scope requests are refused before embeddings/retrieval.
- Other requests pass the optional corpus-centroid gate and retrieval validation.
"""

from __future__ import annotations

import re

from src.v7.config import v7_config
from src.v7.domain_gate import is_in_domain
from src.v7.state_types import NextAfterIntent, RAGState

# ANCHOR: deterministic product-scope rejection precedes embeddings/retrieval.
# Narrow command/topic patterns avoid rejecting ordinary worker-rights questions.
_SCOPE_RE = re.compile(
    r"^\s*(?:расскажи|придумай|напиши)\s+(?:\w+\s+){0,3}(?:анекдот\w*|шутк[уиа]\b|шуточ\w*|стих(?:и|ов|отворени\w*)?\b)"
    r"|^\s*игнорируй\s+(?:свои\s+|все\s+)?инструкци"
    r"|(?:раздел[а-я]*\s+наследств\w*\s+(?:имуществ|между)|наследств[а-я]*\s+между\s+родственник)"
    r"|^\s*(?:ignore\s+(?:all\s+|your\s+)?instructions)",
    re.IGNORECASE,
)
_REFUSAL = (
    "Не могу выполнить этот запрос. Я помогаю с нормативными вопросами "
    "охраны труда и пожарной безопасности. Уточните вопрос в этих пределах."
)


def _refusal(reason: str) -> RAGState:
    return {
        "intent": "noise",
        "answer": _REFUSAL,
        "route_decision": "abstain",
        "route_reason": reason,
        "abstain_reason": reason,
    }


# ── Chitchat / noise patterns ──────────────────────────────────────────────
# Order matters: more specific patterns first.
# Greeting keywords: query STARTS WITH one of these → noise candidate
_GREETING_RE = re.compile(
    r"^\s*(привет\w*|здравствуй\w*"
    r"|добр(ый|ое|ого|ому|ую|ая)\s+(день|утро|вечер\w*|ночь)"
    r"|хай|хеллоу|hi|hey|hello"
    r"|good\s+(morning|evening|day)"
    r"|доброго\s+времени)",
    re.IGNORECASE,
)

# Full-match noise: farewells, meta-questions, small talk
_NOISE_FULL_PATTERNS: list[re.Pattern] = [
    # Farewells
    re.compile(
        r"^\s*(пока\s*\w*|до\s+свидани\w+|до\s+встречи"
        r"|спасибо\s*\w*|благодарю\s*\w*"
        r"|bye|goodbye|thanks|thank\s+you)\s*[!?,.]?\s*$",
        re.IGNORECASE,
    ),
    # Meta: questions about the bot
    re.compile(
        r"^\s*(кто\s+ты|что\s+ты\s+(умеешь|можешь|знаешь)"
        r"|ты\s+(бот|робот|ии|ai|помощник)"
        r"|чем\s+ты\s+можешь\s+помочь"
        r"|как\s+(тебя\s+зовут|ты\s+работаешь)"
        r"|расскажи\s+о\s+себе)\s*[?,.]?\s*$",
        re.IGNORECASE,
    ),
    # "Как дела" / small talk (standalone)
    re.compile(
        r"^\s*(как\s+(дела|жизнь|настроение|сам\w*)"
        r"|всё\s+хорошо"
        r"|ок\w*|хорошо|понял|понятно|ясно|окей)\s*[?,.]?\s*$",
        re.IGNORECASE,
    ),
]


def _is_noise(q: str) -> bool:
    """Return True if query is chitchat/noise with no domain content.

    Two checks:
    1. Query STARTS WITH a greeting keyword AND is short (≤6 words).
    2. Full-match against farewell / meta / small-talk patterns.
    """
    if len(q) < 3:
        return True
    # Greeting at start + short query (covers "привет как дела", "здравствуй помощник")
    if _GREETING_RE.match(q) and len(q.split()) <= 6:
        return True
    for pattern in _NOISE_FULL_PATTERNS:
        if pattern.match(q):
            return True
    return False


def intent_gate(state: RAGState) -> RAGState:
    """Reads: query. Writes: intent ('noise' | 'domain').

    Steps:
    1. Regex noise check (chitchat, greetings, farewells).
    2. Domain gate: cosine similarity of query embedding to corpus centroid.
       Only active when DOMAIN_GATE_THRESHOLD > 0.
    """
    q = (state.get("query") or "").strip()
    if _SCOPE_RE.search(q):
        return _refusal("out_of_scope_request")
    if _is_noise(q):
        return _refusal("noise")

    threshold = v7_config.DOMAIN_GATE_THRESHOLD
    # Short/ambiguous queries (< 3 tokens) bypass the domain gate and go to the router,
    # which returns a clarification request. Domain-gate cosine similarity is unreliable
    # for ultra-short inputs because their embeddings sit near the origin.
    if threshold > 0.0 and len(q.split()) >= 3:
        from src.infra.llm_factory import get_embedding_model

        embedding_model = get_embedding_model()
        query_embedding = embedding_model.embed_query(q)
        if not is_in_domain(query_embedding, threshold):
            return _refusal("domain_gate")

    return {"intent": "domain"}


def route_by_intent(state: RAGState) -> NextAfterIntent:
    return "end" if state["intent"] == "noise" else "router"
