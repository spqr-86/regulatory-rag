"""V7 RAG pipeline — NLP utilities.

Lemmatization, BM25 index, RRF merge, MMR select.

Libraries (per design 5.5 — no reinventing):
- pymorphy3: morphological analysis / lemmatization
- razdel: Russian tokenization
- rank_bm25: BM25Okapi implementation

Source spec: docs/feature/migration-v7 (lines 350-779).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Optional

import pymorphy3
from razdel import tokenize as razdel_tokenize
from rank_bm25 import BM25Okapi

from src.v7.config import v7_config

# ─── Singleton morph analyzer ──────────────────────────────────────────────

_morph = pymorphy3.MorphAnalyzer()

# ─── Stop words (frozenset for O(1) lookups) ──────────────────────────────

STOP_WORDS: frozenset[str] = frozenset(
    {
        "для",
        "при",
        "что",
        "как",
        "или",
        "это",
        "все",
        "его",
        "они",
        "она",
        "быть",
        "было",
        "будет",
        "также",
        "уже",
        "так",
        "если",
        "только",
        "может",
        "нет",
        "без",
        "над",
        "под",
        "между",
        "через",
        "после",
        "перед",
        "более",
        "менее",
        "очень",
        "который",
        "должен",
        "требование",
        "соответствие",
        "согласно",
        "мочь",
        # interrogative / modal words — not domain content
        "какой",
        "каковой",  # lemma of "какова", "каков", "каковы"
        "такой",  # lemma of "такое" (from "что такое X")
        "этот",
        "сколько",
        "когда",
        "где",
        "почему",
        "зачем",
        "можно",
        "нужно",
        "надо",
        "есть",
        "ли",
        # Geographical / attribution words — appear in ALL Russian regulatory docs,
        # provide no domain discriminating power (every standard says "РФ" / "Россия").
        "россия",
        "российский",  # lemma of "Российской", "Российская", etc.
        "федерация",
        "федеральный",
    }
)


# ─── extract_keywords ─────────────────────────────────────────────────────


def extract_keywords(text: str) -> set[str]:
    """Keywords for keyword overlap check.

    pymorphy3 lemmatisation + razdel tokenisation.
    Preserves regulatory document numbers (СП 1.13130, ГОСТ 12.1.004).
    """
    # Extract document numbers BEFORE lemmatisation.
    # Dotted numbers (СП 1.13130, ГОСТ 12.1.004-91) and bare order numbers
    # (приказ 2464, 29н) — the latter would otherwise be dropped as pure digits.
    doc_numbers = set(re.findall(r"\d+(?:\.\d+)+(?:-\d+)?", text))
    doc_numbers |= set(re.findall(r"\b\d{2,}[а-яё]?\b", text.lower()))

    lemmas: set[str] = set()
    for token in razdel_tokenize(text):
        word = token.text.lower()
        if len(word) < 3 or not re.match(r"[а-яёa-z]", word):
            continue
        parsed = _morph.parse(word)
        lemma = parsed[0].normal_form if parsed else word
        if lemma not in STOP_WORDS:
            lemmas.add(lemma)

    return lemmas | doc_numbers


# ─── compute_keyword_overlap ──────────────────────────────────────────────


def compute_keyword_overlap(query: str, passages: List[dict]) -> float:
    """Fraction of query keywords found in passages (0.0–1.0)."""
    query_kw = extract_keywords(query)
    if not query_kw:
        return 1.0
    passage_text = " ".join(p.get("text", "") for p in passages)
    passage_kw = extract_keywords(passage_text)
    return len(query_kw & passage_kw) / len(query_kw)


# ─── compute_doc_diversity ────────────────────────────────────────────────


def compute_doc_diversity(passages: List[dict]) -> tuple[int, float]:
    """(unique_doc_count, max_single_doc_ratio)."""
    if not passages:
        return 0, 1.0
    doc_ids = [p.get("doc_id", "unknown") for p in passages]
    counts = Counter(doc_ids)
    return len(counts), counts.most_common(1)[0][1] / len(passages)


# ─── BM25 ─────────────────────────────────────────────────────────────────


def _lemmatize_for_bm25(text: str) -> List[str]:
    """Tokenise and lemmatise for BM25 index and queries."""
    tokens = []
    for token in razdel_tokenize(text):
        word = token.text.lower()
        if len(word) < 2 or not re.match(r"[а-яёa-z0-9]", word):
            continue
        parsed = _morph.parse(word)
        lemma = parsed[0].normal_form if parsed else word
        tokens.append(lemma)
    return tokens


class BM25Index:
    """BM25 index over rank_bm25.BM25Okapi with pymorphy3 lemmatisation.

    Usage:
        index = BM25Index(passages)  # build once
        results = index.search(query, top_k=12)
    """

    def __init__(self, passages: List[dict]) -> None:
        self._passages = passages
        corpus = [_lemmatize_for_bm25(p.get("text", "")) for p in passages]
        self._bm25 = BM25Okapi(corpus)

    def search(
        self,
        query: str,
        top_k: int = 12,
        filters: Optional[dict] = None,
    ) -> List[dict]:
        tokens = _lemmatize_for_bm25(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)

        candidates = []
        for i, score in enumerate(scores):
            p = self._passages[i]
            if filters:
                skip = False
                for k, v in filters.items():
                    if p.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue
            candidates.append((i, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in candidates[:top_k]:
            p = dict(self._passages[idx])
            p["bm25_score"] = round(float(score), 4)
            if "score" not in p:
                p["score"] = p["bm25_score"]
            results.append(p)
        return results


# ─── Global BM25 index ───────────────────────────────────────────────────

_bm25_index: Optional[BM25Index] = None


def init_bm25_index(passages: List[dict]) -> None:
    """Initialize global BM25 index. Call once at startup with full corpus."""
    global _bm25_index
    _bm25_index = BM25Index(passages)


def bm25_search(
    query: str,
    filters: Optional[dict] = None,
    top_k: int = 12,
) -> List[dict]:
    """BM25 full-text search with pymorphy3 lemmatisation.

    Requires prior init_bm25_index() with corpus.
    """
    if _bm25_index is not None:
        return _bm25_index.search(query, top_k, filters)
    return []


# ─── RRF merge ────────────────────────────────────────────────────────────


def rrf_merge(
    *result_lists: List[dict],
    top_k: int = 12,
    k: int | None = None,
) -> List[dict]:
    """Reciprocal Rank Fusion — merges results from multiple retrievers.

    RRF score = Σ 1 / (k + rank_i) across all lists.
    k=60 — standard value (Cormack et al.).
    Dedup by chunk_id.
    """
    if k is None:
        k = v7_config.RRF_K

    chunk_scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for results in result_lists:
        for rank, p in enumerate(results):
            cid = p.get("chunk_id", f"unknown_{rank}")
            rrf_score = 1.0 / (k + rank + 1)
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rrf_score
            if cid not in chunk_map:
                chunk_map[cid] = p

    ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result = []
    for cid, score in ranked:
        p = dict(chunk_map[cid])
        p["rrf_score"] = round(score, 5)
        result.append(p)

    return result


# ─── MMR select (fallback only) ──────────────────────────────────────────


def mmr_select(
    passages: List[dict],
    top_k: int,
    lambda_param: float | None = None,
) -> List[dict]:
    """Maximal Marginal Relevance — FALLBACK ONLY.

    In production the primary MMR is done natively by Chroma/Qdrant.
    This mmr_select is used ONLY in merge_all_passages(),
    where VectorDB access is unavailable (passages already extracted).

    Diversity penalty based on doc_id.
    """
    if lambda_param is None:
        lambda_param = v7_config.MMR_LAMBDA

    if len(passages) <= top_k:
        return passages

    selected: List[dict] = []
    remaining = list(passages)
    selected_doc_ids: Counter = Counter()

    for _ in range(top_k):
        best_idx = -1
        best_mmr = -1.0

        for i, p in enumerate(remaining):
            relevance = p.get("vector_score", p.get("score", 0.0))
            doc_id = p.get("doc_id", "unknown")
            doc_count = selected_doc_ids.get(doc_id, 0)
            diversity_penalty = doc_count / max(len(selected), 1)
            mmr_score = (
                lambda_param * relevance - (1 - lambda_param) * diversity_penalty
            )

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        if best_idx < 0:
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_doc_ids[chosen.get("doc_id", "unknown")] += 1

    return selected


# ─── merge_all_passages ───────────────────────────────────────────────────


def merge_all_passages(
    attempts: List[dict],
    top_k: int = 12,
    mmr_lambda: float | None = None,
) -> List[dict]:
    """Merge unique passages from ALL retrieval attempts.

    1. Collect all passages from all attempts.
    2. Dedup by chunk_id.
    3. MMR-select top_k for diversity.
    """
    if mmr_lambda is None:
        mmr_lambda = v7_config.MMR_LAMBDA

    seen_chunks: set[str] = set()
    all_passages: List[dict] = []

    for attempt in attempts:
        for p in attempt.get("passages", []):
            cid = p.get("chunk_id", "")
            if cid and cid in seen_chunks:
                continue
            seen_chunks.add(cid)
            all_passages.append(p)

    if not all_passages:
        return []

    return mmr_select(all_passages, top_k, mmr_lambda)
