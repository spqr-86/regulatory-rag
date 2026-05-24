from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Sequence

import numpy as np
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.outputs import Generation
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticCache(BaseCache):
    """Semantic cache implementing LangChain's BaseCache interface.

    Lookup/update use cosine similarity over sentence-transformer embeddings.
    `llm_string` is used as a namespace key to isolate entries per LLM config —
    entries cached under one llm_string are never returned for another.

    Backward-compatible `get(query) -> str | None` and `add(query, answer)` are
    kept as thin wrappers (used by agents/multiagent_rag.py); they operate in
    a default namespace (llm_string="").
    """

    _DEFAULT_NAMESPACE = ""

    def __init__(
        self,
        threshold: float = 0.93,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_file: str = "semantic_cache.json",
    ) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self.cache_file = cache_file

        # Storage parallel arrays indexed together; namespace isolates entries
        # so lookups under one llm_string do not match entries from another.
        self.sentences: List[str] = []
        self.embeddings: List[List[float]] = []
        self.answers: List[str] = []
        self.namespaces: List[str] = []

        # Load model (mocked in tests, real model in prod)
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:  # noqa: BLE001 - third-party may raise broad types
            logger.warning("Failed to load SentenceTransformer model: %s", e)
            self.model = None

        self._load()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not os.path.exists(self.cache_file):
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.sentences = data.get("sentences", [])
            self.embeddings = data.get("embeddings", [])
            self.answers = data.get("answers", [])
            # Older cache files may not have namespaces — backfill with default.
            self.namespaces = data.get(
                "namespaces", [self._DEFAULT_NAMESPACE] * len(self.sentences)
            )
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load cache: %s", e)

    def save(self) -> None:
        """Persist cache to disk."""
        data = {
            "sentences": self.sentences,
            "embeddings": self.embeddings,
            "answers": self.answers,
            "namespaces": self.namespaces,
        }
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error("Failed to save cache: %s", e)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _best_match_index(self, query: str, namespace: str) -> Optional[int]:
        if not self.sentences or self.model is None:
            return None

        # Filter indices by namespace to isolate entries per llm_string.
        ns_indices = [i for i, ns in enumerate(self.namespaces) if ns == namespace]
        if not ns_indices:
            return None

        try:
            query_embedding = self.model.encode(query)
            cache_embeddings = np.array([self.embeddings[i] for i in ns_indices])

            norm_query = np.linalg.norm(query_embedding)
            if norm_query == 0:
                return None

            norm_cache = np.linalg.norm(cache_embeddings, axis=1)
            norm_cache[norm_cache == 0] = 1e-9

            dot_products = np.dot(cache_embeddings, query_embedding)
            similarities = dot_products / (norm_cache * norm_query)

            best_local = int(np.argmax(similarities))
            if similarities[best_local] >= self.threshold:
                return ns_indices[best_local]
        except (ValueError, RuntimeError) as e:
            logger.error("Error in semantic cache similarity search: %s", e)

        return None

    def _add_entry(self, query: str, answer: str, namespace: str) -> None:
        if self.model is None:
            return

        # Exact-match dedupe within namespace.
        for i, sent in enumerate(self.sentences):
            if sent == query and self.namespaces[i] == namespace:
                return

        try:
            embedding = self.model.encode(query)
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            self.sentences.append(query)
            self.embeddings.append(embedding)
            self.answers.append(answer)
            self.namespaces.append(namespace)

            self.save()
        except (ValueError, RuntimeError) as e:
            logger.error("Error adding to semantic cache: %s", e)

    # ------------------------------------------------------------------ #
    # BaseCache interface
    # ------------------------------------------------------------------ #
    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Return cached list of Generations on semantic hit, else None.

        `llm_string` acts as a namespace — entries cached under a different
        llm_string are never returned. The stored value is the serialized
        list of Generations (JSON); deserialize back into Generation objects.
        """
        idx = self._best_match_index(prompt, llm_string)
        if idx is None:
            return None

        raw = self.answers[idx]
        try:
            payload = json.loads(raw)
            if isinstance(payload, list) and all(
                isinstance(g, dict) and "text" in g for g in payload
            ):
                return [Generation(text=g["text"]) for g in payload]
        except (json.JSONDecodeError, TypeError):
            pass

        # Legacy string entries (from old get/add API) — wrap as single Generation.
        return [Generation(text=raw)]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Cache a sequence of Generations under (prompt, llm_string) namespace."""
        if not isinstance(return_val, Sequence):
            return
        serialized = json.dumps(
            [{"text": getattr(g, "text", str(g))} for g in return_val],
            ensure_ascii=False,
        )
        self._add_entry(prompt, serialized, llm_string)

    def clear(self, **kwargs) -> None:
        """Clear all cached entries (BaseCache optional method)."""
        self.sentences = []
        self.embeddings = []
        self.answers = []
        self.namespaces = []
        self.save()

    # ------------------------------------------------------------------ #
    # Backward-compatible legacy API (used by agents/multiagent_rag.py)
    # ------------------------------------------------------------------ #
    def get(self, query: str) -> Optional[str]:
        """Legacy: return a plain string answer (default namespace)."""
        idx = self._best_match_index(query, self._DEFAULT_NAMESPACE)
        if idx is None:
            return None
        raw = self.answers[idx]
        # If entry was stored via update() it'll be JSON — unwrap first text.
        try:
            payload = json.loads(raw)
            if isinstance(payload, list) and payload and "text" in payload[0]:
                return payload[0]["text"]
        except (json.JSONDecodeError, TypeError):
            pass
        return raw

    def add(self, query: str, answer: str) -> None:
        """Legacy: store a plain string answer (default namespace)."""
        self._add_entry(query, answer, self._DEFAULT_NAMESPACE)
