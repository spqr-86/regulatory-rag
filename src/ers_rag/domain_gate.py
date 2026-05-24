"""Domain gate: corpus-centroid cosine similarity filter."""

from __future__ import annotations

import numpy as np
from functools import lru_cache

from src.vector_store import load_vector_store
from utils.logging import logger


@lru_cache(maxsize=1)
def get_corpus_centroid() -> np.ndarray:
    """Compute mean embedding of all corpus documents. Cached per process."""
    vs = load_vector_store()
    result = vs._collection.get(include=["embeddings"])
    embeddings = np.array(result["embeddings"], dtype=np.float32)
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    logger.info(
        f"Domain gate: centroid computed from {len(embeddings)} docs,"
        f" dim={centroid.shape[0]}"
    )
    return centroid


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def is_in_domain(
    query_embedding: list[float] | np.ndarray,
    threshold: float,
) -> bool:
    """Return True if query embedding is close enough to corpus centroid.

    threshold=0.0 disables the gate (always returns True).
    """
    if threshold <= 0.0:
        return True
    centroid = get_corpus_centroid()
    q = np.array(query_embedding, dtype=np.float32)
    sim = cosine_similarity(q, centroid)
    logger.debug("domain gate", cosine_sim=round(sim, 4), threshold=round(threshold, 4))
    return sim >= threshold


def invalidate_corpus_centroid_cache() -> None:
    """Clear the cached centroid — call after corpus reindex."""
    get_corpus_centroid.cache_clear()
