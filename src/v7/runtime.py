"""Dependencies bound to one V7 corpus snapshot, shared by its graphs."""

# ANCHOR: immutable wiring, not query state or a second retrieval algorithm.
# Vector/BM25/fetch callbacks belong to a store; reranker belongs to a model resource.

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Callable


def empty_search(*args, **kwargs) -> list[dict]:
    return []


@dataclass(frozen=True)
class V7Runtime:
    vector_search: Callable = empty_search
    complex_vector_search: Callable | None = None
    bm25_search: Callable = empty_search
    expand: Callable | None = None
    rerank: Callable | None = None
    section_fetch: Callable | None = None
    crossref_expander: Callable | None = None
    visual_proof: Callable | None = None
    generate_simple: Callable | None = None
    generate_complex: Callable | None = None


legacy_runtime_lock = threading.RLock()


def capture_legacy_runtime() -> V7Runtime:
    """Freeze legacy setters at graph construction, before another stack starts."""
    with legacy_runtime_lock:
        return _capture_legacy_runtime()


def _capture_legacy_runtime() -> V7Runtime:
    from src.v7 import nlp_core, pack_context
    from src.v7.nodes import generate_answer, rag_complex, rag_simple, visual_enrichment

    index = nlp_core._bm25_index
    return V7Runtime(
        vector_search=rag_simple._vector_search,
        complex_vector_search=rag_complex._vector_search,
        bm25_search=index.search if index is not None else empty_search,
        expand=rag_simple._expand_fn,
        rerank=rag_complex._rerank_fn,
        section_fetch=rag_complex._section_fetch_fn,
        crossref_expander=pack_context._crossref_expander,
        visual_proof=visual_enrichment._visual_proof_fn,
        generate_simple=generate_answer._generate_fn_simple,
        generate_complex=generate_answer._generate_fn_complex,
    )
