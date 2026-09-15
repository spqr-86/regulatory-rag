"""Concrete search and model functions for the department Q&A service.

# ANCHOR: department qa wiring
# Role: adapt existing retrieval (Chroma + BM25 + RRF) and the LLM to the
#   service's injected search_fn / model_fn, keeping the service unit-testable.
# Search: same scope filter on both paths; Chroma gets it as $and-wrapped where.
# Model: with_structured_output(ModelAnswer, json_schema, include_raw); one retry
#   with the validation error sent back to the model, then the error propagates
#   (service maps it to failed/generation_failed).
"""

from __future__ import annotations

from typing import Callable, List

from langchain_core.messages import AIMessage, HumanMessage

from src.department_qa.contract import ModelAnswer
from src.v7.nlp_core import bm25_search, rrf_merge
from src.v7.scope_filter import to_chroma_where


def make_hybrid_search_fn(
    vector_store, bm25_fn: Callable[..., List[dict]] = bm25_search
) -> Callable[..., List[dict]]:
    def _search(query: str, filters: dict | None = None, top_k: int = 8) -> List[dict]:
        vector_hits = []
        for doc, distance in vector_store.similarity_search_with_score(
            query, k=top_k, filter=to_chroma_where(filters)
        ):
            meta = dict(doc.metadata or {})
            passage = {
                "text": doc.page_content,
                "metadata": meta,
                "score": round(1.0 / (1.0 + distance), 4),
            }
            if "chunk_id" in meta:
                passage["chunk_id"] = meta["chunk_id"]
            vector_hits.append(passage)
        bm25_hits = bm25_fn(query, filters=filters, top_k=top_k)
        return rrf_merge(vector_hits, bm25_hits, top_k=top_k)

    return _search


def make_model_fn(llm) -> Callable[[str], ModelAnswer]:
    structured = llm.with_structured_output(
        ModelAnswer, method="json_schema", include_raw=True
    )

    def _call(prompt: str) -> ModelAnswer:
        messages = [HumanMessage(content=prompt)]
        result = structured.invoke(messages)
        if result.get("parsing_error") is None and result.get("parsed") is not None:
            return result["parsed"]

        error = result.get("parsing_error") or ValueError("empty structured output")
        raw = result.get("raw")
        retry = messages + [
            raw if isinstance(raw, AIMessage) else AIMessage(content=""),
            HumanMessage(
                content=f"Ответ не прошёл проверку схемы: {error}. Верни ответ строго по схеме."
            ),
        ]
        result = structured.invoke(retry)
        if result.get("parsing_error") is not None or result.get("parsed") is None:
            raise result.get("parsing_error") or ValueError("empty structured output")
        return result["parsed"]

    return _call
