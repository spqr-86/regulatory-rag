"""Concrete search and model functions for the department Q&A service.

# ANCHOR: department qa wiring
# Role: adapt existing retrieval (Chroma + BM25 + RRF) and the LLM to the
#   service's injected search_fn / model_fn, keeping the service unit-testable.
# Search: same scope filter on both paths; Chroma gets it as $and-wrapped where.
# Model: with_structured_output(ModelAnswer, json_schema, include_raw); one retry
#   with the validation error sent back to the model, then the error propagates
#   (service maps it to failed/generation_failed).
# Mode: DEPARTMENT_QA_MODE picks one bundle atomically (spec object-profile §2.4 п. 7):
#   v1 = sheets in index, no profiles, prompt v1, ModelAnswerV1;
#   v2 = sheets out of index, load_profiles, prompt v3, ModelAnswer. The prompt version
#   moves inside the bundle: v3 = v2 plus the threshold rule (design decision 10, q1);
#   v2 stays in the registry so the 2026-09-15 run can be re-rendered.
# The Chroma store is a process-wide singleton from settings, so a mode never
#   switches the store; ensure_store_matches fails fast on a mismatch instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage

from src.department_qa.contract import ModelAnswer, ModelAnswerV1
from src.department_qa.object_profile import ObjectProfile, load_profiles
from src.indexing.manifest import Manifest
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


@dataclass(frozen=True)
class ModeConfig:
    mode: str
    chroma_db_path: str
    collection: str
    prompt_version: str
    schema: type[ModelAnswerV1]
    profiles: dict[str, ObjectProfile] = field(default_factory=dict)


def build_mode_config(
    mode: str, manifest: Manifest, source_dir: str, cfg
) -> ModeConfig:
    if mode == "v1":
        return ModeConfig(
            mode="v1",
            chroma_db_path=cfg.DEPARTMENT_V1_CHROMA_DB_PATH,
            collection=cfg.DEPARTMENT_V1_COLLECTION,
            prompt_version="v1",
            schema=ModelAnswerV1,
        )
    if mode == "v2":
        return ModeConfig(
            mode="v2",
            chroma_db_path=cfg.DEPARTMENT_V2_CHROMA_DB_PATH,
            collection=cfg.DEPARTMENT_V2_COLLECTION,
            prompt_version="v3",
            schema=ModelAnswer,
            profiles=load_profiles(manifest, source_dir),
        )
    raise ValueError(f"DEPARTMENT_QA_MODE={mode!r}: expected v1 or v2")


def ensure_store_matches(
    config: ModeConfig, chroma_db_path: str, collection: str
) -> None:
    same_path = os.path.abspath(chroma_db_path) == os.path.abspath(
        config.chroma_db_path
    )
    if not same_path or collection != config.collection:
        raise RuntimeError(
            f"DEPARTMENT_QA_MODE={config.mode} needs CHROMA_DB_PATH={config.chroma_db_path} "
            f"and CHROMA_COLLECTION_NAME={config.collection}; "
            f"got {chroma_db_path} / {collection}"
        )


def make_model_fn(
    llm,
    schema: type[ModelAnswerV1] = ModelAnswer,
    recorder: Optional[Callable[[dict], None]] = None,
) -> Callable[[str], ModelAnswerV1]:
    structured = llm.with_structured_output(
        schema, method="json_schema", include_raw=True
    )

    def _invoke(messages, attempt: int) -> dict:
        result = structured.invoke(messages)
        if recorder is not None:
            raw = result.get("raw")
            error = result.get("parsing_error")
            recorder(
                {
                    "attempt": attempt,
                    "raw": raw.content if isinstance(raw, AIMessage) else "",
                    "parsing_error": str(error) if error is not None else None,
                }
            )
        return result

    def _call(prompt: str) -> ModelAnswerV1:
        messages = [HumanMessage(content=prompt)]
        result = _invoke(messages, 1)
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
        result = _invoke(retry, 2)
        if result.get("parsing_error") is not None or result.get("parsed") is None:
            raise result.get("parsing_error") or ValueError("empty structured output")
        return result["parsed"]

    return _call


@dataclass
class DepartmentStack:
    config: ModeConfig
    manifest: Manifest
    search_fn: Callable[..., List[dict]]
    model_fn: Callable[[str], ModelAnswerV1]
    store: object


def build_department_stack(
    recorder: Optional[Callable[[dict], None]] = None,
) -> DepartmentStack:
    """The only assembly of the department Q&A stack: Streamlit page and eval share it."""
    from config.settings import settings
    from src.backends.vector_store import get_vector_store_backend
    from src.indexing.manifest import load_manifest
    from src.infra.llm_factory import get_simple_llm
    from src.v7.bridge import init_v7_pipeline

    manifest = load_manifest(settings.CORPUS_MANIFEST_PATH)
    config = build_mode_config(
        settings.DEPARTMENT_QA_MODE, manifest, settings.SOURCE_DOCS_PATH, settings
    )
    ensure_store_matches(
        config, settings.CHROMA_DB_PATH, settings.CHROMA_COLLECTION_NAME
    )
    store = get_vector_store_backend(load_existing=True)
    init_v7_pipeline(store)  # builds the BM25 index over the same collection
    model_fn = make_model_fn(get_simple_llm(), schema=config.schema, recorder=recorder)
    return DepartmentStack(
        config=config,
        manifest=manifest,
        search_fn=make_hybrid_search_fn(store),
        model_fn=model_fn,
        store=store,
    )
