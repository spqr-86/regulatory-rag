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
#   v2 = sheets out of index, load_profiles, prompt v4, ModelAnswer. The prompt version
#   moves inside the bundle: v4 adds typed object fields to the v3 threshold rule;
#   v2/v3 stay in the registry so the 2026-09-15/16 runs can be re-rendered.
# The Chroma store is a process-wide singleton from settings, so a mode never
#   switches the store; ensure_store_matches fails fast on a mismatch instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from src.department_qa.contract import ModelAnswer, ModelAnswerV1, VerificationResult
from src.department_qa.object_profile import ObjectProfile, load_profiles
from src.indexing.manifest import Manifest
from src.v7.nlp_core import bm25_search, rrf_merge
from src.v7.scope_filter import to_chroma_where
from src.v7.pack_context import PackLimits
from src.v7.retrieval import (
    RequestEmbeddingMemo,
    bind_request_embedding,
    retrieve_context,
)


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
            prompt_version="v4",
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


def ensure_manifest_matches(manifest: Manifest, store) -> None:
    """Reject a Department store built from another snapshot or with profiles indexed."""
    profile_files = set(manifest.object_profiles.values())
    expected = {
        metadata["document_id"]: metadata
        for filename, metadata in manifest.documents.items()
        if filename not in profile_files
    }
    actual_ids: set[str] = set()
    seen = 0
    for row in store.iter_all_documents():
        seen += 1
        metadata = row.get("metadata") or {}
        if metadata.get("snapshot_id") != manifest.snapshot_id:
            raise RuntimeError(
                "Department index snapshot does not match the V2 manifest"
            )
        document_id = metadata.get("document_id")
        if document_id in {
            manifest.documents[name]["document_id"] for name in profile_files
        }:
            raise RuntimeError("Department V2 index contains an object profile")
        expected_metadata = expected.get(document_id)
        if expected_metadata is None:
            raise RuntimeError(
                "Department V2 index contains a document outside manifest"
            )
        for key in ("source_type", "unit_id", "audience", "organization_id"):
            if key in expected_metadata and metadata.get(key) != expected_metadata[key]:
                raise RuntimeError(
                    f"Department V2 index metadata mismatch for {document_id}: {key}"
                )
        actual_ids.add(document_id)
    if seen == 0:
        raise RuntimeError("Department V2 index is empty")
    if actual_ids != set(expected):
        raise RuntimeError("Department V2 index document set does not match manifest")


def stack_cache_key() -> tuple[int, ...]:
    """Streamlit cache key that changes when local modules are hot-reloaded.

    Streamlit's source watcher evicts *every* watched local module on any file
    change (streamlit/watcher/local_sources_watcher.py, "Delete all watched
    modules"). A stack kept by ``st.cache_resource`` then holds instances of the
    old classes while re-imported modules define new ones, so the first
    cross-module pydantic check fails — e.g. ``PromptVars.object_sections``
    rejects ``ObjectSection`` instances built from the evicted module. The old
    classes stay alive inside the cached stack, so their ids cannot be reused:
    a changed id means "modules were reloaded, rebuild the stack".
    """
    from src.department_qa import contract, object_profile

    return (
        id(contract.ObjectSection),
        id(contract.Evidence),
        id(object_profile.ObjectProfile),
    )


def make_model_fn(
    llm,
    schema: type[BaseModel] = ModelAnswer,
    recorder: Optional[Callable[[dict], None]] = None,
) -> Callable[[str], BaseModel]:
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
                    "usage": (
                        dict(raw.usage_metadata)
                        if isinstance(raw, AIMessage) and raw.usage_metadata
                        else None
                    ),
                }
            )
        return result

    def _call(prompt: str) -> BaseModel:
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
    verifier_fn: Callable[[str], VerificationResult]
    store: object
    retrieve_fn: Optional[Callable] = None
    known_units: set[str] = field(default_factory=set)
    service_limits: object | None = None


def build_department_stack(
    recorder: Optional[Callable[[dict], None]] = None,
    verifier_recorder: Optional[Callable[[dict], None]] = None,
) -> DepartmentStack:
    """The only assembly of the department Q&A stack: Streamlit page and eval share it."""
    from config.settings import settings
    from src.backends.chroma_backend import ChromaBackend
    from src.infra.llm_factory import get_embedding_model
    from src.indexing.manifest import load_manifest
    from src.infra.llm_factory import get_simple_llm
    from src.v7.bridge import build_v7_runtime
    from src.department_qa.service import ServiceLimits

    manifest = load_manifest(settings.CORPUS_MANIFEST_PATH)
    config = build_mode_config(
        settings.DEPARTMENT_QA_MODE, manifest, settings.SOURCE_DOCS_PATH, settings
    )
    ensure_store_matches(
        config, settings.CHROMA_DB_PATH, settings.CHROMA_COLLECTION_NAME
    )
    embeddings = get_embedding_model()
    store = ChromaBackend(
        path=config.chroma_db_path,
        collection=config.collection,
        embeddings=embeddings,
    )
    if config.mode == "v2":
        ensure_manifest_matches(manifest, store)
    runtime = build_v7_runtime(store)

    def scoped_retrieve(
        question: str,
        *,
        corpus: str,
        filters: dict,
        token_budget: int,
        deadline: float,
        embedding_memo: RequestEmbeddingMemo,
        require_multi_doc: bool | None,
    ):
        scoped_runtime = bind_request_embedding(
            runtime,
            memo=embedding_memo,
            space=f"{config.chroma_db_path}:{config.collection}",
            embed=embeddings.embed_query,
        )
        from dataclasses import replace

        scoped_runtime = replace(
            scoped_runtime,
            pack_limits=PackLimits(
                max_passages=8,
                token_budget=token_budget,
            ),
        )
        return retrieve_context(
            question,
            runtime=scoped_runtime,
            filters=filters,
            strict_scope=True,
            deadline=deadline,
            require_multi_doc=require_multi_doc,
        )

    llm = get_simple_llm()
    model_fn = make_model_fn(llm, schema=config.schema, recorder=recorder)
    verifier_fn = make_model_fn(
        llm, schema=VerificationResult, recorder=verifier_recorder
    )
    return DepartmentStack(
        config=config,
        manifest=manifest,
        search_fn=make_hybrid_search_fn(store, runtime.bm25_search),
        model_fn=model_fn,
        verifier_fn=verifier_fn,
        store=store,
        retrieve_fn=scoped_retrieve,
        known_units={
            metadata["unit_id"]
            for metadata in manifest.documents.values()
            if metadata.get("unit_id")
        },
        service_limits=ServiceLimits(
            retrieval_timeout_s=settings.DEPARTMENT_RETRIEVAL_TIMEOUT_S,
            max_workers=settings.DEPARTMENT_RETRIEVAL_WORKERS,
            max_pending=settings.DEPARTMENT_RETRIEVAL_PENDING,
        ),
    )
