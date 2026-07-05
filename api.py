"""FastAPI REST API for Safety Incident Analyzer v7 pipeline.

Exposes the v7 RAG graph as a service so external apps (WTA, etc.) can query it.

Endpoints:
    POST /query    — ask a question, get answer + passages (full v7 pipeline)
    POST /retrieve — retrieval-only hybrid search (no LLM), for batch clients (WTA)
    GET  /corpus   — unique source documents in the index
    GET  /health   — liveness check

Run:
    uvicorn api:app --host 0.0.0.0 --port 8503
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

load_dotenv()

# Must happen before any langchain_google_genai import (which occurs inside src modules).
from src.infra.llm_factory import apply_ipv6_patch_for_googleapis  # noqa: E402

apply_ipv6_patch_for_googleapis()

logger = structlog.get_logger()

# Rate limiter: 30/minute default, 10/minute on query endpoints
limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ChromaDB and initialize v7 pipeline on startup.

    Graceful degradation: init failures leave pipeline=None so /health and
    /query return 503 instead of crashing the process.
    """
    app.state.pipeline = None
    app.state.vector_store = None
    app.state.corpus_sources = None

    logger.info("api.startup: loading vector store and v7 pipeline")
    try:
        from src.backends.vector_store import get_vector_store_backend
        from src.v7.bridge import init_v7_pipeline
        from src.v7.graph import build_graph

        vector_store = get_vector_store_backend(load_existing=True)
        init_v7_pipeline(vector_store)
        app.state.vector_store = vector_store
        app.state.pipeline = build_graph().compile()
        logger.info("api.startup: v7 pipeline ready")
    except Exception as exc:
        logger.error(
            "api.startup: main pipeline init failed", error=str(exc), exc_info=True
        )

    yield
    app.state.pipeline = None
    app.state.vector_store = None
    logger.info("api.shutdown: pipeline cleared")


app = FastAPI(
    title="Regulatory RAG API",
    description="RAG API for Russian regulatory documents (ГОСТ, СНиП, ТК РФ, etc.)",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a short request ID to every request for log correlation."""
    request.state.request_id = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "rate limit exceeded — please slow down"},
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Passage(BaseModel):
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    passages: list[Passage]
    path: str
    elapsed_sec: float


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    source_filter: str | None = Field(default=None, max_length=500)


class RetrieveResponse(BaseModel):
    passages: list[Passage]
    elapsed_sec: float


class CorpusResponse(BaseModel):
    sources: list[str]


def _hybrid_retrieve(
    question: str,
    k: int,
    source_filter: str | None = None,
) -> list[dict]:
    """Retrieval-only hybrid search: vector + BM25 → RRF merge (+ optional rerank).

    No LLM calls (no multi-query expand, no generation) — built for batch
    clients (WTA GOST-check) that need low latency.

    source_filter is applied as a post-filter on metadata.source for both
    branches (ChromaBackend.similarity_search_with_score does not expose
    filters; BM25Index filters match top-level keys, not metadata), with
    over-fetch to keep enough candidates after filtering.
    """
    from src.v7.nlp_core import bm25_search, rrf_merge
    from src.v7.nodes import rag_simple as rag_simple_mod

    fetch_k = max(k * 6, 30) if source_filter else max(k * 2, 12)
    vector_results = rag_simple_mod._vector_search(query=question, top_k=fetch_k)
    bm25_results = bm25_search(query=question, top_k=fetch_k)

    if source_filter:

        def _matches(p: dict) -> bool:
            return p.get("metadata", {}).get("source") == source_filter

        vector_results = [p for p in vector_results if _matches(p)]
        bm25_results = [p for p in bm25_results if _matches(p)]

    # rrf_merge dedups by chunk_id with an "unknown_{rank}" fallback — passages
    # without chunk_id collide across lists. Derive a stable id from text.
    for p in vector_results + bm25_results:
        p.setdefault("chunk_id", p.get("text", "")[:80])

    passages = rrf_merge(vector_results, bm25_results, top_k=max(k, 12))
    if not passages:
        return []

    rerank_fn = getattr(rag_simple_mod, "_reranker_fn", None)
    if rerank_fn is not None:
        try:
            reranked = rerank_fn(question, passages[: max(k, 12)], k)
            if reranked:
                passages = reranked
        except Exception as exc:
            logger.warning(
                "api.retrieve: rerank failed, using RRF order", error=str(exc)
            )

    return passages[:k]


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(request: Request, req: QueryRequest) -> QueryResponse:
    """Ask a question about regulatory documents."""
    pipeline_app = request.app.state.pipeline
    if pipeline_app is None:
        raise HTTPException(status_code=503, detail="pipeline not initialized")

    rid = getattr(request.state, "request_id", "no-rid")
    t0 = time.perf_counter()
    try:
        result = pipeline_app.invoke({"query": req.question.strip()})
    except Exception as exc:
        logger.error(
            "api.query: pipeline error",
            request_id=rid,
            question=req.question[:80],
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"internal error (request_id={rid})"
        ) from exc
    elapsed = round(time.perf_counter() - t0, 2)

    if result.get("clarify_message"):
        answer = result["clarify_message"]
    elif result.get("abstain_reason"):
        answer = f"Не могу ответить: {result['abstain_reason']}"
    else:
        answer = result.get("answer") or ""

    raw_passages = result.get("final_passages") or []
    passages = [
        Passage(
            text=p.get("text", ""),
            source=p.get("metadata", {}).get("source", ""),
            score=float(p.get("score", 0.0)),
        )
        for p in raw_passages
    ]

    path = _infer_path(result)

    logger.info(
        "api.query: done",
        request_id=rid,
        question=req.question[:80],
        path=path,
        passages=len(passages),
        elapsed_sec=elapsed,
    )
    return QueryResponse(
        answer=answer, passages=passages, path=path, elapsed_sec=elapsed
    )


@app.post("/retrieve", response_model=RetrieveResponse)
@limiter.limit("600/minute")
def retrieve(request: Request, req: RetrieveRequest) -> RetrieveResponse:
    """Retrieval-only hybrid search — no LLM, for batch clients (WTA)."""
    if getattr(request.app.state, "vector_store", None) is None:
        raise HTTPException(status_code=503, detail="vector store not initialized")

    rid = getattr(request.state, "request_id", "no-rid")
    t0 = time.perf_counter()
    try:
        raw_passages = _hybrid_retrieve(
            question=req.question.strip(),
            k=req.k,
            source_filter=req.source_filter,
        )
    except Exception as exc:
        logger.error(
            "api.retrieve: retrieval error",
            request_id=rid,
            question=req.question[:80],
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"internal error (request_id={rid})"
        ) from exc
    elapsed = round(time.perf_counter() - t0, 2)

    passages = [
        Passage(
            text=p.get("text", ""),
            source=p.get("metadata", {}).get("source", ""),
            score=float(p.get("score", 0.0)),
        )
        for p in raw_passages
    ]
    logger.info(
        "api.retrieve: done",
        request_id=rid,
        question=req.question[:80],
        k=req.k,
        source_filter=req.source_filter,
        passages=len(passages),
        elapsed_sec=elapsed,
    )
    return RetrieveResponse(passages=passages, elapsed_sec=elapsed)


@app.get("/corpus", response_model=CorpusResponse)
@limiter.limit("600/minute")
def corpus(request: Request) -> CorpusResponse:
    """Unique metadata.source values in the index (cached after first call)."""
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(status_code=503, detail="vector store not initialized")

    cached: list[str] | None = getattr(request.app.state, "corpus_sources", None)
    if cached is not None:
        return CorpusResponse(sources=cached)

    rid = getattr(request.state, "request_id", "no-rid")
    try:
        sources = sorted(
            {
                (doc.get("metadata") or {}).get("source", "")
                for doc in vector_store.iter_all_documents()
            }
            - {""}
        )
    except Exception as exc:
        logger.error(
            "api.corpus: failed to list sources",
            request_id=rid,
            error=str(exc),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"internal error (request_id={rid})"
        ) from exc

    request.app.state.corpus_sources = sources
    logger.info("api.corpus: cached", request_id=rid, sources=len(sources))
    return CorpusResponse(sources=sources)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Liveness check."""
    if request.app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok"}


def _infer_path(result: dict) -> str:
    """Derive human-readable pipeline path from state."""
    if result.get("clarify_message"):
        return "intent_gate → END (chitchat/oos)"
    if result.get("abstain_reason"):
        return "... → abstain → END"
    if result.get("complex_passages"):
        return "rag_simple → evaluate_triage → rag_complex → generate_answer → END"
    return "rag_simple → evaluate_triage → generate_answer → END"
