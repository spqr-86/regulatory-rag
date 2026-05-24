"""FastAPI REST API for Safety Incident Analyzer v7 pipeline.

Exposes the v7 RAG graph as a service so external apps (WTA, etc.) can query it.

Endpoints:
    POST /query  — ask a question, get answer + passages
    GET  /health — liveness check

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
    app.state.gosts_pipeline = None
    app.state.gosts_ready = False

    logger.info("api.startup: loading vector store and v7 pipeline")
    try:
        from src.backends.vector_store import get_vector_store_backend
        from src.v7.bridge import init_v7_pipeline
        from src.v7.graph import build_graph

        vector_store = get_vector_store_backend(load_existing=True)
        init_v7_pipeline(vector_store)
        app.state.pipeline = build_graph().compile()
        logger.info("api.startup: v7 pipeline ready")
    except Exception as exc:
        logger.error(
            "api.startup: main pipeline init failed", error=str(exc), exc_info=True
        )
        # leave pipeline=None; /query returns 503, /health returns 503, process stays alive

    # Gosts pipeline (ers_rag fork: V7 graph + DeepSeek LLM поверх chroma_db_gosts/wta_gosts)
    # Старый src/gosts_pipeline.py оставлен на месте на случай отката.
    try:
        from src.ers_rag import build_graph as _ers_build_graph
        from src.ers_rag import init_ers_rag_from_chroma as _ers_init

        _ers_init()
        app.state.gosts_pipeline = _ers_build_graph().compile()
        app.state.gosts_ready = True
        logger.info("api.startup: gosts (ers_rag) pipeline ready")
    except Exception as exc:
        logger.warning("api.startup: gosts pipeline not available", error=str(exc))

    yield
    app.state.pipeline = None
    app.state.gosts_pipeline = None
    app.state.gosts_ready = False
    logger.info("api.shutdown: pipeline cleared")


app = FastAPI(
    title="Safety Incident Analyzer API",
    description="RAG API for Russian workplace safety regulations (ГОСТ, СНиП, ТК РФ, etc.)",
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


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(request: Request, req: QueryRequest) -> QueryResponse:
    """Ask a question about workplace safety regulations."""
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

    # Extract answer
    if result.get("clarify_message"):
        answer = result["clarify_message"]
    elif result.get("abstain_reason"):
        answer = f"Не могу ответить: {result['abstain_reason']}"
    else:
        answer = result.get("answer") or ""

    # Extract passages
    raw_passages = result.get("final_passages") or []
    passages = [
        Passage(
            text=p.get("text", ""),
            source=p.get("metadata", {}).get("source", ""),
            score=float(p.get("score", 0.0)),
        )
        for p in raw_passages
    ]

    # Infer path from state flags
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


@app.post("/query/gosts", response_model=QueryResponse)
@limiter.limit("10/minute")
def query_gosts(request: Request, req: QueryRequest) -> QueryResponse:
    """Ask a question about technical standards (ГОСТ, СНиП, СП) for water treatment."""
    gosts_app = request.app.state.gosts_pipeline
    if not request.app.state.gosts_ready or gosts_app is None:
        raise HTTPException(status_code=503, detail="not ready")

    rid = getattr(request.state, "request_id", "no-rid")
    t0 = time.perf_counter()
    try:
        result = gosts_app.invoke({"query": req.question.strip()})
    except Exception as exc:
        logger.error(
            "api.gosts: pipeline error",
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

    raw_passages = result.get("final_passages") or result.get("fallback_passages") or []
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
        "api.gosts: done",
        request_id=rid,
        question=req.question[:80],
        passages=len(passages),
        elapsed_sec=elapsed,
    )
    return QueryResponse(
        answer=answer,
        passages=passages,
        path=path,
        elapsed_sec=elapsed,
    )


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
