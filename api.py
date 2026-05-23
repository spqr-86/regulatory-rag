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
from contextlib import asynccontextmanager
from typing import Any

import structlog
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

logger = structlog.get_logger()

# Pipeline state — initialized once on startup
_pipeline: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ChromaDB and initialize v7 pipeline on startup."""
    logger.info("api.startup: loading vector store and v7 pipeline")
    try:
        from src.v7.bridge import init_v7_from_chroma
        from src.v7.graph import build_graph
        from src.vector_store import load_vector_store

        vector_store = load_vector_store()
        init_v7_from_chroma(vector_store)
        _pipeline["app"] = build_graph().compile()
        logger.info("api.startup: v7 pipeline ready")
    except Exception as exc:
        logger.error("api.startup: pipeline init failed", error=str(exc))
        raise

    # Gosts pipeline (ers_rag fork: V7 graph + DeepSeek LLM поверх chroma_db_gosts/wta_gosts)
    # Старый src/gosts_pipeline.py оставлен на месте на случай отката.
    try:
        from src.ers_rag import build_graph as _ers_build_graph
        from src.ers_rag import init_ers_rag_from_chroma as _ers_init

        _ers_init()
        _pipeline["gosts_app"] = _ers_build_graph().compile()
        _pipeline["gosts_ready"] = True
        logger.info("api.startup: gosts (ers_rag) pipeline ready")
    except Exception as exc:
        logger.warning("api.startup: gosts pipeline not available", error=str(exc))
        _pipeline["gosts_ready"] = False

    yield
    _pipeline.clear()
    logger.info("api.shutdown: pipeline cleared")


app = FastAPI(
    title="Safety Incident Analyzer API",
    description="RAG API for Russian workplace safety regulations (ГОСТ, СНиП, ТК РФ, etc.)",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str


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
def query(req: QueryRequest) -> QueryResponse:
    """Ask a question about workplace safety regulations."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    pipeline_app = _pipeline.get("app")
    if pipeline_app is None:
        raise HTTPException(status_code=503, detail="pipeline not initialized")

    t0 = time.perf_counter()
    try:
        result = pipeline_app.invoke({"query": req.question.strip()})
    except Exception as exc:
        logger.error("api.query: pipeline error", question=req.question, error=str(exc))
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc
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
        question=req.question[:80],
        path=path,
        passages=len(passages),
        elapsed_sec=elapsed,
    )
    return QueryResponse(
        answer=answer, passages=passages, path=path, elapsed_sec=elapsed
    )


@app.post("/query/gosts", response_model=QueryResponse)
def query_gosts(req: QueryRequest) -> QueryResponse:
    """Ask a question about technical standards (ГОСТ, СНиП, СП) for water treatment."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    gosts_app = _pipeline.get("gosts_app")
    if not _pipeline.get("gosts_ready") or gosts_app is None:
        raise HTTPException(
            status_code=503,
            detail="gosts pipeline not available — run index_gosts.py first",
        )

    t0 = time.perf_counter()
    try:
        result = gosts_app.invoke({"query": req.question.strip()})
    except Exception as exc:
        logger.error("api.gosts: pipeline error", question=req.question, error=str(exc))
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc
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
def health() -> dict[str, str]:
    """Liveness check."""
    if _pipeline.get("app") is None:
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return {"status": "ok"}


def _infer_path(result: dict[str, Any]) -> str:
    """Derive human-readable pipeline path from state."""
    if result.get("clarify_message"):
        return "intent_gate → END (chitchat/oos)"
    if result.get("abstain_reason"):
        return "... → abstain → END"
    if result.get("complex_passages"):
        return "rag_simple → evaluate_triage → rag_complex → generate_answer → END"
    return "rag_simple → evaluate_triage → generate_answer → END"
