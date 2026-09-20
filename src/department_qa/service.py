"""Department Q&A flow: two scoped searches → one prompt → checked structured answer.

# ANCHOR: department answer service
# Role: vertical slice of spec §4–§9 for the Streamlit screen and eval.
# Input: question, unit_id (or None), search_fn(query, filters, top_k) -> passages,
#   model_fn(prompt) -> ModelAnswer(V1) (structured output, validated by schema),
#   profile (ObjectProfile or None), prompt_version.
# Output: DepartmentResponse (spec §8 contract) with evidence resolved from stored
#   chunks and object profile sections, never from model text; evidence = cited ids
#   only; out_of_scope shows no model content.
# Failure modes: retrieval error → failed/retrieval_failed (not "not found");
#   model or schema error → failed/generation_failed; bad citation →
#   failed/citation_invalid with bases hidden (spec П6, no regeneration); profile
#   for another unit or with unit_id=None → failed/profile_mismatch before search
#   (integration error).
"""

from __future__ import annotations

import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date
import threading
import time
from typing import Callable, Optional

import structlog
from pydantic import BaseModel, Field

from src.department_qa.contract import (
    AppliedConclusion,
    Basis,
    Corpus,
    DepartmentPromptVarsV5,
    Evidence,
    ModelAnswerV1,
    ObjectFact,
    PromptVars,
    RequestContext,
    Status,
    VerificationResult,
    cited_ids,
    decide,
)
from src.department_qa.object_profile import (
    ObjectProfile,
    profile_evidence,
    profile_prompt_block,
    typed_fields_prompt_lines,
)
from src.infra.prompt_manager import PromptManager
from src.v7.scope_filter import build_scope_filters
from src.v7.retrieval import RequestEmbeddingMemo, ScopedRetrievalResult

logger = structlog.get_logger()

SearchFn = Callable[..., list[dict]]
ModelFn = Callable[[str], ModelAnswerV1]
VerifierFn = Callable[[str], VerificationResult]
ProgressFn = Callable[[str], None]

PROMPT_ID = "department_answer"
VERIFY_PROMPT_ID = "department_verify"
TOP_K_PER_LEVEL = 8
SCOPED_PROMPT_VERSION = "v5"

# Real backend stages reported to the UI (spec streamlit-portfolio-demo §11);
# no invented stages. A stage is emitted only when it actually runs.
STAGE_RETRIEVAL_STARTED = "retrieval_started"
STAGE_RETRIEVAL_COMPLETED = "retrieval_completed"
STAGE_GENERATION_STARTED = "generation_started"
STAGE_GENERATION_COMPLETED = "generation_completed"
STAGE_VERIFICATION_STARTED = "verification_started"
STAGE_VERIFICATION_COMPLETED = "verification_completed"

_NEXT_STEP: dict[str, str] = {
    "answered": "",
    "needs_context": "Уточните вопрос и повторите запрос.",
    "needs_review": "Передать специалисту по пожарной безопасности для проверки указанных положений.",
    "out_of_scope": "Вопрос вне поддерживаемой темы (пожарная безопасность).",
    "failed": "Повторите запрос позже; при повторе ошибки обратитесь к специалисту.",
}


class DepartmentResponse(BaseModel):
    answer: str = ""
    external_basis: list[Basis] = Field(default_factory=list)
    internal_basis: list[Basis] = Field(default_factory=list)
    object_facts: list[ObjectFact] = Field(default_factory=list)
    applied_conclusions: list[AppliedConclusion] = Field(default_factory=list)
    status: Status
    reason_codes: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    next_step: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    trace_id: str
    snapshot_id: Optional[str] = None
    unit_id: Optional[str] = None
    profile_as_of_date: Optional[date] = None
    profile_sha256: Optional[str] = None
    verification: Optional[VerificationResult] = None
    requested_corpora: tuple[Corpus, ...] = Field(default_factory=tuple)
    retrieval_trace: dict[str, dict] = Field(default_factory=dict)
    retrieval_stats: dict[str, dict[str, int]] = Field(default_factory=dict)
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class ServiceLimits:
    """Bounded shared executor and prompt budgets for the scoped path."""

    retrieval_timeout_s: float = 30.0
    prompt_token_budget: int = 70_000
    response_reserve_tokens: int = 4_096
    schema_retry_reserve_tokens: int = 4_096
    max_workers: int = 4
    max_pending: int = 8

    def __post_init__(self) -> None:
        if self.retrieval_timeout_s <= 0:
            raise ValueError("retrieval_timeout_s must be positive")
        if self.prompt_token_budget <= 0 or self.max_workers <= 0:
            raise ValueError("prompt budget and max_workers must be positive")
        if self.max_pending < 0:
            raise ValueError("max_pending must not be negative")
        if self.response_reserve_tokens < 0 or self.schema_retry_reserve_tokens < 0:
            raise ValueError("token reserves must not be negative")


class BoundedRetrievalExecutor:
    """Process-wide pool with bounded admitted work and non-blocking failure return."""

    def __init__(self, *, max_workers: int, max_pending: int) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="department-retrieval"
        )
        self._slots = threading.BoundedSemaphore(max_workers + max_pending)

    def submit(self, fn, /, *args, timeout: float, **kwargs) -> Future:
        if not self._slots.acquire(timeout=max(0.0, timeout)):
            raise TimeoutError("department retrieval queue is full")
        try:
            future = self._pool.submit(fn, *args, **kwargs)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _future: self._slots.release())
        return future


_EXECUTORS: dict[tuple[int, int], BoundedRetrievalExecutor] = {}
_EXECUTORS_LOCK = threading.Lock()


def _shared_executor(limits: ServiceLimits) -> BoundedRetrievalExecutor:
    key = (limits.max_workers, limits.max_pending)
    with _EXECUTORS_LOCK:
        executor = _EXECUTORS.get(key)
        if executor is None:
            executor = BoundedRetrievalExecutor(
                max_workers=limits.max_workers, max_pending=limits.max_pending
            )
            _EXECUTORS[key] = executor
        return executor


def _scope_for(corpus: Corpus, unit_id: Optional[str]) -> dict:
    external, internal = build_scope_filters(unit_id)
    return external if corpus == "external" else internal


def _trace(result: ScopedRetrievalResult, queue_wait_ms: float) -> dict:
    return {
        "outcome": result.outcome,
        "route": result.route,
        "reason": result.reason,
        "final_passages": len(result.final_context),
        "attempts": result.attempts,
        "retrieval_ms": result.elapsed_ms,
        "queue_wait_ms": queue_wait_ms,
        "technical_failure": result.technical_failure,
    }


def _normative_budget(
    *,
    question: str,
    context: RequestContext,
    profile: Optional[ObjectProfile],
    prompts: PromptManager,
    limits: ServiceLimits,
) -> int:
    """Reserve the complete fixed prompt/profile plus answer and schema retry."""
    object_label, sections = profile_prompt_block(context.unit_id, profile)
    fixed = DepartmentPromptVarsV5(
        question=question,
        unit_label=(
            f"Подразделение: {context.unit_id}"
            if context.unit_id
            else "Подразделение не указано: доступны только общекорпоративные документы."
        ),
        external_evidence=[],
        internal_evidence=[],
        object_label=object_label,
        object_sections=sections,
        typed_fields=typed_fields_prompt_lines(profile) if profile else [],
        requested_corpora=context.corpora,
        corpus_outcomes={corpus: "pending" for corpus in context.corpora},
    )
    prompt = prompts.render(
        PROMPT_ID, version=SCOPED_PROMPT_VERSION, **fixed.model_dump()
    )
    fixed_tokens = len(prompt) // 4
    available = (
        limits.prompt_token_budget
        - fixed_tokens
        - limits.response_reserve_tokens
        - limits.schema_retry_reserve_tokens
    )
    if available <= 0:
        raise ValueError("mandatory prompt exceeds token budget")
    return available


def answer_scoped_question(
    question: str,
    context: RequestContext,
    retrieve_fn: Callable[..., ScopedRetrievalResult],
    model_fn: ModelFn,
    *,
    known_units: set[str],
    snapshot_id: Optional[str] = None,
    prompts: Optional[PromptManager] = None,
    profiles: Optional[dict[str, ObjectProfile]] = None,
    limits: ServiceLimits = ServiceLimits(),
    executor: Optional[BoundedRetrievalExecutor] = None,
    progress_fn: Optional[ProgressFn] = None,
) -> DepartmentResponse:
    """Run selected corpora concurrently, then perform exactly one generation."""
    started = time.monotonic()
    trace_id = uuid.uuid4().hex
    embedding_memo = RequestEmbeddingMemo()
    profiles = profiles or {}
    base = {
        "trace_id": trace_id,
        "snapshot_id": snapshot_id,
        "unit_id": context.unit_id,
        "requested_corpora": context.corpora,
    }

    def emit(stage: str) -> None:
        if progress_fn is None:
            return
        try:
            progress_fn(stage)
        except Exception as exc:
            logger.warning(
                "department_qa.progress_failed", trace_id=trace_id, error=str(exc)
            )

    def fail(reason: str, **extra) -> DepartmentResponse:
        logger.warning(
            "department_qa.scoped_failed",
            trace_id=trace_id,
            reason=reason,
            requested_corpora=context.corpora,
            total_ms=round((time.monotonic() - started) * 1000, 1),
        )
        return DepartmentResponse(
            status="failed",
            reason_codes=[reason],
            next_step=_NEXT_STEP["failed"],
            total_ms=(time.monotonic() - started) * 1000,
            retrieval_stats={"embedding": embedding_memo.stats()},
            **base,
            **extra,
        )

    if context.unit_id is not None and context.unit_id not in known_units:
        return fail("invalid_unit")
    profile = profiles.get(context.unit_id) if context.unit_id else None
    if context.include_object_profile and profile is None:
        return fail("profile_unavailable")
    if profile is not None and profile.unit_id != context.unit_id:
        return fail("profile_mismatch")
    if not context.include_object_profile:
        profile = None

    manager = prompts or PromptManager()
    try:
        total_normative_budget = _normative_budget(
            question=question,
            context=context,
            profile=profile,
            prompts=manager,
            limits=limits,
        )
    except ValueError:
        return fail("prompt_budget_exceeded")
    branch_budget = total_normative_budget // len(context.corpora)

    deadline = started + limits.retrieval_timeout_s
    retrieval_started = time.monotonic()
    emit(STAGE_RETRIEVAL_STARTED)
    pool = executor or _shared_executor(limits)
    futures: dict[Future, tuple[Corpus, float]] = {}

    def run_branch(corpus: Corpus, submitted: float):
        queue_wait_ms = (time.monotonic() - submitted) * 1000
        result = retrieve_fn(
            question,
            corpus=corpus,
            filters=_scope_for(corpus, context.unit_id),
            token_budget=branch_budget,
            deadline=deadline,
            embedding_memo=embedding_memo,
            require_multi_doc=False if len(context.corpora) == 2 else None,
        )
        return result, queue_wait_ms

    try:
        for corpus in context.corpora:
            submitted = time.monotonic()
            future = pool.submit(
                run_branch,
                corpus,
                submitted,
                timeout=deadline - submitted,
            )
            futures[future] = (corpus, submitted)
    except (TimeoutError, RuntimeError):
        for future in futures:
            future.cancel()
        return fail("retrieval_overloaded")

    results: dict[Corpus, ScopedRetrievalResult] = {}
    traces: dict[str, dict] = {}
    pending = set(futures)
    fatal_reason: Optional[str] = None
    while pending and fatal_reason is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            fatal_reason = "retrieval_timeout"
            break
        done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
        if not done:
            fatal_reason = "retrieval_timeout"
            break
        for future in done:
            corpus, submitted = futures[future]
            try:
                result, queue_wait_ms = future.result()
            except Exception:
                fatal_reason = "retrieval_failed"
                break
            traces[corpus] = _trace(result, queue_wait_ms)
            if result.outcome == "failed" or result.technical_failure:
                fatal_reason = "retrieval_failed"
                break
            results[corpus] = result
    traces = {
        corpus: (
            traces.get(corpus, {"outcome": "failed"})
            if corpus in context.corpora
            else {"outcome": "not_requested"}
        )
        for corpus in ("external", "internal")
    }
    if fatal_reason:
        for future in pending:
            future.cancel()
        return fail(fatal_reason, retrieval_trace=traces)

    ready = {
        corpus: result
        for corpus, result in results.items()
        if result.outcome == "ready"
    }
    emit(STAGE_RETRIEVAL_COMPLETED)
    if not ready:
        reasons = (
            ["retrieval_clarification"]
            if any(r.outcome == "clarification" for r in results.values())
            else ["insufficient_normative_evidence"]
        )
        status: Status = (
            "needs_context"
            if reasons[0] == "retrieval_clarification"
            else "needs_review"
        )
        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info(
            "department_qa.scoped_retrieval_incomplete",
            trace_id=trace_id,
            requested_corpora=context.corpora,
            status=status,
            reason_codes=reasons,
            retrieval_ms=round((time.monotonic() - retrieval_started) * 1000, 1),
            total_ms=round(elapsed_ms, 1),
            embedding=embedding_memo.stats(),
        )
        return DepartmentResponse(
            status=status,
            reason_codes=reasons,
            next_step=_NEXT_STEP[status],
            retrieval_trace=traces,
            retrieval_stats={"embedding": embedding_memo.stats()},
            retrieval_ms=(time.monotonic() - retrieval_started) * 1000,
            total_ms=elapsed_ms,
            **base,
        )

    external = _to_evidence(
        ready.get(
            "external", ScopedRetrievalResult([], "empty", None, None, [], 0, False)
        ).final_context,
        "ext",
        "external",
    )
    internal = _to_evidence(
        ready.get(
            "internal", ScopedRetrievalResult([], "empty", None, None, [], 0, False)
        ).final_context,
        "int",
        "internal",
    )
    objects = profile_evidence(profile) if profile else []
    ordered = external + internal + objects
    registry = {item.id: item for item in ordered}
    object_label, sections = profile_prompt_block(context.unit_id, profile)
    prompt_vars = DepartmentPromptVarsV5(
        question=question,
        unit_label=(
            f"Подразделение: {context.unit_id}"
            if context.unit_id
            else "Подразделение не указано: доступны только общекорпоративные документы."
        ),
        external_evidence=external,
        internal_evidence=internal,
        object_label=object_label,
        object_sections=sections,
        typed_fields=typed_fields_prompt_lines(profile) if profile else [],
        requested_corpora=context.corpora,
        corpus_outcomes={corpus: results[corpus].outcome for corpus in context.corpora},
    )
    prompt = manager.render(
        PROMPT_ID, version=SCOPED_PROMPT_VERSION, **prompt_vars.model_dump()
    )
    if (
        len(prompt) // 4
        + limits.response_reserve_tokens
        + limits.schema_retry_reserve_tokens
        > limits.prompt_token_budget
    ):
        return fail("prompt_budget_exceeded", retrieval_trace=traces)
    llm_started = time.monotonic()
    emit(STAGE_GENERATION_STARTED)
    try:
        answer = model_fn(prompt)
    except Exception:
        return fail("generation_failed", retrieval_trace=traces)
    llm_ms = (time.monotonic() - llm_started) * 1000
    emit(STAGE_GENERATION_COMPLETED)
    status, reasons = decide(
        answer,
        registry,
        profile_as_of=profile.as_of_date if profile else None,
        requested_corpora=context.corpora,
        include_object_profile=context.include_object_profile,
        require_corpus_basis=True,
    )
    missing = [
        f"{corpus}_evidence_missing"
        for corpus in context.corpora
        if results[corpus].outcome != "ready"
    ]
    if missing and status not in {"failed", "out_of_scope", "needs_context"}:
        status = "needs_review"
        reasons = list(dict.fromkeys([*reasons, *missing]))
    if reasons == ["citation_invalid"]:
        return fail("citation_invalid", retrieval_trace=traces)
    cited = cited_ids(answer)
    total_ms = (time.monotonic() - started) * 1000
    logger.info(
        "department_qa.scoped_answer",
        trace_id=trace_id,
        requested_corpora=context.corpora,
        status=status,
        reason_codes=reasons,
        corpus_routes={corpus: trace.get("route") for corpus, trace in traces.items()},
        corpus_reasons={
            corpus: trace.get("reason") for corpus, trace in traces.items()
        },
        retrieval_ms=round(llm_started - retrieval_started, 1),
        llm_ms=round(llm_ms, 1),
        total_ms=round(total_ms, 1),
        embedding=embedding_memo.stats(),
    )
    return DepartmentResponse(
        answer="" if status == "out_of_scope" else answer.answer,
        external_basis=[] if status == "out_of_scope" else answer.external_basis,
        internal_basis=[] if status == "out_of_scope" else answer.internal_basis,
        object_facts=(
            [] if status == "out_of_scope" else getattr(answer, "object_facts", [])
        ),
        applied_conclusions=(
            []
            if status == "out_of_scope"
            else getattr(answer, "applied_conclusions", [])
        ),
        status=status,
        reason_codes=reasons,
        clarifying_questions=(
            [] if status == "out_of_scope" else answer.clarifying_questions
        ),
        next_step=_NEXT_STEP[status],
        evidence=(
            []
            if status == "out_of_scope"
            else [item for item in ordered if item.id in cited]
        ),
        profile_as_of_date=profile.as_of_date if profile else None,
        profile_sha256=profile.content_sha256 if profile else None,
        retrieval_trace=traces,
        retrieval_stats={"embedding": embedding_memo.stats()},
        retrieval_ms=llm_started * 1000 - retrieval_started * 1000,
        llm_ms=llm_ms,
        total_ms=total_ms,
        **base,
    )


def _to_evidence(passages: list[dict], prefix: str, level) -> list[Evidence]:
    out = []
    for i, p in enumerate(passages, 1):
        meta = p.get("metadata") or {}
        out.append(
            Evidence(
                id=f"{prefix}_{i:03d}",
                level=level,
                text=p.get("text", ""),
                source=meta.get("source", ""),
                title=meta.get("title") or meta.get("source", ""),
                locator=meta.get("locator") or meta.get("parent_section"),
                document_id=meta.get("document_id"),
                chunk_id=meta.get("chunk_id", p.get("chunk_id")),
                retrieval_score=p.get("score"),
            )
        )
    return out


def answer_question(
    question: str,
    unit_id: Optional[str],
    search_fn: SearchFn,
    model_fn: ModelFn,
    snapshot_id: Optional[str] = None,
    prompts: Optional[PromptManager] = None,
    profile: Optional[ObjectProfile] = None,
    prompt_version: Optional[str] = None,
    verifier_fn: Optional[VerifierFn] = None,
    verify_prompt_version: str = "v1",
    progress_fn: Optional[ProgressFn] = None,
) -> DepartmentResponse:
    trace_id = uuid.uuid4().hex
    base = {
        "trace_id": trace_id,
        "snapshot_id": snapshot_id,
        "unit_id": unit_id,
        "profile_as_of_date": profile.as_of_date if profile else None,
        "profile_sha256": profile.content_sha256 if profile else None,
    }

    def _emit(stage: str) -> None:
        # A broken UI callback must never turn a valid answer into a failure.
        if progress_fn is None:
            return
        try:
            progress_fn(stage)
        except Exception as exc:
            logger.warning(
                "department_qa.progress_failed", trace_id=trace_id, error=str(exc)
            )

    def _fail(reason: str) -> DepartmentResponse:
        return DepartmentResponse(
            status="failed",
            reason_codes=[reason],
            next_step=_NEXT_STEP["failed"],
            **base,
        )

    if profile is not None and (unit_id is None or profile.unit_id != unit_id):
        logger.warning(
            "department_qa.profile_mismatch",
            trace_id=trace_id,
            unit_id=unit_id,
            profile_unit_id=profile.unit_id,
        )
        return _fail("profile_mismatch")

    ext_filter, int_filter = build_scope_filters(unit_id)
    _emit(STAGE_RETRIEVAL_STARTED)
    try:
        external = _to_evidence(
            search_fn(question, filters=ext_filter, top_k=TOP_K_PER_LEVEL),
            "ext",
            "external",
        )
        internal = _to_evidence(
            search_fn(question, filters=int_filter, top_k=TOP_K_PER_LEVEL),
            "int",
            "internal",
        )
    except Exception as exc:
        logger.warning(
            "department_qa.retrieval_failed", trace_id=trace_id, error=str(exc)
        )
        return _fail("retrieval_failed")
    _emit(STAGE_RETRIEVAL_COMPLETED)

    objects = profile_evidence(profile) if profile else []
    ordered = external + internal + objects
    evidence = {e.id: e for e in ordered}
    object_label, object_sections = profile_prompt_block(unit_id, profile)
    prompt_vars = PromptVars(
        question=question,
        unit_label=(
            f"Подразделение: {unit_id}"
            if unit_id
            else "Подразделение не указано: доступны только общекорпоративные документы."
        ),
        external_evidence=external,
        internal_evidence=internal,
        object_label=object_label,
        object_sections=object_sections,
        typed_fields=typed_fields_prompt_lines(profile) if profile else [],
    )
    try:
        _emit(STAGE_GENERATION_STARTED)
        prompt = (prompts or PromptManager()).render(
            PROMPT_ID, version=prompt_version, **prompt_vars.model_dump()
        )
        model_answer = model_fn(prompt)
    except Exception as exc:
        logger.warning(
            "department_qa.generation_failed", trace_id=trace_id, error=str(exc)
        )
        return _fail("generation_failed")
    _emit(STAGE_GENERATION_COMPLETED)

    _emit(STAGE_VERIFICATION_STARTED)
    status, reasons = decide(
        model_answer, evidence, profile_as_of=profile.as_of_date if profile else None
    )
    verification = None
    clarifying_questions = list(model_answer.clarifying_questions)
    if (
        verifier_fn is not None
        and status == "answered"
        and getattr(model_answer, "applied_conclusions", [])
    ):
        try:
            verify_prompt = (prompts or PromptManager()).render(
                VERIFY_PROMPT_ID,
                version=verify_prompt_version,
                question=question,
                answer_json=model_answer.model_dump_json(indent=2),
                evidence=ordered,
            )
            verification = verifier_fn(verify_prompt)
            if verification.verdict == "missing":
                status = "needs_context"
                reasons = ["verification_missing_facts"]
                clarifying_questions.extend(verification.missing_fields)
            elif verification.verdict == "contradiction":
                status = "needs_review"
                reasons = ["verification_contradiction"]
        except Exception as exc:
            logger.warning(
                "department_qa.verification_failed", trace_id=trace_id, error=str(exc)
            )
            status = "needs_review"
            reasons = ["verification_failed"]
    _emit(STAGE_VERIFICATION_COMPLETED)
    logger.info(
        "department_qa.answer",
        trace_id=trace_id,
        snapshot_id=snapshot_id,
        status=status,
        reason_codes=reasons,
        n_external=len(external),
        n_internal=len(internal),
        n_object=len(objects),
        prompt_version=prompt_version,
        profile_sha256=base["profile_sha256"],
        verification_verdict=verification.verdict if verification else None,
    )
    if reasons == ["citation_invalid"]:
        return _fail("citation_invalid")
    if status == "out_of_scope":
        return DepartmentResponse(
            status=status, reason_codes=[], next_step=_NEXT_STEP[status], **base
        )

    cited = cited_ids(model_answer)
    return DepartmentResponse(
        answer=model_answer.answer,
        external_basis=model_answer.external_basis,
        internal_basis=model_answer.internal_basis,
        object_facts=getattr(model_answer, "object_facts", []),
        applied_conclusions=getattr(model_answer, "applied_conclusions", []),
        status=status,
        reason_codes=reasons,
        clarifying_questions=clarifying_questions,
        next_step=_NEXT_STEP[status],
        evidence=[e for e in ordered if e.id in cited],
        verification=verification,
        **base,
    )
