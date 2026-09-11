"""Paired context-risk calibration for the V7 triage policy (issue #9)."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.v7.nlp_core import passage_identity  # noqa: E402
from src.v7.contract import PackStatus  # noqa: E402
from src.v7.decide import accept, decide_simple, terminal_update  # noqa: E402
from src.v7.validate import required_obligations, validate_context  # noqa: E402

# ANCHOR: typed annotation boundary and pure coverage metrics for triage calibration.
# Input: JSONL questions plus packed contexts; output: validated records and element scores.
# Key rule: chunks within an element are OR alternatives; elements compose with AND.


class RequiredElement(BaseModel):
    """One independently required fact, condition, category, or exception."""

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    acceptable_chunk_ids: list[str] = Field(min_length=1)
    critical: bool = True


class CalibrationQuestion(BaseModel):
    """A retrieval question with an explicit context-completeness contract."""

    question: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    source: str = Field(min_length=1)
    required_elements: list[RequiredElement] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_element_ids(self) -> "CalibrationQuestion":
        ids = [element.id for element in self.required_elements]
        if len(ids) != len(set(ids)):
            raise ValueError("required_elements ids must be unique within a question")
        return self


class CalibrationProfile(BaseModel):
    """Simple-route controls varied by the development calibration grid."""

    hard_gate_threshold: float = 0.50
    min_passages: int = 5
    min_keyword_overlap_active: float = 0.15
    min_keyword_overlap_original: float = 0.10
    comparison_require_multi_doc: bool = True


class PreparedCandidate(BaseModel):
    """One already packed complex candidate reusable across grid profiles."""

    origin: str
    final_context: list[dict]
    active_query: str
    plan: dict
    pack_status: PackStatus
    verdict_basis: dict


class PairedSnapshot(BaseModel):
    """Retrieval and packing output that threshold profiles cannot change."""

    simple_context: list[dict]
    simple_plan: dict
    simple_pack_status: PackStatus
    simple_retrieval_error: bool
    simple_verdict_basis: dict
    active_query: str
    complex_candidates: list[PreparedCandidate]
    simple_latency_ms: float
    complex_latency_ms: float


def apply_profile_to_plan(
    plan: Mapping[str, object],
    profile: CalibrationProfile,
    *,
    comparison_query: bool,
    on_complex: bool = False,
) -> dict:
    """Apply only controls that are effective on the selected retrieval path."""
    calibrated = dict(plan)
    if not on_complex:
        calibrated.update(
            {
                "threshold": profile.hard_gate_threshold,
                "min_passages": profile.min_passages,
                "min_keyword_overlap": profile.min_keyword_overlap_active,
            }
        )
    calibrated["min_keyword_overlap_original"] = profile.min_keyword_overlap_original
    if comparison_query:
        calibrated["require_multi_doc"] = profile.comparison_require_multi_doc
    return calibrated


def build_grid_profiles(
    baseline: CalibrationProfile,
    *,
    hard_gate_thresholds: Sequence[float],
    min_passages_values: Sequence[int],
    active_overlap_values: Sequence[float],
    original_overlap_values: Sequence[float],
    comparison_policy_values: Sequence[bool],
    has_comparison_questions: bool,
) -> list[CalibrationProfile]:
    """Build a deterministic grid without tuning an unobserved query policy."""
    comparison_values = (
        list(comparison_policy_values)
        if has_comparison_questions
        else [baseline.comparison_require_multi_doc]
    )
    return [
        CalibrationProfile(
            hard_gate_threshold=threshold,
            min_passages=min_passages,
            min_keyword_overlap_active=active_overlap,
            min_keyword_overlap_original=original_overlap,
            comparison_require_multi_doc=require_multi_doc,
        )
        for threshold, min_passages, active_overlap, original_overlap, require_multi_doc in itertools.product(
            hard_gate_thresholds,
            min_passages_values,
            active_overlap_values,
            original_overlap_values,
            comparison_values,
        )
    ]


def select_grid_profile(
    rows: Sequence[Mapping[str, object]],
    *,
    baseline_unsafe: int,
    baseline_critical_misses: int,
    baseline_profile: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Select the cheapest risk-safe profile, avoiding arbitrary config drift."""
    eligible = [
        row
        for row in rows
        if row["summary"]["unsafe_generate"]["count"] <= baseline_unsafe
        and row["summary"]["critical_miss_generate"]["count"]
        <= baseline_critical_misses
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            row["summary"]["escalated"]["count"],
            -row["summary"]["full_coverage_generate"]["count"],
            row["summary"]["latency_ms"]["terminal"]["p95"],
            row["profile"] != baseline_profile,
        ),
    )


def _is_comparison_query(question: str) -> bool:
    from src.v7.nodes.router import COMPARISON_MARKERS

    lowered = question.lower()
    return any(marker in lowered for marker in COMPARISON_MARKERS)


def _rethreshold_verdict(verdict_basis: Mapping[str, object], plan: Mapping) -> dict:
    """Recompute threshold-dependent verdict fields from cached context signals."""
    details = dict(verdict_basis["details"])
    top_score = float(details["top_score"])
    passage_count = int(details["passage_count"])
    overlap_active = float(details["keyword_overlap_active"])
    overlap_original = float(details["keyword_overlap_original"])
    max_doc_ratio = float(details["max_doc_ratio"])

    above_threshold = top_score >= float(plan.get("threshold", 0.0))
    enough_evidence = passage_count >= int(plan.get("min_passages", 1))
    keyword_overlap_ok = overlap_active >= float(plan.get("min_keyword_overlap", 0.0))
    diversity_ok = max_doc_ratio <= float(plan.get("max_single_doc_ratio", 1.0))
    require_multi_doc = bool(plan.get("require_multi_doc", False))
    escalation_hint = require_multi_doc and not diversity_ok
    hard_ok = all(
        [
            above_threshold,
            enough_evidence,
            keyword_overlap_ok,
            not escalation_hint,
        ]
    )
    if hard_ok:
        triage = "sufficient"
    elif (
        top_score < float(plan.get("borderline_threshold", 0.0)) or not enough_evidence
    ):
        triage = "clearly_bad"
    else:
        triage = "borderline"

    details.update(
        {
            "above_threshold": above_threshold,
            "enough_evidence": enough_evidence,
            "keyword_overlap_ok": keyword_overlap_ok,
            "diversity_ok": diversity_ok,
            "escalation_hint": escalation_hint,
            "sufficient": hard_ok,
            "triage": triage,
        }
    )
    unmet = [
        obligation
        for obligation in verdict_basis["obligations_unmet"]
        if obligation != "original_query_relevant"
    ]
    original_floor = float(plan.get("min_keyword_overlap_original", 0.0))
    if overlap_original <= 0.0 or overlap_original < original_floor:
        unmet.append("original_query_relevant")
    return {
        **dict(verdict_basis),
        "triage": triage,
        "hard_ok": hard_ok,
        "details": details,
        "obligations_unmet": unmet,
        "top_score": top_score,
    }


def evaluate_paired_snapshot(
    question: CalibrationQuestion,
    snapshot: PairedSnapshot,
    profile: CalibrationProfile,
) -> dict:
    """Re-evaluate one fixed retrieval snapshot under a threshold profile."""
    required = required_obligations(question.question)
    comparison_query = _is_comparison_query(question.question)
    simple_plan = apply_profile_to_plan(
        snapshot.simple_plan,
        profile,
        comparison_query=comparison_query,
    )
    simple_verdict = _rethreshold_verdict(snapshot.simple_verdict_basis, simple_plan)
    simple_route, simple_reason = decide_simple(
        snapshot.simple_context,
        simple_verdict,
        retrieval_error=snapshot.simple_retrieval_error,
        abstain_on_empty=True,
    )
    simple_update = terminal_update(
        route=simple_route,
        reason=simple_reason,
        final_context=snapshot.simple_context,
        candidate_context=snapshot.simple_context,
        verdict=simple_verdict,
        required=required,
        technical_failure=(
            snapshot.simple_retrieval_error or snapshot.simple_pack_status == "degraded"
        ),
    )

    complex_update = None
    rejected: list[dict] = []
    for index, candidate in enumerate(snapshot.complex_candidates):
        complex_plan = apply_profile_to_plan(
            candidate.plan,
            profile,
            comparison_query=comparison_query,
            on_complex=candidate.origin != "fallback_snapshot",
        )
        verdict = _rethreshold_verdict(candidate.verdict_basis, complex_plan)
        is_last = index == len(snapshot.complex_candidates) - 1
        if accept(
            candidate.final_context,
            verdict,
            on_complex=True,
            is_last_candidate=is_last,
        ):
            reason = {
                "merged": "complex_sufficient",
                "last_attempt": "complex_sufficient",
                "fallback_snapshot": "complex_fallback_accepted",
            }.get(candidate.origin, "complex_sufficient")
            if verdict["obligations_unmet"] == ["refs_resolved"]:
                reason = "refs_best_effort"
            elif verdict["obligations_unmet"]:
                reason = "enumeration_best_effort"
            complex_update = terminal_update(
                route="generate",
                reason=reason,
                final_context=candidate.final_context,
                candidate_context=candidate.final_context,
                verdict=verdict,
                required=required,
                technical_failure=candidate.pack_status == "degraded",
                rejected=rejected,
            )
            break
        rejected.append(
            {
                "origin": candidate.origin,
                "obligations_unmet": list(verdict["obligations_unmet"]),
                "triage": verdict["triage"],
                "passages": len(candidate.final_context),
                "pack_status": candidate.pack_status,
            }
        )

    if complex_update is None:
        empty_plan = apply_profile_to_plan(
            snapshot.simple_plan,
            profile,
            comparison_query=comparison_query,
            on_complex=True,
        )
        empty_verdict = validate_context(
            [],
            question.question,
            snapshot.active_query,
            empty_plan,
            required,
        )
        complex_update = terminal_update(
            route="abstain",
            reason="complex_exhausted",
            final_context=[],
            candidate_context=[],
            verdict=empty_verdict,
            required=required,
            technical_failure=False,
            rejected=rejected,
        )

    return make_audit_record(
        question,
        simple_update,
        complex_update,
        simple_latency_ms=snapshot.simple_latency_ms,
        complex_latency_ms=snapshot.complex_latency_ms,
    )


class AnnotationManifest(BaseModel):
    """Sidecar that classifies every row without duplicating the base GT."""

    base_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["draft", "reviewed"]
    count: int = Field(gt=0)
    factoid_indices: list[int] = Field(default_factory=list)
    elements_by_index: dict[int, list[RequiredElement]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def covers_each_index_once(self) -> "AnnotationManifest":
        factoids = set(self.factoid_indices)
        custom = set(self.elements_by_index)
        expected = set(range(1, self.count + 1))
        if factoids & custom:
            raise ValueError("an index cannot be both factoid and custom")
        if factoids | custom != expected:
            missing = sorted(expected - (factoids | custom))
            extra = sorted((factoids | custom) - expected)
            raise ValueError(
                f"annotation coverage mismatch: missing={missing}, extra={extra}"
            )
        if any(not elements for elements in self.elements_by_index.values()):
            raise ValueError("custom annotations must contain required elements")
        return self


def load_calibration_gt(path: Path) -> list[CalibrationQuestion]:
    """Load JSONL and fail closed when any completeness annotation is absent."""
    records: list[CalibrationQuestion] = []
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(CalibrationQuestion.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return records


def load_calibration_split(
    base_path: Path, annotations_path: Path, *, allow_draft: bool = False
) -> list[CalibrationQuestion]:
    """Join a hash-pinned base GT with an exhaustive annotation sidecar."""
    manifest = AnnotationManifest.model_validate_json(
        annotations_path.read_text(encoding="utf-8")
    )
    if manifest.status != "reviewed" and not allow_draft:
        raise ValueError(
            "annotation manifest is draft; review it or pass --allow-draft for diagnostics"
        )
    actual_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
    if actual_hash != manifest.base_sha256:
        raise ValueError(
            "base GT SHA-256 does not match annotation manifest: "
            f"expected {manifest.base_sha256}, got {actual_hash}"
        )

    raw_rows = [
        json.loads(line)
        for line in base_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: manifest.count]
    if len(raw_rows) != manifest.count:
        raise ValueError(
            f"base GT has {len(raw_rows)} rows, manifest requires {manifest.count}"
        )

    questions: list[CalibrationQuestion] = []
    for index, raw in enumerate(raw_rows, start=1):
        if index in manifest.factoid_indices:
            raw["required_elements"] = [
                {
                    "id": "answer",
                    "label": "достаточный нормативный ответ",
                    "acceptable_chunk_ids": raw.get("relevant_chunk_ids")
                    or [raw.get("chunk_id")],
                    "critical": True,
                }
            ]
        else:
            raw["required_elements"] = [
                element.model_dump() for element in manifest.elements_by_index[index]
            ]
        questions.append(CalibrationQuestion.model_validate(raw))
    return questions


def score_context(
    passages: Iterable[dict],
    required_elements: Sequence[RequiredElement],
) -> dict:
    """Measure required-element coverage on an exact packed context."""
    present = {passage_identity(passage) for passage in passages if passage}
    covered: list[str] = []
    missing: list[str] = []
    critical_misses: list[str] = []

    for element in required_elements:
        if present.intersection(element.acceptable_chunk_ids):
            covered.append(element.id)
        else:
            missing.append(element.id)
            if element.critical:
                critical_misses.append(element.id)

    required = len(required_elements)
    return {
        "covered": len(covered),
        "required": required,
        "coverage": len(covered) / required if required else 0.0,
        "all_required_covered": not missing and required > 0,
        "critical_misses": critical_misses,
        "covered_element_ids": covered,
        "missing_element_ids": missing,
    }


def binary_hit(passages: Iterable[dict], relevant_chunk_ids: Sequence[str]) -> bool:
    """Preserve the legacy OR semantics for the eight-cell diagnostic matrix."""
    present = {passage_identity(passage) for passage in passages if passage}
    return bool(present.intersection(relevant_chunk_ids))


def make_audit_record(
    question: CalibrationQuestion,
    simple_update: Mapping[str, object],
    complex_update: Mapping[str, object],
    *,
    simple_latency_ms: float,
    complex_latency_ms: float,
) -> dict:
    """Build one auditable paired record from both terminal decisions."""
    simple_context = list(simple_update.get("final_context") or [])
    complex_context = list(complex_update.get("final_context") or [])
    escalated = simple_update.get("route_decision") == "complex"
    terminal = complex_update if escalated else simple_update
    terminal_route = str(terminal.get("route_decision") or "abstain")
    final_context = (
        list(terminal.get("final_context") or [])
        if terminal_route == "generate"
        else []
    )

    simple_score = score_context(simple_context, question.required_elements)
    complex_score = score_context(complex_context, question.required_elements)
    final_score = score_context(final_context, question.required_elements)

    return {
        "question": question.question,
        "source": question.source,
        "escalated": escalated,
        "simple_hit": binary_hit(simple_context, question.relevant_chunk_ids),
        "complex_hit": binary_hit(complex_context, question.relevant_chunk_ids),
        "simple_coverage": simple_score["coverage"],
        "complex_coverage": complex_score["coverage"],
        "final_coverage": final_score["coverage"],
        "simple_missing_elements": simple_score["missing_element_ids"],
        "complex_missing_elements": complex_score["missing_element_ids"],
        "final_missing_elements": final_score["missing_element_ids"],
        "final_critical_misses": final_score["critical_misses"],
        "terminal_route": terminal_route,
        "terminal_reason": str(terminal.get("route_reason") or "unknown"),
        "simple_route": str(simple_update.get("route_decision") or "abstain"),
        "simple_reason": str(simple_update.get("route_reason") or "unknown"),
        "hypothetical_complex_route": str(
            complex_update.get("route_decision") or "abstain"
        ),
        "hypothetical_complex_reason": str(
            complex_update.get("route_reason") or "unknown"
        ),
        "simple_context_ids": [
            passage_identity(passage) for passage in simple_context if passage
        ],
        "complex_context_ids": [
            passage_identity(passage) for passage in complex_context if passage
        ],
        "simple_latency_ms": round(simple_latency_ms, 3),
        "complex_latency_ms": round(complex_latency_ms, 3),
    }


def _latency_summary(values: Sequence[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "max": 0.0}

    def percentile(p: float) -> float:
        index = max(0, math.ceil(p * len(ordered)) - 1)
        return round(ordered[index], 3)

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "mean": round(sum(ordered) / len(ordered), 3),
        "max": round(ordered[-1], 3),
    }


def summarize_audit(records: Sequence[Mapping[str, object]]) -> dict:
    """Aggregate a paired route audit without hiding numerators."""
    matrix = {
        f"{route}|{simple}|{complex_}": 0
        for route in ("not_escalated", "escalated")
        for simple in ("simple_miss", "simple_hit")
        for complex_ in ("complex_miss", "complex_hit")
    }
    routes: dict[str, int] = {}
    reasons: dict[str, int] = {}
    unsafe_generate = 0
    critical_miss_generate = 0
    full_coverage_generate = 0

    for record in records:
        key = "|".join(
            [
                "escalated" if record["escalated"] else "not_escalated",
                "simple_hit" if record["simple_hit"] else "simple_miss",
                "complex_hit" if record["complex_hit"] else "complex_miss",
            ]
        )
        matrix[key] += 1

        route = str(record["terminal_route"])
        reason = str(record["terminal_reason"])
        routes[route] = routes.get(route, 0) + 1
        reasons[reason] = reasons.get(reason, 0) + 1
        if route == "generate" and float(record["final_coverage"]) < 1.0:
            unsafe_generate += 1
        if route == "generate" and record["final_critical_misses"]:
            critical_miss_generate += 1
        if route == "generate" and float(record["final_coverage"]) == 1.0:
            full_coverage_generate += 1

    n = len(records)

    def rate(count: int) -> dict:
        return {"count": count, "rate": count / n if n else 0.0}

    def mean(field: str) -> float:
        return sum(float(record[field]) for record in records) / n if n else 0.0

    return {
        "n": n,
        "matrix": matrix,
        "escalated": rate(sum(bool(record["escalated"]) for record in records)),
        "unsafe_generate": rate(unsafe_generate),
        "critical_miss_generate": rate(critical_miss_generate),
        "full_coverage_generate": rate(full_coverage_generate),
        "mean_coverage": {
            "simple": mean("simple_coverage"),
            "complex": mean("complex_coverage"),
            "final": mean("final_coverage"),
        },
        "routes": dict(sorted(routes.items())),
        "reasons": dict(sorted(reasons.items())),
        "latency_ms": {
            "simple": _latency_summary(
                [float(record["simple_latency_ms"]) for record in records]
            ),
            "complex": _latency_summary(
                [float(record["complex_latency_ms"]) for record in records]
            ),
            "terminal": _latency_summary(
                [
                    float(record["simple_latency_ms"])
                    + (
                        float(record["complex_latency_ms"])
                        if record["escalated"]
                        else 0.0
                    )
                    for record in records
                ]
            ),
        },
    }


def run_paired_audit(questions: Sequence[CalibrationQuestion]) -> dict:
    """Run simple plus hypothetical complex retrieval once for every question."""
    from src.v7.nodes.evaluate_complex import evaluate_complex
    from src.v7.nodes.evaluate_triage import evaluate_triage
    from src.v7.nodes.rag_complex import rag_complex
    from src.v7.nodes.rag_simple import rag_simple
    from src.v7.nodes.router import router

    records: list[dict] = []
    for index, question in enumerate(questions, start=1):
        state: dict = {"query": question.question, "filters": None}
        state.update(router(state))
        if state.get("clarify_message"):
            raise ValueError(
                f"calibration question routed to clarify: {question.question}"
            )

        started = time.perf_counter()
        state.update(rag_simple(state))
        simple_latency_ms = (time.perf_counter() - started) * 1000
        simple_update = evaluate_triage(state)

        complex_state = dict(state)
        complex_state.update(simple_update)
        if not complex_state.get("fallback_snapshot"):
            complex_state["fallback_snapshot"] = {
                "passages": list(simple_update.get("final_context") or []),
                "plan": dict(state.get("plan") or {}),
                "active_query": state.get("active_query", question.question),
                "origin": "fallback_snapshot",
                "packed": True,
                "pack_status": "ok",
            }

        prior_attempts = list(complex_state.get("retrieval_attempts") or [])
        started = time.perf_counter()
        complex_retrieval = rag_complex(complex_state)
        complex_latency_ms = (time.perf_counter() - started) * 1000
        complex_state.update(complex_retrieval)
        complex_state["retrieval_attempts"] = prior_attempts + list(
            complex_retrieval.get("retrieval_attempts") or []
        )
        complex_update = evaluate_complex(complex_state)

        records.append(
            make_audit_record(
                question,
                simple_update,
                complex_update,
                simple_latency_ms=simple_latency_ms,
                complex_latency_ms=complex_latency_ms,
            )
        )
        print(
            f"[{index}/{len(questions)}] "
            f"{records[-1]['simple_route']} → {records[-1]['terminal_route']} "
            f"coverage={records[-1]['final_coverage']:.2f}"
        )

    return {"summary": summarize_audit(records), "records": records}


def prepare_paired_snapshots(
    questions: Sequence[CalibrationQuestion],
) -> list[PairedSnapshot]:
    """Retrieve and pack each simple/complex candidate exactly once for a grid."""
    from src.v7.nodes.evaluate_complex import _candidates, _prepare
    from src.v7.nodes.rag_complex import rag_complex
    from src.v7.nodes.rag_simple import rag_simple
    from src.v7.nodes.router import router
    from src.v7.nodes.visual_enrichment import enrich_passages
    from src.v7.pack_context import pack_context

    snapshots: list[PairedSnapshot] = []
    for index, question in enumerate(questions, start=1):
        state: dict = {"query": question.question, "filters": None}
        state.update(router(state))
        if state.get("clarify_message"):
            raise ValueError(
                f"calibration question routed to clarify: {question.question}"
            )

        started = time.perf_counter()
        state.update(rag_simple(state))
        simple_latency_ms = (time.perf_counter() - started) * 1000
        attempts = list(state.get("retrieval_attempts") or [])
        if not attempts:
            raise ValueError(
                f"simple retrieval produced no attempt: {question.question}"
            )
        simple_attempt = attempts[-1]
        simple_plan = dict(simple_attempt.get("attempt_plan") or state["plan"])
        simple_packed = pack_context(
            enrich_passages(simple_attempt.get("passages") or []),
            state.get("active_query", question.question),
            simple_plan,
        )
        required = required_obligations(question.question)
        simple_verdict_basis = validate_context(
            simple_packed["final_context"],
            question.question,
            state.get("active_query", question.question),
            simple_plan,
            required,
            pack_status=simple_packed["status"],
        )

        complex_state = dict(state)
        complex_state["fallback_snapshot"] = {
            "passages": list(simple_packed["final_context"]),
            "plan": simple_plan,
            "active_query": state.get("active_query", question.question),
            "origin": "fallback_snapshot",
            "packed": True,
            "pack_status": simple_packed["status"],
        }
        prior_attempts = list(complex_state.get("retrieval_attempts") or [])
        started = time.perf_counter()
        complex_retrieval = rag_complex(complex_state)
        complex_latency_ms = (time.perf_counter() - started) * 1000
        complex_state.update(complex_retrieval)
        complex_state["retrieval_attempts"] = prior_attempts + list(
            complex_retrieval.get("retrieval_attempts") or []
        )

        cache: dict = {}
        prepared_candidates: list[PreparedCandidate] = []
        for candidate in _candidates(complex_state):
            packed = _prepare(candidate, cache)
            verdict_basis = validate_context(
                packed["final_context"],
                question.question,
                candidate["active_query"],
                candidate["plan"],
                required,
                pack_status=packed["status"],
                prior_context=simple_packed["final_context"],
            )
            prepared_candidates.append(
                PreparedCandidate(
                    origin=candidate["origin"],
                    final_context=packed["final_context"],
                    active_query=candidate["active_query"],
                    plan=dict(candidate["plan"]),
                    pack_status=packed["status"],
                    verdict_basis=verdict_basis,
                )
            )

        snapshots.append(
            PairedSnapshot(
                simple_context=simple_packed["final_context"],
                simple_plan=simple_plan,
                simple_pack_status=simple_packed["status"],
                simple_retrieval_error=bool(simple_attempt.get("retrieval_error")),
                simple_verdict_basis=simple_verdict_basis,
                active_query=state.get("active_query", question.question),
                complex_candidates=prepared_candidates,
                simple_latency_ms=simple_latency_ms,
                complex_latency_ms=complex_latency_ms,
            )
        )
        print(f"[{index}/{len(questions)}] prepared grid snapshot")
    return snapshots


def run_calibration_grid(questions: Sequence[CalibrationQuestion]) -> dict:
    """Evaluate a bounded profile grid over one immutable paired retrieval run."""
    from src.v7.config import v7_config

    snapshots = prepare_paired_snapshots(questions)
    baseline_profile = CalibrationProfile(
        hard_gate_threshold=v7_config.HARD_GATE_THRESHOLD,
        min_passages=v7_config.MIN_PASSAGES,
        min_keyword_overlap_active=v7_config.MIN_KEYWORD_OVERLAP_ACTIVE,
        min_keyword_overlap_original=v7_config.MIN_KEYWORD_OVERLAP_ORIGINAL,
        comparison_require_multi_doc=True,
    )
    comparison_count = sum(_is_comparison_query(q.question) for q in questions)
    profiles = build_grid_profiles(
        baseline_profile,
        hard_gate_thresholds=[0.45, 0.50, 0.55],
        min_passages_values=[3, 5, 8],
        active_overlap_values=[0.10, 0.15, 0.20],
        original_overlap_values=[0.05, 0.10, 0.15],
        comparison_policy_values=[True, False],
        has_comparison_questions=comparison_count > 0,
    )

    rows: list[dict] = []
    for profile in profiles:
        records = [
            evaluate_paired_snapshot(question, snapshot, profile)
            for question, snapshot in zip(questions, snapshots, strict=True)
        ]
        rows.append(
            {
                "profile": profile.model_dump(),
                "summary": summarize_audit(records),
            }
        )

    baseline_records = [
        evaluate_paired_snapshot(question, snapshot, baseline_profile)
        for question, snapshot in zip(questions, snapshots, strict=True)
    ]
    baseline = {
        "profile": baseline_profile.model_dump(),
        "summary": summarize_audit(baseline_records),
        "records": baseline_records,
    }
    selected = select_grid_profile(
        rows,
        baseline_unsafe=baseline["summary"]["unsafe_generate"]["count"],
        baseline_critical_misses=baseline["summary"]["critical_miss_generate"]["count"],
        baseline_profile=baseline["profile"],
    )
    return {
        "comparison_questions": comparison_count,
        "profile_count": len(rows),
        "baseline": baseline,
        "selected": selected,
        "profiles": rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    from eval.run_retrieval_eval import init_engine

    args = _parse_args()
    questions = (
        load_calibration_split(args.gt, args.annotations, allow_draft=args.allow_draft)
        if args.annotations
        else load_calibration_gt(args.gt)
    )
    if args.limit is not None:
        questions = questions[: args.limit]
    init_engine()
    result = (
        run_calibration_grid(questions) if args.grid else run_paired_audit(questions)
    )
    result.update({"date": date.today().isoformat(), "gt_path": str(args.gt)})

    summary = result["baseline"]["summary"] if args.grid else result["summary"]
    print(
        "\n"
        f"n={summary['n']} escalation={summary['escalated']['rate']:.3f} "
        f"unsafe_generate={summary['unsafe_generate']['count']}/{summary['n']} "
        f"coverage={summary['mean_coverage']['final']:.3f} "
        f"complex_p95={summary['latency_ms']['complex']['p95']:.1f}ms"
    )
    if args.grid:
        selected = result["selected"]
        print(
            f"grid_profiles={result['profile_count']} "
            f"comparison_questions={result['comparison_questions']} "
            f"selected={selected['profile'] if selected else None}"
        )

    out = args.out or (
        PROJECT_ROOT
        / "benchmarks"
        / f"triage_calibration_{date.today().isoformat()}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Полный результат: {out}")


if __name__ == "__main__":
    main()
