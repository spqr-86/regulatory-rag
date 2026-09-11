"""Paired context-risk calibration for the V7 triage policy (issue #9)."""

from __future__ import annotations

import argparse
import hashlib
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--allow-draft", action="store_true")
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
    result = run_paired_audit(questions)
    result.update({"date": date.today().isoformat(), "gt_path": str(args.gt)})

    summary = result["summary"]
    print(
        "\n"
        f"n={summary['n']} escalation={summary['escalated']['rate']:.3f} "
        f"unsafe_generate={summary['unsafe_generate']['count']}/{summary['n']} "
        f"coverage={summary['mean_coverage']['final']:.3f} "
        f"complex_p95={summary['latency_ms']['complex']['p95']:.1f}ms"
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
