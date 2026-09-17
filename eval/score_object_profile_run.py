"""Offline deterministic scorer for object-profile runs (variant B, step 1).

# ANCHOR: object profile run scorer
# Role: read the run expectations and a saved run catalog (q*.json) and report,
#   per question, which typed object fields (obj_f_*) the answer cited, whether
#   each is known or unknown in the expectation, the contract status and the
#   deterministic violations. No model calls, no judge — semantics stay manual.
# Input: eval/data/object_profile_pair_expectations.yaml and <run_dir>/q*.json
#   produced by run_object_profile_pair.py.
# Output: QuestionReport per question + a summary; violations are
#   ``applied_on_unknown`` (an applied conclusion rests on a field the sheet
#   marks unknown) and ``forbidden_conclusion`` (a forbidden string appears
#   verbatim). ``contract_ok`` compares (status, reasons) with the expectation.
# Failure modes: a question missing from the run or an expectation without
#   ``typed_fields`` raises ValueError naming what is wrong.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.department_qa.object_profile import FIELD_SPECS  # noqa: E402

DEFAULT_EXPECTATIONS = (
    REPO_ROOT / "eval" / "data" / "object_profile_pair_expectations.yaml"
)

UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
_FIELD_PREFIX = "obj_f_"
_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class FieldRef:
    """A typed object field cited by the answer, with its expected state."""

    id: str
    key: str
    label: str
    state: str  # known | unknown | not_applicable | unmapped


@dataclass(frozen=True)
class Violation:
    code: str
    detail: str


@dataclass(frozen=True)
class QuestionReport:
    n: int
    mode: str
    unit_id: str | None
    status: str
    reason_codes: tuple[str, ...]
    cited_fields: tuple[FieldRef, ...]
    violations: tuple[Violation, ...]
    contract_ok: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


def load_expectations(path: Path) -> dict[int, dict]:
    """Expectations by question number; ``typed_fields`` is optional."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    questions = data.get("questions") or []
    if not questions:
        raise ValueError(f"{path}: no questions")
    return {q["n"]: q for q in questions}


def load_run(run_dir: Path) -> dict[int, dict]:
    """Run records by question number from ``q*.json`` in a mode directory."""
    run_dir = Path(run_dir)
    records: dict[int, dict] = {}
    for path in sorted(run_dir.glob("q*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[record["n"]] = record
    if not records:
        raise ValueError(f"{run_dir}: no q*.json records")
    return records


def cited_field_ids(response: dict) -> list[str]:
    """``obj_f_*`` ids cited by object facts and applied conclusions, deduped."""
    ids: list[str] = []
    for item in (response.get("object_facts") or []) + (
        response.get("applied_conclusions") or []
    ):
        for eid in item.get("evidence_ids") or []:
            if eid.startswith(_FIELD_PREFIX) and eid not in ids:
                ids.append(eid)
    return ids


def field_states(expectation: dict) -> dict[str, str]:
    """Expected ``key`` -> state from ``object_facts.typed_fields``."""
    typed = (expectation.get("object_facts") or {}).get("typed_fields") or {}
    states: dict[str, str] = {}
    for key, value in typed.items():
        if value == UNKNOWN:
            states[key] = UNKNOWN
        elif value == NOT_APPLICABLE:
            states[key] = NOT_APPLICABLE
        else:
            states[key] = "known"
    return states


def _field_ref(eid: str, states: dict[str, str]) -> FieldRef:
    key = eid[len(_FIELD_PREFIX) :]
    label = FIELD_SPECS.get(key, (None, key, None))[1]
    return FieldRef(id=eid, key=key, label=label, state=states.get(key, "unmapped"))


def applied_on_unknown(response: dict, states: dict[str, str]) -> list[Violation]:
    """Unconditional applied conclusions that cite an unknown object field.

    This mirrors ``department_qa.contract.decide``: clarifying questions take
    precedence and make the response ``applicability_unclear``. In that case an
    applied conclusion may state the rule conditionally while naming the facts
    still needed, so citing the unknown field is not itself a violation.
    """
    if response.get("clarifying_questions"):
        return []
    out: list[Violation] = []
    for conclusion in response.get("applied_conclusions") or []:
        unknown = [
            eid
            for eid in conclusion.get("evidence_ids") or []
            if eid.startswith(_FIELD_PREFIX)
            and states.get(eid[len(_FIELD_PREFIX) :]) == UNKNOWN
        ]
        if unknown:
            out.append(
                Violation(
                    code="applied_on_unknown",
                    detail=f"{' '.join(unknown)} :: {conclusion.get('statement', '')}",
                )
            )
    return out


def _response_texts(response: dict) -> list[str]:
    texts = [response.get("answer") or ""]
    for group in (
        "external_basis",
        "internal_basis",
        "object_facts",
        "applied_conclusions",
    ):
        for item in response.get(group) or []:
            texts.append(item.get("statement") or "")
    texts.extend(response.get("clarifying_questions") or [])
    return texts


def forbidden_conclusion_hits(response: dict, forbidden: list[str]) -> list[Violation]:
    """Forbidden conclusions matched verbatim (casefolded, whitespace-normalized)."""
    haystack = " \n ".join(
        _WS.sub(" ", t) for t in _response_texts(response)
    ).casefold()
    out: list[Violation] = []
    for phrase in forbidden:
        needle = _WS.sub(" ", phrase).strip().casefold()
        if needle and needle in haystack:
            out.append(Violation(code="forbidden_conclusion", detail=phrase))
    return out


def _matches_expected(block: dict, status: str, reasons: list[str]) -> bool:
    expected = (status, tuple(sorted(reasons)))
    primary = (block.get("status"), tuple(sorted(block.get("reasons") or [])))
    if expected == primary:
        return True
    for alt in block.get("alternatives") or []:
        if expected == (alt.get("status"), tuple(sorted(alt.get("reasons") or []))):
            return True
    return False


def score_question(expectation: dict, record: dict) -> QuestionReport:
    response = record.get("response") or {}
    mode = record.get("mode") or ""
    status = response.get("status") or ""
    reasons = list(response.get("reason_codes") or [])
    states = field_states(expectation)

    expected_block = (expectation.get("expected") or {}).get(mode)
    contract_ok = (
        None
        if expected_block is None
        else _matches_expected(expected_block, status, reasons)
    )
    violations = tuple(
        applied_on_unknown(response, states)
        + forbidden_conclusion_hits(
            response, expectation.get("forbidden_conclusions") or []
        )
    )
    return QuestionReport(
        n=record["n"],
        mode=mode,
        unit_id=record.get("unit_id"),
        status=status,
        reason_codes=tuple(reasons),
        cited_fields=tuple(
            _field_ref(eid, states) for eid in cited_field_ids(response)
        ),
        violations=violations,
        contract_ok=contract_ok,
    )


def score_run(
    expectations: dict[int, dict],
    run_dir: Path,
    only: set[int] | None = None,
) -> list[QuestionReport]:
    """Score selected expectations against the run, in expectation order."""
    selected = expectations
    if only is not None:
        unknown = only - set(expectations)
        if unknown:
            raise ValueError(f"unknown question numbers: {sorted(unknown)}")
        selected = {n: q for n, q in expectations.items() if n in only}
    records = load_run(run_dir)
    missing = sorted(set(selected) - set(records))
    if missing:
        raise ValueError(f"{run_dir}: no records for questions {missing}")
    return [score_question(selected[n], records[n]) for n in selected]


def summarize(reports: list[QuestionReport]) -> dict:
    return {
        "questions": len(reports),
        "contract_ok": sum(1 for r in reports if r.contract_ok),
        "applied_on_unknown": sum(
            1
            for r in reports
            if any(v.code == "applied_on_unknown" for v in r.violations)
        ),
        "forbidden_conclusion": sum(
            1
            for r in reports
            if any(v.code == "forbidden_conclusion" for v in r.violations)
        ),
        "violations": sum(len(r.violations) for r in reports),
        "cited_fields": sum(len(r.cited_fields) for r in reports),
    }


def render_markdown(reports: list[QuestionReport], summary: dict) -> str:
    lines = [
        "# Офлайн-оценка прогона (детерминированная)",
        "",
        "| q | статус | reasons | поля obj_f_* | нарушения |",
        "|---:|---|---|---|---|",
    ]
    for r in reports:
        fields = ", ".join(f"{f.id}={f.state}" for f in r.cited_fields) or "—"
        violations = ", ".join(v.code for v in r.violations) or "—"
        lines.append(
            f"| {r.n} | {r.status} | {'; '.join(r.reason_codes) or '—'} "
            f"| {fields} | {violations} |"
        )
    lines.append("")
    lines.append(
        "Итог: "
        f"контракт {summary['contract_ok']}/{summary['questions']}; "
        f"applied_on_unknown {summary['applied_on_unknown']}; "
        f"запрещённые строки {summary['forbidden_conclusion']}; "
        f"всего нарушений {summary['violations']}."
    )
    lines.append("")
    for r in reports:
        for v in r.violations:
            lines.append(f"- q{r.n} `{v.code}`: {v.detail}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="mode directory")
    parser.add_argument("--expectations", type=Path, default=DEFAULT_EXPECTATIONS)
    parser.add_argument(
        "--only", help="comma-separated question numbers to score, e.g. 1,7"
    )
    parser.add_argument("--json", action="store_true", help="output JSON, not markdown")
    args = parser.parse_args()

    try:
        expectations = load_expectations(args.expectations)
        only = (
            {int(value) for value in args.only.split(",") if value.strip()}
            if args.only
            else None
        )
        reports = score_run(expectations, args.run, only=only)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    summary = summarize(reports)
    if args.json:
        print(
            json.dumps(
                {
                    "summary": summary,
                    "questions": [r.to_dict() for r in reports],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(render_markdown(reports, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
