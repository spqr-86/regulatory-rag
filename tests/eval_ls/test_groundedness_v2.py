"""Contract regressions using saved baseline replies; no judge-quality claims."""

# ANCHOR: recorded invalid replies stay invalid; explicit per-fragment evidence
# supports combined claims without manufacturing a concatenated verbatim quote.

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from eval.judges.citations import (
    JudgeFormatError,
    is_complete,
    validate_and_score,
)
from eval.ls.groundedness import make_groundedness_evaluator, parse_groundedness

pytestmark = pytest.mark.unit
FIXTURE = json.loads(
    (
        Path(__file__).parents[1]
        / "fixtures/groundedness_baseline_contract_failures.json"
    ).read_text(encoding="utf-8")
)


def case(cid="G027"):
    return next(row for row in FIXTURE["groundedness"] if row["case_id"] == cid)


def payload():
    # A proposed v2 reply assembled from the real G027 evidence, not a captured
    # v2 model reply. Tests establish validation, not model alignment.
    return {
        "contract_version": 2,
        "checks": [
            {
                "answer_quote": case()["invalid_legacy_checks"][0]["answer_quote"],
                "evidence": [
                    {
                        "fragment_id": 1,
                        "quote": "393. В инструкции о мерах пожарной безопасности необходимо отражать следующие вопросы:",
                    },
                    {
                        "fragment_id": 2,
                        "quote": "394. В инструкции о мерах пожарной безопасности указываются лица, ответственные за обеспечение пожарной безопасности, в том числе за:",
                    },
                ],
                "status": "supported",
                "reason": "The two recorded provisions support the combined claim.",
            }
        ],
    }


def parse(data):
    row = case()
    return parse_groundedness(json.dumps(data), row["answer"], row["context"])


def test_separate_evidence_from_two_real_fragments_is_valid():
    result = parse(payload())
    assert result["score"] == 1
    assert result["contract_version"] == 2
    assert len(result["checks"][0]["evidence"]) == 2


@pytest.mark.parametrize("cid", ["G005", "G008", "G027", "G052", "G057", "G058"])
def test_invalid_recorded_legacy_replies_are_not_silently_repaired(cid):
    row = case(cid)
    with pytest.raises(JudgeFormatError):
        parse_groundedness(
            json.dumps({"checks": row["invalid_legacy_checks"]}),
            row["answer"],
            row["context"],
        )


def test_legacy_concatenated_quote_cannot_be_a_v2_evidence_quote():
    data = payload()
    data["checks"][0]["evidence"] = [
        {"fragment_id": 1, "quote": case()["invalid_legacy_checks"][0]["context_quote"]}
    ]
    with pytest.raises(JudgeFormatError, match="evidence"):
        parse(data)


@pytest.mark.parametrize("fragment_id", [2, 999, 0, True, "1"])
def test_evidence_must_belong_to_its_declared_fragment(fragment_id):
    data = payload()
    data["checks"][0]["evidence"][0]["fragment_id"] = fragment_id
    with pytest.raises(JudgeFormatError):
        parse(data)


@pytest.mark.parametrize(
    "quote",
    [
        "394. В инструкции о мерах пожарной безопасности необходимо отражать следующие вопросы:",
        "393. В инструкции о мерах пожарной безопасности не необходимо отражать следующие вопросы:",
        "393. В инструкции ... необходимо отражать следующие вопросы:",
    ],
)
def test_v2_evidence_rejects_changed_number_negation_or_ellipsis(quote):
    data = payload()
    data["checks"][0]["evidence"][0]["quote"] = quote
    with pytest.raises(JudgeFormatError, match="evidence"):
        parse(data)


def test_v2_rejects_quote_spanning_fragment_boundary_even_if_verbatim():
    row = case()
    start = row["context"].index("393. В инструкции")
    end = row["context"].index("394. В инструкции") + len("394. В инструкции")
    data = payload()
    data["checks"][0]["evidence"] = [
        {"fragment_id": 1, "quote": row["context"][start:end]}
    ]
    with pytest.raises(JudgeFormatError, match="evidence"):
        parse(data)


@pytest.mark.parametrize(
    "status, expected", [("supported", None), ("unsupported", 0), ("unverifiable", 0)]
)
def test_missing_evidence_never_earns_a_supported_pass(status, expected):
    data = payload()
    data["checks"][0].update(status=status, evidence=[])
    if status == "supported":
        with pytest.raises(JudgeFormatError, match="supported requires evidence"):
            parse(data)
    else:
        assert parse(data)["score"] == expected


def test_supported_without_evidence_is_not_dropped_from_other_valid_checks():
    data = payload()
    bad = copy.deepcopy(data["checks"][0])
    bad["evidence"] = []
    data["checks"].append(bad)
    with pytest.raises(JudgeFormatError, match="supported requires evidence"):
        parse(data)


def test_v2_no_claims_is_unscored():
    assert parse({"contract_version": 2, "checks": []})["score"] is None


@pytest.mark.parametrize("version", [None, True, 2.0, "2", 3])
def test_v2_contract_version_is_an_explicit_supported_integer(version):
    data = payload()
    data["contract_version"] = version
    with pytest.raises(JudgeFormatError, match="contract_version"):
        parse(data)


def test_duplicate_fragment_headers_are_rejected_instead_of_overwriting_evidence():
    row = case()
    context = row["context"].replace("[2] (HIGH)", "[1] (HIGH)")
    assert context != row["context"]
    with pytest.raises(JudgeFormatError, match="Duplicate"):
        parse_groundedness(json.dumps(payload()), row["answer"], context)


def test_crlf_context_preserves_fragment_boundaries():
    row = case()
    context = row["context"].replace("\n", "\r\n")
    assert (
        parse_groundedness(json.dumps(payload()), row["answer"], context)["score"] == 1
    )
    data = payload()
    data["checks"][0]["evidence"][0]["fragment_id"] = 2
    with pytest.raises(JudgeFormatError, match="evidence"):
        parse_groundedness(json.dumps(data), row["answer"], context)


def test_malformed_numbered_headers_do_not_become_one_implicit_fragment():
    row = case()
    context = (
        row["context"]
        .replace("(HIGH)", "(UNKNOWN)")
        .replace("(MED)", "(UNKNOWN)")
        .replace("(LOW)", "(UNKNOWN)")
    )
    with pytest.raises(JudgeFormatError, match="headers"):
        parse_groundedness(json.dumps(payload()), row["answer"], context)


def test_one_malformed_header_cannot_merge_foreign_evidence_into_previous_fragment():
    row = case()
    context = row["context"].replace("[2] (HIGH)", "[2] (UNKNOWN)")
    data = payload()
    # If the malformed second header is ignored, its text incorrectly belongs to 1.
    data["checks"][0]["evidence"][1]["fragment_id"] = 1
    with pytest.raises(JudgeFormatError, match="headers"):
        parse_groundedness(json.dumps(data), row["answer"], context)


def test_live_evaluator_requires_the_new_contract_without_rewriting_legacy():
    row = case()
    legacy = {"checks": row["invalid_legacy_checks"]}
    model = NS(invoke=lambda _: NS(content=json.dumps(legacy), response_metadata={}))
    result = make_groundedness_evaluator(model, keep_raw=True)(
        {"question": "Что содержит инструкция?"}, row, {}
    )
    assert result["score"] is None
    assert "contract_version" in result["comment"]
    assert json.loads(result["raw"]) == legacy


def test_live_v2_result_preserves_version_in_audit_and_alignment(tmp_path):
    from eval.ls.align_groundedness import judge_rows
    from eval.ls.artifacts import Artifacts
    from eval.ls.audit import audited

    row = case()
    model = NS(invoke=lambda _: NS(content=json.dumps(payload()), response_metadata={}))
    evaluator = make_groundedness_evaluator(model)
    result = evaluator({"question": "q"}, row, {})
    assert result["score"] == 1
    assert result["evaluator_info"]["groundedness_contract_version"] == 2
    aligned = judge_rows(
        [
            {
                "review_id": "fixture",
                "case_id": "G027",
                "inputs": {"question": "q"},
                "outputs": row,
                "reference": {},
                "human": 1,
            }
        ],
        evaluator,
    )
    assert aligned[0]["groundedness_contract_version"] == 2
    audited_result = audited(
        evaluator, ("groundedness",), Artifacts(tmp_path, {"run_id": "x"}), {}
    )(
        NS(id="r", inputs={"question": "q"}, outputs=row),
        NS(id="e", metadata={}, outputs={}),
    )
    assert audited_result["results"][0]["evaluator_info"] == {
        "groundedness_contract_version": 2,
        "assessment_id": "x",
    }


def test_saved_citation_reply_accepts_yo_typography_but_not_new_numbers():
    row = FIXTURE["citations_G052"]
    assert validate_and_score(row["verdict"], row["answer"], row["context"]) == "pass"
    assert is_complete(row["verdict"])
    bad = copy.deepcopy(row["verdict"])
    bad["checks"][-1]["answer_quote"] = "статьей 8"
    with pytest.raises(JudgeFormatError):
        validate_and_score(bad, row["answer"], row["context"])


def test_manifest_versions_effective_groundedness_policy(tmp_path):
    from eval.ls.run import run_experiment

    run_experiment(dry_run=True, limit=1, output_dir=tmp_path / "dry")
    policy = json.loads((tmp_path / "dry/manifest.json").read_text())["judge_policy"]
    assert policy["groundedness_contract_version"] == 2
    assert policy["groundedness_max_tokens"] == 32000
    assert policy["quote_normalization_version"] == 2
