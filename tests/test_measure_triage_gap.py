"""Tests for the triage-gap measurement script (issue #13, stage B2).

The script runs router → rag_simple → evaluate_triage without generation;
these tests pin its pure parts: which passage list the triage hands on, the
aggregation, and the per-question regression check.
"""

from __future__ import annotations

import pytest

from eval.measure_triage_gap import (
    compare_runs,
    format_report,
    passages_after_triage,
    summarize,
)


class TestPassagesAfterTriage:
    @pytest.mark.unit
    def test_sufficient_hands_on_final_passages(self):
        retrieved = [{"text": "a"}]
        update = {"sufficient": True, "final_passages": [{"text": "a"}, {"text": "b"}]}
        assert passages_after_triage(update, retrieved) == update["final_passages"]

    @pytest.mark.unit
    def test_escalation_hands_on_fallback_passages(self):
        """On escalation rag_complex starts from the fallback, so that is the output."""
        retrieved = [{"text": "a"}]
        update = {"sufficient": False, "fallback_passages": [{"text": "a"}, {"text": "b"}]}
        assert passages_after_triage(update, retrieved) == update["fallback_passages"]

    @pytest.mark.unit
    def test_no_fallback_falls_back_to_retrieved(self):
        retrieved = [{"text": "a"}]
        assert passages_after_triage({"sufficient": False}, retrieved) == retrieved


class TestSummarize:
    @pytest.mark.unit
    def test_counts_escalations_gaps_and_hit_rate(self):
        records = [
            {"question": "q1", "escalated": True, "hit": False, "gap_open": True, "gap_seen": True},
            {"question": "q2", "escalated": False, "hit": True, "gap_open": False, "gap_seen": True},
            {"question": "q3", "escalated": False, "hit": True, "gap_open": False, "gap_seen": False},
            {"question": "q4", "escalated": False, "hit": False, "gap_open": False, "gap_seen": False},
        ]
        s = summarize(records, k=12)
        assert s["n"] == 4
        assert s["escalation_rate"] == 0.25
        assert s["hit_rate@12"] == 0.5
        assert s["gaps_seen"] == 2
        assert s["gaps_closed"] == 1

    @pytest.mark.unit
    def test_empty_batch_is_zeroed_not_a_crash(self):
        s = summarize([], k=12)
        assert s == {
            "n": 0,
            "escalation_rate": 0.0,
            "hit_rate@12": 0.0,
            "gaps_seen": 0,
            "gaps_closed": 0,
        }


class TestCompareRuns:
    @pytest.mark.unit
    def test_passes_when_escalations_drop_and_no_question_regresses(self):
        base = {"records": [
            {"question": "q1", "escalated": True, "hit": True},
            {"question": "q2", "escalated": True, "hit": False},
        ]}
        new = {"records": [
            {"question": "q1", "escalated": False, "hit": True},
            {"question": "q2", "escalated": True, "hit": True},
        ]}
        verdict = compare_runs(base, new)
        assert verdict["escalation_delta"] == -0.5
        assert verdict["regressed"] == []
        assert verdict["passed"] is True

    @pytest.mark.unit
    def test_fails_when_a_single_question_loses_its_hit(self):
        base = {"records": [
            {"question": "q1", "escalated": True, "hit": True},
            {"question": "q2", "escalated": True, "hit": True},
        ]}
        new = {"records": [
            {"question": "q1", "escalated": False, "hit": False},
            {"question": "q2", "escalated": False, "hit": True},
        ]}
        verdict = compare_runs(base, new)
        assert verdict["regressed"] == ["q1"]
        assert verdict["passed"] is False

    @pytest.mark.unit
    def test_fails_when_escalation_rate_does_not_drop(self):
        base = {"records": [{"question": "q1", "escalated": True, "hit": True}]}
        new = {"records": [{"question": "q1", "escalated": True, "hit": True}]}
        verdict = compare_runs(base, new)
        assert verdict["escalation_delta"] == 0.0
        assert verdict["passed"] is False

    @pytest.mark.unit
    def test_question_sets_that_differ_fail_the_comparison(self):
        base = {"records": [{"question": "q1", "escalated": True, "hit": True}]}
        new = {"records": [{"question": "q2", "escalated": False, "hit": True}]}
        verdict = compare_runs(base, new)
        assert verdict["missing"] == ["q1", "q2"]
        assert verdict["passed"] is False


class TestReport:
    @pytest.mark.unit
    def test_warns_when_v8_flag_makes_the_measurement_meaningless(self):
        """With V8 on, _legacy_triage never runs and stage B2 cannot show up."""
        result = {
            "n": 1, "errors": 0, "elapsed_s": 1.0, "k": 12,
            "escalation_rate": 0.0, "hit_rate@12": 1.0,
            "gaps_seen": 0, "gaps_closed": 0,
            "v8_evidence_assess": True,
        }
        assert "V8" in format_report(result)

    @pytest.mark.unit
    def test_no_warning_on_the_legacy_path(self):
        result = {
            "n": 1, "errors": 0, "elapsed_s": 1.0, "k": 12,
            "escalation_rate": 0.0, "hit_rate@12": 1.0,
            "gaps_seen": 0, "gaps_closed": 0,
            "v8_evidence_assess": False,
        }
        assert "V8" not in format_report(result)
