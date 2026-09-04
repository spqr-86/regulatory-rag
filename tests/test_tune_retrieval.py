"""Unit tests for the retrieval parameter tuner (eval/tune_retrieval.py).

Covers the engine-free core: the config override contract (restore on the way
out, including after an exception), the scan loop over candidate values, the
Hit Rate@k curve, the recommendation rule and report formatting.

The tuner must never leave a mutated ``v7_config`` behind and must never write
the winning value into the config — the roadmap requires a manual PR after
review, not an automatic apply.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from eval.tune_retrieval import (
    best_value,
    format_scan_report,
    hit_rate_curve,
    override_config,
    scan_param,
)
from src.v7.config import v7_config

pytestmark = pytest.mark.unit


def _gt(n: int = 3) -> list[dict]:
    return [
        {
            "question": f"Вопрос {i}?",
            "chunk_id": f"a.pdf#{i}",
            "relevant_chunk_ids": [f"a.pdf#{i}"],
            "source": "a.pdf",
        }
        for i in range(1, n + 1)
    ]


class TestOverrideConfig:
    def test_sets_value_inside_and_restores_after(self):
        original = v7_config.RRF_K
        with override_config(RRF_K=7):
            assert v7_config.RRF_K == 7
        assert v7_config.RRF_K == original

    def test_restores_after_exception(self):
        original = v7_config.RRF_K
        with pytest.raises(RuntimeError):
            with override_config(RRF_K=7):
                raise RuntimeError("boom")
        assert v7_config.RRF_K == original

    def test_rejects_unknown_parameter(self):
        with pytest.raises(ValueError, match="NO_SUCH_PARAM"):
            with override_config(NO_SUCH_PARAM=1):
                pass
        assert not hasattr(v7_config, "NO_SUCH_PARAM")


class TestScanParam:
    def test_runs_once_per_value_and_keeps_order(self):
        seen: list[int] = []

        def factory():
            seen.append(v7_config.RRF_K)
            return lambda q: ["a.pdf#1"]

        scan = scan_param("RRF_K", [10, 20, 30], _gt(), factory, ks=(5,))
        assert seen == [10, 20, 30]
        assert [r["value"] for r in scan["results"]] == [10, 20, 30]

    def test_value_is_visible_to_the_retrieval_call(self):
        """The scanned value must reach retrieval, not just the factory."""

        def factory():
            return lambda q: [f"a.pdf#{v7_config.RRF_K}"]

        scan = scan_param("RRF_K", [1, 2], _gt(2), factory, ks=(5,))
        first, second = scan["results"]
        # gt has a.pdf#1 and a.pdf#2 → RRF_K=1 hits one of two, RRF_K=2 the other
        assert first["metrics"]["hit_rate@5"] == pytest.approx(0.5)
        assert second["metrics"]["hit_rate@5"] == pytest.approx(0.5)

    def test_metrics_differ_when_retrieval_differs(self):
        def factory():
            k = v7_config.RRF_K
            return lambda q: [] if k == 99 else ["a.pdf#1", "a.pdf#2", "a.pdf#3"]

        scan = scan_param("RRF_K", [60, 99], _gt(3), factory, ks=(5,))
        assert scan["results"][0]["metrics"]["hit_rate@5"] == pytest.approx(1.0)
        assert scan["results"][1]["metrics"]["hit_rate@5"] == pytest.approx(0.0)

    def test_restores_config_after_the_scan(self):
        original = v7_config.RRF_K
        scan_param("RRF_K", [10, 20], _gt(), lambda: (lambda q: []), ks=(5,))
        assert v7_config.RRF_K == original

    def test_records_the_parameter_name_and_question_count(self):
        scan = scan_param("RRF_K", [10], _gt(3), lambda: (lambda q: []), ks=(5,))
        assert scan["param"] == "RRF_K"
        assert scan["n"] == 3

    def test_rejects_an_empty_value_list(self):
        with pytest.raises(ValueError, match="values"):
            scan_param("RRF_K", [], _gt(), lambda: (lambda q: []), ks=(5,))


class TestBestValue:
    def _scan(self, pairs, metric="mrr"):
        return {
            "param": "RRF_K",
            "n": 3,
            "ks": [5],
            "results": [
                {"value": v, "metrics": {metric: m, "hit_rate@5": m}}
                for v, m in pairs
            ],
        }

    def test_picks_the_highest_metric(self):
        scan = self._scan([(10, 0.4), (60, 0.7), (120, 0.5)])
        assert best_value(scan, metric="mrr") == 60

    def test_ties_go_to_the_first_value_scanned(self):
        scan = self._scan([(10, 0.7), (60, 0.7)])
        assert best_value(scan, metric="mrr") == 10

    def test_unknown_metric_is_an_error(self):
        scan = self._scan([(10, 0.4)])
        with pytest.raises(KeyError):
            best_value(scan, metric="ndcg")


class TestHitRateCurve:
    def test_counts_ranks_at_each_cutoff(self):
        records = [{"rank": 1}, {"rank": 3}, {"rank": None}, {"rank": 12}]
        curve = hit_rate_curve(records, max_k=12)
        assert curve[1] == pytest.approx(0.25)
        assert curve[3] == pytest.approx(0.5)
        assert curve[11] == pytest.approx(0.5)
        assert curve[12] == pytest.approx(0.75)

    def test_is_monotonic_non_decreasing(self):
        records = [{"rank": r} for r in (1, 2, 5, None, 9)]
        curve = hit_rate_curve(records, max_k=10)
        values = [curve[k] for k in sorted(curve)]
        assert values == sorted(values)

    def test_empty_batch_is_all_zeroes(self):
        curve = hit_rate_curve([], max_k=3)
        assert set(curve.values()) == {0.0}


class TestFormatScanReport:
    def test_lists_every_scanned_value_and_marks_the_winner(self):
        scan = {
            "param": "RRF_K",
            "n": 43,
            "ks": [5, 12],
            "metric": "mrr",
            "results": [
                {
                    "value": 10,
                    "metrics": {"hit_rate@5": 0.80, "hit_rate@12": 0.88, "mrr": 0.60},
                },
                {
                    "value": 60,
                    "metrics": {"hit_rate@5": 0.81, "hit_rate@12": 0.88, "mrr": 0.64},
                },
            ],
            "best": 60,
            "baseline": 60,
        }
        report = format_scan_report(scan)
        assert "RRF_K" in report
        assert "10" in report and "60" in report
        assert "0.640" in report
        # the winner is marked, and the report says nothing was applied
        assert "←" in report or "*" in report
        assert "не применено" in report.lower() or "не приме" in report.lower()

    def test_warns_when_the_spread_is_within_noise(self):
        """A flat scan must say so — otherwise the 4th decimal looks like a win."""
        scan = {
            "param": "RRF_K",
            "n": 43,
            "ks": [5],
            "metric": "mrr",
            "results": [
                {"value": 10, "metrics": {"hit_rate@5": 0.81, "mrr": 0.6371}},
                {"value": 60, "metrics": {"hit_rate@5": 0.81, "mrr": 0.6370}},
            ],
            "best": 10,
            "baseline": 60,
        }
        assert "шум" in format_scan_report(scan).lower()

    def test_no_noise_warning_when_the_spread_is_real(self):
        scan = {
            "param": "RRF_K",
            "n": 43,
            "ks": [5],
            "metric": "mrr",
            "results": [
                {"value": 10, "metrics": {"hit_rate@5": 0.81, "mrr": 0.70}},
                {"value": 60, "metrics": {"hit_rate@5": 0.75, "mrr": 0.60}},
            ],
            "best": 10,
            "baseline": 60,
        }
        assert "шум" not in format_scan_report(scan).lower()
