"""Tests for eval/pricing.py — token counts → dollars, and latency percentiles."""

import pytest

from eval.pricing import (
    PRICE_PER_1M,
    calc_total_price,
    cost_for_usages,
    percentile,
    price_for,
)


class TestPriceFor:
    def test_known_model(self):
        assert price_for("gpt-4o-mini") == PRICE_PER_1M["gpt-4o-mini"]

    def test_unknown_model_raises_with_hint(self):
        with pytest.raises(ValueError, match="No price for model"):
            price_for("gpt-42")


class TestCalcTotalPrice:
    def test_sums_input_and_output(self):
        cost = calc_total_price(
            [{"input": 1_000_000, "output": 1_000_000}], model="gpt-4o-mini"
        )
        assert cost == pytest.approx(0.15 + 0.60)


class TestCostForUsages:
    def test_prices_each_record_by_its_own_model(self):
        usages = [
            {
                "model": "gpt-4o-mini",
                "prompt_tokens": 1_000_000,
                "completion_tokens": 0,
            },
            {"model": "gpt-4o", "prompt_tokens": 0, "completion_tokens": 1_000_000},
        ]
        result = cost_for_usages(usages)
        assert result["cost_usd"] == pytest.approx(0.15 + 10.00)
        assert result["unpriced_models"] == []

    def test_unknown_model_is_reported_not_silently_zero(self):
        """A run must not report a cheerful low price because a model has no rate."""
        result = cost_for_usages(
            [{"model": "gemini-9-ultra", "prompt_tokens": 100, "completion_tokens": 10}]
        )
        assert result["cost_usd"] == 0.0
        assert result["unpriced_models"] == ["gemini-9-ultra"]

    def test_empty(self):
        assert cost_for_usages([]) == {"cost_usd": 0.0, "unpriced_models": []}


class TestPercentile:
    def test_p50_of_odd_sample(self):
        assert percentile([1, 2, 3], 50) == 2

    def test_p50_interpolates_even_sample(self):
        assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_p95_close_to_max(self):
        values = list(range(1, 101))
        assert percentile(values, 95) == pytest.approx(95.05, abs=0.1)

    def test_unsorted_input(self):
        assert percentile([5, 1, 3], 50) == 3

    def test_empty_is_zero(self):
        assert percentile([], 95) == 0.0
