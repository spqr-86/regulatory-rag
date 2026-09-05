"""Cost and latency reporting in the v7 eval runner (roadmap step 4a).

Before: a run reported quality and a mean latency; the price of a query was a
figure from 22.05.2026 taken on N=10 against a different pipeline. Now every
run prices itself from the tokens the pipeline actually spent, split by path.
"""

import pytest

from eval.run_v7_eval import run_query, summarize_cost


class _FakeGraph:
    def __init__(self, state):
        self._state = state

    def invoke(self, _inputs):
        return self._state


class TestRunQueryUsage:
    def test_returns_tokens_and_cost_from_state(self):
        graph = _FakeGraph(
            {
                "answer": "ответ",
                "final_passages": [{"text": "t"}],
                "retrieval_attempts": [{"stage": "simple"}],
                "llm_usage": [
                    {
                        "model": "gpt-4o-mini",
                        "node": "generate",
                        "stage": "simple",
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 0,
                    }
                ],
            }
        )
        result = run_query(graph, "вопрос")
        assert result["prompt_tokens"] == 1_000_000
        assert result["completion_tokens"] == 0
        assert result["cost_usd"] == pytest.approx(0.15)
        assert result["llm_calls"] == 1

    def test_per_call_breakdown_is_kept_for_audit(self):
        """The cost figure must be checkable: which model, which node, how many tokens."""
        calls = [
            {
                "model": "gpt-4o-mini",
                "node": "expand",
                "stage": "simple",
                "prompt_tokens": 300,
                "completion_tokens": 40,
            },
            {
                "model": "gpt-4o",
                "node": "generate",
                "stage": "complex",
                "prompt_tokens": 7000,
                "completion_tokens": 200,
            },
        ]
        graph = _FakeGraph(
            {
                "answer": "a",
                "retrieval_attempts": [{"stage": "simple"}, {"stage": "complex"}],
                "llm_usage": calls,
            }
        )
        result = run_query(graph, "вопрос")
        assert result["usage"] == calls
        assert result["cost_usd"] == pytest.approx(
            300 / 1e6 * 0.15 + 40 / 1e6 * 0.60 + 7000 / 1e6 * 2.5 + 200 / 1e6 * 10
        )

    def test_no_usage_means_zero_not_missing(self):
        graph = _FakeGraph({"answer": "a", "retrieval_attempts": []})
        result = run_query(graph, "вопрос")
        assert result["cost_usd"] == 0.0
        assert result["prompt_tokens"] == 0
        assert result["llm_calls"] == 0
        assert result["usage"] == []


class TestSummarizeCost:
    def _rows(self):
        return [
            {
                "path": "simple",
                "elapsed_sec": 2.0,
                "cost_usd": 0.001,
                "prompt_tokens": 1000,
                "completion_tokens": 100,
            },
            {
                "path": "simple",
                "elapsed_sec": 4.0,
                "cost_usd": 0.002,
                "prompt_tokens": 2000,
                "completion_tokens": 200,
            },
            {
                "path": "complex",
                "elapsed_sec": 30.0,
                "cost_usd": 0.02,
                "prompt_tokens": 8000,
                "completion_tokens": 400,
            },
        ]

    def test_totals_and_mean_cost(self):
        s = summarize_cost(self._rows())
        assert s["total_cost_usd"] == pytest.approx(0.023)
        assert s["mean_cost_usd"] == pytest.approx(0.023 / 3)

    def test_split_by_path_is_mandatory(self):
        """Complex costs an order of magnitude more; a single mean hides what we pay for."""
        s = summarize_cost(self._rows())
        assert s["by_path"]["simple"]["mean_cost_usd"] == pytest.approx(0.0015)
        assert s["by_path"]["complex"]["mean_cost_usd"] == pytest.approx(0.02)
        assert s["by_path"]["simple"]["queries"] == 2

    def test_latency_percentiles_not_only_mean(self):
        s = summarize_cost(self._rows())
        assert s["latency_p50_sec"] == pytest.approx(4.0)
        assert s["latency_p95_sec"] == pytest.approx(27.4, abs=0.5)
        assert s["by_path"]["complex"]["latency_p50_sec"] == pytest.approx(30.0)

    def test_token_totals(self):
        s = summarize_cost(self._rows())
        assert s["prompt_tokens"] == 11000
        assert s["completion_tokens"] == 700

    def test_unpriced_models_are_propagated(self):
        rows = [
            {
                "path": "simple",
                "elapsed_sec": 1.0,
                "cost_usd": 0.0,
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "unpriced_models": ["gemini-2.5-flash"],
            }
        ]
        s = summarize_cost(rows)
        assert s["unpriced_models"] == ["gemini-2.5-flash"]

    def test_empty_results(self):
        s = summarize_cost([])
        assert s["total_cost_usd"] == 0.0
        assert s["by_path"] == {}
