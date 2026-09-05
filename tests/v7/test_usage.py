"""Tests for src/v7/usage.py — per-call LLM token accounting.

Roadmap step 4a: usage must reach graph state, not only the log.
"""

from src.v7.usage import (
    LLM_USAGE_KEY,
    stamp_stage,
    sum_usage,
    unpack,
    usage_from_response,
)


class _Resp:
    def __init__(self, usage_metadata=None, response_metadata=None):
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata
        if response_metadata is not None:
            self.response_metadata = response_metadata


class TestUsageFromResponse:
    def test_langchain_standard_keys(self):
        r = _Resp(usage_metadata={"input_tokens": 120, "output_tokens": 30})
        u = usage_from_response(r, model="gpt-4o-mini", node="generate")
        assert u == {
            "model": "gpt-4o-mini",
            "node": "generate",
            "prompt_tokens": 120,
            "completion_tokens": 30,
        }

    def test_openai_legacy_keys(self):
        r = _Resp(usage_metadata={"prompt_tokens": 7, "completion_tokens": 5})
        u = usage_from_response(r, model="gpt-4o", node="expand")
        assert u["prompt_tokens"] == 7
        assert u["completion_tokens"] == 5

    def test_falls_back_to_response_metadata(self):
        r = _Resp(
            response_metadata={
                "token_usage": {"prompt_tokens": 9, "completion_tokens": 2}
            }
        )
        u = usage_from_response(r, model="gpt-4o", node="generate")
        assert u["prompt_tokens"] == 9
        assert u["completion_tokens"] == 2

    def test_missing_usage_is_zero_not_none(self):
        """A provider that reports nothing must not poison arithmetic downstream."""
        u = usage_from_response(_Resp(), model="gpt-4o", node="generate")
        assert u["prompt_tokens"] == 0
        assert u["completion_tokens"] == 0

    def test_none_values_coerced_to_zero(self):
        r = _Resp(usage_metadata={"input_tokens": None, "output_tokens": 4})
        u = usage_from_response(r, model="m", node="generate")
        assert u["prompt_tokens"] == 0
        assert u["completion_tokens"] == 4


class TestUnpack:
    def test_tuple_result_splits_into_value_and_usage(self):
        value, usages = unpack(
            ("answer", [{"model": "m", "prompt_tokens": 1, "completion_tokens": 2}])
        )
        assert value == "answer"
        assert len(usages) == 1

    def test_bare_value_keeps_backward_compatibility(self):
        """Old injected fns (and stubs) return only the value — no usage, no crash."""
        value, usages = unpack("answer")
        assert value == "answer"
        assert usages == []

    def test_single_usage_dict_wrapped_into_list(self):
        value, usages = unpack(
            ("answer", {"model": "m", "prompt_tokens": 1, "completion_tokens": 0})
        )
        assert usages == [{"model": "m", "prompt_tokens": 1, "completion_tokens": 0}]

    def test_tuple_value_is_not_mistaken_for_usage(self):
        """A list result (expand returns list[str]) stays the value."""
        value, usages = unpack(["a", "b"])
        assert value == ["a", "b"]
        assert usages == []


class TestStampStage:
    def test_adds_stage_to_each_usage(self):
        usages = [{"model": "m", "prompt_tokens": 1, "completion_tokens": 1}]
        out = stamp_stage(usages, "complex")
        assert out[0]["stage"] == "complex"

    def test_does_not_mutate_input(self):
        usages = [{"model": "m", "prompt_tokens": 1, "completion_tokens": 1}]
        stamp_stage(usages, "simple")
        assert "stage" not in usages[0]


class TestSumUsage:
    def test_sums_by_model(self):
        usages = [
            {"model": "a", "prompt_tokens": 10, "completion_tokens": 1},
            {"model": "a", "prompt_tokens": 5, "completion_tokens": 2},
            {"model": "b", "prompt_tokens": 1, "completion_tokens": 1},
        ]
        assert sum_usage(usages) == {
            "a": {"prompt_tokens": 15, "completion_tokens": 3},
            "b": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    def test_empty(self):
        assert sum_usage([]) == {}


def test_state_key_name_is_stable():
    assert LLM_USAGE_KEY == "llm_usage"
