"""usage_from_response carries reasoning tokens and the serving provider."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from src.v7.usage import usage_from_response


@pytest.mark.unit
def test_reasoning_tokens_and_provider_read_from_langchain_message():
    msg = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 5000,
            "output_tokens": 6293,
            "total_tokens": 11293,
            "output_token_details": {"reasoning": 5030},
        },
        response_metadata={"openrouter_provider": "SiliconFlow"},
    )

    usage = usage_from_response(msg, "deepseek/deepseek-v4.1-flash", "generate")

    assert usage["completion_tokens"] == 6293
    assert usage["reasoning_tokens"] == 5030
    assert usage["provider"] == "SiliconFlow"


@pytest.mark.unit
def test_reasoning_tokens_read_from_raw_token_usage():
    class _Resp:
        usage_metadata = None
        response_metadata = {
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 40},
            }
        }

    usage = usage_from_response(_Resp(), "m", "generate")

    assert usage["reasoning_tokens"] == 40
    assert usage["provider"] == ""


@pytest.mark.unit
def test_missing_reasoning_is_zero():
    msg = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
    )
    usage = usage_from_response(msg, "m", "generate")
    assert usage["reasoning_tokens"] == 0
