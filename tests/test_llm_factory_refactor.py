"""Tests for the LLM factory registry refactor (CARD-1.1).

Verifies that get_llm() routes to the right provider, forwards kwargs to
Gemini, and silently drops Gemini-specific kwargs for OpenAI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_registry_contains_supported_chat_providers():
    """Registry must expose both providers after refactor."""
    from src.infra.llm_factory import _LLM_PROVIDERS

    assert "gemini" in _LLM_PROVIDERS
    assert "openai" in _LLM_PROVIDERS
    assert "openrouter" in _LLM_PROVIDERS


@pytest.mark.unit
def test_openrouter_uses_its_key_and_endpoint(monkeypatch):
    import src.infra.llm_factory as lf

    spy = MagicMock(name="ChatOpenAI")
    monkeypatch.setattr(lf, "_OpenRouterChat", spy)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    lf._create_openrouter_llm(model_name="openai/gpt-4o-mini")

    kwargs = spy.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["api_key"] == "router-key"
    assert kwargs["base_url"] == "https://openrouter.ai/api/v1"


# --- CARD-5.1: AFC disabled via public bind(), no monkey-patch ----------------


@pytest.mark.unit
def test_get_gemini_llm_does_not_monkeypatch_build_request_config(monkeypatch):
    """get_gemini_llm must NOT replace the private _build_request_config method.

    The previous implementation overrode an internal bound method of
    ChatGoogleGenerativeAI to inject AutomaticFunctionCallingConfig(disable=True).
    After CARD-5.1, AFC is disabled via the public Runnable.bind() API, so
    the returned LLM's _build_request_config must remain the original method.
    """

    import src.infra.llm_factory as lf

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(lf.settings, "REQUEST_TIMEOUT", 60.0, raising=False)

    raw_instance = MagicMock(name="ChatGoogleGenerativeAI_instance")
    original_build = raw_instance._build_request_config
    bound = MagicMock(name="bound_runnable")
    raw_instance.bind.return_value = bound

    monkeypatch.setattr(
        lf, "ChatGoogleGenerativeAI", MagicMock(return_value=raw_instance)
    )

    llm = lf.get_gemini_llm()

    # No monkey-patch: original bound method is intact.
    assert raw_instance._build_request_config is original_build, (
        "get_gemini_llm must not replace _build_request_config; "
        "use Runnable.bind() instead."
    )
    # bind() was called with AFC disabled.
    raw_instance.bind.assert_called_once()
    bind_kwargs = raw_instance.bind.call_args.kwargs
    assert "automatic_function_calling" in bind_kwargs
    afc = bind_kwargs["automatic_function_calling"]
    # disable=True on AutomaticFunctionCallingConfig
    assert getattr(afc, "disable", None) is True
    # Returned object is the bound runnable (AFC injected on every invoke).
    assert llm is bound


# --- Model name from settings must reach the OpenAI constructor --------------
#
# Regression: `kwargs.setdefault("model_name", model)` was present only in the
# gemini branch of get_simple/complex/judge_llm, so for provider="openai" the
# resolved settings model never reached _create_openai_llm and ChatOpenAI fell
# back to its own default ("gpt-4o-mini"). JUDGE_MODEL_NAME / COMPLEX_MODEL_NAME
# were silently ignored.


@pytest.fixture
def openai_spy(monkeypatch):
    """Patch ChatOpenAI and return the mock so tests can read the model kwarg."""
    import src.infra.llm_factory as lf

    spy = MagicMock(name="ChatOpenAI")
    monkeypatch.setattr(lf, "ChatOpenAI", spy)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return spy


@pytest.mark.unit
@pytest.mark.parametrize(
    ("getter", "provider_attr", "model_attr", "model_name"),
    [
        ("get_judge_llm", "JUDGE_LLM_PROVIDER", "JUDGE_MODEL_NAME", "gpt-4o"),
        ("get_complex_llm", "COMPLEX_LLM_PROVIDER", "COMPLEX_MODEL_NAME", "gpt-4o"),
        ("get_simple_llm", "SIMPLE_LLM_PROVIDER", "SIMPLE_MODEL_NAME", "gpt-4o-mini"),
    ],
)
def test_openai_getters_pass_settings_model(
    monkeypatch, openai_spy, getter, provider_attr, model_attr, model_name
):
    """The model named in settings must be what ChatOpenAI is constructed with."""
    import src.infra.llm_factory as lf

    monkeypatch.setattr(lf.settings, provider_attr, "openai", raising=False)
    monkeypatch.setattr(lf.settings, model_attr, model_name, raising=False)

    getattr(lf, getter)()

    openai_spy.assert_called_once()
    assert openai_spy.call_args.kwargs["model"] == model_name


@pytest.mark.unit
def test_explicit_model_name_overrides_settings(monkeypatch, openai_spy):
    """An explicit model_name kwarg still wins over the settings default."""
    import src.infra.llm_factory as lf

    monkeypatch.setattr(lf.settings, "JUDGE_LLM_PROVIDER", "openai", raising=False)
    monkeypatch.setattr(lf.settings, "JUDGE_MODEL_NAME", "gpt-4o", raising=False)

    lf.get_judge_llm(model_name="gpt-4o-mini")

    assert openai_spy.call_args.kwargs["model"] == "gpt-4o-mini"


# --- OpenRouter reasoning / output caps / provider routing --------------------


def _openrouter_kwargs(monkeypatch, **factory_kwargs):
    import src.infra.llm_factory as lf

    spy = MagicMock(name="ChatOpenAI")
    monkeypatch.setattr(lf, "_OpenRouterChat", spy)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")
    lf._create_openrouter_llm(
        model_name="deepseek/deepseek-v4.1-flash", **factory_kwargs
    )
    return spy.call_args.kwargs


@pytest.mark.unit
def test_openrouter_defaults_leave_request_unchanged(monkeypatch):
    """No settings → no reasoning/max_tokens/provider fields: current behaviour."""
    import src.infra.llm_factory as lf

    monkeypatch.setattr(lf.settings, "OPENROUTER_REASONING_EFFORT", None)
    monkeypatch.setattr(lf.settings, "OPENROUTER_MAX_TOKENS", None)
    monkeypatch.setattr(lf.settings, "OPENROUTER_PROVIDER_SORT", None)

    kwargs = _openrouter_kwargs(monkeypatch, thinking_budget=4096)

    assert "max_tokens" not in kwargs
    assert not kwargs.get("extra_body")


@pytest.mark.unit
def test_openrouter_sends_native_reasoning_effort_max_tokens_and_sort(monkeypatch):
    import src.infra.llm_factory as lf

    monkeypatch.setattr(lf.settings, "OPENROUTER_REASONING_EFFORT", "low")
    monkeypatch.setattr(lf.settings, "OPENROUTER_MAX_TOKENS", 2000)
    monkeypatch.setattr(lf.settings, "OPENROUTER_PROVIDER_SORT", "latency")

    kwargs = _openrouter_kwargs(monkeypatch, thinking_budget=4096)

    assert kwargs["max_tokens"] == 2000
    assert kwargs["extra_body"] == {
        "reasoning": {"effort": "low"},
        "provider": {"sort": "latency"},
    }


@pytest.mark.unit
def test_openrouter_thinking_budget_zero_disables_reasoning(monkeypatch):
    """thinking_budget=0 means 'no thinking' — honoured, not silently dropped."""
    import src.infra.llm_factory as lf

    monkeypatch.setattr(lf.settings, "OPENROUTER_REASONING_EFFORT", "low")
    monkeypatch.setattr(lf.settings, "OPENROUTER_MAX_TOKENS", None)
    monkeypatch.setattr(lf.settings, "OPENROUTER_PROVIDER_SORT", None)

    kwargs = _openrouter_kwargs(monkeypatch, thinking_budget=0)

    assert kwargs["extra_body"] == {"reasoning": {"enabled": False}}


@pytest.mark.unit
def test_openrouter_chat_exposes_serving_provider(monkeypatch):
    """OpenRouter's top-level `provider` must reach response_metadata."""
    import src.infra.llm_factory as lf

    llm = lf._OpenRouterChat(model="deepseek/deepseek-v4.1-flash", api_key="k")
    result = llm._create_chat_result(
        {
            "id": "gen-1",
            "model": "deepseek/deepseek-v4.1-flash",
            "provider": "DeepInfra",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 50,
                "total_tokens": 60,
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
        }
    )
    message = result.generations[0].message
    assert message.response_metadata["openrouter_provider"] == "DeepInfra"
