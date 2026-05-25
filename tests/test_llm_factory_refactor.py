"""Tests for the LLM factory registry refactor (CARD-1.1).

Verifies that get_llm() routes to the right provider, forwards kwargs to
Gemini, and silently drops Gemini-specific kwargs for OpenAI.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.mark.unit
def test_registry_contains_gemini_and_openai():
    """Registry must expose both providers after refactor."""
    from src.infra.llm_factory import _LLM_PROVIDERS

    assert "gemini" in _LLM_PROVIDERS
    assert "openai" in _LLM_PROVIDERS


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

    monkeypatch.setattr(lf.settings, "GEMINI_API_KEY", "test-key", raising=False)
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
