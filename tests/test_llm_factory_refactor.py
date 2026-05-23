"""Tests for the LLM factory registry refactor (CARD-1.1).

Verifies that get_llm() routes to the right provider, forwards kwargs to
Gemini, and silently drops Gemini-specific kwargs for OpenAI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.unit
def test_get_llm_gemini_forwards_kwargs():
    """Gemini factory must pass thinking_budget/response_mime_type through."""
    from src.llm_factory import get_llm

    with (
        patch("src.llm_factory.settings") as s,
        patch("src.llm_factory.get_gemini_llm") as mock_g,
    ):
        s.LLM_PROVIDER = "gemini"
        get_llm(thinking_budget=4096, response_mime_type="application/json")
        mock_g.assert_called_once_with(
            thinking_budget=4096, response_mime_type="application/json"
        )


@pytest.mark.unit
def test_get_llm_openai_drops_gemini_kwargs():
    """OpenAI factory must silently drop kwargs it does not understand."""
    from src.llm_factory import get_llm

    with (
        patch("src.llm_factory.settings") as s,
        patch("src.llm_factory.ChatOpenAI") as mock_cls,
    ):
        s.LLM_PROVIDER = "openai"
        s.MODEL_NAME = "gpt-4o-mini"
        s.TEMPERATURE = 0.0
        s.REQUEST_TIMEOUT = 120.0
        mock_cls.return_value = MagicMock()

        get_llm(thinking_budget=4096, response_mime_type="application/json")

        call_kwargs = mock_cls.call_args.kwargs
        assert "thinking_budget" not in call_kwargs
        assert "response_mime_type" not in call_kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"


@pytest.mark.unit
def test_get_llm_unknown_provider_raises():
    """Unknown provider must raise ValueError listing available ones."""
    from src.llm_factory import get_llm

    with patch("src.llm_factory.settings") as s:
        s.LLM_PROVIDER = "anthropic"
        with pytest.raises(ValueError, match="anthropic"):
            get_llm()


@pytest.mark.unit
def test_registry_contains_gemini_and_openai():
    """Registry must expose both providers after refactor."""
    from src.llm_factory import _LLM_PROVIDERS

    assert "gemini" in _LLM_PROVIDERS
    assert "openai" in _LLM_PROVIDERS
