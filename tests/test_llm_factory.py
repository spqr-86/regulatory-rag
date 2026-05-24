import pytest
from unittest.mock import patch, MagicMock


@patch("src.llm_factory.ChatOpenAI")
def test_get_llm_openai(mock_openai):
    from src.llm_factory import get_llm

    mock_openai.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.MODEL_NAME = "gpt-4o"
        mock_settings.TEMPERATURE = 0.0
        mock_settings.REQUEST_TIMEOUT = 120.0
        llm = get_llm()
        assert llm is not None


def test_get_llm_unknown_provider_raises():
    from src.llm_factory import get_llm

    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "unknown_provider"
        with pytest.raises(ValueError, match="unknown_provider"):
            get_llm()


@patch("src.llm_factory.ChatGoogleGenerativeAI")
def test_gemini_llm_max_output_tokens_leaves_answer_room_above_thinking_budget(
    mock_chat,
):
    """max_output_tokens must exceed thinking_budget by an answer allowance.

    gemini-3 counts reasoning tokens inside max_output_tokens. If the cap is at or
    below thinking_budget, reasoning consumes the whole budget and the answer
    truncates mid-word.
    """
    from src.llm_factory import get_gemini_llm

    mock_chat.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.GEMINI_FAST_MODEL = "gemini-3-flash-preview"
        mock_settings.REQUEST_TIMEOUT = 120.0
        get_gemini_llm(thinking_budget=4096)

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["max_output_tokens"] >= 4096 + 2048


@patch("src.llm_factory.ChatGoogleGenerativeAI")
def test_gemini_llm_max_output_tokens_scales_with_thinking_budget(mock_chat):
    """A smaller thinking_budget still gets an answer allowance on top."""
    from src.llm_factory import get_gemini_llm

    mock_chat.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.GEMINI_FAST_MODEL = "gemini-3-flash-preview"
        mock_settings.REQUEST_TIMEOUT = 120.0
        get_gemini_llm(thinking_budget=1024)

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["max_output_tokens"] >= 1024 + 2048


@patch("src.llm_factory.ChatGoogleGenerativeAI")
def test_get_simple_llm_uses_gemini_simple_model(mock_chat):
    """get_simple_llm() must instantiate Gemini with GEMINI_SIMPLE_MODEL,
    not GEMINI_FAST_MODEL — that's the whole point of the split.
    """
    from src.llm_factory import get_simple_llm

    mock_chat.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.GEMINI_FAST_MODEL = "gemini-3-flash-preview"
        mock_settings.GEMINI_SIMPLE_MODEL = "gemini-2.5-flash"
        mock_settings.REQUEST_TIMEOUT = 120.0
        get_simple_llm(thinking_budget=1024)

    assert mock_chat.call_args.kwargs["model"] == "gemini-2.5-flash"


@patch("src.llm_factory.ChatGoogleGenerativeAI")
def test_get_simple_llm_falls_back_to_fast_when_simple_empty(mock_chat):
    """Empty GEMINI_SIMPLE_MODEL → fall back to GEMINI_FAST_MODEL (safe default)."""
    from src.llm_factory import get_simple_llm

    mock_chat.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.GEMINI_FAST_MODEL = "gemini-3-flash-preview"
        mock_settings.GEMINI_SIMPLE_MODEL = ""
        mock_settings.REQUEST_TIMEOUT = 120.0
        get_simple_llm(thinking_budget=1024)

    assert mock_chat.call_args.kwargs["model"] == "gemini-3-flash-preview"


@patch("src.llm_factory.ChatGoogleGenerativeAI")
def test_get_simple_llm_differs_from_get_gemini_llm(mock_chat):
    """get_simple_llm and get_gemini_llm must pick distinct models when
    GEMINI_SIMPLE_MODEL is set. Guards against accidental collapse of the
    cost-saving split.
    """
    from src.llm_factory import get_gemini_llm, get_simple_llm

    mock_chat.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.GEMINI_FAST_MODEL = "gemini-3-flash-preview"
        mock_settings.GEMINI_SIMPLE_MODEL = "gemini-2.5-flash"
        mock_settings.REQUEST_TIMEOUT = 120.0
        get_gemini_llm(thinking_budget=1024)
        fast_model = mock_chat.call_args.kwargs["model"]
        get_simple_llm(thinking_budget=1024)
        simple_model = mock_chat.call_args.kwargs["model"]

    assert fast_model != simple_model
    assert simple_model == "gemini-2.5-flash"


@patch("src.llm_factory.ChatGoogleGenerativeAI")
def test_gemini_llm_passes_max_retries(mock_chat):
    """Gemini factory must pass max_retries=3 so LangChain handles transient
    5xx/429 retries itself (replaces the previous tenacity wrapper in bridge).
    """
    from src.llm_factory import get_gemini_llm

    mock_chat.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.GEMINI_FAST_MODEL = "gemini-3-flash-preview"
        mock_settings.REQUEST_TIMEOUT = 120.0
        get_gemini_llm(thinking_budget=1024)

    kwargs = mock_chat.call_args.kwargs
    assert kwargs["max_retries"] == 3
