from __future__ import annotations

import os

import structlog as _structlog
from dotenv import load_dotenv
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config.settings import settings

load_dotenv()


# IPv6 patch for Gemini API: VPS datacenter IPv4 is ASN-blocked by Google,
# but IPv6 is not. When no HTTPS_PROXY is set, prefer IPv6 for googleapis.com.
# Call apply_ipv6_patch_for_googleapis() explicitly in entry points (api.py, app.py)
# before importing any langchain_google_genai modules.
_ipv6_patch_applied = False


def apply_ipv6_patch_for_googleapis() -> None:
    """Override getaddrinfo to prefer IPv6 for googleapis.com connections.

    Idempotent — safe to call multiple times. Must be called before the first
    ChatGoogleGenerativeAI import (i.e. before src.llm_factory is imported).
    """
    global _ipv6_patch_applied
    if _ipv6_patch_applied:
        return
    import socket  # noqa: PLC0415

    if os.getenv("HTTPS_PROXY") or os.getenv("https_proxy"):
        _ipv6_patch_applied = True
        return  # proxy handles routing, skip patch
    _orig = socket.getaddrinfo

    def _prefer_ipv6(host: str | None, port, family=0, *args, **kwargs):
        if host and (host == "googleapis.com" or host.endswith(".googleapis.com")):
            try:
                results = _orig(host, port, socket.AF_INET6, *args, **kwargs)
                if results:
                    return results
            except OSError:
                pass
        return _orig(host, port, family, *args, **kwargs)

    socket.getaddrinfo = _prefer_ipv6  # type: ignore[assignment]
    _ipv6_patch_applied = True


apply_ipv6_patch_for_googleapis()

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from google.genai.types import AutomaticFunctionCallingConfig
except ImportError:
    ChatGoogleGenerativeAI = None
    AutomaticFunctionCallingConfig = None


_log = _structlog.get_logger()
_GEMINI_ONLY_KWARGS = {"thinking_budget", "response_mime_type"}


def _create_openai_llm(**kwargs):
    dropped = {k: kwargs.pop(k) for k in list(kwargs) if k in _GEMINI_ONLY_KWARGS}
    if dropped:
        _log.warning(
            "llm_factory.kwarg_drop",
            provider="openai",
            dropped=dropped,
            note="provider-specific kwargs ignored — may cause behavior drift",
        )
    model = kwargs.pop("model_name", "gpt-4o-mini")
    temperature = kwargs.pop("temperature", settings.TEMPERATURE)
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=settings.REQUEST_TIMEOUT,
        max_retries=3,
        **kwargs,
    )


def _create_gemini_llm(**kwargs):
    """Forward kwargs (thinking_budget, response_mime_type, ...) to get_gemini_llm."""
    return get_gemini_llm(**kwargs)


_LLM_PROVIDERS = {
    "openai": _create_openai_llm,
    "gemini": _create_gemini_llm,
}


# gemini-3 counts reasoning tokens inside max_output_tokens, so the cap must cover
# thinking_budget AND the visible answer — otherwise reasoning eats the whole budget
# and the answer truncates mid-word (finish_reason=MAX_TOKENS).
_GEMINI_ANSWER_TOKEN_ALLOWANCE = 4096


def get_gemini_llm(
    model_name: str = None,
    temperature: float = 0.0,
    thinking_budget: int = None,
    response_mime_type: str = None,
):
    """
    Factory for Gemini LLM with model selection and thinking budget.

    Args:
        model_name: Model to use (default: gemini-3-flash-preview)
        temperature: Temperature for generation (default: 0.0)
        thinking_budget: Token budget for Gemini thinking mode (None = disabled)
        response_mime_type: Response format, e.g. "application/json" (None = text)

    Returns:
        ChatGoogleGenerativeAI instance
    """
    if ChatGoogleGenerativeAI is None:
        raise ImportError(
            "Google Gemini package (langchain-google-genai) not installed. "
            "Please install it or use another provider."
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Set it in .env or environment variables."
        )

    model = model_name or "gemini-3-flash-preview"

    max_output_tokens = (thinking_budget or 0) + _GEMINI_ANSWER_TOKEN_ALLOWANCE
    kwargs = dict(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout=settings.REQUEST_TIMEOUT,
        max_retries=3,
    )
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    if response_mime_type is not None:
        kwargs["response_mime_type"] = response_mime_type

    llm = ChatGoogleGenerativeAI(**kwargs)

    # Disable Gemini SDK's Automatic Function Calling (AFC) via the public
    # Runnable.bind() API. AFC creates its own tool-calling loop on top of
    # LangGraph's, causing duplicate tool calls. langchain-google-genai 4.2.x
    # exposes no constructor parameter for AFC, but `_prepare_request` forwards
    # **kwargs into `GenerateContentConfig`, where `automatic_function_calling`
    # is a first-class field — so binding the kwarg is the supported path and
    # avoids monkey-patching the private `_build_request_config` method.
    # WARNING: uses .bind(AutomaticFunctionCallingConfig) API —
    # version is pinned in requirements.txt (langchain-google-genai==4.2.1).
    # To upgrade: bump pin, re-run scripts/trace_v7.py and verify pipeline ends
    # with answer (not duplicate tool calls in trace).
    if AutomaticFunctionCallingConfig is not None:
        llm = llm.bind(
            automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)
        )

    return llm


def _resolve_provider_and_model(provider: str, model: str) -> tuple[str, str]:
    return provider.lower(), model


def get_simple_llm(**kwargs):
    """LLM for the simple RAG path (SIMPLE_LLM_PROVIDER + SIMPLE_MODEL_NAME)."""
    provider, model = _resolve_provider_and_model(
        settings.SIMPLE_LLM_PROVIDER,
        settings.SIMPLE_MODEL_NAME,
    )
    if provider == "gemini":
        kwargs.setdefault("model_name", model)
        return get_gemini_llm(**kwargs)
    factory = _LLM_PROVIDERS.get(provider)
    if not factory:
        available = ", ".join(sorted(_LLM_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown SIMPLE_LLM_PROVIDER={provider!r}. Available: {available}"
        )
    return factory(**kwargs)


def get_complex_llm(**kwargs):
    """LLM for the complex RAG path (COMPLEX_LLM_PROVIDER + COMPLEX_MODEL_NAME)."""
    provider, model = _resolve_provider_and_model(
        settings.COMPLEX_LLM_PROVIDER,
        settings.COMPLEX_MODEL_NAME,
    )
    if provider == "gemini":
        kwargs.setdefault("model_name", model)
        return get_gemini_llm(**kwargs)
    factory = _LLM_PROVIDERS.get(provider)
    if not factory:
        available = ", ".join(sorted(_LLM_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown COMPLEX_LLM_PROVIDER={provider!r}. Available: {available}"
        )
    return factory(**kwargs)


def get_judge_llm(**kwargs):
    """LLM for eval judge (JUDGE_LLM_PROVIDER + JUDGE_MODEL_NAME). Independent from pipeline."""
    provider, model = _resolve_provider_and_model(
        settings.JUDGE_LLM_PROVIDER,
        settings.JUDGE_MODEL_NAME,
    )
    if provider == "gemini":
        kwargs.setdefault("model_name", model)
        return get_gemini_llm(**kwargs)
    factory = _LLM_PROVIDERS.get(provider)
    if not factory:
        available = ", ".join(sorted(_LLM_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown JUDGE_LLM_PROVIDER={provider!r}. Available: {available}"
        )
    return factory(**kwargs)


def get_vision_llm():
    """
    Возвращает LLM с поддержкой Vision (зрения).
    Используется для анализа скриншотов документов.
    """
    return get_gemini_llm()


def _create_hf_embeddings():
    model = settings.EMBEDDING_MODEL_NAME or "intfloat/multilingual-e5-base"
    api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or ""
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=api_key,
        model_name=model,
    )


def _create_local_embeddings():
    model = settings.EMBEDDING_MODEL_NAME
    return HuggingFaceEmbeddings(
        model_name=model or "ai-forever/sbert_large_nlu_ru",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


_EMBEDDING_PROVIDERS = {
    "openai": lambda: OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL_NAME or "text-embedding-3-small"
    ),
    "hf_api": _create_hf_embeddings,
    "local": _create_local_embeddings,
    "huggingface": _create_local_embeddings,
}


def get_embedding_model():
    provider = (settings.EMBEDDING_PROVIDER or "").lower()
    factory = _EMBEDDING_PROVIDERS.get(provider)
    if not factory:
        available = ", ".join(sorted(_EMBEDDING_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER={provider}. Available: {available}"
        )
    return factory()
