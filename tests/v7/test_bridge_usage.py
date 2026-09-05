"""Bridge returns token usage alongside the generated text (roadmap 4a)."""

from unittest.mock import MagicMock

from src.v7.bridge import make_expand_fn, make_generate_fn, model_name_of


class _FakeResponse:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata or {}


class _FakeLLM:
    """Minimal stand-in for a chat model: records prompts, returns canned text."""

    def __init__(self, text, usage=None, model_name="gpt-4o-mini"):
        self._text = text
        self._usage = (
            {"input_tokens": 100, "output_tokens": 20} if usage is None else usage
        )
        self.model_name = model_name
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _FakeResponse(self._text, self._usage)


class TestModelNameOf:
    def test_openai_style_model_name(self):
        llm = MagicMock(spec=["model_name"])
        llm.model_name = "gpt-4o"
        assert model_name_of(llm) == "gpt-4o"

    def test_gemini_style_model_attr(self):
        llm = MagicMock(spec=["model"])
        llm.model = "models/gemini-2.5-flash"
        assert model_name_of(llm) == "models/gemini-2.5-flash"

    def test_unknown_llm_does_not_raise(self):
        assert model_name_of(object()) == "unknown"


class TestGenerateFnUsage:
    def test_returns_answer_and_usage(self):
        llm = _FakeLLM("Ответ по норме.")
        fn = make_generate_fn(llm)
        answer, usage = fn("вопрос", "вопрос", [{"text": "норма", "metadata": {}}])
        assert answer == "Ответ по норме."
        assert usage["model"] == "gpt-4o-mini"
        assert usage["node"] == "generate"
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 20

    def test_usage_is_zero_when_provider_reports_nothing(self):
        llm = _FakeLLM("Ответ.", usage={})
        answer, usage = make_generate_fn(llm)("q", "q", [{"text": "t", "metadata": {}}])
        assert answer == "Ответ."
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0

    def test_fallback_path_still_returns_usage_shape(self):
        """LLM failure falls back to a stub answer — the caller must not need to
        special-case the return shape."""
        llm = MagicMock(spec=["invoke", "model_name"])
        llm.model_name = "gpt-4o-mini"
        llm.invoke.side_effect = RuntimeError("boom")
        answer, usage = make_generate_fn(llm)(
            "q", "q", [{"text": "текст нормы", "metadata": {}}]
        )
        assert isinstance(answer, str)
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0


class TestExpandFnUsage:
    def test_returns_alternatives_and_usage(self):
        llm = _FakeLLM("вариант 1\nвариант 2\nвариант 3")
        alts, usage = make_expand_fn(llm, n=3)("вопрос")
        assert alts == ["вариант 1", "вариант 2", "вариант 3"]
        assert usage["node"] == "expand"
        assert usage["prompt_tokens"] == 100

    def test_failure_returns_empty_list_and_zero_usage(self):
        llm = MagicMock(spec=["invoke", "model_name"])
        llm.model_name = "gpt-4o-mini"
        llm.invoke.side_effect = RuntimeError("boom")
        alts, usage = make_expand_fn(llm)("вопрос")
        assert alts == []
        assert usage["prompt_tokens"] == 0


class TestGenerateReportsContextSize:
    """Issue #22: the event must say how many passages reached the LLM.

    ``final_passages`` is what retrieval found; cross-reference expansion inside
    the generate fn adds more, so only the fn itself knows the real number.
    """

    def test_usage_carries_number_of_passages_sent_to_llm(self):
        llm = _FakeLLM("ответ")
        fn = make_generate_fn(llm)
        passages = [{"text": f"фрагмент {i}", "metadata": {}} for i in range(3)]

        _answer, usage = fn("вопрос", "вопрос", passages)

        assert usage["n_passages"] == 3

    def test_no_passages_means_zero(self):
        llm = _FakeLLM("ответ")
        fn = make_generate_fn(llm)

        _answer, usage = fn("вопрос", "вопрос", [])

        assert usage["n_passages"] == 0
