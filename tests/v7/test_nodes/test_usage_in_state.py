"""Nodes put LLM token usage into graph state (roadmap 4a).

Before this, usage went to the log only and the eval runner reported quality
without price.
"""

import pytest

from src.v7.nodes import generate_answer as ga_mod
from src.v7.nodes import rag_simple as rs_mod
from src.v7.usage import LLM_USAGE_KEY


@pytest.fixture(autouse=True)
def _restore_generate_fns():
    yield
    ga_mod.set_generate_fns(None, None)


def _usage(tokens=(10, 3), node="generate"):
    return {
        "model": "gpt-4o-mini",
        "node": node,
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
    }


class TestGenerateAnswerUsage:
    def test_usage_reaches_state_with_stage(self):
        ga_mod.set_generate_fns(
            simple=lambda q, aq, p: ("ответ", _usage()),
            complex_=lambda q, aq, p: ("другое", _usage()),
        )
        out = ga_mod.generate_answer(
            {
                "query": "q",
                "final_passages": [{"text": "t"}],
                "retrieval_attempts": [{"stage": "simple"}],
            }
        )
        assert out["answer"] == "ответ"
        assert out[LLM_USAGE_KEY] == [
            {
                "model": "gpt-4o-mini",
                "node": "generate",
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "stage": "simple",
            }
        ]

    def test_complex_path_stamped_complex(self):
        ga_mod.set_generate_fns(
            simple=lambda q, aq, p: ("s", _usage()),
            complex_=lambda q, aq, p: ("c", _usage((100, 50))),
        )
        out = ga_mod.generate_answer(
            {
                "query": "q",
                "final_passages": [{"text": "t"}],
                "retrieval_attempts": [{"stage": "simple"}, {"stage": "complex"}],
            }
        )
        assert out["answer"] == "c"
        assert out[LLM_USAGE_KEY][0]["stage"] == "complex"
        assert out[LLM_USAGE_KEY][0]["prompt_tokens"] == 100

    def test_plain_string_fn_still_works(self):
        """Back-compat: an fn that returns only text yields no usage, no crash."""
        ga_mod.set_generate_fns(
            simple=lambda q, aq, p: "просто текст", complex_=None
        )
        out = ga_mod.generate_answer(
            {"query": "q", "final_passages": [{"text": "t"}], "retrieval_attempts": []}
        )
        assert out["answer"] == "просто текст"
        assert out.get(LLM_USAGE_KEY, []) == []

    def test_stub_generation_reports_no_usage(self):
        out = ga_mod.generate_answer(
            {"query": "q", "final_passages": [{"text": "t"}], "retrieval_attempts": []}
        )
        assert out.get(LLM_USAGE_KEY, []) == []


class TestRagSimpleExpandUsage:
    def test_expand_usage_reaches_state(self, monkeypatch):
        monkeypatch.setattr(rs_mod.v7_config, "V8_ENABLE_MULTI_QUERY", True)
        monkeypatch.setattr(rs_mod.v7_config, "V8_EXPAND_N", 2)
        rs_mod.set_expand_fn(lambda q, n=2: (["alt1", "alt2"], _usage((7, 4), "expand")))
        monkeypatch.setattr(rs_mod, "_vector_search", lambda **kw: [])
        monkeypatch.setattr(rs_mod, "bm25_search", lambda **kw: [])

        out = rs_mod.rag_simple(
            {
                "query": "q",
                "plan": {"top_k": 5},
                "retrieval_id": "r1",
                "retrieval_attempts": [],
            }
        )
        usages = out.get(LLM_USAGE_KEY, [])
        assert len(usages) == 1
        assert usages[0]["node"] == "expand"
        assert usages[0]["stage"] == "simple"
        rs_mod.set_expand_fn(None)

    def test_legacy_expand_fn_returning_list(self, monkeypatch):
        monkeypatch.setattr(rs_mod.v7_config, "V8_ENABLE_MULTI_QUERY", True)
        monkeypatch.setattr(rs_mod.v7_config, "V8_EXPAND_N", 2)
        rs_mod.set_expand_fn(lambda q, n=2: ["alt1"])
        monkeypatch.setattr(rs_mod, "_vector_search", lambda **kw: [])
        monkeypatch.setattr(rs_mod, "bm25_search", lambda **kw: [])

        out = rs_mod.rag_simple(
            {
                "query": "q",
                "plan": {"top_k": 5},
                "retrieval_id": "r1",
                "retrieval_attempts": [],
            }
        )
        assert out.get(LLM_USAGE_KEY, []) == []
        rs_mod.set_expand_fn(None)
