"""Tests for generate_answer node."""

from __future__ import annotations

import pytest

from src.v7.nodes.generate_answer import generate_answer, set_generate_fn


class TestGenerateAnswer:
    @pytest.mark.unit
    def test_generates_answer_from_passages(self):
        """When final_passages are present — stub generates an answer."""
        state = {
            "query": "требования к пожарному извещателю",
            "active_query": "требования к пожарному извещателю",
            "final_passages": [
                {
                    "text": "Пожарные извещатели должны устанавливаться на потолке.",
                    "score": 0.8,
                },
                {"text": "Расстояние между извещателями не менее 4 м.", "score": 0.7},
            ],
        }
        result = generate_answer(state)
        assert "answer" in result
        assert result["answer"]
        assert len(result["answer"]) > 10

    @pytest.mark.unit
    def test_returns_empty_when_no_passages(self):
        """Without passages — answer is empty / fallback."""
        state = {
            "query": "вопрос",
            "active_query": "вопрос",
            "final_passages": [],
        }
        result = generate_answer(state)
        assert "answer" in result

    @pytest.mark.unit
    def test_returns_empty_when_passages_missing(self):
        """final_passages not set — does not raise, returns answer."""
        state = {
            "query": "вопрос",
            "active_query": "вопрос",
        }
        result = generate_answer(state)
        assert "answer" in result

    @pytest.mark.unit
    def test_stub_includes_passage_text(self):
        """Stub includes passage text in the answer."""
        passages = [{"text": "Уникальный текст АБВГД", "score": 0.9}]
        state = {
            "query": "q",
            "active_query": "q",
            "final_passages": passages,
        }
        result = generate_answer(state)
        assert "АБВГД" in result["answer"]

    @pytest.mark.unit
    def test_injectable_generate_fn(self):
        """set_generate_fn replaces the implementation."""
        called_with = {}

        def my_fn(query, active_query, passages):
            called_with["query"] = query
            called_with["passages"] = passages
            return "custom answer"

        set_generate_fn(my_fn)
        try:
            state = {
                "query": "тест",
                "active_query": "тест",
                "final_passages": [{"text": "текст", "score": 0.5}],
            }
            result = generate_answer(state)
            assert result["answer"] == "custom answer"
            assert called_with["query"] == "тест"
        finally:
            set_generate_fn(None)  # restore stub
