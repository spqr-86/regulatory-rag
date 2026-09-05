"""Tests for src/v7/telemetry.py — a finished query turned into one event row.

No Postgres, no network: the writer is a protocol and the tests inject fakes
(issue #16, monitoring module 05).
"""

import json

import pytest
import structlog

from src.pricing import cost_for_usages
from src.v7 import telemetry


def make_state(**overrides) -> dict:
    """A completed simple-path query, the shape the graph leaves behind."""
    state = {
        "query": "Какая периодичность проверки огнетушителей?",
        "answer": "Раз в год.",
        "final_passages": [{"id": "a"}, {"id": "b"}],
        "retrieval_attempts": [{"retrieval_id": "r1", "stage": "simple"}],
        "llm_usage": [
            {
                "model": "gpt-4o-mini",
                "node": "generate",
                "stage": "simple",
                "prompt_tokens": 1000,
                "completion_tokens": 200,
            }
        ],
    }
    state.update(overrides)
    return state


class FakeWriter:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


class BrokenWriter:
    def write(self, event):
        raise RuntimeError("postgres is down")


class TestBuildEvent:
    def test_all_spec_fields_present_and_not_none(self):
        event = telemetry.build_event(make_state(), source="ui", latency_ms=5300)

        for field in telemetry.EVENT_FIELDS:
            assert field in event, f"missing field {field}"
            if field != "error":
                assert event[field] is not None, f"{field} must not be None"

    def test_carries_question_answer_len_and_passages(self):
        event = telemetry.build_event(make_state(), source="api", latency_ms=1200)

        assert event["question"] == "Какая периодичность проверки огнетушителей?"
        assert event["answer_len"] == len("Раз в год.")
        assert event["n_passages"] == 2
        assert event["source"] == "api"
        assert event["latency_ms"] == 1200
        assert event["error"] is None

    def test_token_sums_and_cost_match_pricing(self):
        state = make_state()
        event = telemetry.build_event(state, source="eval", latency_ms=900)

        assert event["prompt_tokens"] == 1000
        assert event["completion_tokens"] == 200
        assert event["cost_usd"] == cost_for_usages(state["llm_usage"])["cost_usd"]
        assert event["models"] == state["llm_usage"]

    def test_generates_query_id_when_absent(self):
        first = telemetry.build_event(make_state(), source="ui", latency_ms=1)
        second = telemetry.build_event(make_state(), source="ui", latency_ms=1)

        assert first["query_id"] != second["query_id"]

    def test_keeps_query_id_passed_from_outside(self):
        event = telemetry.build_event(
            make_state(), source="ui", latency_ms=1, query_id="fixed-id"
        )

        assert event["query_id"] == "fixed-id"

    def test_empty_usage_costs_nothing(self):
        event = telemetry.build_event(
            make_state(llm_usage=[], answer="", final_passages=[]),
            source="ui",
            latency_ms=40,
        )

        assert event["prompt_tokens"] == 0
        assert event["completion_tokens"] == 0
        assert event["cost_usd"] == 0.0
        assert event["unpriced_models"] == []
        assert event["n_passages"] == 0

    def test_unpriced_model_is_named_not_silently_free(self):
        state = make_state(
            llm_usage=[
                {"model": "gemini-3-pro", "prompt_tokens": 500, "completion_tokens": 100},
                {"model": "gpt-4o-mini", "prompt_tokens": 1000, "completion_tokens": 200},
            ]
        )
        event = telemetry.build_event(state, source="ui", latency_ms=1)

        assert event["unpriced_models"] == ["gemini-3-pro"]
        assert event["cost_usd"] > 0

    def test_failed_query_carries_error(self):
        event = telemetry.build_event(
            make_state(answer="", llm_usage=[]),
            source="api",
            latency_ms=200,
            error="TimeoutError: provider timed out",
        )

        assert event["error"] == "TimeoutError: provider timed out"
        assert event["answer_len"] == 0


class TestPathResolution:
    def test_abstain_wins_over_everything(self):
        state = make_state(
            abstain_reason="нет нормы в корпусе",
            clarify_message="уточните",
            retrieval_attempts=[{"stage": "complex"}],
        )
        assert telemetry.build_event(state, source="ui", latency_ms=1)["path"] == "abstain"

    def test_clarify_when_no_abstain(self):
        state = make_state(clarify_message="уточните год", retrieval_attempts=[])
        assert telemetry.build_event(state, source="ui", latency_ms=1)["path"] == "clarify"

    def test_complex_when_escalated(self):
        state = make_state(
            retrieval_attempts=[{"stage": "simple"}, {"stage": "complex"}]
        )
        assert telemetry.build_event(state, source="ui", latency_ms=1)["path"] == "complex"

    def test_simple_by_default(self):
        assert telemetry.build_event(make_state(), source="ui", latency_ms=1)["path"] == "simple"


class TestSafeWrite:
    def test_passes_event_to_writer(self):
        writer = FakeWriter()
        event = telemetry.build_event(make_state(), source="ui", latency_ms=1)

        assert telemetry.write_event(writer, event) is True
        assert writer.events == [event]

    def test_broken_writer_does_not_raise_and_warns_with_query_id(self):
        event = telemetry.build_event(
            make_state(), source="ui", latency_ms=1, query_id="q-42"
        )

        with structlog.testing.capture_logs() as logs:
            assert telemetry.write_event(BrokenWriter(), event) is False

        warnings = [entry for entry in logs if entry["log_level"] == "warning"]
        assert len(warnings) == 1
        assert warnings[0]["query_id"] == "q-42"


class TestJsonlWriter:
    def test_appends_one_valid_json_line_per_event(self, tmp_path):
        path = tmp_path / "events.jsonl"
        writer = telemetry.JsonlWriter(path)

        writer.write(telemetry.build_event(make_state(), source="ui", latency_ms=1))
        writer.write(telemetry.build_event(make_state(), source="api", latency_ms=2))

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert [json.loads(line)["source"] for line in lines] == ["ui", "api"]

    def test_creates_missing_directory(self, tmp_path):
        path = tmp_path / "logs" / "nested" / "events.jsonl"
        telemetry.JsonlWriter(path).write(
            telemetry.build_event(make_state(), source="ui", latency_ms=1)
        )

        assert path.exists()

    def test_unwritable_path_is_swallowed_by_write_event(self, tmp_path):
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        writer = telemetry.JsonlWriter(blocker / "events.jsonl")
        event = telemetry.build_event(make_state(), source="ui", latency_ms=1)

        with structlog.testing.capture_logs() as logs:
            assert telemetry.write_event(writer, event) is False

        assert [entry["log_level"] for entry in logs] == ["warning"]

        with pytest.raises(OSError):
            writer.write(event)


class TestContextSize:
    """Issue #22: n_passages is what reached the LLM, not what retrieval found."""

    def test_prefers_the_count_reported_by_the_generate_call(self):
        state = make_state(
            final_passages=[{"id": str(i)} for i in range(14)],
            llm_usage=[
                {"model": "gpt-4o-mini", "node": "expand", "prompt_tokens": 50, "completion_tokens": 10},
                {
                    "model": "gpt-4o-mini",
                    "node": "generate",
                    "prompt_tokens": 7000,
                    "completion_tokens": 120,
                    "n_passages": 30,
                },
            ],
        )

        event = telemetry.build_event(state, source="ui", latency_ms=1)

        assert event["n_passages"] == 30
        assert event["n_passages_found"] == 14

    def test_refusal_without_generation_is_zero_on_both(self):
        state = make_state(
            answer="",
            abstain_reason="нет нормы в корпусе",
            final_passages=[],
            llm_usage=[],
        )

        event = telemetry.build_event(state, source="ui", latency_ms=1)

        assert event["n_passages"] == 0
        assert event["n_passages_found"] == 0

    def test_generation_that_reported_nothing_falls_back_to_found(self):
        """Stub generators and the pre-#22 contract report no context size.

        Reporting zero next to a real answer would read as a bug, so the found
        count stands in — and it is the same number the old field carried.
        """
        state = make_state(final_passages=[{"id": "a"}, {"id": "b"}])

        event = telemetry.build_event(state, source="ui", latency_ms=1)

        assert event["n_passages"] == 2
        assert event["n_passages_found"] == 2
