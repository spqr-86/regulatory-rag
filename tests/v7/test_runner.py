"""Tests for src/v7/runner.py — the one place a query passes through (issue #17).

The runner owns nothing but timing, the telemetry call and the query id; the
graph and the writer are both injected, so nothing here touches Postgres, the
network or a real pipeline.
"""

import json

import pytest
import structlog

from src.v7 import pg_writer, runner, telemetry


class FakeGraph:
    """Stands in for a compiled LangGraph app."""

    def __init__(self, result=None, raises=None):
        self.result = result if result is not None else {"answer": "ответ"}
        self.raises = raises
        self.calls = []

    def invoke(self, state):
        self.calls.append(state)
        if self.raises:
            raise self.raises
        return self.result


class FakeWriter:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


class BrokenWriter:
    def write(self, event):
        raise RuntimeError("postgres is down")


def simple_state(**overrides):
    state = {
        "query": "Периодичность проверки огнетушителей?",
        "answer": "Раз в год.",
        "final_passages": [{"id": "a"}],
        "retrieval_attempts": [{"stage": "simple"}],
        "llm_usage": [
            {"model": "gpt-4o-mini", "prompt_tokens": 100, "completion_tokens": 20}
        ],
    }
    state.update(overrides)
    return state


class TestHappyPath:
    def test_invokes_graph_with_query_and_returns_result(self):
        graph = FakeGraph(result=simple_state())
        writer = FakeWriter()

        result, query_id = runner.run_query(
            graph, "Периодичность проверки огнетушителей?", source="ui", writer=writer
        )

        assert result == graph.result
        assert graph.calls == [{"query": "Периодичность проверки огнетушителей?"}]
        assert query_id

    def test_writes_exactly_one_event_with_caller_source(self):
        writer = FakeWriter()

        runner.run_query(
            FakeGraph(result=simple_state()), "вопрос", source="api", writer=writer
        )

        assert len(writer.events) == 1
        assert writer.events[0]["source"] == "api"

    def test_returned_query_id_matches_the_written_row(self):
        writer = FakeWriter()

        _, query_id = runner.run_query(
            FakeGraph(result=simple_state()), "вопрос", source="ui", writer=writer
        )

        assert writer.events[0]["query_id"] == query_id

    def test_caller_supplied_query_id_is_kept(self):
        writer = FakeWriter()

        _, query_id = runner.run_query(
            FakeGraph(result=simple_state()),
            "вопрос",
            source="ui",
            writer=writer,
            query_id="q-7",
        )

        assert query_id == "q-7"
        assert writer.events[0]["query_id"] == "q-7"

    def test_latency_covers_the_whole_call(self):
        writer = FakeWriter()

        runner.run_query(
            FakeGraph(result=simple_state()), "вопрос", source="ui", writer=writer
        )

        assert writer.events[0]["latency_ms"] >= 0
        assert isinstance(writer.events[0]["latency_ms"], int)

    def test_filters_are_passed_into_the_graph(self):
        graph = FakeGraph(result=simple_state())

        runner.run_query(
            graph,
            "вопрос",
            source="mcp",
            writer=FakeWriter(),
            filters={"doc_type": "СП"},
        )

        assert graph.calls == [{"query": "вопрос", "filters": {"doc_type": "СП"}}]

    def test_source_is_required(self):
        with pytest.raises(TypeError):
            runner.run_query(FakeGraph(), "вопрос", writer=FakeWriter())


class TestRefusalPaths:
    def test_abstain_is_recorded(self):
        writer = FakeWriter()
        state = simple_state(answer="", abstain_reason="нет нормы в корпусе")

        runner.run_query(FakeGraph(result=state), "вопрос", source="ui", writer=writer)

        assert writer.events[0]["path"] == "abstain"

    def test_clarify_is_recorded(self):
        writer = FakeWriter()
        state = simple_state(
            answer="", clarify_message="уточните год", retrieval_attempts=[]
        )

        runner.run_query(FakeGraph(result=state), "вопрос", source="ui", writer=writer)

        assert writer.events[0]["path"] == "clarify"

    def test_empty_state_still_produces_a_row(self):
        writer = FakeWriter()

        runner.run_query(FakeGraph(result={}), "шум", source="ui", writer=writer)

        assert writer.events[0]["path"] == "simple"
        assert writer.events[0]["cost_usd"] == 0.0


class TestFailures:
    def test_graph_failure_is_recorded_then_reraised(self):
        writer = FakeWriter()
        graph = FakeGraph(raises=TimeoutError("provider timed out"))

        with pytest.raises(TimeoutError):
            runner.run_query(graph, "вопрос", source="api", writer=writer)

        assert len(writer.events) == 1
        assert "provider timed out" in writer.events[0]["error"]
        assert writer.events[0]["source"] == "api"

    def test_broken_writer_does_not_break_the_answer(self):
        graph = FakeGraph(result=simple_state())

        with structlog.testing.capture_logs() as logs:
            result, _ = runner.run_query(
                graph, "вопрос", source="ui", writer=BrokenWriter()
            )

        assert result == graph.result
        assert [entry["log_level"] for entry in logs] == ["warning"]


class TestWiring:
    def test_default_writer_appends_to_the_configured_journal(
        self, tmp_path, monkeypatch
    ):
        journal = tmp_path / "events.jsonl"
        monkeypatch.setenv("V7_TELEMETRY_JSONL_PATH", str(journal))
        monkeypatch.setenv("V7_TELEMETRY_ENABLED", "true")

        writer = runner.default_writer()
        assert writer is not None

        runner.run_query(
            FakeGraph(result=simple_state()), "вопрос", source="ui", writer=writer
        )

        row = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
        assert row["source"] == "ui"

    def test_disabled_telemetry_writes_nothing(self, tmp_path, monkeypatch):
        journal = tmp_path / "events.jsonl"
        monkeypatch.setenv("V7_TELEMETRY_JSONL_PATH", str(journal))
        monkeypatch.setenv("V7_TELEMETRY_ENABLED", "false")

        writer = runner.default_writer()
        assert writer is None

        runner.run_query(
            FakeGraph(result=simple_state()), "вопрос", source="ui", writer=writer
        )

        assert not journal.exists()

    def test_postgres_writer_is_chosen_by_config(self, monkeypatch):
        """Switching journals is a config value, not a change at the call sites (#23)."""
        monkeypatch.setenv("V7_TELEMETRY_ENABLED", "true")
        monkeypatch.setenv("V7_TELEMETRY_WRITER", "postgres")
        monkeypatch.setenv(
            "V7_TELEMETRY_PG_DSN", "postgresql://regrag:pw@127.0.0.1:5432/regrag"
        )

        writer = runner.default_writer()

        assert isinstance(writer, telemetry.FallbackWriter)
        assert isinstance(writer.primary, pg_writer.PostgresWriter)
        assert writer.primary.dsn == "postgresql://regrag:pw@127.0.0.1:5432/regrag"

    def test_postgres_writer_keeps_the_journal_behind_it(self, tmp_path, monkeypatch):
        """Спек, критерий 4: упавшая база не должна стоить события."""
        monkeypatch.setenv("V7_TELEMETRY_ENABLED", "true")
        monkeypatch.setenv("V7_TELEMETRY_WRITER", "postgres")
        monkeypatch.setenv(
            "V7_TELEMETRY_PG_DSN", "postgresql://regrag:pw@127.0.0.1:5432/regrag"
        )
        journal = tmp_path / "events.jsonl"
        monkeypatch.setenv("V7_TELEMETRY_JSONL_PATH", str(journal))

        writer = runner.default_writer()
        assert isinstance(writer.backup, telemetry.JsonlWriter)
        assert writer.backup.path == journal

    def test_jsonl_stays_the_default_writer(self, tmp_path, monkeypatch):
        monkeypatch.setenv("V7_TELEMETRY_ENABLED", "true")
        monkeypatch.delenv("V7_TELEMETRY_WRITER", raising=False)
        monkeypatch.setenv("V7_TELEMETRY_JSONL_PATH", str(tmp_path / "events.jsonl"))

        assert isinstance(runner.default_writer(), telemetry.JsonlWriter)

    def test_unusable_postgres_config_falls_back_to_the_journal(
        self, tmp_path, monkeypatch
    ):
        """A missing DSN must not cost the whole answer path (#23)."""
        monkeypatch.setenv("V7_TELEMETRY_ENABLED", "true")
        monkeypatch.setenv("V7_TELEMETRY_WRITER", "postgres")
        monkeypatch.delenv("V7_TELEMETRY_PG_DSN", raising=False)
        for var in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
            monkeypatch.delenv(var, raising=False)
        journal = tmp_path / "events.jsonl"
        monkeypatch.setenv("V7_TELEMETRY_JSONL_PATH", str(journal))

        writer = runner.default_writer()

        assert isinstance(writer, telemetry.JsonlWriter)


class TestRunId:
    """The runner carries the caller's run id into the row (issue #18)."""

    def test_run_id_reaches_the_written_event(self):
        writer = FakeWriter()

        runner.run_query(
            FakeGraph(result=simple_state()),
            "вопрос",
            source="eval",
            writer=writer,
            run_id="eval-20260905T091500Z-a3f1",
        )

        assert writer.events[0]["run_id"] == "eval-20260905T091500Z-a3f1"

    def test_without_run_id_the_row_says_none(self):
        writer = FakeWriter()

        runner.run_query(
            FakeGraph(result=simple_state()), "вопрос", source="ui", writer=writer
        )

        assert writer.events[0]["run_id"] is None

    def test_failed_query_keeps_the_run_id(self):
        writer = FakeWriter()

        with pytest.raises(RuntimeError):
            runner.run_query(
                FakeGraph(raises=RuntimeError("boom")),
                "вопрос",
                source="eval",
                writer=writer,
                run_id="eval-20260905T091500Z-a3f1",
            )

        assert writer.events[0]["run_id"] == "eval-20260905T091500Z-a3f1"
