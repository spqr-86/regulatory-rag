"""Tests for scripts/ingest_events.py — the journal replayed into Postgres (#23).

The writer is injected, so the script is tested for what it owns: reading the
journal, skipping junk, and not double-counting a second run.
"""

import json

import pytest

from scripts import ingest_events


class RecordingWriter:
    """Counts what a real PostgresWriter would insert, without a database."""

    def __init__(self, raises_on=None):
        self.events = []
        self.raises_on = raises_on

    def write(self, event):
        if self.raises_on and event["query_id"] == self.raises_on:
            raise RuntimeError("insert failed")
        self.events.append(event)


def journal(tmp_path, *events) -> str:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8",
    )
    return str(path)


def event(query_id="a", **overrides):
    row = {"query_id": query_id, "source": "ui", "question": "вопрос"}
    row.update(overrides)
    return row


class TestIngest:
    def test_every_line_reaches_the_writer(self, tmp_path):
        writer = RecordingWriter()
        path = journal(tmp_path, event("a"), event("b"))

        report = ingest_events.ingest(path, writer)

        assert [e["query_id"] for e in writer.events] == ["a", "b"]
        assert report.written == 2

    def test_second_run_is_a_no_op_at_the_database(self, tmp_path):
        """Idempotency is the insert's ON CONFLICT; the script must not filter it away."""
        writer = RecordingWriter()
        path = journal(tmp_path, event("a"))

        ingest_events.ingest(path, writer)
        report = ingest_events.ingest(path, writer)

        # The row is offered again — the unique query_id is what makes it land once.
        assert len(writer.events) == 2
        assert report.written == 1

    def test_a_malformed_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text(
            json.dumps(event("a")) + "\nnot json\n" + json.dumps(event("b")) + "\n",
            encoding="utf-8",
        )
        writer = RecordingWriter()

        report = ingest_events.ingest(str(path), writer)

        assert [e["query_id"] for e in writer.events] == ["a", "b"]
        assert report.skipped == 1

    def test_a_row_without_query_id_is_skipped(self, tmp_path):
        path = journal(tmp_path, {"source": "ui"}, event("b"))
        writer = RecordingWriter()

        report = ingest_events.ingest(path, writer)

        assert [e["query_id"] for e in writer.events] == ["b"]
        assert report.skipped == 1

    def test_a_failed_insert_is_counted_and_does_not_stop_the_run(self, tmp_path):
        path = journal(tmp_path, event("a"), event("bad"), event("c"))
        writer = RecordingWriter(raises_on="bad")

        report = ingest_events.ingest(path, writer)

        assert [e["query_id"] for e in writer.events] == ["a", "c"]
        assert report.failed == 1
        assert report.written == 2

    def test_missing_journal_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest_events.ingest(str(tmp_path / "nope.jsonl"), RecordingWriter())
