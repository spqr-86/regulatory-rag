"""Tests for src/v7/feedback.py — 👍/👎 lands in Postgres (issue #20).

No live database: the connection is injected, so these tests describe the SQL and
the failure behaviour. What needs the stack — that the constraint really collapses
a repeated vote — is pinned by the migration test below reading the DDL.
"""

import re
from pathlib import Path

import pytest

from src.v7 import feedback

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "db" / "migrations"


class FakeCursor:
    def __init__(self, raises=None):
        self.raises = raises
        self.executed = []

    def execute(self, sql, params=None):
        if self.raises:
            raise self.raises
        self.executed.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, raises=None, closed=False):
        self.cursors = []
        self.raises = raises
        self.commits = 0
        self.closed = closed

    def cursor(self):
        cur = FakeCursor(raises=self.raises)
        self.cursors.append(cur)
        return cur

    def commit(self):
        self.commits += 1

    def close(self):
        self.closed = True

    @property
    def executed(self):
        return [row for cur in self.cursors for row in cur.executed]


def connector(*connections):
    calls = []

    def connect(dsn):
        calls.append(dsn)
        return connections[len(calls) - 1]

    connect.calls = calls
    return connect


QUERY_ID = "11111111-1111-1111-1111-111111111111"


def test_vote_is_one_row_bound_to_the_query_id():
    conn = FakeConnection()
    writer = feedback.FeedbackWriter("dsn", connect=connector(conn))

    writer.record(QUERY_ID, 1)

    sql, params = conn.executed[0]
    assert "INSERT INTO feedback" in sql
    assert params == (QUERY_ID, 1, None)
    assert conn.commits == 1


def test_repeated_vote_overwrites_instead_of_piling_up():
    """AC: pressing twice must not leave two rows for one answer."""
    conn = FakeConnection()
    writer = feedback.FeedbackWriter("dsn", connect=connector(conn))

    writer.record(QUERY_ID, -1, "промахнулся мимо вопроса")

    sql, params = conn.executed[0]
    assert "ON CONFLICT (query_id) DO UPDATE" in sql
    assert "verdict" in sql.split("DO UPDATE", 1)[1]
    assert params == (QUERY_ID, -1, "промахнулся мимо вопроса")


def test_blank_comment_is_stored_as_null_not_as_empty_text():
    """An empty string in the panel reads as "commented, said nothing"."""
    conn = FakeConnection()
    writer = feedback.FeedbackWriter("dsn", connect=connector(conn))

    writer.record(QUERY_ID, -1, "   ")

    assert conn.executed[0][1] == (QUERY_ID, -1, None)


@pytest.mark.parametrize("verdict", [0, 2, -2, "1", None])
def test_only_thumbs_up_or_down_are_accepted(verdict):
    """The check constraint would reject it anyway; fail before the round trip."""
    conn = FakeConnection()
    writer = feedback.FeedbackWriter("dsn", connect=connector(conn))

    with pytest.raises(ValueError):
        writer.record(QUERY_ID, verdict)

    assert conn.executed == []


def test_a_dead_connection_is_reopened_once():
    """The container restarts; the vote should not be lost to a stale socket."""
    dead = FakeConnection(raises=RuntimeError("server closed the connection"))
    fresh = FakeConnection()
    writer = feedback.FeedbackWriter("dsn", connect=connector(dead, fresh))

    writer.record(QUERY_ID, 1)

    assert dead.closed
    assert fresh.executed[0][1] == (QUERY_ID, 1, None)


def test_a_write_that_keeps_failing_raises():
    """Monitoring may not break the answer — the caller decides to swallow, not us."""
    first = FakeConnection(raises=RuntimeError("down"))
    second = FakeConnection(raises=RuntimeError("still down"))
    writer = feedback.FeedbackWriter("dsn", connect=connector(first, second))

    with pytest.raises(RuntimeError):
        writer.record(QUERY_ID, 1)


def test_close_drops_the_connection():
    conn = FakeConnection()
    writer = feedback.FeedbackWriter("dsn", connect=connector(conn))
    writer.record(QUERY_ID, 1)

    writer.close()

    assert conn.closed


def test_default_writer_is_none_when_events_do_not_go_to_postgres(monkeypatch):
    """Without the events table there is no query_id to point a vote at."""
    monkeypatch.setenv("V7_TELEMETRY_WRITER", "jsonl")
    assert feedback.default_feedback_writer() is None


def test_default_writer_is_none_when_credentials_are_missing(monkeypatch):
    """A misconfigured stack hides the buttons instead of raising under the answer."""
    monkeypatch.setenv("V7_TELEMETRY_WRITER", "postgres")
    monkeypatch.setenv("V7_TELEMETRY_PG_DSN", "")
    for var in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
        monkeypatch.delenv(var, raising=False)

    assert feedback.default_feedback_writer() is None


def test_default_writer_uses_the_configured_dsn(monkeypatch):
    monkeypatch.setenv("V7_TELEMETRY_WRITER", "postgres")
    monkeypatch.setenv("V7_TELEMETRY_PG_DSN", "postgresql://u:p@h:5432/db")

    writer = feedback.default_feedback_writer()

    assert isinstance(writer, feedback.FeedbackWriter)
    assert writer.dsn == "postgresql://u:p@h:5432/db"


def test_migration_makes_one_vote_per_query_the_rule():
    """The upsert needs a unique key on query_id, or ON CONFLICT is a syntax error."""
    ddl = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(MIGRATIONS.glob("*.sql"))
    )
    assert re.search(
        r"CREATE\s+UNIQUE\s+INDEX[^;]*\bON\s+feedback\s*\(\s*query_id\s*\)",
        ddl,
        re.IGNORECASE,
    ), "no unique index on feedback (query_id)"
