"""Tests for src/v7/pg_writer.py — the event row lands in Postgres (issue #23).

No live database: the connection is injected, so these tests describe the SQL
and the failure behaviour, not the driver.
"""

import pytest
import structlog

from src.v7 import pg_writer, telemetry


def make_event(**overrides) -> dict:
    event = {
        "query_id": "11111111-1111-1111-1111-111111111111",
        "run_id": None,
        "ts": "2026-09-05T10:00:00+00:00",
        "source": "ui",
        "question": "Какая периодичность проверки огнетушителей?",
        "path": "simple",
        "answer_len": 10,
        "n_passages": 2,
        "n_passages_found": 4,
        "latency_ms": 1234,
        "prompt_tokens": 1000,
        "completion_tokens": 200,
        "cost_usd": 0.00042,
        "models": [{"model": "gpt-4o-mini", "node": "generate"}],
        "unpriced_models": ["mystery-model"],
        "error": None,
    }
    event.update(overrides)
    return event


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
    """Enough of psycopg.Connection for the writer: cursor, commit, close."""

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
    """A connect() that hands out the given connections in order."""
    calls = []
    pool = list(connections)

    def connect(dsn):
        calls.append(dsn)
        if not pool:
            raise AssertionError("connect called more times than expected")
        conn = pool.pop(0)
        if isinstance(conn, Exception):
            raise conn
        return conn

    connect.calls = calls
    return connect


class TestWrite:
    def test_writes_one_row_with_every_event_field(self):
        conn = FakeConnection()
        writer = pg_writer.PostgresWriter("dsn", connect=connector(conn))

        writer.write(make_event())

        assert len(conn.executed) == 1
        sql, params = conn.executed[0]
        assert "INSERT INTO queries" in sql
        # Column list and values follow EVENT_FIELDS, so schema drift shows up here.
        for field in telemetry.EVENT_FIELDS:
            assert field in sql
        assert len(params) == len(telemetry.EVENT_FIELDS)
        assert conn.commits == 1

    def test_json_and_array_fields_are_adapted_for_postgres(self):
        conn = FakeConnection()
        writer = pg_writer.PostgresWriter("dsn", connect=connector(conn))

        writer.write(make_event())

        _, params = conn.executed[0]
        by_field = dict(zip(telemetry.EVENT_FIELDS, params))
        # jsonb column: a bare list would be adapted as an array, not as json.
        assert isinstance(by_field["models"], pg_writer.Jsonb)
        # text[] column: psycopg adapts a plain list of str directly.
        assert by_field["unpriced_models"] == ["mystery-model"]

    def test_repeated_query_id_does_not_raise_and_writes_once(self):
        conn = FakeConnection()
        writer = pg_writer.PostgresWriter("dsn", connect=connector(conn))
        event = make_event()

        writer.write(event)
        writer.write(event)

        sql, _ = conn.executed[0]
        assert "ON CONFLICT (query_id) DO NOTHING" in sql

    def test_connection_is_reused_across_events(self):
        conn = FakeConnection()
        connect = connector(conn)
        writer = pg_writer.PostgresWriter("dsn", connect=connect)

        writer.write(make_event())
        writer.write(make_event(query_id="22222222-2222-2222-2222-222222222222"))

        assert len(connect.calls) == 1
        assert len(conn.executed) == 2


class TestFailure:
    def test_dead_connection_is_retried_once_on_a_fresh_one(self):
        dead = FakeConnection(raises=RuntimeError("server closed the connection"))
        alive = FakeConnection()
        writer = pg_writer.PostgresWriter("dsn", connect=connector(dead, alive))

        writer.write(make_event())

        assert len(alive.executed) == 1
        assert dead.closed is True

    def test_database_down_raises_for_write_event_to_swallow(self):
        writer = pg_writer.PostgresWriter(
            "dsn",
            connect=connector(
                RuntimeError("connection refused"), RuntimeError("connection refused")
            ),
        )

        with pytest.raises(RuntimeError):
            writer.write(make_event())

    def test_write_event_keeps_the_answer_alive_when_postgres_is_down(self):
        structlog.configure(processors=[structlog.testing.LogCapture()])
        writer = pg_writer.PostgresWriter(
            "dsn",
            connect=connector(
                RuntimeError("connection refused"), RuntimeError("connection refused")
            ),
        )

        landed = telemetry.write_event(writer, make_event())

        assert landed is False


class TestDsn:
    def test_explicit_dsn_wins(self):
        env = {"V7_TELEMETRY_PG_DSN": "postgresql://explicit/db", "POSTGRES_USER": "u"}
        assert pg_writer.dsn_from_env(env) == "postgresql://explicit/db"

    def test_built_from_the_compose_variables_when_dsn_is_empty(self):
        env = {
            "POSTGRES_USER": "regrag",
            "POSTGRES_PASSWORD": "s3cret",
            "POSTGRES_DB": "regrag",
        }
        assert pg_writer.dsn_from_env(env) == (
            "postgresql://regrag:s3cret@127.0.0.1:5432/regrag"
        )

    def test_custom_port_is_honoured(self):
        env = {
            "POSTGRES_USER": "regrag",
            "POSTGRES_PASSWORD": "s3cret",
            "POSTGRES_DB": "regrag",
            "POSTGRES_PORT": "5433",
        }
        assert pg_writer.dsn_from_env(env).endswith(":5433/regrag")

    def test_password_with_special_characters_is_quoted(self):
        env = {
            "POSTGRES_USER": "regrag",
            "POSTGRES_PASSWORD": "p@ss word/1",
            "POSTGRES_DB": "regrag",
        }
        dsn = pg_writer.dsn_from_env(env)
        assert "p%40ss%20word%2F1" in dsn

    def test_missing_credentials_are_an_error_not_a_silent_localhost(self):
        with pytest.raises(ValueError):
            pg_writer.dsn_from_env({})
