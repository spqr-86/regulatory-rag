"""Event rows go to Postgres instead of the JSONL journal (issue #23).

The missing link between the telemetry module (#16), which builds the event, and
the stack from #15, whose ``queries`` table is shaped exactly like
``EVENT_FIELDS``. Nothing above this module changes: the writer protocol is the
one ``run_query`` already injects, so switching journals is a config value.

Two rules carried over from #16:

* **Monitoring may not break the answer.** This class raises on a dead database
  — swallowing belongs to :func:`telemetry.write_event`, which logs it and lets
  the user's reply through. Keeping the raise here is what makes the failure
  visible in the log at all.
* **The column list is derived from EVENT_FIELDS, never retyped.** A field added
  to the event and not to the migration fails loudly on the insert instead of
  being dropped silently.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import quote

import structlog
from psycopg.types.json import Jsonb

from src.v7.telemetry import EVENT_FIELDS

logger = structlog.get_logger()

# Columns whose Python value is a dict or list bound for a jsonb column: psycopg
# adapts a bare list to a Postgres array, which the jsonb column would reject.
_JSON_FIELDS = ("models",)

_COLUMNS = ", ".join(EVENT_FIELDS)
_PLACEHOLDERS = ", ".join(["%s"] * len(EVENT_FIELDS))
# The id is the event's own uuid, so a replayed journal (the ingest script) and a
# retried write both land once.
INSERT_SQL = (
    f"INSERT INTO queries ({_COLUMNS}) VALUES ({_PLACEHOLDERS}) "
    "ON CONFLICT (query_id) DO NOTHING"
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = "5432"


def dsn_from_env(env: Optional[Mapping[str, str]] = None) -> str:
    """Where to write: an explicit DSN, else the compose credentials.

    ``V7_TELEMETRY_PG_DSN`` covers a database that is not the local stack. With
    it empty the DSN is built from the same ``POSTGRES_*`` variables
    ``docker-compose.yml`` reads, so the password lives in ``.env`` once.
    """
    env = os.environ if env is None else env

    explicit = (env.get("V7_TELEMETRY_PG_DSN") or "").strip()
    if explicit:
        return explicit

    user = env.get("POSTGRES_USER")
    password = env.get("POSTGRES_PASSWORD")
    db = env.get("POSTGRES_DB")
    if not (user and password and db):
        raise ValueError(
            "Postgres telemetry needs V7_TELEMETRY_PG_DSN, or POSTGRES_USER, "
            "POSTGRES_PASSWORD and POSTGRES_DB (see .env.example). "
            "Refusing to guess a connection."
        )

    host = env.get("POSTGRES_HOST") or DEFAULT_HOST
    port = env.get("POSTGRES_PORT") or DEFAULT_PORT
    # A generated password contains /, @ and : often enough to matter.
    return (
        f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
        f"@{host}:{port}/{quote(db, safe='')}"
    )


def _row(event: Dict[str, Any]) -> tuple:
    """The event as insert parameters, in the column order of EVENT_FIELDS."""
    return tuple(
        Jsonb(event.get(field)) if field in _JSON_FIELDS else event.get(field)
        for field in EVENT_FIELDS
    )


class PostgresWriter:
    """Writes one row per event, holding a single connection open.

    Connecting per event would add a handshake to every user-visible query, so
    the connection is kept and reopened once when it turns out to be dead — the
    ordinary fate of a long-lived connection to a container that restarted.
    """

    def __init__(
        self,
        dsn: str,
        *,
        connect: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.dsn = dsn
        self._connect = connect or _psycopg_connect
        self._conn: Any = None

    def write(self, event: Dict[str, Any]) -> None:
        row = _row(event)
        try:
            self._insert(row)
        except Exception as first:  # noqa: BLE001 — one retry on a fresh connection
            logger.info(
                "telemetry.pg_reconnect",
                query_id=event.get("query_id"),
                error=str(first),
            )
            self._discard()
            self._insert(row)

    def close(self) -> None:
        self._discard()

    def _insert(self, row: tuple) -> None:
        conn = self._conn
        if conn is None:
            conn = self._conn = self._connect(self.dsn)
        with conn.cursor() as cur:
            cur.execute(INSERT_SQL, row)
        conn.commit()

    def _discard(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — closing a broken socket may throw
            pass


def _psycopg_connect(dsn: str) -> Any:
    """The real connection; injected away in tests, so it is the only driver call."""
    import psycopg

    return psycopg.connect(dsn)
