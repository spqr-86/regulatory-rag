"""One reconnecting Postgres connection, shared by the writers (issues #23, #20).

Both journals — the event row and the 👍/👎 vote — want the same thing: a
connection held open across writes, discarded and reopened once when the
container behind it has restarted. The logic lived in
:class:`~src.v7.pg_writer.PostgresWriter` first; it is here so the feedback
writer reuses it instead of growing a second copy that drifts.

Failures are raised, never swallowed: the caller decides whether monitoring is
allowed to break the answer (it is not), and swallowing here would hide the
failure from the log entirely.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import structlog

logger = structlog.get_logger()


def psycopg_connect(dsn: str) -> Any:
    """The real connection; injected away in tests, so it is the only driver call."""
    import psycopg

    return psycopg.connect(dsn)


class ReconnectingConnection:
    """Executes one statement, retrying once on a fresh connection."""

    def __init__(
        self,
        dsn: str,
        *,
        connect: Optional[Callable[[str], Any]] = None,
        log_event: str = "pg.reconnect",
    ) -> None:
        self.dsn = dsn
        self._connect = connect or psycopg_connect
        self._log_event = log_event
        self._conn: Any = None

    def execute(self, sql: str, params: tuple, **log_context: Any) -> None:
        try:
            self._execute(sql, params)
        except Exception as first:  # noqa: BLE001 — one retry on a fresh connection
            logger.info(self._log_event, error=str(first), **log_context)
            self.close()
            self._execute(sql, params)

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            conn.close()
        except Exception:  # noqa: BLE001 — closing a broken socket may throw
            pass

    def _execute(self, sql: str, params: tuple) -> None:
        conn = self._conn
        if conn is None:
            conn = self._conn = self._connect(self.dsn)
        with conn.cursor() as cur:
            cur.execute(sql, params)
        conn.commit()
