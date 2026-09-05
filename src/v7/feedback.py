"""The user's 👍/👎 on an answer (issue #20).

Why this exists: the synthetic golden set is biased in favour of BM25, so it
cannot be the only judge of quality. A vote is the one signal that comes from
outside the evaluation loop, and it is worth nothing unless it points at the
exact query it judges — hence a foreign key on ``query_id``, never a copy of the
question text.

One vote per answer. A second press on the same answer is the user correcting
themselves, not a second opinion: the row is overwritten (``ON CONFLICT``), so
the panel counts answers, not clicks. That needs the unique index from
``db/migrations/002_feedback_one_vote.sql``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import structlog

from src.v7.config import V7Config
from src.v7.pg_conn import ReconnectingConnection
from src.v7.pg_writer import dsn_from_env

logger = structlog.get_logger()

UPSERT_SQL = (
    "INSERT INTO feedback (query_id, verdict, comment) VALUES (%s, %s, %s) "
    "ON CONFLICT (query_id) DO UPDATE SET "
    "verdict = EXCLUDED.verdict, comment = EXCLUDED.comment, ts = now()"
)

VERDICTS = (1, -1)


class FeedbackWriter:
    """Writes one vote per answer over a connection kept open (see pg_conn)."""

    def __init__(
        self,
        dsn: str,
        *,
        connect: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self.dsn = dsn
        self._conn = ReconnectingConnection(
            dsn, connect=connect, log_event="feedback.pg_reconnect"
        )

    def record(
        self, query_id: str, verdict: int, comment: Optional[str] = None
    ) -> None:
        """👍 (``1``) or 👎 (``-1``) for ``query_id``, with an optional comment."""
        # bool is an int in Python, and True would silently become a 👍.
        if isinstance(verdict, bool) or verdict not in VERDICTS:
            raise ValueError(f"verdict must be 1 or -1, got {verdict!r}")
        text = (comment or "").strip() or None
        self._conn.execute(UPSERT_SQL, (query_id, verdict, text), query_id=query_id)

    def close(self) -> None:
        self._conn.close()


def default_feedback_writer(
    config: Optional[V7Config] = None,
) -> Optional[FeedbackWriter]:
    """The writer the app runs with, or ``None`` when there is nowhere to write.

    Votes live next to the events they judge, so they exist only when the events
    go to Postgres: a JSONL journal has no ``query_id`` to point a foreign key
    at. ``None`` is the signal to the UI to hide the buttons rather than offer a
    click that fails under the answer.
    """
    cfg = config or V7Config()
    if not cfg.TELEMETRY_ENABLED or cfg.TELEMETRY_WRITER != "postgres":
        return None
    try:
        dsn = cfg.TELEMETRY_PG_DSN.strip() or dsn_from_env()
    except ValueError as exc:
        logger.warning("feedback.pg_config_unusable", error=str(exc))
        return None
    return FeedbackWriter(dsn)
