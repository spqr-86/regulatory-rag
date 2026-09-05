"""One finished query → one event row (monitoring module 05, issue #16).

The graph already carries token usage in its state (``src/v7/usage.py``, key
``llm_usage``); this module turns a completed state into the event described by
the spec and hands it to an injected writer.

Two rules shape the design:

* **The writer is injected, never imported by the graph.** Tests and eval runs
  must work with no database and no network, so the pipeline knows a protocol
  with one ``write`` method, not Postgres.
* **Monitoring may not break the answer.** Anything the writer throws is logged
  and swallowed by :func:`write_event`; the user's reply is never lost to a
  telemetry failure.

Three values do not exist in ``RAGState`` and are passed in by the caller (the
graph wrapper, issue #17): ``source``, ``latency_ms`` and — when the caller
already showed it to the user — ``query_id``.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Protocol

import structlog

from src.pricing import cost_for_usages

logger = structlog.get_logger()

Source = Literal["ui", "api", "eval", "mcp"]
Path_ = Literal["simple", "complex", "clarify", "abstain"]

# The spec's event row, in the order the table declares it.
EVENT_FIELDS = (
    "query_id",
    # Which batch run wrote the row; ``None`` for live traffic (issue #18).
    "run_id",
    "ts",
    "source",
    "question",
    "path",
    "answer_len",
    "n_passages",
    "n_passages_found",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "cost_usd",
    "models",
    "unpriced_models",
    "error",
)


def new_query_id() -> str:
    """The id the row is written under and the UI shows to the user."""
    return str(uuid.uuid4())


def resolve_path(state: Dict[str, Any]) -> Path_:
    """Which route the query took — derived, because the state has no such field.

    Order matters: a query that escalated to the complex path and then refused to
    answer is an ``abstain``, not a ``complex``. What the user got back wins over
    how much work it took to get there.
    """
    if state.get("abstain_reason"):
        return "abstain"
    if state.get("clarify_message"):
        return "clarify"
    for attempt in state.get("retrieval_attempts") or []:
        if attempt.get("stage") == "complex":
            return "complex"
    return "simple"


def build_event(
    state: Dict[str, Any],
    *,
    source: Source,
    latency_ms: int,
    query_id: str | None = None,
    run_id: str | None = None,
    error: str | None = None,
    ts: datetime | None = None,
) -> Dict[str, Any]:
    """Collect the event row from a finished graph state.

    Missing counts are zeros, never ``None``: a row with holes in it is worse
    than a row that honestly says nothing happened. Two fields are allowed to be
    ``None`` and mean it: ``error`` (that is what a successful query looks like)
    and ``run_id`` (a live query belongs to no batch run).
    """
    usage: List[dict] = list(state.get("llm_usage") or [])
    found = len(state.get("final_passages") or [])
    priced = cost_for_usages(usage)

    return {
        "query_id": query_id or new_query_id(),
        "run_id": run_id,
        "ts": ts or datetime.now(timezone.utc),
        "source": source,
        "question": state.get("query", ""),
        "path": resolve_path(state),
        "answer_len": len(state.get("answer") or ""),
        # What actually went into the prompt, per the spec. Retrieval's own count
        # is kept beside it: the two differ whenever cross-refs expanded the set.
        "n_passages": _passages_sent_to_llm(usage, found),
        "n_passages_found": found,
        "latency_ms": int(latency_ms),
        "prompt_tokens": sum(int(u.get("prompt_tokens", 0)) for u in usage),
        "completion_tokens": sum(int(u.get("completion_tokens", 0)) for u in usage),
        "cost_usd": priced["cost_usd"],
        "models": usage,
        # A model with no rate card is named, not silently priced at zero.
        "unpriced_models": priced["unpriced_models"],
        "error": error,
    }


def _passages_sent_to_llm(usage: List[dict], found: int) -> int:
    """How many passages reached the model's prompt.

    Reported by the generate call itself (issue #22). A generator that reports
    nothing — the stub, or the pre-#22 contract — falls back to the retrieved
    count: a zero printed next to a real answer reads as a bug, and the fallback
    is exactly what the field used to mean. No generation at all means zero.
    """
    reported = [u["n_passages"] for u in usage if "n_passages" in u]
    if reported:
        return sum(int(n) for n in reported)
    return found


class EventWriter(Protocol):
    """Anything that can persist an event. Postgres arrives in #15/#17."""

    def write(self, event: Dict[str, Any]) -> None: ...


class JsonlWriter:
    """Append-only JSONL journal — the fallback that works without a database.

    One event is one ``write`` of a single line, so two processes (Streamlit and
    the API) appending to the same file do not tear each other's rows.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, event: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def write_event(writer: EventWriter, event: Dict[str, Any]) -> bool:
    """Write the event, swallowing any failure. Returns whether it landed.

    Deliberately catches everything: a driver raising something exotic must not
    reach the user, and there is no failure of monitoring worth a lost answer.
    """
    try:
        writer.write(event)
        return True
    except Exception as exc:  # noqa: BLE001 — monitoring must never propagate
        logger.warning(
            "telemetry.write_failed",
            query_id=event.get("query_id"),
            writer=type(writer).__name__,
            error=str(exc),
        )
        return False
