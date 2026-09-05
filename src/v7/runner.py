"""The single door into the v7 pipeline (monitoring module 05, issue #17).

Every entry point — Streamlit, FastAPI, the MCP server, the eval runner — calls
:func:`run_query` instead of ``graph.invoke`` directly, so one query produces
exactly one telemetry row no matter who asked.

Why here and not inside the graph: a terminal node would have to be patched in
four places (noise, clarify, answer, abstain) and none of them knows how long
the whole query took — a node sees only its own step. Why not FastAPI
middleware: ``app.py`` calls the graph directly, bypassing ``api.py``, so
middleware would miss the Streamlit traffic entirely.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from src.v7 import telemetry
from src.v7.config import V7Config, v7_config


def default_writer(config: Optional[V7Config] = None) -> Optional[telemetry.EventWriter]:
    """The writer the app runs with: a JSONL journal, or nothing when disabled.

    Config is read at call time (not import time) so a test or a deployment can
    flip the switch without reimporting the module. Postgres arrives in #15 and
    replaces the return value here, not the call sites.
    """
    cfg = config or V7Config()
    if not cfg.TELEMETRY_ENABLED:
        return None
    return telemetry.JsonlWriter(cfg.TELEMETRY_JSONL_PATH)


def run_query(
    pipeline: Any,
    query: str,
    *,
    source: telemetry.Source,
    writer: Optional[telemetry.EventWriter] = None,
    query_id: Optional[str] = None,
    run_id: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    """Run one query through the compiled graph and record what happened.

    ``source`` is keyword-only and has no default on purpose: a row that says
    "ui" because nobody passed anything is worse than a missing row — it makes
    eval traffic look like users.

    ``run_id`` groups the rows of one batch run (the eval runner passes its own,
    issue #18). Live traffic leaves it ``None``: there is no run behind it.

    Returns the graph result and the ``query_id`` of the row, which the caller
    shows to the user so feedback (#20) can point back at it.
    """
    event_id = query_id or telemetry.new_query_id()
    started = time.perf_counter()

    try:
        result = pipeline.invoke(_graph_input(query, filters))
    except Exception as exc:
        # The failure is recorded, then handed on: monitoring must not turn a
        # broken query into a silent success.
        _record(
            writer,
            {},
            source=source,
            latency_ms=_elapsed_ms(started),
            query_id=event_id,
            run_id=run_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    _record(
        writer,
        result or {},
        source=source,
        latency_ms=_elapsed_ms(started),
        query_id=event_id,
        run_id=run_id,
    )
    return result, event_id


def _graph_input(query: str, filters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    state: Dict[str, Any] = {"query": query}
    if filters:
        state["filters"] = filters
    return state


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _record(
    writer: Optional[telemetry.EventWriter],
    state: Dict[str, Any],
    **kwargs: Any,
) -> None:
    """No writer means telemetry is off — not an error, and not a warning."""
    if writer is None:
        return
    event = telemetry.build_event(state, **kwargs)
    telemetry.write_event(writer, event)
