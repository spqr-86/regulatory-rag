from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structlog once per process (idempotent).

    Uses a human-readable ConsoleRenderer on a TTY and JSONRenderer otherwise
    (e.g. when logs are piped to a file or a log collector in production).
    """
    global _configured
    if _configured:
        return

    is_tty = sys.stderr.isatty()
    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer()
        if is_tty
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    _configured = True


logger = structlog.get_logger()
