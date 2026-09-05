#!/usr/bin/env python3
"""Replay the JSONL journal into the Postgres ``queries`` table (issue #23).

For the events written before the stack existed, and for whatever the journal
collected while the database was down. Safe to run twice: the insert carries
``ON CONFLICT (query_id) DO NOTHING``, so a row that already landed is skipped
by Postgres rather than by a lookup here — one round trip, no read-then-write
race between two runs.

    python scripts/ingest_events.py logs/events.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v7 import pg_writer  # noqa: E402  — after the path fix above


class Writer(Protocol):
    def write(self, event: Dict[str, Any]) -> None: ...


@dataclass
class Report:
    """What the run did, for the operator reading the last line."""

    written: int = 0
    skipped: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return (
            f"offered {self.written}, skipped {self.skipped} unreadable, "
            f"failed {self.failed}"
        )


def _rows(path: Path) -> Iterator[tuple[int, str]]:
    with path.open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if line:
                yield number, line


def _fill_found_count(event: Dict[str, Any]) -> Dict[str, Any]:
    """Give a pre-#22 event the field the table requires.

    Before the split, ``n_passages`` counted the passages retrieval found; the
    column that now holds that number did not exist. Copying it over is not a
    guess — it is the same measurement under its later name. An event with
    neither count is left as it is: Postgres rejecting it is better than a
    number invented here.
    """
    if event.get("n_passages_found") is None and event.get("n_passages") is not None:
        event["n_passages_found"] = event["n_passages"]
    return event


def ingest(journal: str | Path, writer: Writer) -> Report:
    """Offer every readable event to the writer; never stop on one bad row."""
    path = Path(journal)
    if not path.exists():
        raise FileNotFoundError(f"journal not found: {path}")

    report = Report()
    for number, line in _rows(path):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"line {number}: not JSON, skipped", file=sys.stderr)
            report.skipped += 1
            continue

        # Without an id the row cannot be deduplicated, and a replay would
        # multiply it on every run.
        if not isinstance(event, dict) or not event.get("query_id"):
            print(f"line {number}: no query_id, skipped", file=sys.stderr)
            report.skipped += 1
            continue

        try:
            writer.write(_fill_found_count(event))
        except Exception as exc:  # noqa: BLE001 — one bad row is not the run
            print(f"line {number}: insert failed ({exc})", file=sys.stderr)
            report.failed += 1
            continue
        report.written += 1

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "journal",
        nargs="?",
        default="logs/events.jsonl",
        help="JSONL journal to replay (default: logs/events.jsonl)",
    )
    parser.add_argument(
        "--dsn",
        default="",
        help="Postgres DSN; default is built from the POSTGRES_* variables in .env",
    )
    args = parser.parse_args(argv)

    writer = pg_writer.PostgresWriter(args.dsn.strip() or pg_writer.dsn_from_env())
    try:
        report = ingest(args.journal, writer)
    finally:
        writer.close()

    print(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
