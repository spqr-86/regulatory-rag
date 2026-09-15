"""Anonymization gate for the public corpus (issue #36).

The stop-list lives outside the repo (it contains real names). One Python regex
per line, matched case-insensitively; `#` comments and blank lines are skipped.

Usage: python scripts/check_stoplist.py <stoplist> <corpus_dir>
Exit code 1 if any line of any text file matches.
"""

# ANCHOR: anonymization gate
# Role: block publishing corpus text that matches the private stop-list.
# Input: stop-list path (regex per line), corpus directory (.md/.txt/.yaml).
# Output: file:line: matches /pattern/ per hit; exit 1 on any hit, 2 on bad usage.
# Scope: regex catches names and known tokens only; dates and order numbers
#   still need manual review before commit.

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".txt", ".yaml", ".yml"}


@dataclass(frozen=True)
class Hit:
    path: Path
    line_no: int
    pattern: str


def load_patterns(path: str | Path) -> list[re.Pattern]:
    patterns = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


def find_hits(corpus_dir: str | Path, patterns: list[re.Pattern]) -> list[Hit]:
    hits = []
    for path in sorted(Path(corpus_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            for pattern in patterns:
                if pattern.search(line):
                    hits.append(Hit(path, line_no, pattern.pattern))
    return hits


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    hits = find_hits(argv[1], load_patterns(argv[0]))
    for hit in hits:
        # Pattern only, not the matched text: the log must not leak what it guards.
        print(f"{hit.path}:{hit.line_no}: matches /{hit.pattern}/")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
