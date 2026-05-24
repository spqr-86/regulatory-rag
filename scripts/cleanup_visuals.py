"""LRU cleanup for static/visuals/. Run via cron — see CLAUDE.md."""

from __future__ import annotations

from pathlib import Path
import time

VISUALS_DIR = Path("static/visuals")
MAX_AGE_DAYS = 7
MAX_TOTAL_MB = 100


def cleanup() -> None:
    if not VISUALS_DIR.exists():
        return
    now = time.time()
    cutoff = now - MAX_AGE_DAYS * 86400
    # Age-based pass
    for f in VISUALS_DIR.glob("*.png"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
    # Size-based pass (LRU)
    files = sorted(
        ((f.stat().st_mtime, f.stat().st_size, f) for f in VISUALS_DIR.glob("*.png")),
        key=lambda x: x[0],
    )
    total = sum(s for _, s, _ in files)
    max_bytes = MAX_TOTAL_MB * 1024 * 1024
    while total > max_bytes and files:
        _, size, f = files.pop(0)
        f.unlink()
        total -= size


if __name__ == "__main__":
    cleanup()
