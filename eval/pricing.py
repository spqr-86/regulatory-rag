"""Re-export of the project's single rate card — the code now lives in ``src/pricing.py``.

Moved there 05.09.2026 (issue #16): telemetry inside the pipeline must price an event with
exactly the same code as the eval runner, and ``src/`` does not import ``eval/``. Eval scripts
keep importing ``eval.pricing`` unchanged.
"""

from __future__ import annotations

from src.pricing import (  # noqa: F401
    PRICE_PER_1M,
    calc_total_price,
    cost_for_usages,
    percentile,
    price_for,
)

__all__ = [
    "PRICE_PER_1M",
    "calc_total_price",
    "cost_for_usages",
    "percentile",
    "price_for",
]
