"""Model rate cards and cost maths — one place for every eval script.

The pipeline counts tokens (``src/v7/usage.py``); the money is counted here.
Extracted from ``eval/generate_retrieval_gt.py`` (roadmap step 4a) so the v7
eval runner prices a query with the same rate card the GT generator used.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

# OpenAI list prices, USD per 1M tokens.
# gpt-4o / gpt-4o-mini перепроверены 05.09.2026 и не менялись; но OpenAI сняла их
# с официальной страницы цен (developers.openai.com/api/docs/pricing) — модели
# остались в API как legacy, их цену больше нельзя подтвердить первоисточником.
# Значит при следующем расхождении в счёте сверять по консоли биллинга, а не по сайту.
PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Промо-цена, действует минимум до конца ноября 2026 (базовая — $5/$30).
    # Длинный контекст (>272K) идёт вдвое дороже; наши промпты в него не входят.
    "gpt-5.6-sol": {"input": 4.00, "output": 20.00},
    "gpt-5.6-terra": {"input": 2.00, "output": 12.00},
}


def price_for(model: str) -> dict:
    """Rate card for ``model``. Unknown model is an error, not a default: pricing
    it at some other model's rate is exactly how the cost guard gets fooled."""
    try:
        return PRICE_PER_1M[model]
    except KeyError:
        known = ", ".join(sorted(PRICE_PER_1M))
        raise ValueError(
            f"No price for model {model!r}. Known: {known}. "
            "Add its rate to PRICE_PER_1M before running."
        ) from None


def calc_total_price(usages: Iterable[dict], model: str) -> float:
    """Sum USD cost from a list of ``{"input": n, "output": n}`` token counts."""
    rate = price_for(model)
    total = 0.0
    for u in usages:
        total += u.get("input", 0) / 1_000_000 * rate["input"]
        total += u.get("output", 0) / 1_000_000 * rate["output"]
    return total


def cost_for_usages(usages: Iterable[dict]) -> dict:
    """Price ``LLMUsage`` records, each by its own model.

    A model with no rate card contributes 0 and is named in
    ``unpriced_models``: a run that silently prices an unknown model at zero
    reports a cheap pipeline that isn't. The caller must surface that list.
    """
    total = 0.0
    unpriced: List[str] = []
    for u in usages:
        model = u.get("model", "unknown")
        try:
            rate = price_for(model)
        except ValueError:
            if model not in unpriced:
                unpriced.append(model)
            continue
        total += u.get("prompt_tokens", 0) / 1_000_000 * rate["input"]
        total += u.get("completion_tokens", 0) / 1_000_000 * rate["output"]
    return {"cost_usd": total, "unpriced_models": unpriced}


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (same convention as numpy's default).

    Latency is reported as p50/p95, not as a mean: the CrossEncoder in the
    complex path adds seconds to a minority of queries and a mean smears that.
    """
    data = sorted(v for v in values if v is not None)
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])
    pos = (len(data) - 1) * (p / 100.0)
    low = int(pos)
    high = min(low + 1, len(data) - 1)
    frac = pos - low
    return float(data[low] + (data[high] - data[low]) * frac)
