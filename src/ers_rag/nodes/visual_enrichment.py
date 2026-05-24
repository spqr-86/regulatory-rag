"""V7 node: visual_enrichment — enrich passages with visual context before generation.

Triggered for passages that are:
- From a table (element_type contains "Table")
- Truncated / incomplete (detect_incomplete_chunk)
- Very short (< 150 chars)

Only processes passages that have source + page_no + bbox in metadata.
Limit: MAX_VISUAL_PROOFS per query (default 3).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.infra.parsers import detect_incomplete_chunk
from src.ers_rag.state_types import RAGState

logger = logging.getLogger(__name__)

MAX_VISUAL_PROOFS = 3  # overridden in init if settings available

# ─── DI interface ────────────────────────────────────────────────────────

_visual_proof_fn: Optional[Callable[[str, int, list, str], str]] = None


def set_visual_proof_fn(fn: Optional[Callable[[str, int, list, str], str]]) -> None:
    """Inject visual proof implementation. Pass None to disable."""
    global _visual_proof_fn
    _visual_proof_fn = fn


# ─── Helpers ──────────────────────────────────────────────────────────────


def _needs_visual(passage: dict) -> bool:
    """Return True if this passage should be enriched with visual context."""
    meta = passage.get("metadata", {})
    # Must have coordinates to render
    if not (meta.get("source") and meta.get("page_no") and meta.get("bbox")):
        return False
    element_type = str(meta.get("element_type", "")).lower()
    text = passage.get("text", "")
    return (
        "table" in element_type
        or detect_incomplete_chunk(text)
        or len(text.strip()) < 150
    )


# ─── Node ─────────────────────────────────────────────────────────────────


def visual_enrichment(state: RAGState) -> RAGState:
    """Enrich final_passages with visual context before answer generation.

    Reads:  final_passages
    Writes: final_passages (updated in-place copy, only changed passages)
    No-op if: no visual_proof_fn injected, no passages, or no passages need enrichment.
    """
    fn = _visual_proof_fn
    passages = state.get("final_passages") or []

    if not fn or not passages:
        return {}

    try:
        max_proofs = MAX_VISUAL_PROOFS
        from config.settings import settings as _settings

        max_proofs = _settings.MAX_VISUAL_PROOFS
    except Exception:
        pass

    enriched = list(passages)
    count = 0

    for i, p in enumerate(enriched):
        if count >= max_proofs:
            break
        if not _needs_visual(p):
            continue

        meta = p["metadata"]
        element_type = str(meta.get("element_type", "")).lower()
        mode = "analyze" if "table" in element_type else "show"

        bbox = meta["bbox"]
        if isinstance(bbox, str):
            import json

            try:
                bbox = json.loads(bbox)
            except (ValueError, TypeError):
                logger.warning(
                    "visual_enrichment: could not parse bbox %r, skipping", bbox
                )
                continue

        try:
            result = fn(meta["source"], int(meta["page_no"]), bbox, mode)
            if not result:
                continue
            if mode == "analyze":
                enriched[i] = {
                    **p,
                    "text": p["text"] + "\n\n[Таблица — визуальный анализ]:\n" + result,
                }
            else:
                enriched[i] = {**p, "image_path": result}
            count += 1
        except Exception as exc:
            logger.warning("visual_enrichment failed for passage %d: %s", i, exc)

    if count == 0:
        return {}

    return {"final_passages": enriched}
