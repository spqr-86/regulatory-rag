"""Context text the generator sees: numbered passages with source and section headers.

One place for the format, so that eval (citation judge) checks the answer against
exactly the text the generator got. Pure, no heavy imports.
"""

from __future__ import annotations

from typing import List

# ANCHOR: shared generator/judge headers preserve indexed legal addresses.
_SECTION_MAX = 120


def _score_label(score: float) -> str:
    if score >= 0.6:
        return "HIGH"
    if score >= 0.4:
        return "MED"
    return "LOW"


def _short_source(passage: dict) -> str:
    """Extract short document name from metadata.source."""
    raw = passage.get("metadata", {}).get("source", "")
    # "Трудовой кодекс РФ - Система Охрана труда. Премиальная версия.pdf"
    # → "Трудовой кодекс РФ"
    name = raw.split(" - ")[0].replace(".pdf", "").strip()
    return name or "Unknown source"


def _section_label(passage: dict) -> str:
    """Extract short section label (parent_section / heading_path) for the chunk.

    Helps the LLM and reranker distinguish similar wording across different
    normative sections (e.g. Ст.228 vs Ст.228.1).
    """
    meta = passage.get("metadata", {}) or {}
    raw = (meta.get("parent_section") or meta.get("heading_path") or "").strip()
    if not raw:
        return ""
    # Truncate very long section titles
    return raw[:_SECTION_MAX] + ("…" if len(raw) > _SECTION_MAX else "")


def _chunk_header(i: int, passage: dict) -> str:
    score = _score_label(passage.get("score", 0.0))
    src = _short_source(passage)
    section = _section_label(passage)
    fields = [f"Источник: {src}"]
    if section:
        fields.append(f"Раздел: {section}")
    meta = passage.get("metadata") or {}
    for key, label in (
        ("article", "Статья"),
        ("part", "Часть"),
        ("clause", "Пункт"),
        ("point", "Пункт"),
    ):
        if meta.get(key) is not None and meta[key] != "":
            value = str(meta[key])
            if (
                key == "part"
                and meta.get("part_end")
                and str(meta["part_end"]) != value
            ):
                value += f"–{meta['part_end']}"
            fields.append(f"{label}: {value}")
    return f"[{i + 1}] ({score}) [{'; '.join(fields)}]"


def format_generation_context(passages: List[dict]) -> str:
    """Passages → the `context` variable of the generate_answer prompt."""
    return "\n\n".join(
        f"{_chunk_header(i, p)}\n{p.get('text', '')}" for i, p in enumerate(passages)
    )
