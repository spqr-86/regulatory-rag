"""Parsers for LLM responses."""

from __future__ import annotations

import re


def extract_text(content) -> str:
    """Extract plain text from AIMessage content (str or Gemini-style list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def detect_incomplete_chunk(text: str) -> bool:
    """
    Detects if a text chunk is likely incomplete or requires visual analysis.
    This is a heuristic and can be improved with more sophisticated NLP.
    """
    text = text.strip()
    if not text:
        return False

    # Check for common continuation markers
    continuation_markers = [
        "...",
        "продолжение следует",
        "см. далее",
        "таблица",
        "схема",
    ]
    if any(marker in text.lower() for marker in continuation_markers):
        return True

    # Check if it ends abruptly without punctuation
    if not re.search(r"[.?!;:]$", text) and len(text.split()) > 5:
        return True

    # Check for bullet points or numbered lists that don't start at the beginning
    if re.search(r"^\s*[-*\d]+\s", text, re.MULTILINE) and not re.match(
        r"^\s*(1\.|-|\*)\s", text
    ):
        return True

    return False
