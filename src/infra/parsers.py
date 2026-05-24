"""Parsers for LLM responses, search results, and status blocks."""

from __future__ import annotations

import json
import re
from typing import List

from src.infra.types import RAGStatus, ChunkInfo


def parse_json_from_response(raw: str) -> dict:
    """Extract JSON from LLM response with multiple fallback strategies."""
    code_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if code_match:
        raw_json = code_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", raw, re.DOTALL)
        raw_json = brace_match.group(1) if brace_match else "{}"

    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        return {}


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


def parse_status_block(text: str) -> tuple[RAGStatus, str, list[str]]:
    """Parse ===STATUS===, ===ANSWER===, and ===UNANSWERED=== blocks from agent output."""
    status = RAGStatus.FOUND
    status_match = re.search(r"===STATUS===\s*\n\s*(\w+)", text)
    if status_match:
        raw_status = status_match.group(1).strip().upper()
        if raw_status in (s.value for s in RAGStatus):
            status = RAGStatus(raw_status)

    answer = text
    answer_match = re.search(
        r"===ANSWER===\s*\n(.*?)(?:\n```|===UNANSWERED===|\Z)", text, re.DOTALL
    )
    if answer_match:
        answer = answer_match.group(1).strip()

    unanswered = []
    unanswered_match = re.search(r"===UNANSWERED===\s*\n(.*?)$", text, re.DOTALL)
    if unanswered_match:
        for line in unanswered_match.group(1).strip().splitlines():
            line = line.strip().lstrip("- ")
            if line:
                unanswered.append(line)

    return status, answer, unanswered


def parse_search_results(search_output: str) -> List[ChunkInfo]:
    """Parse search_documents output into structured ChunkInfo list."""
    chunks = []

    result_pattern = re.compile(
        r"\[Result \d+\] File: ([^\|]+)\| Page: ([^\|]+)\| BBox: ([^\|\n]+)(?:\| Similarity: ([^\|\n]+))?\n"
        r"(?:Extended Context:\n)?(.*?)(?:\(IDs: [^\)]+\))?(?=\[Result|\Z)",
        re.DOTALL,
    )

    for match in result_pattern.finditer(search_output):
        source = match.group(1).strip()
        page_str = match.group(2).strip()
        bbox_str = match.group(3).strip()

        sim_str = match.group(4)
        similarity = None
        if sim_str:
            try:
                similarity = float(sim_str.strip())
            except ValueError:
                pass

        content = match.group(5).strip() if match.group(5) else ""

        try:
            page_no = int(page_str) if page_str != "N/A" else None
        except ValueError:
            page_no = None

        bbox = None
        if bbox_str and bbox_str not in ("N/A", "None"):
            try:
                bbox = json.loads(bbox_str.replace("'", '"'))
            except json.JSONDecodeError:
                pass

        # We already extracted similarity from regex group 4
        # sim_match block removed as it is now redundant

        chunks.append(
            ChunkInfo(
                content=content,
                source=source,
                page_no=page_no,
                bbox=bbox,
                visual_text=None,
                similarity=similarity,
            )
        )

    if (
        not chunks
        and search_output
        and "No relevant documents found" not in search_output
    ):
        chunks.append(
            ChunkInfo(
                content=search_output,
                source="unknown",
                page_no=None,
                bbox=None,
                visual_text=None,
            )
        )

    return chunks


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
