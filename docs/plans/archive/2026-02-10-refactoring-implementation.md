# Project Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Incremental refactoring — remove duplication, fix latency, eliminate global state, clean architecture across 5 phases.

**Architecture:** Bottom-up approach: extract utilities first, then fix performance, then improve testability/configurability, then clean providers, then polish UI. Each phase leaves system working.

**Tech Stack:** Python 3.11+, LangChain, LangGraph, ChromaDB, Streamlit, Pydantic Settings

---

## Phase 1: Extract Utilities & Remove Duplication

### Task 1.1: Create `src/chroma_helpers.py` — Deduplicate ChromaDB queries

**Files:**
- Create: `src/chroma_helpers.py`
- Create: `tests/test_chroma_helpers.py`
- Modify: `src/agent_tools.py:38-78,222-258`

**Step 1: Write the failing tests**

```python
# tests/test_chroma_helpers.py
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.chroma_helpers import query_chunks_by_range, chroma_results_to_documents


class TestChromaResultsToDocuments:
    def test_converts_results_to_documents(self):
        result = {
            "documents": ["text1", "text2"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }
        docs = chroma_results_to_documents(result)
        assert len(docs) == 2
        assert docs[0].page_content == "text1"
        assert docs[0].metadata == {"source": "a.pdf"}

    def test_empty_results(self):
        assert chroma_results_to_documents(None) == []
        assert chroma_results_to_documents({"documents": None}) == []
        assert chroma_results_to_documents({"documents": []}) == []


class TestQueryChunksByRange:
    def test_queries_chroma_and_returns_sorted_docs(self):
        mock_vs = MagicMock()
        mock_vs.get.return_value = {
            "documents": ["chunk2", "chunk1"],
            "metadatas": [
                {"source": "a.pdf", "chunk_id": 5},
                {"source": "a.pdf", "chunk_id": 3},
            ],
        }
        docs = query_chunks_by_range(mock_vs, "a.pdf", 3, 5)
        assert len(docs) == 2
        assert docs[0].metadata["chunk_id"] == 3  # sorted
        mock_vs.get.assert_called_once_with(
            where={
                "$and": [
                    {"source": "a.pdf"},
                    {"chunk_id": {"$gte": 3}},
                    {"chunk_id": {"$lte": 5}},
                ]
            }
        )

    def test_returns_empty_on_error(self):
        mock_vs = MagicMock()
        mock_vs.get.side_effect = Exception("DB error")
        assert query_chunks_by_range(mock_vs, "a.pdf", 1, 5) == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chroma_helpers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.chroma_helpers'`

**Step 3: Write minimal implementation**

```python
# src/chroma_helpers.py
"""Shared helpers for ChromaDB operations — deduplicates query logic."""
from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def chroma_results_to_documents(result: Optional[dict]) -> List[Document]:
    """Convert raw Chroma .get() result dict to a list of Documents."""
    if not result or not result.get("documents"):
        return []
    docs = []
    for i, text in enumerate(result["documents"]):
        meta = result["metadatas"][i] if result.get("metadatas") else {}
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def query_chunks_by_range(
    vs, source: str, start: int, end: int
) -> List[Document]:
    """Query Chroma for chunks in [start, end] range for a given source.

    Returns Documents sorted by chunk_id. Returns [] on error.
    """
    try:
        result = vs.get(
            where={
                "$and": [
                    {"source": source},
                    {"chunk_id": {"$gte": start}},
                    {"chunk_id": {"$lte": end}},
                ]
            }
        )
        docs = chroma_results_to_documents(result)
        docs.sort(key=lambda x: x.metadata.get("chunk_id", 0))
        return docs
    except Exception as e:
        logger.error("Error querying chunks range %d-%d for %s: %s", start, end, source, e)
        return []
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chroma_helpers.py -v`
Expected: PASS (all 5 tests)

**Step 5: Refactor `agent_tools.py` to use new helpers**

Replace `_fetch_neighboring_chunks` (lines 38-78) to use `query_chunks_by_range`:

```python
# In agent_tools.py — replace _fetch_neighboring_chunks body
from src.chroma_helpers import query_chunks_by_range, chroma_results_to_documents

def _fetch_neighboring_chunks(
    base_id: int, source: str, window: int = 2
) -> List[Document]:
    """Fetch chunks with IDs in range [base_id - window, base_id + window]."""
    vs = load_vector_store()
    min_id = max(0, base_id - window)
    max_id = base_id + window
    return query_chunks_by_range(vs, source, min_id, max_id)
```

Replace the inline duplicate in `search_documents` (lines 228-250) to use `query_chunks_by_range`:

```python
# In search_documents, inside "for start, end in ranges:" loop — replace lines 228-250 with:
try:
    vs = load_vector_store()
    range_docs = query_chunks_by_range(vs, source, start, end)
    if range_docs:
        merged = _merge_chunks(range_docs)
        if merged:
            final_blocks.append(merged)
except Exception as e:
    print(f"Error processing range {start}-{end} for {source}: {e}")
    continue
```

Remove the `from src.vector_store import load_vector_store` duplicate import inside the function (line 230).

**Step 6: Run existing tests**

Run: `pytest tests/ -v`
Expected: All existing tests pass (no behavior change)

**Step 7: Commit**

```bash
git add src/chroma_helpers.py tests/test_chroma_helpers.py src/agent_tools.py
git commit -m "refactor: extract chroma_helpers to deduplicate ChromaDB query logic"
```

---

### Task 1.2: Create `src/parsers.py` — Extract parsers from multiagent_rag.py

**Files:**
- Create: `src/parsers.py`
- Create: `tests/test_parsers.py`
- Modify: `agents/multiagent_rag.py:181-299`

**Step 1: Write the failing tests**

```python
# tests/test_parsers.py
import pytest
from src.parsers import (
    parse_json_from_response,
    extract_text,
    parse_status_block,
    parse_search_results,
)


class TestParseJsonFromResponse:
    def test_parses_markdown_code_block(self):
        raw = '```json\n{"status": "approved"}\n```'
        assert parse_json_from_response(raw) == {"status": "approved"}

    def test_parses_bare_json(self):
        raw = 'Some text {"key": "value"} more text'
        assert parse_json_from_response(raw) == {"key": "value"}

    def test_returns_empty_dict_on_invalid(self):
        assert parse_json_from_response("no json here") == {}


class TestExtractText:
    def test_string_passthrough(self):
        assert extract_text("hello") == "hello"

    def test_gemini_style_blocks(self):
        content = [{"text": "part1"}, {"text": "part2"}]
        assert extract_text(content) == "part1\npart2"

    def test_mixed_list(self):
        content = [{"text": "block"}, "plain"]
        assert extract_text(content) == "block\nplain"

    def test_fallback_to_str(self):
        assert extract_text(42) == "42"


class TestParseStatusBlock:
    def test_parses_all_sections(self):
        text = (
            "===STATUS===\nFOUND\n"
            "===ANSWER===\nОтвет тут\n"
            "===UNANSWERED===\n- вопрос 1\n- вопрос 2"
        )
        status, answer, unanswered = parse_status_block(text)
        assert status.value == "FOUND"
        assert answer == "Ответ тут"
        assert unanswered == ["вопрос 1", "вопрос 2"]

    def test_defaults_when_no_markers(self):
        status, answer, unanswered = parse_status_block("Just plain text")
        assert status.value == "FOUND"
        assert answer == "Just plain text"
        assert unanswered == []


class TestParseSearchResults:
    def test_parses_structured_results(self):
        text = (
            "[Result 0] File: doc.pdf | Page: 5 | BBox: [1,2,3,4]\n"
            "Extended Context:\nSome content here\n(IDs: [10, 11])"
        )
        chunks = parse_search_results(text)
        assert len(chunks) == 1
        assert chunks[0]["source"] == "doc.pdf"
        assert chunks[0]["page_no"] == 5
        assert chunks[0]["content"] == "Some content here"

    def test_fallback_for_unstructured(self):
        chunks = parse_search_results("Some relevant text without structure")
        assert len(chunks) == 1
        assert chunks[0]["source"] == "unknown"

    def test_no_results_found(self):
        chunks = parse_search_results("No relevant documents found")
        assert chunks == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parsers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.parsers'`

**Step 3: Write minimal implementation**

Move functions from `agents/multiagent_rag.py` lines 181-299 to `src/parsers.py`:

```python
# src/parsers.py
"""Parsers for LLM responses, search results, and status blocks."""
from __future__ import annotations

import json
import re
from typing import List, Optional

# Import enums from multiagent_rag to avoid circular deps — use string values
# These will be imported by consumers who need them
from agents.multiagent_rag import RAGStatus, ChunkInfo


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
    answer_match = re.search(r"===ANSWER===\s*\n(.*?)(?:===UNANSWERED===|\Z)", text, re.DOTALL)
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
        r"\[Result \d+\] File: ([^\|]+)\| Page: ([^\|]+)\| BBox: ([^\n]+)\n"
        r"(?:Extended Context:\n)?(.*?)(?:\(IDs: [^\)]+\))?(?=\[Result|\Z)",
        re.DOTALL,
    )

    for match in result_pattern.finditer(search_output):
        source = match.group(1).strip()
        page_str = match.group(2).strip()
        bbox_str = match.group(3).strip()
        content = match.group(4).strip() if match.group(4) else ""

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

        chunks.append(
            ChunkInfo(
                content=content, source=source, page_no=page_no,
                bbox=bbox, visual_text=None,
            )
        )

    if not chunks and search_output and "No relevant documents found" not in search_output:
        chunks.append(
            ChunkInfo(
                content=search_output, source="unknown",
                page_no=None, bbox=None, visual_text=None,
            )
        )

    return chunks
```

**IMPORTANT:** To avoid circular imports (parsers.py importing from multiagent_rag.py), we need to move `RAGStatus`, `ChunkInfo`, and other shared types to a separate file. Create `src/types.py`:

```python
# src/types.py
"""Shared types and enums for the multi-agent RAG system."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, TypedDict


class RAGStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    PARTIAL = "PARTIAL"


class VerifyStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"


class RouteType(str, Enum):
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"
    RAG = "rag"


class ChunkInfo(TypedDict):
    content: str
    source: str
    page_no: Optional[int]
    bbox: Optional[List[float]]
    visual_text: Optional[str]
```

Then update imports:
- `src/parsers.py`: `from src.types import RAGStatus, ChunkInfo`
- `agents/multiagent_rag.py`: `from src.types import RAGStatus, VerifyStatus, RouteType, ChunkInfo` and `from src.parsers import parse_json_from_response, extract_text, parse_status_block, parse_search_results`
- Remove the original function definitions and enum/TypedDict definitions from `multiagent_rag.py` (lines 134-299)

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parsers.py -v`
Expected: PASS (all 11 tests)

**Step 5: Update `multiagent_rag.py` imports and remove extracted code**

Replace local references:
- `_parse_json_from_response` → `parse_json_from_response` (line 480)
- `_extract_text` → `extract_text` (lines 480, 544, 551, 569, 577)
- `_parse_status_block` → `parse_status_block` (line 570)
- `_parse_search_results` → `parse_search_results` (line 544)

Delete lines 134-299 (enum definitions, TypedDict definitions, parser functions) from `multiagent_rag.py`.

**Step 6: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add src/types.py src/parsers.py tests/test_parsers.py agents/multiagent_rag.py
git commit -m "refactor: extract parsers and shared types from multiagent_rag"
```

---

### Task 1.3: Deduplicate `final_chain.py` — Retriever factory

**Files:**
- Modify: `src/final_chain.py:44-94`

**Step 1: Write the failing test**

```python
# tests/test_final_chain.py
import pytest
from unittest.mock import MagicMock, patch


@patch("src.final_chain.Ranker")
@patch("src.final_chain.get_llm")
def test_build_reranked_retriever_creates_retriever(mock_get_llm, mock_ranker):
    from src.final_chain import build_reranked_retriever

    mock_vs = MagicMock()
    mock_bm25 = MagicMock()
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm

    retriever = build_reranked_retriever(mock_vs, mock_bm25, mock_llm)
    assert retriever is not None


@patch("src.final_chain.Ranker")
@patch("src.final_chain.get_llm")
def test_build_reranked_retriever_respects_query_expansion_flag(mock_get_llm, mock_ranker):
    from src.final_chain import build_reranked_retriever

    mock_vs = MagicMock()
    mock_bm25 = MagicMock()
    mock_llm = MagicMock()

    r1 = build_reranked_retriever(mock_vs, mock_bm25, mock_llm, query_expansion=True)
    r2 = build_reranked_retriever(mock_vs, mock_bm25, mock_llm, query_expansion=False)
    # Both should return a ContextualCompressionRetriever
    assert r1 is not None
    assert r2 is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_final_chain.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_reranked_retriever'`

**Step 3: Write implementation — extract factory function**

Add to `src/final_chain.py`:

```python
def build_reranked_retriever(vector_store, bm25_retriever, llm, query_expansion=True):
    """Build an ApplicabilityRetriever with FlashRank reranking.

    Args:
        vector_store: Chroma vector store
        bm25_retriever: BM25Retriever instance
        llm: LLM for query expansion
        query_expansion: Whether to use LLM query expansion (disable for agent mode)
    """
    ensemble = ApplicabilityRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        llm=llm,
        search_kwargs={"k": settings.VECTOR_SEARCH_K},
        query_expansion=query_expansion,
    )
    flashrank_client = Ranker(
        model_name=settings.RERANKING_MODEL,
        cache_dir=getattr(settings, "FLASHRANK_CACHE_DIR", None),
    )
    compressor = FlashrankRerank(client=flashrank_client, top_n=12)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble
    )
```

Then refactor `create_final_hybrid_chain()` to use it:

```python
def create_final_hybrid_chain():
    print("Создание финальной гибридной RAG-цепочки...")
    vector_store = load_vector_store()

    semantic_retriever = vector_store.as_retriever(
        search_kwargs={"k": settings.VECTOR_SEARCH_K}
    )

    all_data = vector_store.get(include=["metadatas", "documents"])
    all_docs_as_objects = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    keyword_retriever = BM25Retriever.from_documents(all_docs_as_objects)
    keyword_retriever.k = settings.VECTOR_SEARCH_K

    llm = get_llm()

    final_retriever = build_reranked_retriever(vector_store, keyword_retriever, llm, query_expansion=True)

    prompt_manager = PromptManager()

    def render_prompt(inputs):
        text = prompt_manager.render("final_chain", **inputs)
        return [HumanMessage(content=text)]

    final_chain = (
        {
            "context": itemgetter("question") | final_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | RunnableLambda(render_prompt)
        | llm
        | StrOutputParser()
    )

    agent_reranker = build_reranked_retriever(vector_store, keyword_retriever, llm, query_expansion=False)

    print("Финальная гибридная цепочка успешно создана.")
    return final_chain, final_retriever, agent_reranker
```

**Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add src/final_chain.py tests/test_final_chain.py
git commit -m "refactor: extract build_reranked_retriever factory in final_chain"
```

---

### Task 1.4: Lint and verify Phase 1

**Step 1: Run linters**

```bash
black . && ruff check . --fix
```

**Step 2: Run all tests**

```bash
pytest tests/ -v
```

**Step 3: Commit lint fixes if any**

```bash
git add -u
git commit -m "style: lint fixes after Phase 1 refactoring"
```

---

## Phase 2: Fix Latency & Inefficiencies

### Task 2.1: Move `load_vector_store()` out of nested loop in `agent_tools.py`

**Files:**
- Modify: `src/agent_tools.py:131-276`

**Step 1: Refactor search_documents**

Move `vs = load_vector_store()` from inside the `for start, end in ranges:` loop to the top of the function, right after the retriever check. Pass `vs` to `query_chunks_by_range`.

The key change in `search_documents`:

```python
@tool
def search_documents(query: str) -> str:
    """..."""
    global _retriever, _search_call_count
    if not _retriever:
        return "Error: Retriever not initialized."

    _search_call_count += 1
    if _search_call_count > MAX_SEARCH_CALLS:
        return (...)

    initial_docs = _retriever.invoke(query)
    if not initial_docs:
        return "No relevant documents found."

    # ... hits grouping logic stays the same ...

    # Load vector store ONCE for all range queries
    vs = load_vector_store()

    for source, cids in hits_by_source.items():
        cids.sort()
        # ... range merging logic stays the same ...

        for start, end in ranges:
            try:
                range_docs = query_chunks_by_range(vs, source, start, end)
                if range_docs:
                    merged = _merge_chunks(range_docs)
                    if merged:
                        final_blocks.append(merged)
            except Exception as e:
                print(f"Error processing range {start}-{end} for {source}: {e}")
                continue

    # ... format output stays the same ...
```

**Step 2: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add src/agent_tools.py
git commit -m "perf: move load_vector_store() out of nested loop in search_documents"
```

---

### Task 2.2: Cache glossary loading in `multiagent_rag.py`

**Files:**
- Modify: `agents/multiagent_rag.py:67-97`

**Step 1: Add `@lru_cache` to glossary loading and pre-compile patterns**

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def _load_glossary(path: str = str(GLOSSARY_PATH)) -> dict[str, str]:
    """Load term glossary (cached — called once per process)."""
    p = Path(path)
    if not p.exists():
        logger.warning("Term glossary not found at %s", p)
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        terms = data.get("terms", {}) if data else {}
        return {key.lower(): val["official"] for key, val in terms.items() if "official" in val}
    except Exception as e:
        logger.error("Failed to load glossary: %s", e)
        return {}


@lru_cache(maxsize=1)
def _compiled_glossary_patterns() -> list[tuple[re.Pattern, str, str]]:
    """Pre-compile regex patterns for all glossary terms (cached)."""
    glossary = _load_glossary()
    patterns = []
    for short_term, official in glossary.items():
        pattern = _make_term_pattern(short_term)
        patterns.append((pattern, short_term, official))
    return patterns


def _expand_query(query: str) -> str:
    """Expand unofficial abbreviations using cached glossary patterns."""
    patterns = _compiled_glossary_patterns()
    if not patterns:
        return query
    expansions = []
    for pattern, short_term, official in patterns:
        if pattern.search(query):
            expansions.append(f"{short_term} → {official}")
    if expansions:
        return query + "\n\n[Глоссарий: " + "; ".join(expansions) + "]"
    return query
```

Note: `_load_glossary` signature changes to accept `str` instead of `Path` (because `@lru_cache` needs hashable args). Update `__init__` accordingly:
```python
# In __init__:
# Remove: self.glossary = _load_glossary()
# In invoke:
# Change: expanded_query = _expand_query(query, self.glossary)
# To:     expanded_query = _expand_query(query)
```

**Step 2: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add agents/multiagent_rag.py
git commit -m "perf: cache glossary loading and pre-compile term regex patterns"
```

---

### Task 2.3: Reuse LLM instance in `final_chain.py`

**Files:**
- Modify: `src/final_chain.py`

**Step 1: Use single LLM instance**

Already done in Task 1.3 — `create_final_hybrid_chain` now creates `llm = get_llm()` once and passes it to both `build_reranked_retriever` calls and the chain. Verify no duplicate `get_llm()` calls remain.

**Step 2: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

---

### Task 2.4: Lint and verify Phase 2

**Step 1: Run linters**

```bash
black . && ruff check . --fix
```

**Step 2: Run all tests**

```bash
pytest tests/ -v
```

**Step 3: Commit if needed**

```bash
git add -u
git commit -m "style: lint fixes after Phase 2 performance refactoring"
```

---

## Phase 3: Remove Global State & Hardcoded Constants

### Task 3.1: Move hardcoded constants to `config/settings.py`

**Files:**
- Modify: `config/settings.py`
- Modify: `agents/multiagent_rag.py:34-38`
- Modify: `src/agent_tools.py:22-23`

**Step 1: Add settings fields**

Add to `Settings` class in `config/settings.py`:

```python
    # Agent workflow settings
    THINKING_BUDGET: int = 8192
    THINKING_VERIFIER: int = 1024
    MAX_REVISIONS: int = 1
    MAX_AGENT_STEPS: int = 16
    MAX_SEARCH_CALLS: int = 2
    MAX_VISUAL_PROOF_CALLS: int = 1
```

**Step 2: Update consumers**

In `agents/multiagent_rag.py`, remove lines 35-38 and replace usages:
- `THINKING_BUDGET` → `settings.THINKING_BUDGET`
- `THINKING_VERIFIER` → `settings.THINKING_VERIFIER`
- `MAX_REVISIONS` → `settings.MAX_REVISIONS`
- `MAX_AGENT_STEPS` → `settings.MAX_AGENT_STEPS`

In `src/agent_tools.py`, remove lines 22-23 and replace:
- `MAX_SEARCH_CALLS` → `settings.MAX_SEARCH_CALLS`
- `MAX_VISUAL_PROOF_CALLS` → `settings.MAX_VISUAL_PROOF_CALLS`

Add `from config.settings import settings` if not already imported.

**Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add config/settings.py agents/multiagent_rag.py src/agent_tools.py
git commit -m "refactor: move hardcoded constants to pydantic Settings"
```

---

### Task 3.2: Replace global state with `ToolContext`

**Files:**
- Modify: `src/agent_tools.py`
- Modify: `agents/multiagent_rag.py`
- Modify: `tests/test_agent_tools.py`

**Step 1: Write failing test**

```python
# Add to tests/test_agent_tools.py
from src.agent_tools import create_tool_context, make_tools


def test_tool_context_isolation():
    """Two contexts don't share state."""
    ctx1 = create_tool_context(retriever=MagicMock())
    ctx2 = create_tool_context(retriever=MagicMock())
    ctx1.search_call_count = 5
    assert ctx2.search_call_count == 0
```

**Step 2: Implement ToolContext**

```python
# In src/agent_tools.py — add near top, replacing global state

from dataclasses import dataclass, field


@dataclass
class ToolContext:
    """Isolated context for tool invocations — no global state."""
    retriever: BaseRetriever
    search_call_count: int = 0
    visual_proof_call_count: int = 0


def create_tool_context(retriever: BaseRetriever) -> ToolContext:
    """Create a fresh tool context for a workflow invocation."""
    return ToolContext(retriever=retriever)


def make_tools(ctx: ToolContext):
    """Create tool functions bound to a specific ToolContext.

    Returns (search_documents, visual_proof) tools.
    """
    @tool
    def search_documents(query: str) -> str:
        """Search for information in the safety regulations. ..."""
        ctx.search_call_count += 1
        if ctx.search_call_count > settings.MAX_SEARCH_CALLS:
            return (
                f"Лимит поисков достигнут ({settings.MAX_SEARCH_CALLS}). "
                "Сформулируй ответ на основе уже найденных данных."
            )
        if not ctx.retriever:
            return "Error: Retriever not initialized."

        # ... rest of search_documents logic using ctx.retriever ...

    @tool
    def visual_proof(file_name: str, page_no: int, bbox: List[float], mode: str = "show") -> str:
        """Generate visual proof or analyze content. ..."""
        ctx.visual_proof_call_count += 1
        if ctx.visual_proof_call_count > settings.MAX_VISUAL_PROOF_CALLS:
            return (...)

        # ... rest of visual_proof logic (unchanged) ...

    return [search_documents, visual_proof]
```

Keep the old global functions temporarily with a deprecation comment for backward compatibility during transition. Remove them in Phase 5.

**Step 3: Update `multiagent_rag.py`**

```python
# In MultiAgentRAGWorkflow.__init__:
from src.agent_tools import create_tool_context, make_tools

# Replace:
#   set_global_retriever(retriever)
# With:
self.tool_ctx = create_tool_context(retriever)
self.tools = make_tools(self.tool_ctx)

# In _rag_agent_node:
# Replace:
#   tools = [search_documents, visual_proof]
# With:
tools = self.tools

# In invoke():
# Replace:
#   reset_tool_counters()
# With:
self.tool_ctx.search_call_count = 0
self.tool_ctx.visual_proof_call_count = 0
```

**Step 4: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 5: Commit**

```bash
git add src/agent_tools.py agents/multiagent_rag.py tests/test_agent_tools.py
git commit -m "refactor: replace global state with ToolContext for isolation and testability"
```

---

### Task 3.3: Accept tools via constructor

**Files:**
- Modify: `agents/multiagent_rag.py`

**Step 1: Update constructor**

```python
def __init__(self, retriever: BaseRetriever, llm_provider: str = "gemini", tools=None):
    self.retriever = retriever
    self.llm_provider = llm_provider.lower()

    # Tool context
    self.tool_ctx = create_tool_context(retriever)
    self.tools = tools or make_tools(self.tool_ctx)

    # ... rest of __init__ ...
```

**Step 2: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add agents/multiagent_rag.py
git commit -m "refactor: accept tools via constructor for extensibility"
```

---

### Task 3.4: Lint and verify Phase 3

```bash
black . && ruff check . --fix
pytest tests/ -v
git add -u && git commit -m "style: lint fixes after Phase 3"
```

---

## Phase 4: Clean LLM Factory & Retriever Architecture

### Task 4.1: Provider registry in `llm_factory.py`

**Files:**
- Modify: `src/llm_factory.py`
- Create: `tests/test_llm_factory.py`

**Step 1: Write failing test**

```python
# tests/test_llm_factory.py
import pytest
from unittest.mock import patch, MagicMock


@patch("src.llm_factory.ChatOpenAI")
def test_get_llm_openai(mock_openai):
    from src.llm_factory import get_llm
    mock_openai.return_value = MagicMock()
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.MODEL_NAME = "gpt-4o"
        mock_settings.TEMPERATURE = 0.0
        mock_settings.REQUEST_TIMEOUT = 120.0
        llm = get_llm()
        assert llm is not None


def test_get_llm_unknown_provider_raises():
    from src.llm_factory import get_llm
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "unknown_provider"
        with pytest.raises(ValueError, match="unknown_provider"):
            get_llm()
```

**Step 2: Refactor to registry pattern**

```python
# src/llm_factory.py

def _create_openai_llm():
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        timeout=settings.REQUEST_TIMEOUT,
        max_retries=3,
    )


_LLM_PROVIDERS = {
    "openai": _create_openai_llm,
}


def get_llm():
    """Create LLM instance based on LLM_PROVIDER setting."""
    provider = settings.LLM_PROVIDER.lower()
    factory = _LLM_PROVIDERS.get(provider)
    if not factory:
        available = ", ".join(sorted(_LLM_PROVIDERS.keys()))
        raise ValueError(
            f"Неизвестный провайдер LLM: {provider}. Доступные: {available}"
        )
    return factory()


_EMBEDDING_PROVIDERS = {
    "openai": lambda: OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME or "text-embedding-3-small"),
    "hf_api": _create_hf_embeddings,
    "local": _create_local_embeddings,
    "huggingface": _create_local_embeddings,
}


def get_embedding_model():
    provider = (settings.EMBEDDING_PROVIDER or "").lower()
    factory = _EMBEDDING_PROVIDERS.get(provider)
    if not factory:
        available = ", ".join(sorted(_EMBEDDING_PROVIDERS.keys()))
        raise ValueError(f"Unknown EMBEDDING_PROVIDER={provider}. Available: {available}")
    return factory()
```

Extract `_create_hf_embeddings` and `_create_local_embeddings` as top-level functions.

**Step 3: Run tests**

Run: `pytest tests/test_llm_factory.py tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add src/llm_factory.py tests/test_llm_factory.py
git commit -m "refactor: replace if/else chains with provider registry in llm_factory"
```

---

### Task 4.2: Deduplicate Chroma init in `vector_store.py`

**Files:**
- Modify: `src/vector_store.py:72-127`

**Step 1: Extract helper**

```python
def _create_chroma_instance(embeddings) -> Chroma:
    """Create a Chroma instance with standard settings."""
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_DB_PATH,
    )
```

Use in both `create_vector_store` and `load_vector_store`.

**Step 2: Make `load_vector_store` a cached singleton**

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def load_vector_store() -> Chroma:
    """Load existing Chroma collection (singleton — cached per process)."""
    if not os.path.isdir(settings.CHROMA_DB_PATH):
        raise FileNotFoundError(f"Chroma DB не найдена: {settings.CHROMA_DB_PATH}")

    embeddings = get_embedding_model()
    vs = _create_chroma_instance(embeddings)

    count = vs._collection.count()
    if count == 0:
        raise ValueError(
            f"Chroma DB пуста (0 документов): {settings.CHROMA_DB_PATH}. "
            "Запустите 'python index.py' для индексации."
        )
    logger.info(f"Chroma DB загружена: {count} документов")
    return vs
```

**Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 4: Commit**

```bash
git add src/vector_store.py
git commit -m "refactor: deduplicate Chroma init and cache load_vector_store as singleton"
```

---

### Task 4.3: Cache query expansion in `applicability_retriever.py`

**Files:**
- Modify: `src/applicability_retriever.py:24-43`

**Step 1: Add instance-level cache**

```python
class ApplicabilityRetriever(BaseRetriever):
    # ... existing fields ...
    _expansion_cache: dict = {}  # query → expanded queries

    def model_post_init(self, __context):
        self._expansion_cache = {}

    def _generate_queries(self, original_query: str) -> List[str]:
        if original_query in self._expansion_cache:
            return self._expansion_cache[original_query]

        # ... existing LLM call logic ...

        self._expansion_cache[original_query] = queries
        return queries
```

**Step 2: Run tests**

Run: `pytest tests/ -v`
Expected: All pass

**Step 3: Commit**

```bash
git add src/applicability_retriever.py
git commit -m "perf: cache query expansion results in ApplicabilityRetriever"
```

---

### Task 4.4: Lint and verify Phase 4

```bash
black . && ruff check . --fix
pytest tests/ -v
git add -u && git commit -m "style: lint fixes after Phase 4"
```

---

## Phase 5: Simplify `app.py` & Final Cleanup

### Task 5.1: Extract UI helpers from `app.py`

**Files:**
- Create: `src/ui_helpers.py`
- Modify: `app.py`

**Step 1: Create `src/ui_helpers.py`**

```python
# src/ui_helpers.py
"""Pure helper functions for Streamlit UI — no st imports in functions."""
from __future__ import annotations

import os
import re
from typing import List, Optional

PROOF_IMAGE_PATTERN = re.compile(r"(static/visuals/proof_[a-f0-9]+\.png)")


def find_proof_images(text: str) -> List[str]:
    """Extract proof image paths from text. Returns only paths that exist on disk."""
    return [p for p in PROOF_IMAGE_PATTERN.findall(text) if os.path.exists(p)]
```

**Step 2: Update `app.py`**

Replace both duplicate image-rendering blocks (lines 179-184 and 214-221) with:

```python
from src.ui_helpers import find_proof_images

# In chat history rendering:
if m["role"] == "assistant":
    for img_path in find_proof_images(m["content"]):
        st.image(img_path, caption="Визуальное доказательство", width=600)

# In new message rendering:
for img_path in find_proof_images(answer):
    st.image(img_path, caption="Визуальное доказательство", width=600)
```

**Step 3: Run app manually to verify**

```bash
streamlit run app.py
```

**Step 4: Commit**

```bash
git add src/ui_helpers.py app.py
git commit -m "refactor: extract UI helpers from app.py, deduplicate image rendering"
```

---

### Task 5.2: Pass sources through agent state (remove redundant retrieval)

**Files:**
- Modify: `agents/multiagent_rag.py` (RAGState)
- Modify: `app.py:248-253`

**Step 1: Add `retrieved_docs` to RAGState**

In `src/types.py`, extend the state (or add to RAGState in multiagent_rag.py):

The RAGState TypedDict already has `chunks_found` which contains the retrieved chunks. We can use these directly in the UI instead of re-retrieving.

**Step 2: Update `app.py` to use agent's chunks**

Replace lines 248-276 (the sources display section) in the multi-agent branch:

```python
if mas_mode and agent:
    # Use chunks already found by the agent
    chunks = result.get("chunks_found", [])
    if chunks:
        with st.expander(f"🔎 Показать источники ({len(chunks)})", expanded=False):
            for i, chunk in enumerate(chunks, start=1):
                src = chunk.get("source", "N/A")
                page = chunk.get("page_no", "")
                preview = chunk.get("content", "")[:500].strip().replace("\n", " ")
                st.markdown(f"**{i}. Источник:** `{src}`" + (f" · стр. {page}" if page else ""))
                st.code(preview, language="markdown")
                st.divider()
    else:
        st.caption("Источники не найдены.")
else:
    # Legacy RAG mode — use hybrid retriever
    try:
        retrieved_docs = hybrid_retriever.invoke(user_query)[:show_sources_n]
    except Exception:
        retrieved_docs = []
    # ... existing source display logic ...
```

**Step 3: Run app manually**

```bash
streamlit run app.py
```

**Step 4: Commit**

```bash
git add app.py
git commit -m "perf: use agent's found chunks for sources display, remove redundant retrieval"
```

---

### Task 5.3: Final cleanup — remove deprecated globals, lint, verify

**Files:**
- Modify: `src/agent_tools.py` — remove old global functions if no longer used
- All files — lint

**Step 1: Remove deprecated global state from `agent_tools.py`**

Remove:
- `_retriever`, `_search_call_count`, `_visual_proof_call_count` globals
- `set_global_retriever()`, `reset_tool_counters()` functions
- Old module-level `search_documents` and `visual_proof` if fully replaced by `make_tools`

Check that `app.py` doesn't import these anymore. Update any remaining references.

**Step 2: Run full lint**

```bash
black . && ruff check . --fix
```

**Step 3: Run all tests**

```bash
pytest tests/ -v
```

**Step 4: Validate prompts**

```bash
python scripts/validate_prompts.py
```

**Step 5: Final commit**

```bash
git add -u
git commit -m "refactor: final cleanup — remove deprecated globals, lint all files"
```

---

## Summary of New Files

| File | Purpose |
|------|---------|
| `src/types.py` | Shared enums and TypedDicts (RAGStatus, VerifyStatus, RouteType, ChunkInfo) |
| `src/parsers.py` | JSON/status/search result parsers extracted from multiagent_rag |
| `src/chroma_helpers.py` | Deduplicated ChromaDB range query and result conversion |
| `src/ui_helpers.py` | Pure UI helper functions (image path extraction) |
| `tests/test_chroma_helpers.py` | Tests for chroma helpers |
| `tests/test_parsers.py` | Tests for parsers |
| `tests/test_final_chain.py` | Tests for retriever factory |
| `tests/test_llm_factory.py` | Tests for LLM provider registry |

## Files Modified

| File | Changes |
|------|---------|
| `agents/multiagent_rag.py` | -250 lines (parsers, enums extracted), +imports, cached glossary, ToolContext, tools via constructor |
| `src/agent_tools.py` | ToolContext replaces globals, uses chroma_helpers, load_vector_store out of loop |
| `src/final_chain.py` | build_reranked_retriever factory, single LLM instance |
| `src/llm_factory.py` | Provider registry pattern |
| `src/vector_store.py` | _create_chroma_instance helper, load_vector_store singleton |
| `src/applicability_retriever.py` | Query expansion cache |
| `config/settings.py` | +6 agent/tool constants |
| `app.py` | UI helpers, agent sources, deduplication |
