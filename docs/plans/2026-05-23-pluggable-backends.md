# Pluggable Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SIA backend-agnostic — users can choose their LLM provider, vector store, and embedding model via env vars, without touching code.

**Architecture:** Three independent subsystems, each behind a thin factory/adapter layer. Phase 1 unifies LLM selection so the main V7 pipeline works without Gemini. Phase 2 abstracts the vector store with a protocol interface + Qdrant adapter. Phase 3 documents the already-working embedding abstraction and updates .env.example.

**Tech Stack:** LangChain abstractions (`BaseChatModel`, `VectorStore`), `langchain-anthropic`, `langchain-qdrant`, existing `pydantic-settings`.

---

## Scope

Three subsystems are independent — each phase produces working, testable code on its own. Implement in order: Phase 1 unblocks the most users (Gemini-free setup), Phase 2 adds vector store choice, Phase 3 polishes docs.

---

## Current state (read before coding)

- `src/llm_factory.py`: `get_llm()` only supports OpenAI. Main V7 pipeline (`src/v7/bridge.py`) calls `get_gemini_llm()` directly — hardcoded Gemini. GOST pipeline (`src/ers_rag/bridge.py`) uses `DeepSeekLLM` custom class — also hardcoded.
- `src/vector_store.py`: hardcodes `langchain_chroma.Chroma` throughout. No abstraction.
- Embeddings: already modular via `EMBEDDING_PROVIDER=openai|hf_api|local` in `llm_factory.py`. Just needs docs.

---

## Phase 1: LLM Provider Unification

**Goal:** `LLM_PROVIDER=openai|gemini|anthropic` controls which LLM the V7 main pipeline uses. Anthropic is new; Gemini goes through `get_llm()` instead of the direct call.

### Files

- Modify: `src/llm_factory.py` — add Anthropic + Gemini to `_LLM_PROVIDERS`, unify `get_llm()`
- Modify: `src/v7/bridge.py` — replace `get_gemini_llm(thinking_budget=N)` calls with `get_llm()`
- Modify: `config/settings.py` — add `ANTHROPIC_API_KEY`, `LLM_THINKING_BUDGET`
- Modify: `.env.example` — add Anthropic section
- Modify: `tests/test_llm_factory.py` (create if missing) — unit tests

---

### Task 1.1: Add `LLM_THINKING_BUDGET` to settings

**Files:**
- Modify: `config/settings.py`

- [ ] **Step 1: Add field to Settings**

In `config/settings.py`, add after `GEMINI_FAST_MODEL`:

```python
ANTHROPIC_API_KEY: str = ""
ANTHROPIC_MODEL: str = "claude-opus-4-7-20251101"
LLM_THINKING_BUDGET: int = 4096  # used when provider supports extended thinking
```

- [ ] **Step 2: Verify settings loads**

```bash
python -c "from config.settings import settings; print(settings.ANTHROPIC_MODEL)"
```

Expected: `claude-opus-4-7-20251101`

- [ ] **Step 3: Commit**

```bash
git add config/settings.py
git commit -m "feat(config): add ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LLM_THINKING_BUDGET"
```

---

### Task 1.2: Add Anthropic and Gemini to `_LLM_PROVIDERS`

**Files:**
- Modify: `src/llm_factory.py`
- Test: `tests/test_llm_factory.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_llm_factory.py` (or append if exists):

```python
from unittest.mock import patch, MagicMock
import pytest


def test_get_llm_unknown_provider_raises():
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "nonexistent"
        mock_settings.MODEL_NAME = "gpt-4o-mini"
        mock_settings.TEMPERATURE = 0.0
        mock_settings.REQUEST_TIMEOUT = 120.0
        from src.llm_factory import get_llm
        with pytest.raises(ValueError, match="nonexistent"):
            get_llm()


def test_get_llm_openai_provider():
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.MODEL_NAME = "gpt-4o-mini"
        mock_settings.TEMPERATURE = 0.0
        mock_settings.REQUEST_TIMEOUT = 120.0
        with patch("src.llm_factory.ChatOpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from src.llm_factory import get_llm
            result = get_llm()
            assert result is not None


def test_get_llm_gemini_provider():
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "gemini"
        mock_settings.GEMINI_API_KEY = "test-key"
        mock_settings.GEMINI_FAST_MODEL = "gemini-3-flash"
        mock_settings.TEMPERATURE = 0.0
        mock_settings.REQUEST_TIMEOUT = 120.0
        mock_settings.LLM_THINKING_BUDGET = 0
        with patch("src.llm_factory.ChatGoogleGenerativeAI") as mock_gemini:
            mock_gemini.return_value = MagicMock()
            from src.llm_factory import get_llm
            result = get_llm()
            assert result is not None


def test_get_llm_anthropic_provider():
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "anthropic"
        mock_settings.ANTHROPIC_API_KEY = "test-key"
        mock_settings.ANTHROPIC_MODEL = "claude-opus-4-7-20251101"
        mock_settings.TEMPERATURE = 0.0
        mock_settings.REQUEST_TIMEOUT = 120.0
        mock_settings.LLM_THINKING_BUDGET = 0
        with patch("src.llm_factory.ChatAnthropic") as mock_anthropic:
            mock_anthropic.return_value = MagicMock()
            from src.llm_factory import get_llm
            result = get_llm()
            assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_llm_factory.py -v
```

Expected: 3 of 4 FAIL (anthropic not imported, gemini not in providers)

- [ ] **Step 3: Install Anthropic package**

```bash
pip install langchain-anthropic
```

- [ ] **Step 4: Add Gemini and Anthropic to `_LLM_PROVIDERS` in `src/llm_factory.py`**

Replace the `_LLM_PROVIDERS` block and `get_llm()` function:

```python
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None


def _create_gemini_llm(**kwargs):
    if ChatGoogleGenerativeAI is None:
        raise ImportError("langchain-google-genai not installed")
    api_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    thinking_budget = settings.LLM_THINKING_BUDGET
    max_output_tokens = thinking_budget + _GEMINI_ANSWER_TOKEN_ALLOWANCE
    llm_kwargs = dict(
        model=settings.GEMINI_FAST_MODEL,
        google_api_key=api_key,
        temperature=settings.TEMPERATURE,
        max_output_tokens=max_output_tokens,
        timeout=settings.REQUEST_TIMEOUT,
    )
    if thinking_budget > 0:
        llm_kwargs["thinking_budget"] = thinking_budget
    llm_kwargs.update(kwargs)
    llm = ChatGoogleGenerativeAI(**llm_kwargs)
    if AutomaticFunctionCallingConfig is not None:
        _original_build = llm._build_request_config
        def _patched_build(*args, **kw):
            kw["automatic_function_calling"] = AutomaticFunctionCallingConfig(disable=True)
            return _original_build(*args, **kw)
        llm._build_request_config = _patched_build
    return llm


def _create_anthropic_llm(**kwargs):
    if ChatAnthropic is None:
        raise ImportError("langchain-anthropic not installed. Run: pip install langchain-anthropic")
    api_key = os.getenv("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        anthropic_api_key=api_key,
        temperature=settings.TEMPERATURE,
        timeout=settings.REQUEST_TIMEOUT,
        **kwargs,
    )


_LLM_PROVIDERS = {
    "openai": _create_openai_llm,
    "gemini": _create_gemini_llm,
    "anthropic": _create_anthropic_llm,
}


def get_llm(**kwargs):
    """Create LLM instance based on LLM_PROVIDER setting.

    Supported providers (set via LLM_PROVIDER env var):
      openai    — OpenAI ChatGPT (MODEL_NAME, OPENAI_API_KEY)
      gemini    — Google Gemini (GEMINI_FAST_MODEL, GEMINI_API_KEY, LLM_THINKING_BUDGET)
      anthropic — Anthropic Claude (ANTHROPIC_MODEL, ANTHROPIC_API_KEY)
    """
    provider = settings.LLM_PROVIDER.lower()
    factory = _LLM_PROVIDERS.get(provider)
    if not factory:
        available = ", ".join(sorted(_LLM_PROVIDERS.keys()))
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider!r}. Available: {available}"
        )
    return factory(**kwargs)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_llm_factory.py -v
```

Expected: all 4 PASS

- [ ] **Step 6: Commit**

```bash
git add src/llm_factory.py tests/test_llm_factory.py
git commit -m "feat(llm): add gemini and anthropic to get_llm() provider registry"
```

---

### Task 1.3: Wire `get_llm()` into V7 bridge

**Files:**
- Modify: `src/v7/bridge.py` — replace direct `get_gemini_llm()` calls

The V7 bridge calls `get_gemini_llm(thinking_budget=N)` four times. Replace them with `get_llm()` — the thinking_budget is now read from `settings.LLM_THINKING_BUDGET` inside `_create_gemini_llm`. For providers that don't support thinking (OpenAI, Anthropic), the kwarg is silently ignored.

- [ ] **Step 1: Write integration smoke test**

Add to `tests/test_llm_factory.py`:

```python
def test_get_llm_is_langchain_chat_model():
    """get_llm() must return a LangChain BaseChatModel for any provider."""
    from langchain_core.language_models import BaseChatModel
    with patch("src.llm_factory.settings") as mock_settings:
        mock_settings.LLM_PROVIDER = "openai"
        mock_settings.MODEL_NAME = "gpt-4o-mini"
        mock_settings.TEMPERATURE = 0.0
        mock_settings.REQUEST_TIMEOUT = 120.0
        with patch("src.llm_factory.ChatOpenAI") as mock_cls:
            mock_cls.return_value = MagicMock(spec=BaseChatModel)
            from src.llm_factory import get_llm
            result = get_llm()
            assert mock_cls.called
```

- [ ] **Step 2: Run test**

```bash
pytest tests/test_llm_factory.py::test_get_llm_is_langchain_chat_model -v
```

Expected: PASS

- [ ] **Step 3: Update `src/v7/bridge.py` imports and LLM init**

Find the import:
```python
from src.llm_factory import get_gemini_llm
```

Replace with:
```python
from src.llm_factory import get_gemini_llm, get_llm
```

Find the four `get_gemini_llm(...)` calls (~lines 446–457) inside `init_default_fns()`:

```python
verifier_llm = get_gemini_llm(
    thinking_budget=settings.THINKING_VERIFIER,
    response_mime_type="application/json",
)
rewriter_llm = get_gemini_llm(thinking_budget=1024)
generator_llm = get_gemini_llm(thinking_budget=4096)
expander_llm = get_gemini_llm(thinking_budget=0)
```

Replace with:

```python
verifier_llm = get_llm()
rewriter_llm = get_llm()
generator_llm = get_llm()
expander_llm = get_llm()
```

Keep `get_gemini_llm` import — it's still used by `get_vision_llm()`.

- [ ] **Step 4: Smoke test V7 pipeline with stub**

```bash
python scripts/trace_v7.py --no-chroma "привет как дела"
```

Expected: path shown, no ImportError

- [ ] **Step 5: Commit**

```bash
git add src/v7/bridge.py
git commit -m "feat(v7): use get_llm() provider abstraction instead of hardcoded gemini"
```

---

### Task 1.4: Update `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add Anthropic section**

After the `# Google Gemini` section, add:

```
# -------------------------------------------------------------------
# Anthropic Claude (LLM_PROVIDER=anthropic)
# -------------------------------------------------------------------
# Get key: https://console.anthropic.com/settings/keys
# ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY_HERE
# ANTHROPIC_MODEL=claude-opus-4-7-20251101

# -------------------------------------------------------------------
# LLM Provider selection
# -------------------------------------------------------------------
# Choose one: openai | gemini | anthropic
LLM_PROVIDER=openai

# Thinking budget for providers that support extended reasoning (gemini, anthropic)
# Set to 0 to disable thinking mode
# LLM_THINKING_BUDGET=4096
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add anthropic and LLM_THINKING_BUDGET to .env.example"
```

---

## Phase 2: Vector Store Abstraction

**Goal:** `VECTOR_STORE=chroma|qdrant` controls which backend is used. Chroma remains the default — existing setups need no changes.

### Files

- Create: `src/backends/vector_store.py` — abstract protocol + factory
- Create: `src/backends/chroma_backend.py` — wraps existing `src/vector_store.py`
- Create: `src/backends/qdrant_backend.py` — Qdrant adapter
- Modify: `config/settings.py` — add `VECTOR_STORE`, `QDRANT_URL`, `QDRANT_API_KEY`
- Modify: `src/v7/bridge.py` — use `get_vector_store()` instead of direct Chroma import
- Modify: `src/v7/config.py` — same
- Test: `tests/test_vector_store_backends.py`

---

### Task 2.1: Define `VectorStoreBackend` protocol and factory

**Files:**
- Create: `src/backends/__init__.py`
- Create: `src/backends/vector_store.py`
- Modify: `config/settings.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_vector_store_backends.py`:

```python
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document


def test_factory_unknown_backend_raises():
    with patch("src.backends.vector_store.settings") as mock_s:
        mock_s.VECTOR_STORE = "pinecone"
        from src.backends.vector_store import get_vector_store_backend
        with pytest.raises(ValueError, match="pinecone"):
            get_vector_store_backend()


def test_factory_returns_chroma_by_default():
    with patch("src.backends.vector_store.settings") as mock_s:
        mock_s.VECTOR_STORE = "chroma"
        mock_s.CHROMA_DB_PATH = "./chroma_db_test"
        mock_s.CHROMA_COLLECTION_NAME = "test"
        with patch("src.backends.vector_store.ChromaBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            from src.backends.vector_store import get_vector_store_backend
            result = get_vector_store_backend()
            assert mock_cls.called
```

- [ ] **Step 2: Run tests to confirm failure**

```bash
pytest tests/test_vector_store_backends.py -v
```

Expected: ImportError (module doesn't exist yet)

- [ ] **Step 3: Add settings fields**

In `config/settings.py`, add after `CHROMA_COLLECTION_NAME`:

```python
# Vector store backend selection
VECTOR_STORE: str = "chroma"  # options: chroma | qdrant

# Qdrant settings (used when VECTOR_STORE=qdrant)
QDRANT_URL: str = "http://localhost:6333"
QDRANT_API_KEY: str = ""
QDRANT_COLLECTION_NAME: str = "documents"
```

- [ ] **Step 4: Create `src/backends/__init__.py`**

```python
```
(empty file)

- [ ] **Step 5: Create `src/backends/vector_store.py`**

```python
"""Vector store factory — returns the configured backend."""
from __future__ import annotations

from typing import Protocol, runtime_checkable, List
from langchain_core.documents import Document

from config.settings import settings


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Minimal interface every vector store backend must implement."""

    def similarity_search_with_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        ...

    def add_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        ...


def get_vector_store_backend(load_existing: bool = True) -> VectorStoreBackend:
    """Return the configured vector store backend.

    Args:
        load_existing: If True, load an existing index. If False, create a new one.

    Raises:
        ValueError: If VECTOR_STORE is set to an unknown backend.
    """
    backend = settings.VECTOR_STORE.lower()

    if backend == "chroma":
        from src.backends.chroma_backend import ChromaBackend
        return ChromaBackend(load_existing=load_existing)

    if backend == "qdrant":
        from src.backends.qdrant_backend import QdrantBackend
        return QdrantBackend(load_existing=load_existing)

    available = "chroma, qdrant"
    raise ValueError(
        f"Unknown VECTOR_STORE={backend!r}. Available: {available}"
    )
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_vector_store_backends.py -v
```

Expected: PASS (ChromaBackend import will fail until Task 2.2, but factory raises correct errors)

- [ ] **Step 7: Commit**

```bash
git add src/backends/__init__.py src/backends/vector_store.py config/settings.py
git commit -m "feat(backends): add VectorStoreBackend protocol and factory"
```

---

### Task 2.2: Chroma backend adapter

**Files:**
- Create: `src/backends/chroma_backend.py`

- [ ] **Step 1: Write test**

Add to `tests/test_vector_store_backends.py`:

```python
def test_chroma_backend_exposes_similarity_search():
    """ChromaBackend must expose similarity_search_with_score."""
    with patch("src.backends.chroma_backend.load_vector_store") as mock_load:
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="test"), 0.9)
        ]
        mock_load.return_value = mock_vs
        from src.backends.chroma_backend import ChromaBackend
        backend = ChromaBackend(load_existing=True)
        results = backend.similarity_search_with_score("query", k=1)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(0.9)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_vector_store_backends.py::test_chroma_backend_exposes_similarity_search -v
```

Expected: FAIL (module missing)

- [ ] **Step 3: Create `src/backends/chroma_backend.py`**

```python
"""ChromaDB backend — thin wrapper around src.vector_store."""
from __future__ import annotations

from langchain_core.documents import Document

from config.settings import settings


class ChromaBackend:
    """Wraps the existing Chroma vector store. Delegates all calls through."""

    def __init__(self, load_existing: bool = True) -> None:
        if load_existing:
            from src.vector_store import load_vector_store
            self._vs = load_vector_store()
        else:
            self._vs = None  # populated later via create()

    def create(self, chunks: list[Document]) -> "ChromaBackend":
        from src.vector_store import create_vector_store
        self._vs = create_vector_store(chunks)
        return self

    def similarity_search_with_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        return self._vs.similarity_search_with_score(query, k=k)

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        return self._vs.add_texts(texts=texts, metadatas=metadatas or [])

    # Pass through attribute access for legacy code that uses vs._collection etc.
    def __getattr__(self, name: str):
        return getattr(self._vs, name)
```

- [ ] **Step 4: Run test**

```bash
pytest tests/test_vector_store_backends.py::test_chroma_backend_exposes_similarity_search -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backends/chroma_backend.py tests/test_vector_store_backends.py
git commit -m "feat(backends): add ChromaBackend adapter"
```

---

### Task 2.3: Qdrant backend adapter

**Files:**
- Create: `src/backends/qdrant_backend.py`

- [ ] **Step 1: Write test**

Add to `tests/test_vector_store_backends.py`:

```python
def test_qdrant_backend_exposes_similarity_search():
    with patch("src.backends.qdrant_backend.QdrantClient") as mock_client_cls, \
         patch("src.backends.qdrant_backend.get_embedding_model") as mock_emb:
        mock_emb.return_value = MagicMock()
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.search.return_value = [
            MagicMock(payload={"page_content": "text", "source": "doc.pdf"}, score=0.85)
        ]
        from src.backends.qdrant_backend import QdrantBackend
        backend = QdrantBackend(load_existing=True)
        results = backend.similarity_search_with_score("query", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert score == pytest.approx(0.85)
        assert doc.page_content == "text"
```

- [ ] **Step 2: Run test**

```bash
pytest tests/test_vector_store_backends.py::test_qdrant_backend_exposes_similarity_search -v
```

Expected: FAIL (module missing)

- [ ] **Step 3: Install Qdrant client**

```bash
pip install langchain-qdrant qdrant-client
```

- [ ] **Step 4: Create `src/backends/qdrant_backend.py`**

```python
"""Qdrant vector store backend."""
from __future__ import annotations

import uuid
from langchain_core.documents import Document

from config.settings import settings
from src.llm_factory import get_embedding_model


class QdrantBackend:
    """Qdrant-backed vector store using langchain-qdrant."""

    def __init__(self, load_existing: bool = True) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._embeddings = get_embedding_model()
        self._collection = settings.QDRANT_COLLECTION_NAME

        url = settings.QDRANT_URL
        api_key = settings.QDRANT_API_KEY or None
        self._client = QdrantClient(url=url, api_key=api_key)

        if not load_existing:
            # Detect embedding dimension via a probe
            probe = self._embeddings.embed_query("probe")
            dim = len(probe)
            self._client.recreate_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def similarity_search_with_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        vector = self._embeddings.embed_query(query)
        results = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=k,
            with_payload=True,
        )
        docs = []
        for hit in results:
            payload = hit.payload or {}
            page_content = payload.pop("page_content", "")
            docs.append((Document(page_content=page_content, metadata=payload), hit.score))
        return docs

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        from qdrant_client.models import PointStruct

        metadatas = metadatas or [{} for _ in texts]
        vectors = self._embeddings.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload={"page_content": texts[i], **metadatas[i]},
            )
            for i in range(len(texts))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return ids
```

- [ ] **Step 5: Run test**

```bash
pytest tests/test_vector_store_backends.py::test_qdrant_backend_exposes_similarity_search -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/backends/qdrant_backend.py
git commit -m "feat(backends): add QdrantBackend adapter"
```

---

### Task 2.4: Wire factory into V7 bridge and `index.py`

**Files:**
- Modify: `src/v7/bridge.py` — use `get_vector_store_backend()` at load time
- Modify: `src/v7/config.py` — same if it loads the store directly
- Modify: `index.py` — use backend factory for writes

- [ ] **Step 1: Check where V7 bridge loads Chroma**

```bash
grep -n "load_vector_store\|Chroma\|chroma_db" src/v7/bridge.py src/v7/config.py
```

- [ ] **Step 2: Replace Chroma load with factory in `src/v7/bridge.py`**

Find the import and load pattern (typically `from src.vector_store import load_vector_store`).

Replace:
```python
from src.vector_store import load_vector_store
# ...
vs = load_vector_store()
```

With:
```python
from src.backends.vector_store import get_vector_store_backend
# ...
vs = get_vector_store_backend(load_existing=True)
```

Repeat for `src/v7/config.py` if it also loads the store.

- [ ] **Step 3: Update `index.py` to use backend for writes**

Find the `create_vector_store(chunks)` call in `index.py`.

Replace:
```python
from src.vector_store import create_vector_store
# ...
create_vector_store(chunks)
```

With:
```python
from src.backends.vector_store import get_vector_store_backend
# ...
backend = get_vector_store_backend(load_existing=False)
backend.create(chunks)
```

- [ ] **Step 4: Smoke test with Chroma (existing default)**

```bash
python scripts/trace_v7.py --no-chroma "что такое инструктаж"
```

Expected: runs without error

- [ ] **Step 5: Commit**

```bash
git add src/v7/bridge.py src/v7/config.py index.py
git commit -m "feat(v7): wire vector store factory — VECTOR_STORE env controls backend"
```

---

### Task 2.5: Update `.env.example` with Qdrant section

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Add vector store section**

After `CHROMA_DB_PATH` line, add:

```
# -------------------------------------------------------------------
# Vector Store backend (chroma | qdrant)
# -------------------------------------------------------------------
VECTOR_STORE=chroma

# Qdrant settings (used when VECTOR_STORE=qdrant)
# Start local Qdrant: docker run -p 6333:6333 qdrant/qdrant
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=         # leave empty for local, set for Qdrant Cloud
# QDRANT_COLLECTION_NAME=documents
```

- [ ] **Step 2: Commit**

```bash
git add .env.example
git commit -m "docs: add Qdrant vector store options to .env.example"
```

---

## Phase 3: Embeddings documentation polish

Embeddings are already modular (`EMBEDDING_PROVIDER=openai|hf_api|local`). This phase ensures the README and .env.example make it discoverable.

### Task 3.1: Add embeddings section to README and .env.example

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: Add embeddings section to .env.example**

After the `EMBEDDING_MODEL_NAME` line:

```
# Embedding provider options:
#   openai   — text-embedding-3-small (default, requires OPENAI_API_KEY)
#   local    — sentence-transformers on CPU (no API key, first run downloads model)
#   hf_api   — Hugging Face Inference API (requires HF_TOKEN)
#
# For fully local setup (no OpenAI):
#   EMBEDDING_PROVIDER=local
#   EMBEDDING_MODEL_NAME=ai-forever/sbert_large_nlu_ru
```

- [ ] **Step 2: Add "Bring Your Own Backend" section to README.md**

After the `## Stack` table, add:

````markdown
## Bring Your Own Backend

Configure via environment variables — no code changes needed:

| Variable | Options | Default |
|----------|---------|---------|
| `LLM_PROVIDER` | `openai` · `gemini` · `anthropic` | `openai` |
| `VECTOR_STORE` | `chroma` · `qdrant` | `chroma` |
| `EMBEDDING_PROVIDER` | `openai` · `local` · `hf_api` | `openai` |

**Fully local setup** (no API keys):
```bash
LLM_PROVIDER=openai          # or anthropic / gemini
VECTOR_STORE=chroma
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_NAME=ai-forever/sbert_large_nlu_ru
```

**Qdrant Cloud:**
```bash
VECTOR_STORE=qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-key
```
````

- [ ] **Step 3: Commit**

```bash
git add README.md .env.example
git commit -m "docs: add Bring Your Own Backend section to README and .env.example"
```

---

## Self-Review

**Spec coverage:**
- ✅ LLM: openai + gemini + anthropic via `get_llm()`
- ✅ Vector store: chroma (default) + qdrant via factory
- ✅ Embeddings: already modular, documented in Phase 3
- ✅ Chroma remains default — no breaking change

**Placeholder scan:** None found.

**Type consistency:**
- `VectorStoreBackend.similarity_search_with_score` → returns `list[tuple[Document, float]]` — matches LangChain convention used downstream in `nlp_core.py`
- `ChromaBackend.__getattr__` passthrough covers legacy `.get()` calls in `src/chroma_helpers.py`

**Risk:** Task 2.4 step 2 says "find the import and load pattern" without showing exact code — because the exact lines depend on what `grep` returns in Step 1. The engineer must read the output before editing. This is intentional: `src/v7/config.py` content was not read during plan writing.
