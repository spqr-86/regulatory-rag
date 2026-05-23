# Pluggable Backends — Kanban Plan

> **For agentic workers:** Pick one card at a time. Update `Status:` field as you go (`TODO → IN PROGRESS → DONE`). Do not start a card whose `Depends on:` is not `DONE`. Each card is atomic and independently committable.

**Goal:** Make SIA backend-agnostic — users choose LLM provider, vector store, and embedding model via `.env`, no code changes.

**Architecture:** Three thin factory layers. Existing Chroma + Gemini paths remain default (zero breaking changes for current users).

**Tech Stack:** `langchain-anthropic`, `langchain-qdrant`, existing `pydantic-settings`, `pytest` with `unittest.mock`.

---

## How to use this plan

1. Read **Code Reality Check** below — it's the ground truth.
2. Open the board for the current phase.
3. Pick a card with `Status: TODO` whose dependencies are `DONE`.
4. Update card to `Status: IN PROGRESS`, do the work, commit, then mark `DONE`.
5. Phase 2 can start only after Phase 1 is fully `DONE`. Phase 3 can start any time.

---

## Code Reality Check (read first)

| Aspect | Reality | Implication |
|--------|---------|-------------|
| `get_llm()` in `llm_factory.py` | Only knows `openai` | Must add `gemini` and `anthropic` |
| V7 main pipeline LLM | `src/v7/bridge.py:446-458` calls `get_gemini_llm()` with **4 different** `thinking_budget` values + `response_mime_type="application/json"` for verifier | `get_llm()` must accept `**kwargs` and forward them to the provider. For non-Gemini, kwargs like `thinking_budget` are silently dropped. |
| GOST pipeline LLM | `src/ers_rag/bridge.py:412` has hardcoded `DeepSeekLLM` class | Out of scope for Phase 1 (separate concern — DeepSeek API has no langchain wrapper that fits cleanly). Add a card to track. |
| Vector store loading | `src/v7/bridge.py:397` takes `vector_store` as **argument**, doesn't call `load_vector_store()` | Replace `load_vector_store()` in **callers**: `api.py:41`, `app.py:174`, `eval/run_eval.py:129`, `eval/run_v7_eval.py:170`, `scripts/measure_cps.py:42`, `scripts/trace_v7.py:198` |
| BM25 corpus build | `src/v7/bridge.py:413` calls `vector_store.get(include=["metadatas", "documents"])` — Chroma-specific | Protocol needs `iter_all_documents()` method, every backend implements it |
| GOST indexer | `index_gosts.py:205` hardcodes `Chroma(...)` separately | Add card to use factory there too |
| Embeddings | `EMBEDDING_PROVIDER` already supports `openai`/`local`/`hf_api` | No code work — just docs |

---

# Board 1 — LLM Provider Unification

Goal: `LLM_PROVIDER=openai|gemini|anthropic` controls main pipeline. Default stays Gemini for existing users.

---

### CARD-1.1 — Add settings for Anthropic + thinking config

**Status:** ⬜ TODO
**Depends on:** —
**Files:** `config/settings.py`

**Do:**
Append to `Settings` class (after `GEMINI_FAST_MODEL`):
```python
ANTHROPIC_API_KEY: str = ""
ANTHROPIC_MODEL: str = "claude-opus-4-7-20251101"
```
(Do NOT add `LLM_THINKING_BUDGET` — different roles need different budgets, kept per-call.)

**Verify:**
```bash
python -c "from config.settings import settings; print(settings.ANTHROPIC_MODEL)"
# Expected: claude-opus-4-7-20251101
```

**Done when:** Settings load without error.

**Commit:** `feat(config): add ANTHROPIC_API_KEY, ANTHROPIC_MODEL`

---

### CARD-1.2 — Add Gemini and Anthropic to `get_llm()` registry

**Status:** ⬜ TODO
**Depends on:** CARD-1.1
**Files:** `src/llm_factory.py`, `tests/test_llm_factory_providers.py` (new), `requirements.txt`

**Do:**

1. `pip install langchain-anthropic` and add to `requirements.txt`.

2. In `src/llm_factory.py`, add imports near top:
```python
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None
```

3. Add two factory functions before `_LLM_PROVIDERS`:
```python
def _create_gemini_llm(**kwargs):
    """Forward kwargs (thinking_budget, response_mime_type, etc.) to get_gemini_llm."""
    return get_gemini_llm(**kwargs)


def _create_anthropic_llm(**kwargs):
    if ChatAnthropic is None:
        raise ImportError(
            "langchain-anthropic not installed. Run: pip install langchain-anthropic"
        )
    api_key = os.getenv("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    # Anthropic does not understand thinking_budget/response_mime_type — drop them.
    kwargs.pop("thinking_budget", None)
    kwargs.pop("response_mime_type", None)
    return ChatAnthropic(
        model=settings.ANTHROPIC_MODEL,
        anthropic_api_key=api_key,
        temperature=settings.TEMPERATURE,
        timeout=settings.REQUEST_TIMEOUT,
        **kwargs,
    )
```

4. Update `_create_openai_llm` to drop Gemini-specific kwargs:
```python
def _create_openai_llm(**kwargs):
    kwargs.pop("thinking_budget", None)
    kwargs.pop("response_mime_type", None)
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        timeout=settings.REQUEST_TIMEOUT,
        max_retries=3,
        **kwargs,
    )
```

5. Update registry:
```python
_LLM_PROVIDERS = {
    "openai": _create_openai_llm,
    "gemini": _create_gemini_llm,
    "anthropic": _create_anthropic_llm,
}
```

6. Update `get_llm()` docstring to list supported providers.

**Test:** Create `tests/test_llm_factory_providers.py`:
```python
from unittest.mock import patch, MagicMock
import pytest


@pytest.mark.unit
def test_get_llm_unknown_provider_raises():
    from src.llm_factory import get_llm
    with patch("src.llm_factory.settings") as s:
        s.LLM_PROVIDER = "pinecone"
        with pytest.raises(ValueError, match="pinecone"):
            get_llm()


@pytest.mark.unit
def test_get_llm_anthropic_drops_gemini_kwargs():
    from src.llm_factory import get_llm
    with patch("src.llm_factory.settings") as s, \
         patch("src.llm_factory.ChatAnthropic") as mock_cls:
        s.LLM_PROVIDER = "anthropic"
        s.ANTHROPIC_API_KEY = "k"
        s.ANTHROPIC_MODEL = "claude-opus-4-7"
        s.TEMPERATURE = 0.0
        s.REQUEST_TIMEOUT = 120.0
        get_llm(thinking_budget=1024, response_mime_type="application/json")
        call_kwargs = mock_cls.call_args.kwargs
        assert "thinking_budget" not in call_kwargs
        assert "response_mime_type" not in call_kwargs


@pytest.mark.unit
def test_get_llm_gemini_forwards_kwargs():
    """thinking_budget must reach get_gemini_llm for Gemini provider."""
    from src.llm_factory import get_llm
    with patch("src.llm_factory.settings") as s, \
         patch("src.llm_factory.get_gemini_llm") as mock_g:
        s.LLM_PROVIDER = "gemini"
        get_llm(thinking_budget=4096)
        mock_g.assert_called_once_with(thinking_budget=4096)
```

```bash
pytest tests/test_llm_factory_providers.py -v
```

**Done when:** All 3 tests pass.

**Commit:** `feat(llm): add gemini and anthropic to get_llm() with kwarg passthrough`

---

### CARD-1.3 — Switch V7 bridge to `get_llm()`

**Status:** ⬜ TODO
**Depends on:** CARD-1.2
**Files:** `src/v7/bridge.py` (lines 446–458)

**Do:**

In `src/v7/bridge.py`, change import:
```python
from src.llm_factory import get_gemini_llm, get_llm
```

Replace the 4 calls in `init_default_fns()`:
```python
# BEFORE
verifier_llm = get_gemini_llm(thinking_budget=1024, response_mime_type="application/json")
rewriter_llm = get_gemini_llm(thinking_budget=1024)
generator_llm = get_gemini_llm(thinking_budget=4096)
expander_llm = get_gemini_llm(thinking_budget=0)

# AFTER
verifier_llm = get_llm(thinking_budget=1024, response_mime_type="application/json")
rewriter_llm = get_llm(thinking_budget=1024)
generator_llm = get_llm(thinking_budget=4096)
expander_llm = get_llm(thinking_budget=0)
```

Keep `get_gemini_llm` import — `get_vision_llm()` still uses it directly.

**Verify:** With existing `.env` (LLM_PROVIDER=gemini or unset → defaults to openai):
```bash
python scripts/trace_v7.py --no-chroma "привет"
```
Expected: pipeline runs, no ImportError. If `LLM_PROVIDER=openai`, verifier may fail at JSON parse — that's a known limitation noted in README later (CARD-3.1).

**Done when:** trace_v7 runs end-to-end with at least one provider.

**Commit:** `feat(v7): route bridge LLM calls through get_llm() provider registry`

---

### CARD-1.4 — Update `.env.example` for LLM choice

**Status:** ⬜ TODO
**Depends on:** CARD-1.3
**Files:** `.env.example`

**Do:**
Replace the existing `LLM_PROVIDER=openai` block with:
```bash
# -------------------------------------------------------------------
# LLM Provider — choose one: openai | gemini | anthropic
# -------------------------------------------------------------------
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o-mini      # for openai
TEMPERATURE=0.0
```

Add new section after Gemini section:
```bash
# -------------------------------------------------------------------
# Anthropic Claude (LLM_PROVIDER=anthropic)
# -------------------------------------------------------------------
# Get key: https://console.anthropic.com/settings/keys
# ANTHROPIC_API_KEY=YOUR_ANTHROPIC_KEY_HERE
# ANTHROPIC_MODEL=claude-opus-4-7-20251101
```

**Done when:** File parses, all three providers documented.

**Commit:** `docs: document anthropic provider in .env.example`

---

### CARD-1.5 — GOST pipeline LLM unification

**Status:** ❌ WONTFIX (decided 2026-05-23)

**Decision:** GOST pipeline stays pinned to DeepSeek. It's a WTA-specific implementation. Future work (not this plan): merge GOST into main pipeline as a single project run with DeepSeek + Chroma — proves end-to-end backend flexibility. Tracked separately, not in scope here.

---

# Board 2 — Vector Store Abstraction

Goal: `VECTOR_STORE=chroma|qdrant` controls backend. Chroma stays default.

---

### CARD-2.1 — Define `VectorStoreBackend` protocol with full API surface

**Status:** ⬜ TODO
**Depends on:** —
**Files:** `src/backends/__init__.py` (new, empty), `src/backends/vector_store.py` (new), `config/settings.py`

**Do:**

1. Add to `Settings` after `CHROMA_COLLECTION_NAME`:
```python
# Vector store backend (chroma | qdrant)
VECTOR_STORE: str = "chroma"

# Qdrant (used when VECTOR_STORE=qdrant)
QDRANT_URL: str = "http://localhost:6333"
QDRANT_API_KEY: str = ""
QDRANT_COLLECTION_NAME: str = "documents"
```

2. Create `src/backends/__init__.py` (empty).

3. Create `src/backends/vector_store.py`:
```python
"""Vector store factory and protocol."""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable
from langchain_core.documents import Document

from config.settings import settings


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Minimum API every backend must implement."""

    def similarity_search_with_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        """Top-k semantic search. Score is similarity (higher=better, 0..1)."""
        ...

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        """Insert documents. Returns assigned IDs."""
        ...

    def iter_all_documents(self) -> Iterator[dict]:
        """Yield every stored doc as {"text": str, "metadata": dict}.
        Used by BM25 corpus build (src/v7/bridge.py:413)."""
        ...

    def count(self) -> int:
        """Total document count."""
        ...

    def get_by_filter(self, where: dict) -> list[Document]:
        """Metadata filter query (used by chroma_helpers.query_chunks_by_range).
        `where` follows Chroma's filter syntax — backends translate as needed."""
        ...


def get_vector_store_backend(load_existing: bool = True) -> VectorStoreBackend:
    backend = settings.VECTOR_STORE.lower()
    if backend == "chroma":
        from src.backends.chroma_backend import ChromaBackend
        return ChromaBackend(load_existing=load_existing)
    if backend == "qdrant":
        from src.backends.qdrant_backend import QdrantBackend
        return QdrantBackend(load_existing=load_existing)
    raise ValueError(
        f"Unknown VECTOR_STORE={backend!r}. Available: chroma, qdrant"
    )
```

**Test:** `tests/test_vector_store_factory.py`:
```python
import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
def test_factory_unknown_raises():
    from src.backends.vector_store import get_vector_store_backend
    with patch("src.backends.vector_store.settings") as s:
        s.VECTOR_STORE = "pinecone"
        with pytest.raises(ValueError, match="pinecone"):
            get_vector_store_backend()
```

```bash
pytest tests/test_vector_store_factory.py -v
```

**Done when:** Test passes, settings load with VECTOR_STORE field.

**Commit:** `feat(backends): add VectorStoreBackend protocol and factory`

---

### CARD-2.2 — ChromaBackend implementing full protocol

**Status:** ⬜ TODO
**Depends on:** CARD-2.1
**Files:** `src/backends/chroma_backend.py` (new), `tests/test_chroma_backend.py` (new)

**Do:**

Create `src/backends/chroma_backend.py`:
```python
"""Chroma backend — wraps existing src/vector_store.py."""
from __future__ import annotations

from typing import Iterator
from langchain_core.documents import Document


class ChromaBackend:
    def __init__(self, load_existing: bool = True) -> None:
        if load_existing:
            from src.vector_store import load_vector_store
            self._vs = load_vector_store()
        else:
            self._vs = None  # filled by create()

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

    def iter_all_documents(self) -> Iterator[dict]:
        data = self._vs.get(include=["metadatas", "documents"])
        for text, meta in zip(data["documents"], data["metadatas"]):
            yield {"text": text, "metadata": meta or {}}

    def count(self) -> int:
        return self._vs._collection.count()

    def get_by_filter(self, where: dict) -> list[Document]:
        from src.chroma_helpers import chroma_results_to_documents
        result = self._vs.get(where=where)
        return chroma_results_to_documents(result)

    # Legacy escape hatch — for callers that still need raw Chroma object
    @property
    def raw(self):
        return self._vs
```

**Test:** Mock `load_vector_store`, verify all 5 protocol methods delegate correctly. Skipped here for brevity — pattern is the same as CARD-1.2 test.

**Done when:** Pytest green, `isinstance(ChromaBackend(...), VectorStoreBackend)` is True.

**Commit:** `feat(backends): add ChromaBackend implementing full protocol`

---

### CARD-2.3 — QdrantBackend implementing full protocol

**Status:** ⬜ TODO
**Depends on:** CARD-2.1
**Files:** `src/backends/qdrant_backend.py` (new), `requirements.txt`

**Do:**

1. `pip install qdrant-client` and add to `requirements.txt`.

2. Create `src/backends/qdrant_backend.py`:
```python
"""Qdrant backend."""
from __future__ import annotations

import uuid
from typing import Iterator
from langchain_core.documents import Document

from config.settings import settings
from src.llm_factory import get_embedding_model


class QdrantBackend:
    def __init__(self, load_existing: bool = True) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self._embeddings = get_embedding_model()
        self._collection = settings.QDRANT_COLLECTION_NAME
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY or None,
        )
        if not load_existing:
            dim = len(self._embeddings.embed_query("probe"))
            self._client.recreate_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def similarity_search_with_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        vec = self._embeddings.embed_query(query)
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vec,
            limit=k,
            with_payload=True,
        )
        out = []
        for hit in hits:
            payload = dict(hit.payload or {})
            text = payload.pop("page_content", "")
            out.append((Document(page_content=text, metadata=payload), hit.score))
        return out

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        from qdrant_client.models import PointStruct
        metas = metadatas or [{} for _ in texts]
        vectors = self._embeddings.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in texts]
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload={"page_content": texts[i], **metas[i]},
            )
            for i in range(len(texts))
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return ids

    def iter_all_documents(self) -> Iterator[dict]:
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = dict(p.payload or {})
                text = payload.pop("page_content", "")
                yield {"text": text, "metadata": payload}
            if offset is None:
                break

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count

    def get_by_filter(self, where: dict) -> list[Document]:
        # Translate Chroma-style filter to Qdrant. Minimum: {"source": "x"} →
        # FieldCondition(key="source", match=MatchValue(value="x")).
        # Range queries ($gte/$lte) → Range(gte=..., lte=...).
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue, Range,
        )
        conditions = []
        for key, val in where.items():
            if isinstance(val, dict):
                rng_kwargs = {}
                if "$gte" in val: rng_kwargs["gte"] = val["$gte"]
                if "$lte" in val: rng_kwargs["lte"] = val["$lte"]
                conditions.append(FieldCondition(key=key, range=Range(**rng_kwargs)))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))
        flt = Filter(must=conditions)
        hits, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=flt,
            limit=1000,
            with_payload=True,
        )
        out = []
        for p in hits:
            payload = dict(p.payload or {})
            text = payload.pop("page_content", "")
            out.append(Document(page_content=text, metadata=payload))
        return out
```

**Note:** `where` filter conversion supports flat `{field: value}` and `{field: {"$gte": ..., "$lte": ...}}`. Chroma's `$and` wrapper used in `chroma_helpers.query_chunks_by_range` is not supported here — those callers need refactoring in CARD-2.5.

**Test:** Use mocked `QdrantClient`. Verify scroll/search/upsert called with right args. Pattern same as CARD-2.2.

**Done when:** Pytest green; `isinstance(QdrantBackend(...), VectorStoreBackend)` is True.

**Commit:** `feat(backends): add QdrantBackend implementing full protocol`

---

### CARD-2.4 — Refactor BM25 corpus build to use protocol

**Status:** ⬜ TODO
**Depends on:** CARD-2.2
**Files:** `src/v7/bridge.py` (lines ~412–418)

**Do:**

Replace:
```python
all_data = vector_store.get(include=["metadatas", "documents"])
corpus = [
    {"text": doc, "metadata": meta}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"])
]
init_bm25_index(corpus)
```

With:
```python
# vector_store may be raw Chroma OR a VectorStoreBackend.
if hasattr(vector_store, "iter_all_documents"):
    corpus = list(vector_store.iter_all_documents())
else:
    # Legacy: raw Chroma object passed directly
    all_data = vector_store.get(include=["metadatas", "documents"])
    corpus = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
init_bm25_index(corpus)
```

Same for `make_vector_search_fn` — accept either raw Chroma or a backend. The duck-typed `similarity_search_with_score` already works for both.

**Verify:** Existing tests still pass:
```bash
pytest tests/test_v7_bridge.py -v  # if exists, else trace_v7.py smoke test
python scripts/trace_v7.py --no-chroma "test"
```

**Done when:** Backward compatibility preserved; both raw Chroma and backend objects work.

**Commit:** `refactor(v7): bridge accepts both raw Chroma and VectorStoreBackend`

---

### CARD-2.5 — Wire factory into all callers

**Status:** ⬜ TODO
**Depends on:** CARD-2.4
**Files:** `api.py`, `app.py`, `eval/run_eval.py`, `eval/run_v7_eval.py`, `scripts/measure_cps.py`, `scripts/trace_v7.py`, `index.py`, `src/chroma_helpers.py`

**Do:**

For each caller, replace:
```python
from src.vector_store import load_vector_store
vector_store = load_vector_store()
```
with:
```python
from src.backends.vector_store import get_vector_store_backend
vector_store = get_vector_store_backend(load_existing=True)
```

`index.py`: replace
```python
from src.vector_store import create_vector_store
create_vector_store(chunks)
```
with:
```python
from src.backends.vector_store import get_vector_store_backend
get_vector_store_backend(load_existing=False).create(chunks)
```

`src/chroma_helpers.py:query_chunks_by_range` — refactor to use `backend.get_by_filter`:
```python
def query_chunks_by_range(backend, source: str, start: int, end: int) -> List[Document]:
    try:
        docs = backend.get_by_filter({
            "source": source,
            "chunk_id": {"$gte": start, "$lte": end},
        })
        docs.sort(key=lambda x: x.metadata.get("chunk_id", 0))
        return docs
    except Exception as e:
        logger.error("Error querying chunks %d-%d for %s: %s", start, end, source, e)
        return []
```
(Backend's `get_by_filter` does the AND of conditions implicitly.)

**Verify:**
```bash
# Chroma path (default)
VECTOR_STORE=chroma python scripts/trace_v7.py --no-chroma "test"
# Qdrant path (requires running Qdrant)
docker run -d -p 6333:6333 qdrant/qdrant
VECTOR_STORE=qdrant python index.py  # writes
VECTOR_STORE=qdrant python scripts/trace_v7.py "test"
```

**Done when:** Both backends work end-to-end. No `load_vector_store()` calls remain in production code (search: `grep -rn "load_vector_store" --include="*.py"`).

**Commit:** `refactor: route all vector store access through backend factory`

---

### CARD-2.6 — Document Qdrant in `.env.example`

**Status:** ⬜ TODO
**Depends on:** CARD-2.5
**Files:** `.env.example`

**Do:**
After `CHROMA_DB_PATH` block add:
```bash
# -------------------------------------------------------------------
# Vector Store backend (chroma | qdrant)
# -------------------------------------------------------------------
VECTOR_STORE=chroma

# Qdrant — start local: docker run -p 6333:6333 qdrant/qdrant
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=
# QDRANT_COLLECTION_NAME=documents
```

**Done when:** File parses; instructions readable.

**Commit:** `docs: document Qdrant vector store in .env.example`

---

### CARD-2.7 — GOST indexer pluggable

**Status:** ❌ WONTFIX (decided 2026-05-23)

**Decision:** `index_gosts.py` stays on Chroma. Same reason as CARD-1.5 — GOST is a WTA-specific implementation, will be merged into main pipeline as future work.

---

# Board 3 — Embeddings & Docs

---

### CARD-3.1 — README: "Bring Your Own Backend" section + verifier caveat

**Status:** ⬜ TODO
**Depends on:** CARD-1.4, CARD-2.6
**Files:** `README.md`

**Do:** Add after the `## Stack` section:

````markdown
## Bring Your Own Backend

All backends configurable via `.env`, no code changes:

| Variable | Options | Default | Notes |
|----------|---------|---------|-------|
| `LLM_PROVIDER` | `openai` · `gemini` · `anthropic` | `openai` | Verifier needs JSON-mode → Gemini gives best results; Anthropic/OpenAI fall back to lenient parsing |
| `VECTOR_STORE` | `chroma` · `qdrant` | `chroma` | |
| `EMBEDDING_PROVIDER` | `openai` · `local` · `hf_api` | `openai` | `local` = sentence-transformers, no API key needed |

**Fully local setup (no API except for LLM):**
```bash
LLM_PROVIDER=anthropic
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

**Done when:** Section appears in README, renders correctly on GitHub.

**Commit:** `docs: add Bring Your Own Backend matrix to README`

---

### CARD-3.2 — Embeddings provider hints in `.env.example`

**Status:** ⬜ TODO
**Depends on:** —
**Files:** `.env.example`

**Do:** After `EMBEDDING_MODEL_NAME` line add:
```bash
# Embedding provider options:
#   openai  — text-embedding-3-small (needs OPENAI_API_KEY)
#   local   — sentence-transformers on CPU, no API key
#   hf_api  — Hugging Face Inference API (needs HF_TOKEN)
# For fully local: EMBEDDING_PROVIDER=local + EMBEDDING_MODEL_NAME=ai-forever/sbert_large_nlu_ru
```

**Done when:** File parses.

**Commit:** `docs: clarify embedding provider options in .env.example`

---

## Future work (out of scope for this plan)

- **Merge GOST pipeline into main.** Remove `src/ers_rag/*`, `src/gosts_pipeline.py`, `index_gosts.py`, `/query/gosts` endpoint. Single pipeline handles both safety docs and GOSTs through one index + one LLM (DeepSeek + Chroma as the proof-of-flexibility setup). Validates backend abstraction end-to-end.

---

## Acceptance — overall

Plan is complete when:
- [ ] All Board 1 cards `DONE` (LLM provider unified)
- [ ] All Board 2 cards (excluding 2.7) `DONE` (vector store abstracted, Qdrant works)
- [ ] All Board 3 cards `DONE` (docs updated)
- [ ] `pytest -m unit` is green
- [ ] `python scripts/trace_v7.py "test"` works with at least 2 different LLM providers and both vector stores
