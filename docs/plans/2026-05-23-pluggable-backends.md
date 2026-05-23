# Pluggable Backends — Refactor Plan (Phase 0)

> **For agentic workers:** Pick one card at a time. Update `Status:` field (`TODO → IN PROGRESS → DONE`). Do not start a card whose `Depends on:` is not `DONE`. Each card is atomic and independently committable.

**Goal:** Remove hardcoded LLM and vector store dependencies. Prepare infrastructure so new providers (Anthropic, OpenAI as main, Qdrant, pgvector) can be added later as one-line changes. **Behavior must not change** — same model (Gemini), same backend (Chroma), same eval score (7.9/10).

**Strategy:** Refactor first, add providers later. Each future provider becomes a separate plan + separate eval run.

**Non-goals (deferred to future plans):**
- Adding OpenAI / Anthropic / DeepSeek to main pipeline
- Adding Qdrant / pgvector backends
- Merging GOST pipeline (`src/ers_rag/*`) with main — stays as-is

---

## Code Reality Check

| Aspect | Reality | Refactor target |
|--------|---------|-----------------|
| `get_llm()` in `llm_factory.py` | Only knows `openai`, never called by main pipeline | Add `gemini` to registry, accept `**kwargs` passthrough |
| V7 main LLM | `src/v7/bridge.py:446-458` calls `get_gemini_llm()` directly 4 times with different kwargs | Route through `get_llm(**kwargs)` — kwargs forward as-is |
| Vector store loading | `src/v7/bridge.py:397` takes `vector_store` as arg, but **callers** hardcode `load_vector_store()` from `src/vector_store.py` (Chroma) | Introduce `VectorStoreBackend` protocol + `ChromaBackend` + factory; callers go through factory |
| BM25 corpus build | `src/v7/bridge.py:413` uses `vector_store.get(include=...)` — Chroma-specific | Backend's `iter_all_documents()` |
| GOST pipeline | `src/ers_rag/bridge.py` hardcodes `DeepSeekLLM`; `index_gosts.py` hardcodes Chroma | Out of scope (decided 2026-05-23: WTA-specific, will be unified later as separate task) |

---

# Board 1 — LLM Factory Refactor

Goal: All main-pipeline LLM creation goes through `get_llm(**kwargs)`. Gemini stays the only registered provider. No behavior change.

---

### CARD-1.1 — `get_llm()` accepts kwargs and routes to Gemini

**Status:** ⬜ TODO
**Depends on:** —
**Files:** `src/llm_factory.py`, `tests/test_llm_factory_refactor.py` (new)

**Do:**

1. In `src/llm_factory.py`, add a Gemini factory function that forwards all kwargs to existing `get_gemini_llm`:
```python
def _create_gemini_llm(**kwargs):
    """Forward kwargs (thinking_budget, response_mime_type, etc.) to get_gemini_llm."""
    return get_gemini_llm(**kwargs)
```

2. Update existing `_create_openai_llm` to absorb but ignore Gemini-specific kwargs (so future caller swaps don't break):
```python
def _create_openai_llm(**kwargs):
    # Drop kwargs not understood by OpenAI (kept for forward compatibility)
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

3. Update `_LLM_PROVIDERS`:
```python
_LLM_PROVIDERS = {
    "openai": _create_openai_llm,
    "gemini": _create_gemini_llm,
}
```

4. Update `get_llm()` to log selected provider and accept kwargs (signature already does — verify it does `**kwargs` pass-through).

**Test:** Create `tests/test_llm_factory_refactor.py`:
```python
from unittest.mock import patch, MagicMock
import pytest


@pytest.mark.unit
def test_get_llm_gemini_forwards_kwargs():
    from src.llm_factory import get_llm
    with patch("src.llm_factory.settings") as s, \
         patch("src.llm_factory.get_gemini_llm") as mock_g:
        s.LLM_PROVIDER = "gemini"
        get_llm(thinking_budget=4096, response_mime_type="application/json")
        mock_g.assert_called_once_with(
            thinking_budget=4096, response_mime_type="application/json"
        )


@pytest.mark.unit
def test_get_llm_openai_drops_gemini_kwargs():
    from src.llm_factory import get_llm
    with patch("src.llm_factory.settings") as s, \
         patch("src.llm_factory.ChatOpenAI") as mock_cls:
        s.LLM_PROVIDER = "openai"
        s.MODEL_NAME = "gpt-4o-mini"
        s.TEMPERATURE = 0.0
        s.REQUEST_TIMEOUT = 120.0
        get_llm(thinking_budget=4096)
        call_kwargs = mock_cls.call_args.kwargs
        assert "thinking_budget" not in call_kwargs


@pytest.mark.unit
def test_get_llm_unknown_provider_raises():
    from src.llm_factory import get_llm
    with patch("src.llm_factory.settings") as s:
        s.LLM_PROVIDER = "anthropic"
        with pytest.raises(ValueError, match="anthropic"):
            get_llm()
```

```bash
pytest tests/test_llm_factory_refactor.py -v
```

**Done when:** All 3 tests pass.

**Commit:** `refactor(llm): add gemini to get_llm() registry with kwarg passthrough`

---

### CARD-1.2 — Route V7 bridge through `get_llm()`

**Status:** ⬜ TODO
**Depends on:** CARD-1.1
**Files:** `src/v7/bridge.py` (lines 446–458)

**Do:**

In `src/v7/bridge.py` `init_default_fns()`, replace 4 calls:
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

Update import line to include `get_llm`:
```python
from src.llm_factory import get_gemini_llm, get_llm
```

(Keep `get_gemini_llm` import — `get_vision_llm()` still uses it.)

**Verify:** With existing `.env` (`LLM_PROVIDER=gemini`):
```bash
python scripts/trace_v7.py "для кого проводится повторный инструктаж?"
```
Expected: full pipeline runs, citations returned. Same path as before.

Optional eval (recommended — confirms no behavior change):
```bash
python eval/run_v7_eval.py --n 10
```
Expected: correctness within ±0.1 of 7.9 baseline.

**Done when:** trace_v7 runs end-to-end successfully; eval (if run) shows no regression.

**Commit:** `refactor(v7): route bridge LLM calls through get_llm() factory`

---

### CARD-1.3 — Set default `LLM_PROVIDER=gemini` in settings

**Status:** ⬜ TODO
**Depends on:** CARD-1.2
**Files:** `config/settings.py`, `.env.example`

**Do:**

In `config/settings.py`, change default:
```python
LLM_PROVIDER: str = "gemini"  # was "openai"
```

In `.env.example`, update the LLM_PROVIDER line and comment:
```bash
# LLM Provider — currently supported: gemini
# Future: openai, anthropic, deepseek (see docs/plans/ for roadmap)
LLM_PROVIDER=gemini
```

**Why:** Current production runs on Gemini. Default should match reality so a fresh clone "just works" once `GEMINI_API_KEY` is filled.

**Verify:**
```bash
python -c "from config.settings import settings; print(settings.LLM_PROVIDER)"
# Expected: gemini
```

**Done when:** Default matches production reality.

**Commit:** `chore(config): set LLM_PROVIDER default to gemini (matches prod)`

---

# Board 2 — Vector Store Refactor

Goal: All vector store access goes through `VectorStoreBackend` protocol. `ChromaBackend` is the only implementation. No behavior change.

---

### CARD-2.1 — Define protocol and factory

**Status:** ⬜ TODO
**Depends on:** —
**Files:** `src/backends/__init__.py` (new, empty), `src/backends/vector_store.py` (new), `config/settings.py`

**Do:**

1. In `Settings` after `CHROMA_COLLECTION_NAME`, add:
```python
# Vector store backend — currently supported: chroma
# Future: qdrant, pgvector (see docs/plans/ for roadmap)
VECTOR_STORE: str = "chroma"
```

2. Create empty `src/backends/__init__.py`.

3. Create `src/backends/vector_store.py`:
```python
"""Vector store factory and protocol."""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable
from langchain_core.documents import Document

from config.settings import settings


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Minimum API every backend must implement.

    Future backends (Qdrant, pgvector) implement this same surface — callers
    do not need to change.
    """

    def similarity_search_with_score(
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        """Top-k semantic search."""
        ...

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        """Insert documents. Returns assigned IDs."""
        ...

    def iter_all_documents(self) -> Iterator[dict]:
        """Yield every stored doc as {"text": str, "metadata": dict}.
        Used by BM25 corpus build in src/v7/bridge.py."""
        ...

    def count(self) -> int:
        """Total document count."""
        ...

    def get_by_filter(self, where: dict) -> list[Document]:
        """Metadata filter query. `where` uses Chroma syntax:
        {"field": value} or {"field": {"$gte": N, "$lte": M}}.
        Backends translate as needed."""
        ...


def get_vector_store_backend(load_existing: bool = True) -> VectorStoreBackend:
    """Return the configured backend.

    Args:
        load_existing: True to load existing index, False to start fresh (for index.py).
    """
    backend = settings.VECTOR_STORE.lower()
    if backend == "chroma":
        from src.backends.chroma_backend import ChromaBackend
        return ChromaBackend(load_existing=load_existing)
    raise ValueError(
        f"Unknown VECTOR_STORE={backend!r}. Currently supported: chroma"
    )
```

**Test:** `tests/test_vector_store_factory.py`:
```python
import pytest
from unittest.mock import patch


@pytest.mark.unit
def test_factory_unknown_raises():
    from src.backends.vector_store import get_vector_store_backend
    with patch("src.backends.vector_store.settings") as s:
        s.VECTOR_STORE = "qdrant"
        with pytest.raises(ValueError, match="qdrant"):
            get_vector_store_backend()
```

```bash
pytest tests/test_vector_store_factory.py -v
```

**Done when:** Test passes, settings load with VECTOR_STORE field.

**Commit:** `feat(backends): add VectorStoreBackend protocol and factory`

---

### CARD-2.2 — ChromaBackend implements protocol

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
            self._vs = None  # populated by create()

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
        # Chroma needs explicit $and wrapper for multi-condition filters
        if len(where) > 1 or any(isinstance(v, dict) for v in where.values()):
            conditions = []
            for k, v in where.items():
                if isinstance(v, dict):
                    for op, val in v.items():
                        conditions.append({k: {op: val}})
                else:
                    conditions.append({k: v})
            chroma_where = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        else:
            chroma_where = where
        result = self._vs.get(where=chroma_where)
        return chroma_results_to_documents(result)

    @property
    def raw(self):
        """Escape hatch for legacy code that needs raw Chroma object."""
        return self._vs
```

**Test:** `tests/test_chroma_backend.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


@pytest.mark.unit
def test_chroma_backend_similarity_search_delegates():
    with patch("src.backends.chroma_backend.__import__"), \
         patch("src.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="t"), 0.9)
        ]
        mock_load.return_value = mock_vs
        from src.backends.chroma_backend import ChromaBackend
        backend = ChromaBackend()
        results = backend.similarity_search_with_score("q", k=5)
        assert results == [(Document(page_content="t"), 0.9)]
        mock_vs.similarity_search_with_score.assert_called_once_with("q", k=5)


@pytest.mark.unit
def test_chroma_backend_iter_all_documents():
    with patch("src.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock()
        mock_vs.get.return_value = {
            "documents": ["doc1", "doc2"],
            "metadatas": [{"source": "a"}, {"source": "b"}],
        }
        mock_load.return_value = mock_vs
        from src.backends.chroma_backend import ChromaBackend
        backend = ChromaBackend()
        docs = list(backend.iter_all_documents())
        assert docs == [
            {"text": "doc1", "metadata": {"source": "a"}},
            {"text": "doc2", "metadata": {"source": "b"}},
        ]


@pytest.mark.unit
def test_chroma_backend_satisfies_protocol():
    from src.backends.vector_store import VectorStoreBackend
    from src.backends.chroma_backend import ChromaBackend
    with patch("src.vector_store.load_vector_store"):
        backend = ChromaBackend()
        assert isinstance(backend, VectorStoreBackend)
```

```bash
pytest tests/test_chroma_backend.py -v
```

**Done when:** All 3 tests pass; `isinstance(ChromaBackend(...), VectorStoreBackend)` is True.

**Commit:** `feat(backends): add ChromaBackend implementing VectorStoreBackend protocol`

---

### CARD-2.3 — V7 bridge accepts both raw Chroma and backend

**Status:** ⬜ TODO
**Depends on:** CARD-2.2
**Files:** `src/v7/bridge.py` (lines ~412–418)

**Do:**

In `init_v7_from_chroma()`, replace BM25 corpus build:
```python
# BEFORE
all_data = vector_store.get(include=["metadatas", "documents"])
corpus = [
    {"text": doc, "metadata": meta}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"])
]
init_bm25_index(corpus)

# AFTER
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

`make_vector_search_fn` already duck-types `similarity_search_with_score` — works for both.

`make_section_fetch_fn` uses `vector_store.get(where=...)`. If it doesn't already, wrap calls to prefer `get_by_filter`:
```python
def make_section_fetch_fn(vector_store):
    use_backend = hasattr(vector_store, "get_by_filter")
    def fetch(source: str, start: int, end: int):
        if use_backend:
            return vector_store.get_by_filter({
                "source": source,
                "chunk_id": {"$gte": start, "$lte": end},
            })
        # Legacy path — call existing chroma_helpers logic
        from src.chroma_helpers import query_chunks_by_range
        return query_chunks_by_range(vector_store, source, start, end)
    return fetch
```

(Check actual `make_section_fetch_fn` body first — adapt the wrapper to match its signature.)

**Verify:** Existing smoke test:
```bash
python scripts/trace_v7.py --no-chroma "test"
python scripts/trace_v7.py "для кого проводится повторный инструктаж?"
```

Both should work — first with stub, second with real Chroma via backend.

**Done when:** Both raw Chroma and ChromaBackend can be passed to `init_v7_from_chroma` and the pipeline runs.

**Commit:** `refactor(v7): bridge accepts both raw Chroma and VectorStoreBackend (duck-typed)`

---

### CARD-2.4 — Route callers through factory

**Status:** ⬜ TODO
**Depends on:** CARD-2.3
**Files:** `api.py`, `app.py`, `eval/run_eval.py`, `eval/run_v7_eval.py`, `scripts/measure_cps.py`, `scripts/trace_v7.py`, `index.py`

**Do:**

For each caller listed above, replace:
```python
from src.vector_store import load_vector_store
vector_store = load_vector_store()
```
with:
```python
from src.backends.vector_store import get_vector_store_backend
vector_store = get_vector_store_backend(load_existing=True)
```

For `index.py`, replace:
```python
from src.vector_store import create_vector_store
create_vector_store(chunks)
```
with:
```python
from src.backends.vector_store import get_vector_store_backend
get_vector_store_backend(load_existing=False).create(chunks)
```

**Verify (must run all):**
```bash
# 1. No load_vector_store imports remain in production code
grep -rn "load_vector_store\|create_vector_store" --include="*.py" \
  --exclude-dir=tests --exclude-dir=__pycache__ --exclude-dir=venv \
  src/ api.py app.py index.py scripts/ eval/
# Expected: only matches inside src/vector_store.py and src/backends/chroma_backend.py

# 2. Trace works
python scripts/trace_v7.py "test"

# 3. API starts
uvicorn api:app --port 8503 &
sleep 5
curl -s http://localhost:8503/health
kill %1
```

**Done when:** Grep shows no leaking imports; trace_v7 and /health both work.

**Commit:** `refactor: route all vector store access through backend factory`

---

### CARD-2.5 — Adapt `chroma_helpers.query_chunks_by_range` (optional)

**Status:** ⬜ TODO
**Depends on:** CARD-2.4
**Files:** `src/chroma_helpers.py`

**Do:**

This function is called with raw Chroma in current code. Decide:
- Option A: leave as-is (still works because Chroma is the only backend now)
- Option B: refactor to accept either raw Chroma or backend, mirror CARD-2.3 pattern

**Recommendation:** Option A for this plan. Mark as TODO in the docstring:
```python
def query_chunks_by_range(vs, source: str, start: int, end: int) -> List[Document]:
    """Query Chroma for chunks in [start, end] range for a given source.

    TODO: When a non-Chroma backend is added, refactor to use
    backend.get_by_filter() — see src/backends/vector_store.py.
    """
    ...
```

**Done when:** Docstring updated.

**Commit:** `docs(chroma_helpers): mark for backend abstraction when second backend lands`

---

# Board 3 — Documentation

---

### CARD-3.1 — README: architecture-ready note

**Status:** ⬜ TODO
**Depends on:** CARD-1.3, CARD-2.4
**Files:** `README.md`

**Do:** Add a short note after the `## Stack` section:

````markdown
## Backend abstraction

LLM and vector store are accessed through factory layers (`src/llm_factory.py`, `src/backends/`), making it straightforward to add new providers without touching pipeline code. Currently shipped:

| Layer | Provider | Configurable via |
|-------|----------|------------------|
| LLM   | Gemini   | `LLM_PROVIDER` (gemini\|openai) |
| Vector store | Chroma | `VECTOR_STORE` (chroma) |
| Embeddings | OpenAI / local / hf_api | `EMBEDDING_PROVIDER` |

Roadmap (see `docs/plans/`): Anthropic, OpenAI for main pipeline, Qdrant, pgvector.
````

**Done when:** Section appears in README, renders on GitHub.

**Commit:** `docs: document backend abstraction architecture in README`

---

### CARD-3.2 — `.env.example` final pass

**Status:** ⬜ TODO
**Depends on:** CARD-2.1
**Files:** `.env.example`

**Do:** Ensure `.env.example` reflects current shipped state:
- `LLM_PROVIDER=gemini` (default)
- Comment mentioning `openai` already in registry
- `VECTOR_STORE=chroma` (only option for now)
- Embeddings already documented

Add after `CHROMA_COLLECTION_NAME` line:
```bash
# -------------------------------------------------------------------
# Vector Store backend — currently: chroma
# Future: qdrant, pgvector (see docs/plans/)
# -------------------------------------------------------------------
VECTOR_STORE=chroma
```

**Done when:** File parses, fully reflects current shipped state.

**Commit:** `docs: update .env.example with vector store section`

---

## Acceptance — overall

- [ ] All Board 1 cards `DONE` (LLM factory refactored, get_gemini_llm calls in V7 bridge replaced)
- [ ] All Board 2 cards `DONE` (vector store abstracted, all callers via factory)
- [ ] All Board 3 cards `DONE` (docs updated)
- [ ] `pytest -m unit` is green
- [ ] `python scripts/trace_v7.py "test"` works
- [ ] Optional: eval correctness unchanged (within ±0.1 of 7.9 baseline)
- [ ] `grep -rn "load_vector_store" --include="*.py" --exclude-dir=tests` shows only definitions in `src/vector_store.py` and `src/backends/chroma_backend.py`

---

## After this plan

Each future provider becomes its own small plan:
- `docs/plans/YYYY-MM-DD-add-openai-llm.md` — register `openai` for main pipeline, eval, document
- `docs/plans/YYYY-MM-DD-add-anthropic-llm.md` — same pattern
- `docs/plans/YYYY-MM-DD-add-qdrant.md` — add `QdrantBackend`, eval, document
- `docs/plans/YYYY-MM-DD-merge-gost-pipeline.md` — drop `src/ers_rag/*`, unify on backend abstraction

Each ships as: one new file + one eval row in README's "Tested with" table.
