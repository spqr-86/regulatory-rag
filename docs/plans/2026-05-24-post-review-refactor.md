# Post-Review Refactor — Security, Concurrency, Architecture Cleanup

**Status: 🟡 TODO** — created 2026-05-24 after two reviews (code-reviewer + adversarial-review skill).

> **For agentic workers:** Pick one card at a time. Update `Status:` field (`TODO → IN PROGRESS → DONE`). Do not start a card whose `Depends on:` is not `DONE`. Each card is atomic and independently committable. Reviews referenced as `[N1]`, `[#3]`, `[S2]` etc. point to findings in `REVIEW_2026-05-24.md` (code-reviewer) and `REVIEW_2026-05-24_adversarial.md` (skill). Read those first.

**Goal:** Fix concrete security/concurrency/architecture issues found in both reviews. No behavior change for end-user (same answers, same correctness 7.9/10) — but production posture, security surface, and senior-readability all improve.

**Strategy:** Cards grouped by priority into 4 boards:
- **Board 1 (P0)** — must ship before 2026-05-27 GenAI Lab interview. Public-API hardening.
- **Board 2 (P1)** — supply chain + concurrency. Should land within 2 weeks.
- **Board 3 (P2)** — architecture cleanup so duck-typing/dead-code don't bite the next backend.
- **Board 4 (P3)** — code quality polish (coding standards, structlog, type hints). No urgency.

**Non-goals:**
- Adding new backends (covered in separate follow-up plans).
- Re-architecting the pipeline (current V7 graph stays).
- Performance optimization (separate plan: Eval Cost Optimization).
- Fixing GOST pipeline (`src/ers_rag/*`) — WTA-specific, deferred.

---

## Code Reality Check

| Issue | Where it lives | Current state | Refactor target |
|-------|---------------|---------------|-----------------|
| Exception leak in API | `api.py:108, 163` | `detail=f"pipeline error: {exc}"` exposes paths/versions/keys | `detail="internal error"` + structlog with request_id |
| No rate limit / length cap | `api.py` `/query`, `/query/gosts` | Any caller can flood Gemini quota with 100KB inputs | `slowapi` rate limit + `max_length=2000` on question |
| `trust_remote_code=True` | `src/llm_factory.py:190` | Local embeddings can RCE via HF model swap | `trust_remote_code=False` (default-safe) |
| Module-global state race | `src/v7/bridge.py:413-498` | 7 `set_*_fn` calls + `init_bm25_index` without lock | `threading.Lock` OR documented "restart required" |
| pysqlite3 vs system sqlite | `app.py:8-10` vs `eval/run_v7_eval.py:32-38` | Eval uses pysqlite3, prod uses system sqlite | Unify driver (or document version requirement) |
| Pickle RCE in cache | `src/file_handler.py:402-410` | `pickle.load` from `document_cache/` is RCE if filesystem write | Custom JSON serializer for Document |
| Duck-typing inverted markers | `src/v7/bridge.py:147, 430` | Two `hasattr(vs, "_collection")` checks with opposite semantics | `isinstance(vs, VectorStoreBackend)` (Protocol is runtime_checkable) |
| Disk-fill via visual_proof | `src/agent_tools.py:323-336` | Up to 600MB per PNG, no global cap, no cleanup | Bbox size cap in pixels + LRU cleanup cron |
| Empty-string DoS in section_fetch | `src/v7/bridge.py:140-153` | `if not section` catches None but not `""` | Truthy check on all metadata fields + explicit limit |
| Silent kwarg drop | `src/llm_factory.py:49-50` | `kwargs.pop` without warning → thinking_budget lost on openai branch | `logger.warning` on drop |
| `init_v7_from_chroma` lies | `src/v7/bridge.py:413` | Name says "from_chroma" but accepts abstract backend | Rename to `init_v7_pipeline` |
| Dead code | `src/file_handler.py:65-87` | `_split_into_children` never called in main pipeline | Delete |
| LangChain private API monkey-patch | `src/llm_factory.py:148-156` | `llm._build_request_config` patched, no version pin | Pin `langchain-google-genai==X.Y.Z` + comment |
| Broad `except Exception` | `eval/run_v7_eval.py:155, 234, 247` | Coding standards forbid | Narrow to specific exception types |
| `evaluate_correctness` prompt hardcoded | `eval/run_v7_eval.py:117-138` | Bypasses `prompts/registry.yaml` pattern | Move to registry |
| Global `_pipeline: dict` | `api.py:29` | Anti-pattern; should use `app.state` | Migrate to `app.state` |

---

# Board 1 — P0: Pre-Interview Security Hardening (deadline 2026-05-27)

Goal: Close the three security holes any senior would point at within the first 5 minutes of code review. Each card is small (≤30 min). All cards in this board must be `DONE` before 2026-05-27.

---

### CARD-1.1 — Hide internal exception details in public API

**Status:** TODO
**Depends on:** —
**Findings:** `[#3]`, `[S1]` (exploit PoC)
**Files:** `api.py`

**Do:**

1. Add request ID middleware so logs and client responses can be correlated:

```python
import uuid
from fastapi import Request

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response
```

2. Replace exception detail strings in both endpoints. Find:

```python
# api.py:108
raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc
# api.py:163
raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc
```

Replace with:

```python
rid = getattr(request.state, "request_id", "no-rid")
logger.error("api.query: pipeline error", request_id=rid, question=req.question[:80], error=str(exc), exc_info=True)
raise HTTPException(status_code=500, detail=f"internal error (request_id={rid})") from exc
```

(Add `request: Request` parameter to both endpoint signatures.)

3. Same treatment for `/health` 503 case (`api.py:201`) — return generic `"not ready"` instead of `"pipeline not ready"`.

4. Same treatment for `/query/gosts` 503 with chroma_db_gosts mention (`api.py:153-156`) — generic message.

**Verify:**

```bash
# Start API locally
source venv/bin/activate
uvicorn api:app --port 8503 &
sleep 5

# Trigger error with oversized question (200KB)
curl -s -X POST http://localhost:8503/query \
  -H 'Content-Type: application/json' \
  --data-raw "{\"question\":\"$(python3 -c 'print("A"*200000)')\"}" \
  -m 30 | jq .

# Expected: detail="internal error (request_id=abc123)" — no paths, no langchain versions
# Check logs (tmux a -t sia, or systemd journal): full stack trace WITH request_id

kill %1
```

**Done when:**
- `curl` returns generic error with `request_id`, no internal paths.
- Logs contain full traceback with matching `request_id`.
- `X-Request-ID` header present on every response.

**Commit:** `feat(api): hide internal exception details, add request_id correlation`

---

### CARD-1.2 — Add rate limit and input length cap

**Status:** TODO
**Depends on:** —
**Findings:** `[#4]` (no rate limit), `[exploit playbook #2]`
**Files:** `api.py`, `requirements.txt`

**Do:**

1. Install `slowapi`:

```bash
source venv/bin/activate
pip install slowapi
echo "slowapi>=0.1.9" >> requirements.txt
```

2. Add length cap on `question` field via Pydantic:

```python
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
```

3. Add rate limit decorator. Top of `api.py` after imports:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
```

4. Decorate endpoints:

```python
@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
def query(request: Request, req: QueryRequest) -> QueryResponse:
    ...
```

(Same for `/query/gosts`. Note: `request: Request` becomes required for slowapi.)

5. Custom 429 handler that doesn't leak detail:

```python
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: FastAPIRequest, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "rate limit exceeded — please slow down"},
    )
```

**Verify:**

```bash
# Length cap
curl -s -X POST http://localhost:8503/query \
  -H 'Content-Type: application/json' \
  -d "{\"question\":\"$(python3 -c 'print("A"*3000)')\"}" | jq .
# Expected: 422 "String should have at most 2000 characters"

# Rate limit (11 quick requests, 11th should 429)
for i in $(seq 1 11); do
  curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:8503/query \
    -H 'Content-Type: application/json' \
    -d '{"question":"test"}'
done
# Expected: first 10 return 200/500, 11th returns 429
```

**Done when:**
- Oversized question returns 422.
- 11th request in a minute returns 429.
- Normal use (≤10/min from one IP) unaffected.

**Commit:** `feat(api): add rate limit (10/min per IP) and 2000-char question cap`

---

### CARD-1.3 — Disable `trust_remote_code` in local embeddings

**Status:** TODO
**Depends on:** —
**Findings:** `[#5]`, `[S2]` (RCE via env-supplied model name)
**Files:** `src/llm_factory.py`

**Do:**

In `_create_local_embeddings()`:

```python
# BEFORE (src/llm_factory.py:186-192)
def _create_local_embeddings():
    model = settings.EMBEDDING_MODEL_NAME
    return HuggingFaceEmbeddings(
        model_name=model or "ai-forever/sbert_large_nlu_ru",
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

# AFTER
def _create_local_embeddings():
    model = settings.EMBEDDING_MODEL_NAME
    return HuggingFaceEmbeddings(
        model_name=model or "ai-forever/sbert_large_nlu_ru",
        # trust_remote_code=False (default) — never execute arbitrary code from HF model repos.
        # If a model legitimately needs it, vet the repo first then add explicit allow-list.
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
```

**Verify:**

```bash
source venv/bin/activate
# Quick import check (ai-forever/sbert_large_nlu_ru does not need trust_remote_code)
python -c "
from src.llm_factory import _create_local_embeddings
from config.settings import settings
settings.EMBEDDING_MODEL_NAME = 'ai-forever/sbert_large_nlu_ru'
emb = _create_local_embeddings()
print(emb.embed_query('test')[:5])
"
# Expected: list of 5 floats, no error
```

**Done when:**
- `trust_remote_code` keyword no longer in source.
- Smoke test loads default local model without errors.

**Commit:** `security(llm): disable trust_remote_code in local embeddings (RCE hardening)`

---

# Board 2 — P1: Supply Chain + Concurrency (within 2 weeks)

Goal: Close vectors that don't have immediate exploit but are landmines as soon as the system grows past one worker / one user / one developer with `.env` access.

---

### CARD-2.1 — Replace pickle cache in `file_handler.py` with JSON serialization

**Status:** TODO
**Depends on:** —
**Findings:** `[N5]` (pickle deserialization RCE) — `src/file_handler.py` part
**Files:** `src/file_handler.py`

**Do:**

1. Verify actual signatures first (do NOT trust quoted snippets — read the file):
```bash
grep -n "_save_to_cache\|_load_from_cache\|_cache_path_for\|import pickle" src/file_handler.py
```
Expected hits:
- line 7: `import pickle`
- line 396: `def _cache_path_for(self, file_hash: str) -> Path:` (returns `<hash>.pkl`)
- line 402: `def _save_to_cache(self, chunks: List[Document], cache_path: Path) -> None:` (chunks first, cache_path second)
- line 405: `pickle.dump(data, f)` — wraps `CacheEntry(timestamp=..., chunks=chunks)`
- line 407: `def _load_from_cache(self, cache_path: Path) -> List[Document]:`
- line 409: `data: CacheEntry = pickle.load(f)`

2. Replace cache extension in `_cache_path_for`:
```python
def _cache_path_for(self, file_hash: str) -> Path:
    key = hashlib.sha256(f"{file_hash}:{PIPELINE_VERSION}".encode("utf-8")).hexdigest()
    return self.cache_dir / f"{key}.json"  # was f"{key}.pkl"
```

3. Rewrite save/load to JSON (preserve EXISTING parameter order — chunks first):
```python
import json

def _document_to_dict(doc: Document) -> dict:
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata or {})}

def _dict_to_document(d: dict) -> Document:
    return Document(page_content=d["page_content"], metadata=d.get("metadata") or {})

def _save_to_cache(self, chunks: List[Document], cache_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "timestamp": datetime.now().timestamp(),
        "chunks": [_document_to_dict(c) for c in chunks],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False))

def _load_from_cache(self, cache_path: Path) -> List[Document]:
    raw = json.loads(cache_path.read_text())
    if raw.get("schema_version") != 1:
        raise ValueError(f"unsupported cache schema: {raw.get('schema_version')}")
    return [_dict_to_document(d) for d in raw["chunks"]]
```

Note: `datetime.now().timestamp()` matches the existing pattern in the file (already imported as `from datetime import datetime, timedelta`). Do NOT use `datetime.now(UTC)` — UTC import is not present and would require Python 3.11+ + extra import.

4. Remove now-unused `pickle` import + `CacheEntry` if it becomes unused. Grep:
```bash
grep -n "CacheEntry\|import pickle" src/file_handler.py
```
If `CacheEntry` is only referenced inside `_save_to_cache`/`_load_from_cache`, remove its `@dataclass` / class definition too. Same for `import pickle` (line 7).

5. **One-shot migration of old `.pkl` files (with safeguard — do NOT auto-delete).**

Add a separate `scripts/migrate_cache_to_json.py` (do NOT auto-run from `__init__` or `process()` — too dangerous):
```python
"""One-time migration: convert document_cache/*.pkl → *.json.

Run MANUALLY after CARD-2.1 lands and verified working:
    python scripts/migrate_cache_to_json.py
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from src.file_handler import _document_to_dict

CACHE_DIR = Path("document_cache")

def main() -> int:
    converted = 0
    failed = 0
    for pkl_path in CACHE_DIR.glob("*.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)  # UNSAFE — only run on trusted cache dir
            chunks = data.chunks
            payload = {
                "schema_version": 1,
                "timestamp": data.timestamp,
                "chunks": [_document_to_dict(c) for c in chunks],
            }
            json_path = pkl_path.with_suffix(".json")
            json_path.write_text(json.dumps(payload, ensure_ascii=False))
            pkl_path.unlink()
            converted += 1
        except Exception as exc:
            print(f"FAIL {pkl_path.name}: {exc}")
            failed += 1
    print(f"Converted {converted}, failed {failed}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
```

The new pipeline (after the rewrite) will simply skip `.pkl` files (only matches `*.json` via `_cache_path_for`). Stale `.pkl` files are harmless until migration is run. If migration is skipped, Docling will just regenerate the cache (slow, but safe).

**Verify:**

```bash
source venv/bin/activate

# 1. Round-trip on a tiny synthetic case (no PDF needed):
python -c "
from src.file_handler import DocumentProcessor, _document_to_dict, _dict_to_document
from langchain_core.documents import Document
from pathlib import Path
import tempfile, json

doc = Document(page_content='hello', metadata={'source': 'test.pdf', 'page': 1})
d = _document_to_dict(doc)
assert d == {'page_content': 'hello', 'metadata': {'source': 'test.pdf', 'page': 1}}
back = _dict_to_document(d)
assert back.page_content == 'hello' and back.metadata['source'] == 'test.pdf'
print('Doc round-trip OK')
"

# 2. Full pipeline cache round-trip (slow — uses real PDF):
python -c "
from src.file_handler import DocumentProcessor
from pathlib import Path
proc = DocumentProcessor()
chunks = proc.process([Path('source_docs/2464.pdf')])
print(f'Processed {len(chunks)} chunks')
chunks2 = proc.process([Path('source_docs/2464.pdf')])  # should hit cache
assert len(chunks) == len(chunks2)
assert chunks[0].page_content == chunks2[0].page_content
print('Cache round-trip OK')
"

# 3. No .pkl extension produced by new code:
ls document_cache/ | grep -c '\.pkl$' && echo "Stale pickles present (run migrate_cache_to_json.py)" || echo "OK: no pickle files"

# 4. Existing tests still pass:
pytest tests/test_index_cache_invalidation.py -v
```

**Done when:**
- `_cache_path_for` returns `.json` extension.
- `_save_to_cache` / `_load_from_cache` use JSON, parameter order unchanged.
- Round-trip preserves `page_content` and `metadata` exactly.
- `pickle` import removed from `src/file_handler.py` (if no other usage).
- `tests/test_index_cache_invalidation.py` green.
- Migration script exists but is NOT auto-invoked.

**Commit:** `security(cache): replace pickle with JSON in DocumentProcessor cache`

---

### CARD-2.1b — Replace pickle BM25 cache in `final_chain.py`

**Status:** TODO
**Depends on:** —
**Findings:** `[N5]` second part — `src/final_chain.py` was missed in first pass of CARD-2.1
**Files:** `src/final_chain.py`

**Do:**

1. Verify pickle locations:
```bash
grep -n "pickle" src/final_chain.py
```
Expected:
- line 1: `import pickle`
- line 70: `keyword_retriever = pickle.load(f)` (loading BM25Retriever from `.bm25_cache.pkl`)
- line 86: `pickle.dump(keyword_retriever, f)`

2. **BM25Retriever object is non-trivial to JSON-serialize** (contains numpy arrays + tokenized corpus). Three options:

   **Option A (recommended): Rebuild on startup, no cache.** BM25 build is O(n_chunks) — on current 1973 chunks takes ~1-2 seconds. For prod 30k+ chunks could be 20-30s startup penalty. Acceptable. Delete cache code entirely.

   **Option B: Pickle stays, but only load if file is owned by current user and not world-writable.** Defense-in-depth (mitigates "attacker writes to cache dir" vector but still leaves RCE if attacker IS the user). Less invasive but doesn't fully close the hole.

   **Option C: Serialize BM25 corpus + IDF table to JSON, rebuild retriever from JSON.** Requires understanding BM25Retriever internals. Higher complexity, full fix.

   Pick Option A by default. Justify in commit message.

3. For Option A — find and delete the load/save block (~lines 65-90 in `src/final_chain.py`). Keep the `BM25Retriever.from_documents(...)` call, just remove the cache wrap.

4. Update `.gitignore` — `.bm25_cache.pkl` ignore line can stay (harmless if file is no longer produced).

5. `index.py:38-41` references `.bm25_cache.pkl` for cache invalidation — change to no-op or delete (verify with grep first).

**Verify:**
```bash
grep -rn "bm25_cache\|pickle" src/final_chain.py index.py
# Expected: no pickle in final_chain.py; index.py reference also removed or commented

# Smoke test:
python scripts/trace_v7.py "тест"
# Should still work — BM25 used inside V7 graph via init_bm25_index, not final_chain
```

**Done when:**
- `pickle` import gone from `src/final_chain.py`.
- BM25 rebuilt on startup; first query may be 1-2s slower, subsequent unaffected.
- No `.bm25_cache.pkl` produced on fresh run.

**Commit:** `security(cache): rebuild BM25 on startup, drop pickle cache in final_chain`

---

### CARD-2.2 — Add lock around `init_v7_from_chroma` (concurrency hardening)

**Status:** TODO
**Depends on:** —
**Findings:** `[N1]` (race condition with concrete exploit)
**Files:** `src/v7/bridge.py`

**Do:**

1. Top of `src/v7/bridge.py`, add module-level lock:

```python
import threading

_init_lock = threading.RLock()
```

2. Wrap `init_v7_from_chroma` body:

```python
def init_v7_from_chroma(vector_store, llm_provider: str | None = "gemini") -> None:
    with _init_lock:
        # ... existing body unchanged ...
```

3. If `make_section_fetch_fn` mutates any global, lock there too (read code to confirm).

4. Document the lock semantics in docstring:

```python
def init_v7_from_chroma(vector_store, llm_provider: str | None = "gemini") -> None:
    """Initialize V7 pipeline from a vector store backend.

    Thread-safe: protected by module-level RLock. Concurrent calls will
    serialize. The 7 module-global injectors (set_*_fn) and BM25 corpus build
    are not atomic individually — the lock prevents readers observing a half-
    initialized pipeline state during reindex.

    Trade-off: while reindex is in progress (~30s for current corpus),
    incoming queries will wait. This is preferable to silent garbage results
    from partially-updated state.
    """
```

**Verify:**

```bash
source venv/bin/activate
# Concurrent init test
python -c "
import threading
from unittest.mock import MagicMock
from src.v7 import bridge

errors = []
def init():
    try:
        vs = MagicMock()
        vs.get.return_value = {'documents': ['x'], 'metadatas': [{}]}
        bridge.init_v7_from_chroma(vs, llm_provider=None)
    except Exception as e:
        errors.append(e)

threads = [threading.Thread(target=init) for _ in range(10)]
for t in threads: t.start()
for t in threads: t.join()
print(f'Errors: {len(errors)}')
assert not errors, errors
print('Concurrent init OK')
"
```

**Done when:**
- 10 concurrent `init_v7_from_chroma` calls complete without errors.
- Smoke test (`scripts/trace_v7.py`) still works.

**Commit:** `fix(v7): serialize init_v7_from_chroma with RLock (concurrency hardening)`

---

### CARD-2.3 — Empty-string guard in `make_section_fetch_fn`

**Status:** TODO
**Depends on:** —
**Findings:** `[N2]` (DoS via empty metadata triggering full table scan)
**Files:** `src/v7/bridge.py`, `src/backends/chroma_backend.py`

**Do:**

1. In `src/v7/bridge.py:143`, harden the guard:

```python
# BEFORE
if not section or not source:
    return []

# AFTER
section = (section or "").strip()
source = (source or "").strip()
if not section or not source:
    return []
```

2. In `src/backends/chroma_backend.py:46-64`, add explicit limit parameter:

```python
def get_by_filter(self, where: dict, limit: int = 200) -> list[Document]:
    """Metadata filter query. Limit caps results to prevent unbounded scans."""
    from src.chroma_helpers import chroma_results_to_documents
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
    result = self._vs.get(where=chroma_where, limit=limit)
    return chroma_results_to_documents(result)
```

3. Update `VectorStoreBackend.get_by_filter` Protocol signature in `src/backends/vector_store.py:43-49` to include `limit: int = 200`.

4. Update bridge call site to pass explicit limit:

```python
docs = vector_store.get_by_filter(
    {"parent_section": section, "source": source},
    limit=max_section_chunks,
)[:max_section_chunks]
```

5. Update tests in `tests/test_chroma_backend.py` to cover empty-string + limit cases.

**Verify:**

```bash
source venv/bin/activate
# Empty-string short-circuit
python -c "
from unittest.mock import MagicMock
from src.v7.bridge import make_section_fetch_fn

vs = MagicMock()
fn = make_section_fetch_fn(vs)
result = fn([{'metadata': {'parent_section': '', 'source': 'x.pdf'}}])
assert result == [], result
assert not vs.get_by_filter.called, 'Should not call backend for empty section'
print('Empty-string guard OK')
"

# Existing tests
pytest tests/test_chroma_backend.py tests/v7/ -v
```

**Done when:**
- Empty/whitespace section short-circuits, no backend call.
- `get_by_filter` accepts `limit` parameter.
- All existing tests green.

**Commit:** `fix(bridge): guard section_fetch against empty metadata, add limit to get_by_filter`

---

### CARD-2.4 — Unify SQLite driver (eval vs prod)

**Status:** TODO
**Depends on:** —
**Findings:** `[N3]` (pysqlite3 in eval, system sqlite in prod)
**Files:** `app.py`, `eval/run_v7_eval.py`, `CLAUDE.md`

**Do:**

Pick ONE strategy and apply consistently:

**Strategy A (recommended) — system sqlite everywhere, document requirement:**

1. Delete pysqlite3 swap in `eval/run_v7_eval.py:32-38`.

2. Add to `CLAUDE.md` under `## Deployment (VPS)`:

```markdown
**SQLite requirement:** ChromaDB needs sqlite ≥ 3.35. Check with `sqlite3 --version`.
Ubuntu 22.04+ ships sqlite 3.37 (OK). On older systems (Ubuntu 20.04 has 3.31),
install `pysqlite3-binary` and add the swap stanza at the top of any entry-point
that imports chromadb directly.
```

3. Verify VPS:

```bash
sqlite3 --version  # should be ≥ 3.35
```

**Strategy B — pysqlite3 everywhere:**

1. Add `pysqlite3-binary` to `requirements.txt`.
2. Move the swap stanza to a shared module `src/sqlite_compat.py`:
   ```python
   import sys
   try:
       import pysqlite3
       sys.modules["sqlite3"] = pysqlite3
   except ImportError:
       pass
   ```
3. Import it at the very top (before any `chromadb` import) in `app.py`, `api.py`, `eval/run_v7_eval.py`, `scripts/trace_v7.py`, `scripts/measure_cps.py`, `index.py`.

4. Delete the Streamlit-Cloud-specific check in `app.py:8-10`.

**Verify (whichever strategy):**

```bash
# Same query through eval and through API should hit same chunks/score
source venv/bin/activate

# Eval path
python eval/run_v7_eval.py --skip-judge --n 1 --filter "повторный инструктаж"

# API path
curl -s -X POST http://localhost:8503/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "для кого проводится повторный инструктаж?"}' | jq '.passages[0].text' | head -c 200

# Visually compare top passage — should match
```

**Done when:**
- One driver is used everywhere (verify with `grep -rn "pysqlite3" --include="*.py"`).
- Eval and API return identical top passage for known query.
- CLAUDE.md documents the requirement.

**Commit:** `fix(sqlite): unify SQLite driver between eval and prod (was: split brain)`

---

### CARD-2.5 — Pin `langchain-google-genai` version

**Status:** TODO
**Depends on:** —
**Findings:** `[#12]` (monkey-patch on private `_build_request_config`)
**Files:** `requirements.txt`, `src/llm_factory.py`

**Do:**

1. Check current installed version:
   ```bash
   source venv/bin/activate
   pip show langchain-google-genai | grep -i version
   ```

2. Pin exact version in `requirements.txt`. Find the line (probably `langchain-google-genai` or `langchain-google-genai>=...`) and replace with `langchain-google-genai==X.Y.Z`.

3. Add comment in `src/llm_factory.py` above the monkey-patch (line ~148):

```python
# WARNING: monkey-patching private `_build_request_config` API of
# ChatGoogleGenerativeAI. This breaks if langchain-google-genai is upgraded
# without re-validating the patch — version is pinned in requirements.txt.
# To upgrade: bump pin, re-run scripts/trace_v7.py and verify pipeline ends
# with answer (not duplicate tool calls in trace).
if AutomaticFunctionCallingConfig is not None:
    _original_build = llm._build_request_config
    ...
```

**Verify:**

```bash
grep "langchain-google-genai==" requirements.txt
# Expected: pinned version line
```

**Done when:**
- `requirements.txt` shows `langchain-google-genai==X.Y.Z`.
- Comment explains why version is pinned and how to upgrade safely.

**Commit:** `deps: pin langchain-google-genai (monkey-patched private API)`

---

### CARD-2.6 — LRU cleanup for `static/visuals/` + bbox area cap

**Status:** TODO
**Depends on:** —
**Findings:** `[N4]` (disk-fill DoS via /query)
**Files:** `src/agent_tools.py`, `scripts/cleanup_visuals.py` (new)

**Do:**

1. Verify actual `_validate_bbox` contract first:
```bash
sed -n '193,231p' src/agent_tools.py
```
Real contract (do NOT replace with `raise`):
- `MAX_BBOX_DIM = 10_000.0  # PDF user-units` (line 193) — coords are in **PDF points** (1/72 inch), not pixels.
- `_validate_bbox(bbox) -> str | None` returns error string or None. Caller (line 244-246) does `bbox_err = _validate_bbox(bbox); if bbox_err: return bbox_err`. Match this contract.
- `pix = page.get_pixmap(clip=rect, dpi=150)` (line 323) does pt→px conversion. Pixel area = (pt_width × 150/72) × (pt_height × 150/72) = pt_area × (150/72)².

2. Add area cap in `_validate_bbox` matching existing return-string contract. Calc:
- Cap target: 4 megapixels output (~28 cm × 28 cm at 150 DPI).
- In PDF points squared: `4_000_000 / (150/72)² ≈ 921_600 pt²`.
- Sanity: A4 page area = 595 × 842 ≈ 501_000 pt². So 921_600 pt² ≈ 1.84 × A4. Comfortable headroom for full-page diagrams, blocks 10000×10000 abuse (10^8 pt² = ~434 MP).

```python
# Add near MAX_BBOX_DIM (line 193):
MAX_BBOX_AREA_PT2 = 921_600.0  # ~4 MP at 150 DPI; ~1.84 A4 pages

# Extend _validate_bbox — append BEFORE `return None`:
def _validate_bbox(bbox) -> str | None:
    """Возвращает None если bbox валиден, иначе текст ошибки."""
    import math
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "Error: bbox must be a list of 4 floats [left, top, right, bottom]."
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return "Error: bbox values must be numeric."
    if not all(math.isfinite(v) for v in vals):
        return "Error: bbox values must be finite."
    if any(v < 0 or v > MAX_BBOX_DIM for v in vals):
        return f"Error: bbox values must be within [0, {MAX_BBOX_DIM}]."
    left, top, right, bottom = vals
    if right <= left or abs(bottom - top) < 1e-6:
        return "Error: bbox has zero/negative area."
    area = (right - left) * (bottom - top)
    if area > MAX_BBOX_AREA_PT2:
        return f"Error: bbox area {area:.0f}pt² exceeds max {MAX_BBOX_AREA_PT2:.0f}pt² (≈4 MP at 150 DPI)."
    return None
```

3. Create `scripts/cleanup_visuals.py`:

```python
"""LRU cleanup for static/visuals/. Run via cron — see CLAUDE.md."""
from __future__ import annotations
from pathlib import Path
import time

VISUALS_DIR = Path("static/visuals")
MAX_AGE_DAYS = 7
MAX_TOTAL_MB = 100

def cleanup() -> None:
    if not VISUALS_DIR.exists():
        return
    now = time.time()
    cutoff = now - MAX_AGE_DAYS * 86400
    # Age-based pass
    for f in VISUALS_DIR.glob("*.png"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
    # Size-based pass (LRU)
    files = sorted(
        ((f.stat().st_mtime, f.stat().st_size, f) for f in VISUALS_DIR.glob("*.png")),
        key=lambda x: x[0],
    )
    total = sum(s for _, s, _ in files)
    max_bytes = MAX_TOTAL_MB * 1024 * 1024
    while total > max_bytes and files:
        _, size, f = files.pop(0)
        f.unlink()
        total -= size

if __name__ == "__main__":
    cleanup()
```

4. Document cron entry in CLAUDE.md (manual install — do NOT auto-add). Use correct VPS path `/home/petr/projects/ai/safety-incident-analyzer/`:

```bash
# Add to crontab manually: crontab -e
0 3 * * * cd /home/petr/projects/ai/safety-incident-analyzer && /home/petr/projects/ai/safety-incident-analyzer/venv/bin/python scripts/cleanup_visuals.py >> /tmp/cleanup_visuals.log 2>&1
```

**Verify:**

```bash
# Bbox area cap (returns string, doesn't raise):
python -c "
from src.agent_tools import _validate_bbox
err = _validate_bbox((0, 0, 10000, 10000))
assert err and 'area' in err, f'expected area error, got {err!r}'
print(f'OK: {err}')

# Within cap should pass:
err = _validate_bbox((100, 100, 500, 500))  # 400×400 = 160_000 pt² — well within
assert err is None, f'unexpected error: {err}'
print('Small bbox accepted')
"

# Cleanup script runs:
python scripts/cleanup_visuals.py
ls static/visuals/ 2>/dev/null | wc -l
```

**Done when:**
- `_validate_bbox` returns area-error string for over-cap bbox (does NOT raise).
- `_validate_bbox` returns None for sub-cap bbox.
- `cleanup_visuals.py` runs without error.
- CLAUDE.md documents cron entry with correct path.

**Commit:** `security(visuals): cap bbox area in PDF points, add LRU cleanup script`

---

### CARD-2.7 — Replace silent kwarg drop with warning

**Status:** TODO
**Depends on:** —
**Findings:** `[#2]` (silent kwarg drop → hidden eval regression)
**Files:** `src/llm_factory.py`

**Do:**

In `_create_openai_llm` (line 46-57), replace silent pops:

```python
# BEFORE
def _create_openai_llm(**kwargs):
    kwargs.pop("thinking_budget", None)
    kwargs.pop("response_mime_type", None)
    return ChatOpenAI(...)

# AFTER
import structlog
_log = structlog.get_logger()

_GEMINI_ONLY_KWARGS = {"thinking_budget", "response_mime_type"}

def _create_openai_llm(**kwargs):
    dropped = {k: kwargs.pop(k) for k in list(kwargs) if k in _GEMINI_ONLY_KWARGS}
    if dropped:
        _log.warning(
            "llm_factory.kwarg_drop",
            provider="openai",
            dropped=dropped,
            note="provider-specific kwargs ignored — may cause behavior drift",
        )
    return ChatOpenAI(
        model=settings.MODEL_NAME,
        temperature=settings.TEMPERATURE,
        timeout=settings.REQUEST_TIMEOUT,
        max_retries=3,
        **kwargs,
    )
```

Update test `test_get_llm_openai_drops_gemini_kwargs` in `tests/test_llm_factory_refactor.py` to also assert that a warning was logged.

**Verify:**

```bash
pytest tests/test_llm_factory_refactor.py -v
```

**Done when:**
- Switching to `LLM_PROVIDER=openai` logs a structured warning for dropped kwargs.
- Tests green.

**Commit:** `fix(llm): warn on dropped provider-specific kwargs (was silent)`

---

# Board 3 — P2: Architecture Cleanup (for senior-readiness)

Goal: Remove the duck-typing smell, dead code, and misleading function names that a senior would call out during interview deep-dive. Each card stands alone — no urgency, but they read like maturity signals.

---

### CARD-3.1 — Replace `hasattr(_collection)` with `isinstance(VectorStoreBackend)` + fix tests

**Status:** TODO
**Depends on:** — (no real dep on CARD-2.3; can run any time)
**Findings:** `[#1]`, `[N6]`, `[S3]` (duck-typing marker inconsistent semantics)
**Files:** `src/v7/bridge.py`, `tests/v7/test_bridge.py` (mock specs MUST be tightened in same commit)

**⚠️ Why mock specs are part of this card, not separate:** Today `tests/v7/test_bridge.py` uses `mock_store = MagicMock()` without spec. With `hasattr(vs, "_collection")` it always returns True → tests go through raw-Chroma branch and pass. After switching to `isinstance(vs, VectorStoreBackend)`: `runtime_checkable` Protocol only checks method names, MagicMock-without-spec has ALL methods → `isinstance` returns True → code goes through `iter_all_documents()` branch → `mock_store.iter_all_documents()` returns a MagicMock (not iterable dict) → `init_bm25_index(corpus)` blows up. The fix MUST include tightening mocks, or all 5 tests in `test_bridge.py` break in the same commit.

**Do:**

1. Import the Protocol at top of `src/v7/bridge.py`:
```python
from src.backends.vector_store import VectorStoreBackend
```

2. Replace `bridge.py:147` (inside `make_section_fetch_fn`). Today logic is: "if no `_collection` attribute → it's a backend → use `get_by_filter`". New logic equivalent:
```python
# BEFORE
if not hasattr(vector_store, "_collection"):
    docs = vector_store.get_by_filter(...)
    return [...]
# old _collection-based branch
col = vector_store._collection
results = col.get(where={...}, ...)

# AFTER
if isinstance(vector_store, VectorStoreBackend):
    docs = vector_store.get_by_filter(...)
    return [...]
# else: raw Chroma (legacy path) — same as before
col = vector_store._collection
results = col.get(where={...}, ...)
```

3. Replace `bridge.py:430` inside `init_v7_from_chroma`. Today: "if `_collection` exists → raw Chroma path". Logically equivalent inverted:
```python
# BEFORE
if hasattr(vector_store, "_collection"):
    all_data = vector_store.get(include=["metadatas", "documents"])
    corpus = [{"text": doc, "metadata": meta} for doc, meta in zip(all_data["documents"], all_data["metadatas"])]
else:
    corpus = list(vector_store.iter_all_documents())

# AFTER
if isinstance(vector_store, VectorStoreBackend):
    corpus = list(vector_store.iter_all_documents())
else:
    # Legacy: raw Chroma object — kept for backward compat
    all_data = vector_store.get(include=["metadatas", "documents"])
    corpus = [{"text": doc, "metadata": meta} for doc, meta in zip(all_data["documents"], all_data["metadatas"])]
```

4. **REQUIRED: Update `tests/v7/test_bridge.py` mocks** — all 5 tests on lines 285, 300, 330, 361, 377. Find:
```python
mock_store = MagicMock()
```
Replace with one of:
```python
# If test exercises raw-Chroma path (most existing tests):
from langchain_chroma import Chroma
mock_store = MagicMock(spec=Chroma)
mock_store._collection = MagicMock()  # raw Chroma has this attribute
mock_store.get.return_value = {"documents": ["x"], "metadatas": [{}]}
# isinstance(mock_store, VectorStoreBackend) → False (Chroma has no iter_all_documents)
# → falls into legacy branch, same as before
```

OR — if you want the test to exercise the backend branch instead:
```python
from src.backends.chroma_backend import ChromaBackend
mock_store = MagicMock(spec=ChromaBackend)
mock_store.iter_all_documents.return_value = iter([{"text": "x", "metadata": {}}])
# isinstance check passes via runtime_checkable Protocol; backend branch exercised
```

Quick rule: if the test calls `mock_store.get(...)` → use `spec=Chroma`. If the test asserts on backend methods → use `spec=ChromaBackend`. Audit each of the 5 tests individually.

5. Add ONE new integration test exercising real ChromaBackend (no mock):
```python
# tests/v7/test_bridge_integration.py
import pytest

@pytest.mark.integration
def test_init_v7_from_chroma_with_real_backend(tmp_path, monkeypatch):
    """Smoke test: init_v7_from_chroma works against a real ChromaBackend."""
    from langchain_core.documents import Document
    from src.backends.chroma_backend import ChromaBackend
    from src.v7.bridge import init_v7_from_chroma

    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "test_chroma"))
    backend = ChromaBackend(load_existing=False)
    backend.create([
        Document(page_content="охрана труда", metadata={"source": "test.pdf", "chunk_id": 1}),
        Document(page_content="инструктаж", metadata={"source": "test.pdf", "chunk_id": 2}),
    ])
    init_v7_from_chroma(backend, llm_provider=None)
    # No assertion needed — just verify no exception in iter_all_documents path
```

**Verify:**
```bash
pytest tests/v7/test_bridge.py -v       # all 5 updated tests green
pytest tests/v7/test_bridge_integration.py -v -m integration
python scripts/trace_v7.py "тест"
```

**Done when:**
- Both `make_section_fetch_fn` and `init_v7_from_chroma` use `isinstance(VectorStoreBackend)`.
- All 5 existing `test_bridge.py` tests updated with explicit `spec=` and pass.
- New integration test passes against real `ChromaBackend`.
- trace_v7 works end-to-end.

**Commit:** `refactor(v7): use isinstance(VectorStoreBackend) instead of hasattr; tighten test mocks`

---

### CARD-3.2 — Rename `init_v7_from_chroma` → `init_v7_pipeline`

**Status:** TODO
**Depends on:** CARD-3.1 (must land first — both touch `init_v7_from_chroma` and we want hasattr→isinstance fix in place before renaming)
**Findings:** `[#8]` (name lies — accepts abstract backend)
**Files:** `src/v7/bridge.py`, `src/v7/__init__.py`, all callers (verified list below)

**Do:**

1. Grep all current usages (verified 2026-05-24, last commit 9e641e7):
```bash
grep -rn "init_v7_from_chroma" --include="*.py" \
  --exclude-dir=venv --exclude-dir=__pycache__ \
  /home/petr/projects/ai/safety-incident-analyzer/
```
Confirmed callers (10 lines across 8 files):
- `src/v7/bridge.py:413` — definition
- `src/v7/__init__.py:3, 23` — re-export + `__all__`
- `app.py:30, 174`
- `api.py:37, 42`
- `eval/run_v7_eval.py:46, 174`
- `eval/run_eval.py:129, 133` ← **don't forget this one (older eval script)**
- `scripts/measure_cps.py:36, 42`
- `scripts/trace_v7.py:199`
- `tests/v7/test_bridge.py:11, 285, 300, 330, 361, 377`

2. In `src/v7/bridge.py:413`:
```python
def init_v7_pipeline(vector_store, llm_provider: str | None = "gemini") -> None:
    """Initialize V7 pipeline from a vector store (raw Chroma or VectorStoreBackend).

    Thread-safe via _init_lock (see CARD-2.2).
    """
    # ... body unchanged ...

# Backward-compat alias — remove after one release cycle.
init_v7_from_chroma = init_v7_pipeline
```

3. In `src/v7/__init__.py`:
```python
# BEFORE
from src.v7.bridge import init_v7_from_chroma, make_vector_search_fn
__all__ = [
    "init_v7_from_chroma",
    ...
]

# AFTER
from src.v7.bridge import init_v7_from_chroma, init_v7_pipeline, make_vector_search_fn
__all__ = [
    "init_v7_pipeline",
    "init_v7_from_chroma",  # deprecated alias, kept for one release
    ...
]
```

4. Update each non-test caller to import and use `init_v7_pipeline`:
- `app.py:30` import + `app.py:174` call
- `api.py:37` import + `api.py:42` call
- `eval/run_v7_eval.py:46` import + `:174` call
- `eval/run_eval.py:129` import + `:133` call
- `scripts/measure_cps.py:36` import + `:42` call
- `scripts/trace_v7.py:199` (local import inside function) + call site

5. **Tests in `tests/v7/test_bridge.py`** — keep using old `init_v7_from_chroma` name OR migrate to new name. Either is fine (alias works). Recommend migrate-on-touch: when you touch a test for CARD-3.1, also rename in that test. Don't make this a separate sweep.

**Verify:**
```bash
# Production callers all use new name:
grep -rn "init_v7_from_chroma" --include="*.py" \
  --exclude-dir=venv --exclude-dir=__pycache__ \
  /home/petr/projects/ai/safety-incident-analyzer/ \
  | grep -v "tests/\|bridge.py\|__init__.py"
# Expected: empty output

# Both names still importable:
python -c "from src.v7 import init_v7_pipeline, init_v7_from_chroma; print(init_v7_pipeline is init_v7_from_chroma)"
# Expected: True

pytest tests/v7/ -v
python scripts/trace_v7.py "тест"
```

**Done when:**
- All production callers use `init_v7_pipeline`.
- Alias `init_v7_from_chroma = init_v7_pipeline` present in bridge.py + exported from `__init__.py`.
- `src/v7/__init__.py` `__all__` lists both names.
- All tests green.

**Commit:** `refactor(v7): rename init_v7_from_chroma → init_v7_pipeline (alias kept for compat)`

---

### CARD-3.3 — Delete dead `_split_into_children` code

**Status:** TODO
**Depends on:** —
**Findings:** `[N2]` (parent-context chunking experiment was reverted, code remains)
**Files:** `src/file_handler.py`

**Do:**

1. Confirm `_split_into_children` is unused:

```bash
grep -rn "_split_into_children" --include="*.py" \
  --exclude-dir=venv --exclude-dir=__pycache__
```

Expected: only the definition in `src/file_handler.py:65-87`, no call sites.

2. Delete the function and its helpers (look for `_child_idx`, `parent_text` keys that exist only for child-chunk path).

3. Run tests:

```bash
pytest tests/test_file_handler*.py -v
```

**Done when:**
- `_split_into_children` no longer in source.
- All tests green (no test referenced it).

**Commit:** `chore(file_handler): remove dead parent-context chunking code (reverted experiment)`

---

### CARD-3.4 — Migrate global `_pipeline` dict to `app.state`

**Status:** TODO
**Depends on:** CARD-1.1 (request_id already in middleware)
**Findings:** `[#22]` (global mutable state anti-pattern)
**Files:** `api.py`

**Do:**

Replace usage of module-global `_pipeline: dict` with `app.state` attributes:

```python
# BEFORE
_pipeline: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    _pipeline["app"] = ...
    yield
    _pipeline.clear()

@app.post("/query")
def query(req: QueryRequest):
    pipeline_app = _pipeline.get("app")
    ...

# AFTER
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = ...  # the compiled graph
    app.state.gosts_pipeline = ...
    app.state.gosts_ready = ...
    yield

@app.post("/query")
def query(request: Request, req: QueryRequest):
    pipeline_app = request.app.state.pipeline
    ...
```

`/health` uses `request.app.state.pipeline`.

**Verify:**

```bash
uvicorn api:app --port 8503 &
sleep 5
curl -s http://localhost:8503/health
curl -s -X POST http://localhost:8503/query -H 'Content-Type: application/json' -d '{"question":"test"}' | jq .answer
kill %1
```

**Done when:**
- No module-global `_pipeline` left.
- API health + query work via `request.app.state`.

**Commit:** `refactor(api): migrate _pipeline dict to app.state (FastAPI idiomatic)`

---

### CARD-3.5 — Graceful degradation in lifespan

**Status:** TODO
**Depends on:** CARD-3.4
**Findings:** `[#19]` (lifespan raise kills uvicorn)
**Files:** `api.py`

**Do:**

In lifespan, catch init failures instead of raising:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    app.state.gosts_pipeline = None
    app.state.gosts_ready = False
    try:
        ...
        app.state.pipeline = build_graph().compile()
    except Exception as exc:
        logger.error("api.startup: main pipeline init failed", error=str(exc), exc_info=True)
        # leave pipeline=None; /query returns 503, /health returns 503, process stays alive

    try:
        ...
        app.state.gosts_pipeline = ...
    except Exception as exc:
        logger.warning("api.startup: gosts pipeline not available", error=str(exc))

    yield
```

`/query` already handles `pipeline_app is None` with 503. `/health` already returns 503 — good.

**Verify:**

```bash
# Simulate broken ChromaDB by pointing to bad path
CHROMA_DB_PATH=/nonexistent uvicorn api:app --port 8503 &
sleep 5
curl -s http://localhost:8503/health
# Expected: 503 with "not ready" (NOT connection refused — process is alive)
curl -s -X POST http://localhost:8503/query -H 'Content-Type: application/json' -d '{"question":"x"}'
# Expected: 503 "pipeline not initialized"
kill %1
```

**Done when:**
- API process stays alive on ChromaDB init failure.
- `/health` returns 503, `/query` returns 503 with generic message.

**Commit:** `fix(api): graceful degradation on pipeline init failure (was: uvicorn crash)`

---

### CARD-3.6 — Remove `ChromaBackend.raw` escape hatch (if unused)

**Status:** TODO
**Depends on:** — (verified independent; grep shows no users today, can run first)
**Findings:** `[#14]` (escape hatch breaks abstraction, grep shows no users)
**Files:** `src/backends/chroma_backend.py`

**Do:**

1. Confirm no callers:

```bash
grep -rn "\.raw\b" --include="*.py" \
  --exclude-dir=venv --exclude-dir=__pycache__ src/ api.py app.py eval/ scripts/ tests/
# Expected: only definition in chroma_backend.py:67
```

2. If grep is clean, delete the `@property raw` block.

3. If any callers exist, document each with rationale in code or refactor them to use Protocol methods.

**Verify:**

```bash
pytest tests/test_chroma_backend.py -v
python scripts/trace_v7.py "тест"
```

**Done when:**
- `raw` property removed (or each remaining caller justified).
- All tests pass.

**Commit:** `refactor(backends): remove unused ChromaBackend.raw escape hatch`

---

# Board 4 — P3: Code Quality Polish (no urgency)

Goal: Bring code in line with `~/.claude/coding-standards.md` and minor style cleanups. None of these are bugs — they accumulate goodwill in code review.

---

### CARD-4.1 — `from __future__ import annotations` + modern type hints

**Status:** TODO
**Files:** `src/llm_factory.py`, others as grep-found

**Do:**

1. Find files missing the annotations future import:

```bash
grep -rL "from __future__ import annotations" --include="*.py" src/ | head -20
```

2. For each file in `src/`, add `from __future__ import annotations` at the top.

3. Replace `str = None` (etc.) with `str | None = None` in `src/llm_factory.py:99-102` and similar.

**Done when:** `grep -L "from __future__ import annotations" src/**/*.py` returns empty.

**Commit:** `style: add __future__ annotations imports, modernize type hints`

---

### CARD-4.2 — Replace `logging.getLogger` with `structlog.get_logger`

**Status:** TODO
**Files:** `src/v7/bridge.py:33`, others

**Do:**

```bash
grep -rn "logging.getLogger" --include="*.py" src/
```

Replace each with `structlog.get_logger()` (matches coding-standards.md). Verify log calls use structured kwargs (`logger.info("x", key=val)` not `f"x {val}"`).

**Done when:** `grep -rn "logging.getLogger" --include="*.py" src/` returns empty.

**Commit:** `style: migrate from logging.getLogger to structlog`

---

### CARD-4.3 — Narrow broad `except Exception` in eval

**Status:** TODO
**Files:** `eval/run_v7_eval.py` (lines from `grep -n "except Exception" eval/run_v7_eval.py` — current hits: 150, 195, 233, 238, 245; do NOT trust hardcoded numbers, grep first)

**Do:**

For each `except Exception:`, identify the actual exception class that can be raised and narrow:

- JSON parse errors → `except (json.JSONDecodeError, ValueError):`
- LLM call failures → `except (APIError, TimeoutError):`

If a true catch-all is needed (defense-in-depth around a flaky external call), log with `exc_info=True` and re-raise selectively.

**Done when:** No bare `except Exception:` left in `eval/run_v7_eval.py`. Coding-standards compliance.

**Commit:** `fix(eval): narrow broad except clauses to specific exception types`

---

### CARD-4.4 — Move `evaluate_correctness` prompt to registry

**Status:** TODO
**Depends on:** —
**Findings:** `[#20]` (judge prompt bypasses `prompts/registry.yaml`)
**Files:** `eval/run_v7_eval.py`, `prompts/evaluators/correctness_v1.j2` (new), `prompts/registry.yaml`

**Do:**

1. Create `prompts/evaluators/correctness_v1.j2` with the current hardcoded prompt body.

2. Register in `prompts/registry.yaml` under `evaluators.correctness` with `active_version: v1`.

3. In `eval/run_v7_eval.py:117-138`, load via registry:

```python
from src.prompt_loader import load_prompt  # or whatever the registry helper is
prompt = load_prompt("evaluators.correctness").render(
    question=q, answer=a, ground_truth=gt
)
```

4. Run `python scripts/validate_prompts.py` to verify registry.

**Done when:** `evaluate_correctness` reads prompt from registry; `validate_prompts.py` green.

**Commit:** `refactor(eval): move evaluate_correctness prompt to registry`

---

### CARD-4.5 — Fix `_is_retryable` substring matching

**Status:** TODO
**Findings:** `[#22]` (substring matching → false positives like "User reported overloaded form")
**Files:** `src/v7/bridge.py:341-347`

**Do:**

Replace string-based heuristic with exception-type matching:

```python
# BEFORE
def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(kw in msg for kw in ("503", "resource_exhausted", "rate limit", "overloaded"))

# AFTER
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
_RETRYABLE_TYPES = (ResourceExhausted, ServiceUnavailable, TimeoutError)

def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    # HTTP 5xx fallback for SDK-wrapped responses
    code = getattr(exc, "code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    return code in (429, 500, 502, 503, 504) if code else False
```

**Done when:** Existing retry tests in `tests/v7/test_bridge.py` still pass. Add new test for `ResourceExhausted` instance → retry, `ValueError("overloaded form")` → no retry.

**Commit:** `fix(v7): retry on exception types, not error message substrings`

---

### CARD-4.6 — Add `spec=Chroma` to MagicMock in backend tests

**Status:** TODO
**Findings:** `[#18]` (MagicMock without spec hides typos)
**Files:** `tests/test_chroma_backend.py`, `tests/test_vector_store_factory.py`

**Do:**

For every `MagicMock()` in these test files:

```python
# BEFORE
mock_vs = MagicMock()

# AFTER
from langchain_chroma import Chroma
mock_vs = MagicMock(spec=Chroma)
```

This makes the mock fail loudly if test code calls a method/attribute that does not exist on the real `Chroma` class.

**Done when:** All Chroma mocks use `spec=`; tests still pass; intentional typo in test source raises `AttributeError`.

**Commit:** `test: tighten MagicMock specs for Chroma to catch typos`

---

### CARD-4.7 — Explicit IPv6 patch + remove import-time side effect

**Status:** TODO
**Findings:** `[#25]` (`_patch_socket_ipv6_for_googleapis` runs on import)
**Files:** `src/llm_factory.py`, `api.py`, `app.py`, callers

**Do:**

1. Remove import-time call in `src/llm_factory.py:36`:

```python
# DELETE
_patch_socket_ipv6_for_googleapis()
```

2. Rename to `apply_ipv6_patch_for_googleapis()` (public function).

3. Add explicit call site in `api.py` and `app.py`:

```python
from src.llm_factory import apply_ipv6_patch_for_googleapis
apply_ipv6_patch_for_googleapis()
```

4. Document in CLAUDE.md under "VPS env requirements" that this must be called before any Gemini import.

5. Also tighten the hostname check from `"googleapis.com" in host` to:

```python
if host and (host == "googleapis.com" or host.endswith(".googleapis.com")):
```

(Defeats `attacker.com.googleapis.com.attacker.com` DNS rebinding.)

**Done when:**
- Import of `src.llm_factory` no longer patches sockets.
- API and Streamlit explicitly call `apply_ipv6_patch_for_googleapis()` at startup.
- Hostname check is anchored (no substring match).

**Commit:** `fix(llm): make IPv6 patch explicit + anchor googleapis.com hostname check`

---

# Board 5 — P2: LangChain Idiomatic Cleanup (переизобретённые колёса)

Аудит 2026-05-24: ~10% кода дублирует готовые примитивы LangChain/LangGraph. Архитектура в целом опирается на LCEL/StateGraph корректно — здесь только адресные замены.

---

### CARD-5.1 — Заменить monkey-patch AFC на параметр конструктора

**Where:** `src/llm_factory.py:148-156`

**Why:** Сейчас в `get_gemini_llm()` патчится приватный `_build_request_config` `ChatGoogleGenerativeAI`, чтобы выключить Automatic Function Calling (предотвращает дубль tool calls). `langchain-google-genai` принимает это публичным параметром конструктора — патчить приватные методы не нужно.

**How:**
```python
# BEFORE
llm = ChatGoogleGenerativeAI(model=..., ...)
_orig = llm._build_request_config
def _patched(...): ...
llm._build_request_config = _patched

# AFTER
llm = ChatGoogleGenerativeAI(
    model=...,
    automatic_function_calling=False,
    ...,
)
```

**Verify:**
- `python scripts/trace_v7.py "для кого проводится повторный инструктаж?"` — пайплайн доходит до answer, в трассе нет дублирующихся tool calls.
- Поиск приватного API не остаётся: `rg "_build_request_config" src/` → пусто.

**Side-effect:** Снимает основание для жёсткого pin в CARD-2.5 (можно ослабить до `~=`/нижней границы). Обновить комментарий в `requirements.txt` соответственно.

**Commit:** `refactor(llm): disable Gemini AFC via constructor param, drop private-API monkey-patch`

---

### CARD-5.2 — Заменить самописный query expansion на `MultiQueryRetriever`

**Where:**
- `src/v7/bridge.py:283-348` (`make_expand_fn`)
- `src/applicability_retriever.py:26-51` (тот же приём + ручной кеш)

**Why:** Оба места независимо реализуют одно и то же: LLM генерирует 3 переформулировки запроса → объединение результатов retrieval. Это в точности `langchain.retrievers.MultiQueryRetriever`. Дублирование в двух местах гарантирует расхождение поведения со временем.

**How:**
1. Заменить `make_expand_fn` на обёртку `MultiQueryRetriever.from_llm(retriever=..., llm=...)`. Сохранить интерфейс ноды графа (вход/выход `RAGState`).
2. В `applicability_retriever.py` — то же самое; ручной кеш переформулировок убрать, либо положить на уровне `LLMChain` через стандартный `BaseCache`.
3. Проверить, что RRF/слияние в графе остаётся консистентным (MultiQueryRetriever по умолчанию uniq-объединяет; если нужен RRF — оставить текущее слияние, заменить только генерацию запросов).

**Verify:**
- Существующие тесты `tests/v7/test_bridge_*` зелёные.
- E2E: `python scripts/trace_v7.py "обучение по охране труда"` — в трассе видны 3 субзапроса (LangSmith / структурный лог).
- Eval не регрессирует: `python eval/run_v7_eval.py --skip-judge` — top_score распределение сопоставимо с baseline.

**Commit:** `refactor(retrieval): replace custom query expansion with MultiQueryRetriever`

---

### CARD-5.3 — Убрать дубль `FlashrankRerank` в `make_rerank_fn`

**Where:** `src/v7/bridge.py:57-93` (`make_rerank_fn`) vs `src/final_chain.py` (уже использует `ContextualCompressionRetriever` + `FlashrankRerank`).

**Why:** Две независимые обёртки над FlashRank. В V7-графе используется самописная обёртка, в legacy chain — стандартный `FlashrankRerank`. Унификация: оставить `langchain_community.document_compressors.FlashrankRerank`, обёртку графа сделать тонким адаптером (passages → rerank → passages с сохранённым `vector_score`).

**How:**
1. В `make_rerank_fn` использовать `FlashrankRerank` напрямую.
2. Сохранить логику CARD-5.X из истории: `vector_score` фиксируется ДО rerank (см. фикс 2026-05-16), не теряется.
3. Проверить, что MMR/gates по-прежнему читают `vector_score`, не FlashRank score.

**Verify:**
- `pytest tests/v7/ -k rerank -v` — зелёные.
- Eval не регрессирует (correctness ≥ 7.9, не упало).

**Commit:** `refactor(rerank): use langchain FlashrankRerank, drop custom wrapper`

---

### CARD-5.4 — Заменить ручной JSON parsing на `.with_structured_output()`

**Where:**
- `src/parsers.py:12-24` (regex + braces fallback)
- `src/parsers.py:42-66` (`===STATUS===` / `===ANSWER===` блоки)
- `src/v7/bridge.py:190-228` (`make_verify_fn` — `parse_json_from_response`)

**Why:** Все три места парсят неструктурированный текст от LLM. Gemini нативно поддерживает `response_mime_type="application/json"`; в LangChain это `ChatGoogleGenerativeAI(...).with_structured_output(Schema)` (Pydantic). Это удаляет fallback-логику и регэкспы.

**How:**
1. Объявить Pydantic-схемы для verifier-ответа и для structured answer (status/answer/score).
2. В `bridge.make_verify_fn`: `llm.with_structured_output(VerifierVerdict).invoke(prompt)` — никакого parse_json.
3. В промптах убрать инструкции «верни JSON в формате…» — структурированный вывод гарантирует Pydantic.
4. Старый `parsers.py` оставить только если кто-то ещё его импортирует; иначе — удалить.

**Verify:**
- `pytest tests/ -k "parser or verify" -v` — зелёные (часть тестов на парсер переписать на schema-based).
- E2E trace показывает verifier verdict без warning'ов «could not parse JSON».

**Commit:** `refactor(parsing): use with_structured_output instead of regex JSON parsing`

---

### CARD-5.5 — Убрать tenacity вокруг LLM, опереться на встроенный `max_retries`

**Where:** `src/v7/bridge.py:350-420` (`make_generate_fn`)

**Why:** Поверх Gemini-вызова навешан tenacity-декоратор с exp backoff. `ChatGoogleGenerativeAI` принимает `max_retries=N` (и сам ходит с экспоненциальным backoff на 5xx). Два уровня retry удваивают latency на хвостах ошибок и маскируют реальные коды.

**How:**
1. `get_gemini_llm(..., max_retries=3)`.
2. Снять tenacity-декоратор в `make_generate_fn`; оставить только обработку финального fallback на stub (3 попытки исчерпаны).
3. Сверить с известным фиксом «503 → stub fallback» (CLAUDE.md): встроенный retry покрывает 503/429.

**Verify:**
- Smoke под симуляцией 503: моком ChatGoogle вернуть 503 один раз → второй call успешен (один retry, не цикл tenacity → встроенный).
- Существующий тест fallback на stub после исчерпания retry — зелёный.

**Commit:** `refactor(llm): rely on ChatGoogleGenerativeAI max_retries, drop tenacity wrapper`

---

### CARD-5.6 — Завернуть `SemanticCache` в `BaseCache` интерфейс

**Where:** `src/semantic_cache.py:14-126`

**Why:** Класс реализует семантический кеш ответов LLM (cosine similarity по embeddings). Сама реализация оправдана (стандартного семантического кеша в LangChain нет), но сейчас он живёт сбоку и не вписан в LLM caching pipeline. Если завернуть в `langchain_core.caches.BaseCache`, можно подключать через `set_llm_cache(SemanticCache(...))` и автоматически кешировать ВСЕ LLM-вызовы, а не только ручные точки.

**How:**
1. Наследовать `SemanticCache` от `BaseCache`, реализовать `lookup(prompt, llm_string)` и `update(prompt, llm_string, return_val)`.
2. В `api.py` startup: `from langchain_core.globals import set_llm_cache; set_llm_cache(SemanticCache(...))`.
3. Старые ручные `cache.get/put` снять.

**Verify:**
- Unit: тот же тестсьют семантического матчинга, но через `BaseCache` API.
- E2E: повтор одного и того же вопроса второй раз — нет похода в Gemini (видно по usage callback).

**Commit:** `refactor(cache): expose SemanticCache as BaseCache, register globally`

---

### CARD-5.7 — Снять обёртку `HFEmbeddingsWrapper`

**Where:** `src/llm_factory.py:176-183`

**Why:** Самописный adapter поверх `HuggingFaceEmbeddings`, не добавляет логики (просто проксирует `embed_documents`/`embed_query`).

**How:** Использовать `HuggingFaceEmbeddings` напрямую; обёртку удалить.

**Verify:** Импорты вычищены ruff'ом; `pytest tests/test_llm_factory*.py -v` зелёные.

**Commit:** `refactor(embeddings): drop trivial HFEmbeddingsWrapper`

---

### CARD-5.8 — Перевести custom evaluator на `langchain.evaluation`

**Where:** `src/custom_evaluators.py:9-91`

**Why:** Самописный LLM-judge (prompt + invoke + JSON parse). В `langchain.evaluation` есть готовые `EvaluatorType.CORRECTNESS` / `LABELED_CRITERIA`, в LangSmith — встроенные evaluators с трейсингом из коробки. Снимает поддержку ещё одного парсера.

**How:**
1. Заменить `evaluate_correctness` на `load_evaluator("labeled_criteria", criteria="correctness", llm=...)`.
2. Eval-скрипт `eval/run_v7_eval.py` подключить к LangSmith evaluators (опционально — runtime флаг `--judge=langsmith|local`).
3. Старый кастомный judge можно оставить за флагом для оффлайн-прогонов без сети.

**Verify:**
- `python eval/run_v7_eval.py` — correctness в том же диапазоне (±0.2 от baseline 7.9).

**Commit:** `refactor(eval): use langchain.evaluation labeled_criteria for correctness judge`

---

## Acceptance — overall

### Board 1 (P0, must ship before 2026-05-27):
- [ ] CARD-1.1 `DONE` — exception leak fixed, request_id added
- [ ] CARD-1.2 `DONE` — rate limit + length cap live
- [ ] CARD-1.3 `DONE` — `trust_remote_code=False`

### Board 2 (P1, within 2 weeks):
- [ ] CARD-2.1 `DONE` — pickle → JSON in `file_handler.py`
- [ ] CARD-2.1b `DONE` — pickle removed from `final_chain.py` (BM25 cache)
- [ ] CARD-2.2 `DONE` — init lock
- [ ] CARD-2.3 `DONE` — empty-metadata guard + limit
- [ ] CARD-2.4 `DONE` — sqlite driver unified
- [ ] CARD-2.5 `DONE` — langchain-google-genai pinned
- [ ] CARD-2.6 `DONE` — visuals cleanup
- [ ] CARD-2.7 `DONE` — kwarg warn

### Board 3 (P2):
- [ ] CARD-3.1 `DONE` — `isinstance(VectorStoreBackend)`
- [ ] CARD-3.2 `DONE` — function rename
- [ ] CARD-3.3 `DONE` — dead code deleted
- [ ] CARD-3.4 `DONE` — `app.state` migration
- [ ] CARD-3.5 `DONE` — graceful lifespan degradation
- [ ] CARD-3.6 `DONE` — `.raw` removed

### Board 5 (P2, LangChain idiomatic):
- [ ] CARD-5.1 `DONE` — AFC через параметр конструктора, monkey-patch снят
- [x] CARD-5.2 `PARTIAL` — `LineListOutputParser` в `applicability_retriever`; `bridge.make_expand_fn` оставлен (RRF + русский промпт)
- [ ] CARD-5.3 `DONE` — `FlashrankRerank` напрямую в V7 graph
- [x] CARD-5.4 `PARTIAL` — verifier через `.with_structured_output()`; `parsers.py` (используется legacy multiagent_rag) не тронут
- [x] CARD-5.5 `DONE` — встроенный `max_retries`, tenacity снят (ers_rag/bridge.py — WONTFIX)
- [x] CARD-5.6 `DONE` — `SemanticCache` как `BaseCache` (без глобальной регистрации, чтобы не загрязнять eval)
- [x] CARD-5.7 `DONE` — `HFEmbeddingsWrapper` снят, `HuggingFaceInferenceAPIEmbeddings` напрямую
- [ ] CARD-5.8 `SKIPPED` — `labeled_criteria` возвращает binary, наш judge на шкале 0-10; миграция сломает baseline 7.9. Отдельная задача с пересчётом baseline

### Board 4 (P3):
- [ ] CARD-4.1 `DONE` — type hints
- [ ] CARD-4.2 `DONE` — structlog migration
- [ ] CARD-4.3 `DONE` — narrow excepts
- [ ] CARD-4.4 `DONE` — judge prompt in registry
- [ ] CARD-4.5 `DONE` — retry by type
- [ ] CARD-4.6 `DONE` — MagicMock spec
- [ ] CARD-4.7 `DONE` — explicit IPv6 patch

### Global:
- [ ] `pytest -m unit` green (current 223 tests pass)
- [ ] `python scripts/trace_v7.py "для кого проводится повторный инструктаж?"` succeeds
- [ ] Eval correctness within ±0.1 of 7.9 baseline (recommended after Board 2 done)
- [ ] `grep -rn "pickle\|trust_remote_code=True\|hasattr.*_collection" --include="*.py" src/` returns nothing significant

---

## Known limitations (deliberately deferred — not bugs, but expect questions)

These findings from the two reviews are NOT addressed in this plan. Each has an explicit reason. Be ready to answer for them on 27.05 instead of pretending they don't exist.

| Finding | Why deferred | Talking point if asked |
|---------|--------------|------------------------|
| `[#6]` `model_post_init` only fires on default `CHROMA_DB_PATH` | Design choice: explicit env override beats implicit auto-suffix. Documenting > patching. | "Treat custom `CHROMA_DB_PATH` as user-knows-what-they're-doing; we document the embedding-DB pairing in CLAUDE.md instead of silently mutating user-set values." |
| `[#7]` kwarg asymmetry (openai drops, gemini `TypeError` on unknown kwargs) | CARD-2.7 makes the drop loud; making them symmetric requires per-provider arg-introspection (3+ hours, low value at one-provider scale). | "Symmetry across providers is on the roadmap — for now, the warning catches the drift case in practice." |
| `[#9]` `lru_cache` on `load_vector_store` + multi-worker uvicorn | Production runs on 1 worker. Multi-worker requires either shared-memory backend or pooled connections — out-of-scope rewrite. | "Single-worker is intentional for the prototype scale. Scaling to N workers needs either Qdrant (network-backed) or per-worker process isolation — sketched in the future-work plans." |
| `[#13]` `iter_all_documents` loads all into RAM | At 1973 chunks ≈ 3MB. Streaming via Chroma's API would require chunked `get()` calls — premature for current scale. | "Acceptable up to ~10k chunks; would switch to chunked iteration for prod-scale corpora." |
| `[#21]` `Iterator[dict]` without parameterization | Cosmetic. Not worth a card. | Just say "yes, missed it." |
| `[#24]` `.env.example` vs `settings.py` model name drift | Cosmetic (a one-line `.env.example` update). | One-line follow-up. |

---

## After this plan

Each remaining ticket from `REVIEW_2026-05-24*.md` not addressed here is either:
- **Low priority polish** → opened as `docs/plans/backlog.md` entry
- **Strategic** (e.g. multi-worker concurrency, Qdrant backend, contextual retrieval) → its own plan

The interview narrative ("Hard truth: code works in prod thanks to luck — one worker, one user. At 10 RPS we'd see N1+N3+N4+N5 cascade. We've fixed N3 and N5 pre-interview as defense-in-depth; N1 has a lock + documented restart-required policy.") is now backed by actual commits.

---

## Plan validation history

- **2026-05-24** — plan draft v1 written.
- **2026-05-24** — plan v1 reviewed by code-reviewer subagent against actual codebase. Found 4 bugs in Do-instructions (CARD-2.1 wrong parameter order + wrong method name + wrong import; CARD-2.6 wrong `_validate_bbox` contract + wrong bbox units; CARD-3.1 broke existing 5 unit tests; CARD-3.2 missed `src/v7/__init__.py` and `eval/run_eval.py` callers). Missed coverage: pickle in `src/final_chain.py`. Wrong deps: CARD-3.1 spurious dep on CARD-2.3; CARD-3.6 spurious dep on CARD-3.1+3.2.
- **2026-05-24** — fixes applied (v2): CARD-2.1 rewritten with verified signatures + safer migration via separate script (no auto-delete of `.pkl`); CARD-2.1b added for `final_chain.py`; CARD-2.6 rewritten matching string-return contract + PDF-points unit math; CARD-3.1 rewritten with explicit mock-spec update for all 5 existing tests; CARD-3.2 grep-verified caller list (including `__init__.py` and `eval/run_eval.py`); CARD-3.1/3.6 deps removed; CARD-4.3 line numbers replaced with grep instruction; Known Limitations section added for `[#6, #7, #9, #13, #21, #24]`.
