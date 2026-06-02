# Indexing Pipeline

How PDF documents are turned into searchable chunks stored in ChromaDB.

---

## Overview

```
source_docs/*.pdf  →  DocumentProcessor  →  List[Document]  →  ChromaDB
```

Entry point: `python index.py` (destructive — drops the old DB entirely).

---

## index.py — Entry Point

1. **Drop ChromaDB** (`shutil.rmtree`) — removes the entire vector database folder
2. **Drop Docling cache** — prevents ghost chunks from surviving a reindex (see below)
3. **Collect file paths** — `_collect_paths()` walks `source_docs/` recursively, filters by `.pdf` / `.md`
4. **Process** — `DocumentProcessor.process(file_paths)` → chunks
5. **Persist** — `ChromaBackend.create(chunks)` embeds and writes to ChromaDB

> **Ghost chunks:** if Docling cache is not invalidated after a destructive reindex,
> the processor reads stale chunks from cache and writes them to the new DB — the new
> chunking logic never runs.

---

## DocumentProcessor (`src/indexing/file_handler.py`)

### `process(files)` — main loop

For each file:
1. Convert path/stream → `BytesIO` + display name (`_get_stream_and_name`)
2. Compute SHA-256 of file bytes → cache key
3. **Cache hit** → load chunks from `document_cache/<hash>.json` (fast)
4. **Cache miss** → parse with Docling + HybridChunker → save to cache (slow, 30-60s/doc)
5. **Deduplication** — skip chunk if SHA-256 of `page_content` already seen
6. **Assign `chunk_id`** — sequential int per source, assigned *after* dedup so ids are contiguous

Cache key = `sha256(file_hash + PIPELINE_VERSION)` — bumping `PIPELINE_VERSION` invalidates all cache.

### `_process_docling_document(doc, source)` — the core

```
Docling res.document
    ↓
HybridChunker(max_tokens=400)  — structural chunks at NPA article boundaries
    ↓
_clean_noise()                 — strip URLs, page markers, timestamps; normalise dashes
    ↓
BLACKLIST_PHRASES filter        — drop watermark lines ("Премиальная версия"...)
    ↓
Extract metadata:
  source, parent_section, heading_path, element_type, page_no, bbox
    ↓
Contextual embedding:
  if chunk doesn't start with heading → prepend "Ст.228.1\n" to page_content
    ↓
Document(page_content=..., metadata={...})
```

**Contextual embedding** is the key quality trick: the section/article title is prepended
to the embedded text. This disambiguates near-duplicate wording across different NPA
(e.g. ТК РФ ст.228.1 vs Приказ 223н — both mention "notifications").

### `_clean_noise(text)`

| Step | What it does |
|------|-------------|
| `unicodedata.normalize("NFC")` | Unify Cyrillic encoding variants |
| `text.replace("–", "—")` | Normalise en-dash → em-dash (prevents search mismatches) |
| `text.replace("-\n", "")` | Glue PDF soft hyphens ("рабо-\nтодатель" → "работодатель") |
| `_NOISE_PATTERNS.sub("", ...)` | Strip URLs, page markers (14/34), timestamps |
| `re.sub(r" {2,}", " ", ...)` | Collapse double spaces |

---

## Document Object

Each chunk is a LangChain `Document`:

```python
Document(
  page_content="Статья 228.1 Порядок извещения\nРаботодатель обязан...",
  metadata={
    "source":         "ТК_РФ.pdf",
    "type":           "hybrid_chunk",
    "parent_section": "Статья 228.1",
    "heading_path":   "Раздел X > Глава 34 > Статья 228.1",
    "element_type":   "text",   # or "table"
    "page_no":        134,
    "bbox":           "[72.0, 680.5, 540.0, 710.3]",
    "chunk_id":       42        # sequential per source, 0-based
  }
)
```

Total corpus: **7718 chunks** from 12 documents (11 PDF + ст.143 УК .md).

---

## create_vector_store (`src/indexing/vector_store.py`)

Takes `List[Document]`, batches by token count (≤280k tokens, ≤128 docs per batch),
calls OpenAI Embeddings API, writes to ChromaDB.

Token-based batching (not count-based) is critical — one chunk can be 10 tokens,
another 400, so a fixed count-based batch may blow the API limit.

---

## ChromaDB Storage

```
chroma_db/
├── chroma.sqlite3          — ids, texts, metadata (key-value rows)
└── <collection-uuid>/
    ├── data_level0.bin     — 43 MB: all 7718 vectors packed as raw floats
    ├── link_lists.bin      — HNSW graph (neighbour links between vectors)
    ├── header.bin          — index parameters
    └── length.bin          — vector lengths
```

**Search flow:**
1. Query text → embedding vector (OpenAI API)
2. HNSW walks `data_level0.bin` via neighbour graph → top-K vector ids
3. SQLite lookup by id → text + metadata
4. Return `List[Document]` with similarity scores

> **Why HNSW?** Brute-force over 7718 vectors is fast enough, but HNSW scales to
> millions without slowing down — it finds neighbours in ~50 graph steps instead of
> scanning everything.

---

## Incremental Operations (Backlog #8)

| Operation | Status | How |
|-----------|--------|-----|
| Add document | ✅ Available | `scripts/add_uk_143.py` pattern |
| Delete document | Not implemented | `collection.delete(where={"source": X})` |
| Update document | Not implemented | delete + add (upsert if chunk_id is stable) |
