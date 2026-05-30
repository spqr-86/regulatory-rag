# Data Pipeline

Describes how source documents are converted to a searchable vector index.

## Steps

### 1. Ingestion

**Module:** `src/indexing/file_handler.py`  
**Tool:** [Docling](https://github.com/DS4SD/docling)

Docling converts PDF/DOCX files to structured document objects, preserving headings, tables, and clause hierarchy.

### 2. Chunking

**Tool:** `HybridChunker` from `docling_core`  
**Config:** `max_tokens=400, merge_peers=True`

HybridChunker splits documents by structural boundaries — headings, clauses, numbered paragraphs — rather than by character count. Adjacent short chunks under the same heading are merged. This produces semantically coherent chunks aligned with the document's legal structure.

> **Why structure-aware chunking (decision).** Legal/regulatory text is hierarchical —
> articles, clauses, sub-clauses. Fixed-size character splitting cuts across clause
> boundaries and strands the number from its rule. HybridChunker aligns chunks to the
> document's own structure, which keeps each rule intact and replaced ~200 lines of a
> hand-rolled sentence-overlap splitter.

### 3. Text Cleaning

Each chunk goes through `_clean_noise()`:
- NFC normalization of Cyrillic text
- Removal of watermark URLs (`https://...`)
- Removal of page markers (`14/34`) and timestamps

### 4. Metadata

Each chunk carries:

| Field | Description |
|-------|-------------|
| `source` | filename (e.g. `2464.pdf`) |
| `chunk_id` | per-source 0-based int, assigned after dedup; `source#chunk_id` is the passage identity used for RRF fusion and cross-ref dedup |
| `type` | `hybrid_chunk` |
| `element_type` | Docling label (`table` / `text` / …) — lets table chunks be handled specially |
| `parent_section` | nearest heading text |
| `heading_path` | full heading breadcrumb |
| `page_no` | page number in source document |
| `bbox` | bounding box coordinates |

### 5. Embeddings and Storage

**Model:** OpenAI embeddings — see [FACTS](FACTS.md#models)
**Store:** ChromaDB (`./chroma_db`)

The embedded text is **contextual**: the parent-section heading is prepended to the chunk
text before embedding (`f"{parent_section}\n{text}"`), a lightweight form of Contextual
Retrieval — the article title lands in the vector without an extra LLM call. A guard avoids
duplicating the heading when the text already starts with it.

A BM25 index is built in-memory at startup (not persisted to disk).

## Running

```bash
python index.py
```

> WARNING: destructive — drops the entire ChromaDB collection (and Docling/BM25 caches)
> before reindexing. Always check `ls source_docs/ | wc -l` first — indexing an empty
> `source_docs/` wipes the index.

Current corpus size: see [FACTS](FACTS.md#corpus).
