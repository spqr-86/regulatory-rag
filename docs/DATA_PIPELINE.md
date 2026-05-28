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
| `type` | `hybrid_chunk` |
| `parent_section` | nearest heading text |
| `heading_path` | full heading breadcrumb |
| `page_no` | page number in source document |
| `bbox` | bounding box coordinates |

### 5. Embeddings and Storage

**Model:** `text-embedding-3-small` (OpenAI)  
**Store:** ChromaDB (`./chroma_db`)

Chunks are embedded and stored in ChromaDB. A BM25 index is built in-memory at startup (not persisted to disk).

## Running

```bash
python index.py
```

> WARNING: destructive — drops the entire ChromaDB collection before reindexing.

Current corpus: 11 PDFs in `source_docs/`. Index not yet rebuilt after switching to v3.0-hybrid chunker.
