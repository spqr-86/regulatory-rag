# Indexing review — 2 October 2026

Status: analysis completed; findings 1–3 implemented after Petr's scope approval.
Petr requested this review before indexing the replacement legal corpus. No production
code, active index, reference key or paid model calls were changed by this review.

## Follow-up implementation

Petr authorized findings 1–3 after this review. `index.py` now constructs a separate
candidate, requires all indexed sources and verifies stored content/provenance before
writing its readiness report. It never removes or switches the active store. With a
department manifest, ObjectProfile sheets remain outside the index and unlisted inputs
are not parsed. Source basenames must be unique; source hashes are checked before/after
parsing. Cached structural chunks exclude source/chunk_id; each input rebinds provenance.
Cache identity includes input format, package versions and chunker/cleaning settings.

The findings below describe the pre-fix pipeline. Legal/table boundaries, tokenizer
choice, normalization of collected HTML exports and verification of NPA editions are
still open. Readiness establishes storage/completeness, not legal accuracy or retrieval
quality. Activation remains an explicit application configuration/restart step.

Validation: 51 targeted tests passed, Black/Ruff and `check_docs --ci` passed. An
isolated real-Chroma smoke used local synthetic embeddings: 130 records survived
reopening in a second process; failure in embedding batch 2 preserved the old store
and produced no readiness report. No active-corpus build or paid calls were made.
The broader run excluding `tests/test_api.py` yielded 1,357 passed, 3 xfailed and
one known telemetry-writer failure (`TestWiring.test_jsonl_stays_the_default_writer`).
The full CI gate stalled at the API health TestClient; it is not reported as passed.

The second approved follow-up also rejects degraded conversion, narrows cleanup,
preserves distinct chunk occurrences, persists parser IR and validates managed
snapshots at load time. Its bounded list/table experiment and final validation are
recorded in [Ingest quality](experiments/ingest-quality-2026-10-02.md). Production
chunking/tokenizer and the active corpus remain unchanged.

## Pipeline at the initial audit

`index.py` discovers allowed files recursively, clears the parser cache, runs Docling
and HybridChunker, optionally attaches department manifest metadata, deletes the target
Chroma directory, then embeds and inserts batches of up to 128 chunks. BM25 is built
from stored documents at runtime.

The installed packages inspected were docling 2.124.0, docling-core 2.93.0,
chromadb 1.5.9 and langchain-chroma 1.1.0. Runtime chunking uses the default
`sentence-transformers/all-MiniLM-L6-v2` tokenizer, max_tokens=400, merge_peers=True.
`CHUNK_SIZE=1200` does not control this HybridChunker budget.

## Confirmed findings

1. **Replacement is not failure-safe.** `index.py` deletes the existing database after
   parsing but before new embeddings are written. An isolated mocked embedding failure
   removed the old-index sentinel. Build and validate a separate snapshot before switching
   the application to it; retain the previous snapshot for rollback.
2. **Missing documents do not fail the build.** `DocumentProcessor.process()` logs
   per-file errors or empty conversion results and continues. The aggregate nonempty
   check can therefore accept an incomplete corpus. Require a per-document processing
   report and completeness validation for mandatory sources before accepting a snapshot.
3. **Cache provenance is incorrect.** Cache keys use file bytes and PIPELINE_VERSION,
   while cached chunks contain `source`. Two equal files named a.pdf/b.pdf produced only
   a.pdf chunks in a real cache-path probe. Separate cached structural content from
   source identity; rebind provenance for each input. Parameter/library fingerprints
   also belong in the cache key. Full indexing currently clears all parser caches.
4. **Legal context is lost.** Stored 29н.pdf chunk 319 contains vehicle categories and
   examination frequency without the parent row identifying driving activities. Stored
   2464.pdf chunks 26 and 105 contain list items without the governing lead-in/clause.
   Preserve clause identity and list stem; serialize table rows with column headings
   and the parent category. The nearest section heading alone is insufficient.
5. **Token budget uses a different tokenizer from embeddings.** For one 217-character
   Russian clause, MiniLM counted 182 tokens and cl100k_base counted 87. On the same
   parsed replacement 2464 HTML, the existing processor emitted 346 chunks (median
   389.5 characters); cl100k_base at 400 tokens emitted 170 (median 705.5). The latter
   sometimes merges neighboring numbered clauses. This is a parsing comparison, not a
   measured retrieval improvement; changing the tokenizer alone is not the proposed fix.
6. **Corpus identity is not enforced for generic indexing.** Read-only SQLite inspection
   found 7,842 documents, including 50 from test.pdf absent from the source folder.
   Use a snapshot manifest with exact source hashes, editions, effective dates,
   embedding configuration and parser/chunker fingerprints. Sequential chunk_id remains
   useful within a snapshot but changes when chunk boundaries change.

The cleanup regex additionally removed `14/34` from a synthetic `приказ № 14/34`.
Review watermark cleaning against legal identifiers before normalizing the new inputs.

## Evidence and limits

- Local probes: `eval/runs/indexing_audit_2026-10-02/probes.json`.
- Parser experiment: `html_chunker_comparison.json` and `2464_docling.json` in that folder.
- The 24 tests in test_index_cache_invalidation.py, test_file_handler_chunk_id.py and
  test_corpus_manifest.py passed. They do not cover the three reproduced failure modes.
- Existing #64 artifacts in the chunkctx worktree: HR@12 on 133 questions remained
  0.827 before/after structural prefix restoration; MRR improved slightly. This patch
  does not establish that table parsing has been repaired.

## Recommended sequence, pending scope agreement

1. Safe snapshot construction, required-document completeness and cache provenance.
2. Persist the Docling structured intermediate representation so parsing and chunking
   can be evaluated independently without repeatedly running PDF layout/OCR.
3. Compare the current chunker, aligned tokenizer and explicit legal/table boundaries
   on a small fixed set of lists and table rows. Bound context size and inspect what
   actually reaches generation. Map relevance labels to legal clauses rather than old
   chunk numbers when boundaries change.
4. Select a candidate on retrieval coverage and regressions, then build a separate full
   index, verify revised references and obtain permission for external judging.

## Collected source candidates

The separate local package `eval/runs/corpus_refresh_2026-10-02/manifest.json` records
four user-supplied 1otruda HTML exports disguised as .doc (2464, 342н, 125-ФЗ, 696н),
an MChS-hosted PDF of 1120 including appendix 4, four articles of 370-ФЗ from
ConsultantPlus, and secondary-host texts of amendments 805, 392н and 327н.
The 342н export includes 327н and notes about future provisions effective in 2027.
The official publication portal was inaccessible; the Mintrud attachment returned 403.
These are candidates, not a verified active corpus. Current ALLOWED_TYPES excludes
.doc and .html; choose and verify format normalization before ingestion. Do not
publish licensed exports or add this mixed candidate/support package as an active corpus.

## Primary technical sources

- [Docling chunking](https://docling-project.github.io/docling/concepts/chunking/):
  tokenizer alignment, contextualize, table headers and line-preserving splitting.
- [Docling hybrid chunking example](https://docling-project.github.io/docling/_generated/examples/hybrid_chunking/).
- [Anthropic contextual retrieval](https://www.anthropic.com/engineering/contextual-retrieval):
  contextual text for embedding and BM25. Its reported gains are not measurements on
  this corpus; deterministic legal context should be tested before adding an LLM step.
