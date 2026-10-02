# Ingest quality and bounded chunking experiment — 2 October 2026

Petr authorized the follow-up to the ingestion audit: reject degraded conversion,
preserve legal text during cleaning, enforce managed-snapshot readiness, persist
structured parser output and compare chunking on a small fixed sample. Activation,
external judging and an embedding/retrieval benchmark were not part of this step.

## Implemented behavior

- Only Docling SUCCESS with no conversion errors enters the structured cache. Partial
  conversion is rejected. Strict indexing also rejects flattened DOCX fallback,
  including when a fallback result was read from the chunk cache.
  Degraded fallback is not cached; strict mode reparses any older fallback cache so
  a transient converter failure does not block recovery until cache expiry.
- Cleanup removes recognized standalone boilerplate lines. Inline URLs, slash-separated
  identifiers, legitimate text containing “Страница”, and hyphens across line breaks
  survive. Occurrence identity includes Docling item references and page/bbox location;
  processing reports count retained chunks after deduplication.
- Docling JSON is cached under `CACHE_DIR/ir` by input hash, format, parser policy and
  Docling versions. This cache is independent of chunking/cleaning settings. Only a
  complete parse is persisted. A real Markdown smoke reused IR with conversion mocked
  to fail if called: one original chunk became two after changing chunker parameters.
- New builds write an early marker and a schema-2 report only after storage validation.
  Managed candidates without that report cannot load, even if a failed build contains
  records. Loading checks path, collection, embedding model, record count and a checksum
  over stored text/metadata. Legacy unmarked stores retain compatibility behavior and
  are not retrospectively certified by these checks.
- Consumer-specific activation settings are documented: legacy/API, Generic and
  Department use different paths/collections. No running service or active index was
  changed.

## Offline experiment

Reproduction (no embeddings or LLM calls):

```bash
PYTHONPATH=. .venv/bin/python scripts/compare_ingest_chunking.py \
  --ir eval/runs/indexing_audit_2026-10-02/2464_docling.json \
  --output eval/runs/ingest_quality_2026-10-02/chunking_comparison.json
```

The sample consists of 18 list items under clauses 16, 33 and 46 of the saved candidate
2464 IR and a synthetic 10-row table with a merged category cell. Coverage means that
the complete selected item and its governing stem coexist in one chunk. Markdown link
syntax and whitespace are normalized for comparison; editorial edition notes are
excluded from the selected item. The raw report retains the stems/items for inspection.

| Variant | Chunks in full 2464 IR | List context coverage | Synthetic table category at last row |
|---|---:|---:|---|
| Current production processing, default tokenizer, 400 tokens | 346 | 3/18 | Preserved |
| Same processing, cl100k_base, 400 tokens | 170 | 7/18 | Preserved |
| Experimental explicit stem per item / headers and expanded category per row | Not a full-document replacement | 18/18 | Preserved |

The explicit sample fits within 170 cl100k_base tokens per unit. Oversized units raise
an error instead of being silently truncated. The experimental serializer is confined
to the comparison script; production chunk boundaries/tokenizer remain unchanged.

## Interpretation and limits

Changing the tokenizer alone does not preserve most selected list context. Repeating
the governing stem directly addresses this boundary failure on the selected sample.
The explicit serializer constructs that association by design; 18/18 is a serialization
check, not an independent retrieval-quality score. There is no held-out evaluation,
generation-context measurement or proof of legal currency from this experiment.

All variants preserved the synthetic table category. This fixture does not reproduce
the observed 29н PDF failure, so this step does not claim to repair real table parsing.
Inspect the actual 29н structured table, including merged cells and page transitions,
before selecting a production table serializer. Also verify parsing and normalization
of the replacement HTML sources separately.

Structural JSON retains table spans that Markdown export can flatten; this is why it
is the reusable intermediate representation rather than plain text. See the official
[Docling serialization documentation](https://docling-project.github.io/docling/concepts/serialization/)
and [chunking documentation](https://docling-project.github.io/docling/concepts/chunking/).

## Next acceptance gate

Use the real table IR and fixed legal-clause labels to compare context reaching the
answer against baseline. Measure retrieval coverage/regressions before adopting the
experimental serializer. Verify the NPA editions and only then build an active-corpus
candidate. Storage readiness alone does not establish legal accuracy.

Independent pre-save review also identified two readiness limitations: model-name
matching does not pin provider, dimensions, revision, pooling or normalization, and
the checksum covers text/metadata rather than vector values. Before activating a
candidate, verify the embedding configuration and query behavior; the current report
does not certify semantic vector integrity.

## Validation

83 targeted tests passed, including recovery after degraded conversion and an older
fallback cache. Black, Ruff and `check_docs --ci` passed. The earlier broader gate
excluding the previously hanging API TestClient yielded 1,373 passed, 3 xfailed and
the known telemetry default-writer failure. That failure and API hang were already
reproduced on the unchanged baseline in the previous step; the full gate is not
reported as green.

A real isolated Chroma smoke persisted 130 records and reopened them through the
managed loader in a second process. A second-batch failure preserved the old store;
the partial candidate and a different embedding model were rejected. A separate real
Markdown parse/rechunk smoke confirmed reuse of structured JSON without conversion.
These smokes used local synthetic embeddings and temporary directories. No paid calls,
active-index changes or production restarts occurred.
