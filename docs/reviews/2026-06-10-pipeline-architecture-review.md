# V7 Pipeline Architecture & Bug Review — 2026-06-10

Scope: retrieval → merging → generation data flow, with focus on score normalization,
cross-reference expansion, and simple/complex path asymmetry. File references point at
the current state of `main` (f2e773f).

Legend: **Impact** High/Medium/Low · **Effort** Easy/Medium/Hard.

---

## A. Score normalization (vector vs BM25 vs reranker)

### A1. Raw BM25 scores leak into the hard-gate `top_score` — threshold gate is effectively disabled on the simple path
**Impact: High · Effort: Medium**

`BM25Index.search` (`src/v7/nlp_core.py:204-207`) copies the *raw* BM25Okapi score into
the passage's `score` field (`if "score" not in p: p["score"] = p["bm25_score"]`).
BM25Okapi scores are unbounded (typically 2–25 for any term match), while
`HARD_GATE_THRESHOLD = 0.50` (`src/v7/config.py:26`) is calibrated for the L2→similarity
mapping `1/(1+d)` where "relevant docs score ~0.44–0.56".

`rag_simple` carefully anchors `RetrievalAttempt.top_score` to vector results only
(`src/v7/nodes/rag_simple.py:87-88`) — but that field is **never used by the gates**.
`_legacy_triage` calls `check_full_triage` on the attempt's passages
(`src/v7/nodes/evaluate_triage.py:86`), and `check_hard_gates` recomputes
`top_score = max(p.get("score", 0.0) for p in passages)` (`src/v7/hard_gates.py:88`).
After `rrf_merge`, BM25-only chunks keep their raw score (vector lists are merged first,
so dual-found chunks keep the vector score — only-BM25 chunks don't), and the
"BM25 guarantee" block (`rag_simple.py:100-106`) appends up to 3 raw-score passages per
query variant unconditionally.

Consequence: whenever BM25 matched *anything* (i.e. almost always, since the query words
exist somewhere in the corpus), `top_score` is ~5–25 and `above_threshold` is trivially
true. The `0.50` acceptance gate and the `0.38` borderline floor are dead in practice;
sufficiency on the simple path actually hinges only on `min_passages=5` and
`keyword_overlap >= 0.15`. This is the most likely root cause of the false-sufficiency
instability (README reports 0.098, target < 0.10).

Same contamination repeats in `evaluate_complex` (`src/v7/nodes/evaluate_complex.py:29,40,55`)
because `check_hard_gates` is called on merged passages that include simple-attempt
BM25 chunks.

**Fix direction:** normalize at the source — never let `score` carry a raw BM25 value
(e.g. min-max/sigmoid-normalize per result list, or keep `bm25_score` separate and make
`check_hard_gates` read an explicitly named `vector_score`/`calibrated_score`).
Alternatively make the gates consume the already-anchored `attempt["top_score"]`.

### A2. `mmr_select` mixes incomparable score scales — BM25 chunks crowd out reranked chunks in the final merge
**Impact: High · Effort: Easy-Medium**

`mmr_select` uses `relevance = p.get("vector_score", p.get("score", 0.0))`
(`src/v7/nlp_core.py:335`). Simple-attempt BM25-only passages have no `vector_score`,
so their raw BM25 score (~5–25) is compared against vector similarities (~0.5) and
sigmoid-squashed CrossEncoder scores (≤1.0) of complex-attempt passages.
`merge_all_passages` (called from `evaluate_complex.py:27` with `top_k=24`) therefore
systematically prefers raw-BM25 chunks from the *fast* attempt over the reranked output
of the *slow* path — the CrossEncoder/FlashRank work is largely thrown away exactly on
the queries that escalated because the fast path wasn't good enough.

### A3. FlashRank scores are fed to gates despite the explicit decision not to
**Impact: Medium · Effort: Easy**

`make_rerank_fn` preserves `vector_score` "so downstream MMR/gates use vector scores
(FlashRank scores are not calibrated for thresholds)" (`src/v7/bridge.py:84-87`), and
`rag_complex` anchors `top_score` to `vector_score` (`src/v7/nodes/rag_complex.py:98-100`).
But the gates never read either: `compute_attempt_metrics` → `check_hard_gates`
(`rag_complex.py:102`, `hard_gates.py:88`) reads `p["score"]`, which after reranking *is*
the FlashRank probability. With `RERANKER_BACKEND=flashrank` (the default,
`config/settings.py:18`), the complex-path threshold of 0.50 is compared against
uncalibrated FlashRank probabilities — the stated design intent is silently violated.
(The sigmoid in `make_crossencoder_rerank_fn` mitigates this only for the non-default
backend.)

### A4. Misleading evidence labels in the generation prompt
**Impact: Low-Medium · Effort: Easy**

`_score_label` in `make_generate_fn` (`src/v7/bridge.py:304-309`) maps `score >= 0.6 →
HIGH`. Raw-BM25 chunks are always labeled HIGH, the semantically best vector chunks
(~0.44–0.56) get MED, and cross-ref chunks (fixed 0.35) get LOW — so the LLM is told the
keyword matches are the strongest evidence regardless of actual relevance. bbox siblings
inherit the parent score (`src/v7/cross_ref.py:179`), propagating the contamination.

---

## B. Cross-reference expansion (`src/v7/cross_ref.py`)

### B1. Substring match on bare ref numbers causes mass false positives
**Impact: High · Effort: Medium**

`_extract_refs` returns bare capture groups: `"пункт 46"` → `"46"`, `"статьи 5"` → `"5"`
(`cross_ref.py:22-40`). Mechanism 1 then does
`if any(ref in doc.page_content for ref in refs)` (`cross_ref.py:111`) — a plain substring
test over every chunk of the source. `"46"` matches "146", "461", "2464", "1946", table
values, page numbers; `"5"` matches nearly every chunk. Each match is appended as a
passage with score 0.35, polluting the context that `make_generate_fn` later caps at 30
(`bridge.py:351`).

### B2. `подпункт «а»` extracts a single Cyrillic letter — pulls in the entire source document
**Impact: High · Effort: Easy**

The pattern `подпункт\w*\s+[«"]?([а-яё])[»"]?` (`cross_ref.py:24`) captures one letter,
e.g. `"а"`. The same substring test (`cross_ref.py:111`) then matches **every chunk**
containing the letter "а" — i.e. all of them. Worse, `_get_source_docs` uses
`backend.get_by_filter(where={"source": source}, limit=500)` where `limit` is a *page
size*, not a cap (`src/backends/chroma_backend.py:62-95` paginates to exhaustion), so a
single passage that mentions "подпункта «а»" appends the whole source (ТК РФ is >500
chunks) to `extra`. The `[:30]` cap in generation hides most of it, but on the simple
path (≈12 base passages) up to ~18 slots get filled with arbitrary chunks in insertion
order — typically the document's preamble/TOC — and latency/memory take the hit every
generation call.

**Fix direction:** match refs structurally — anchor to clause starts (`^46\.` /
`статья 46` as a phrase) or, better, against a `clause_no` metadata field; require
letter refs to be matched as `подпункт «а»` phrases, never bare letters; cap extras.

### B3. Expansion runs inside generation, per call, with full-source scans
**Impact: Medium (latency/cost) · Effort: Medium**

`expand_cross_references` is invoked inside `make_generate_fn` (`bridge.py:343-346`) —
after gating, where its output bypasses all sufficiency checks, and synchronously on the
critical path. Mechanisms 1/3/4 do O(source_chunks × refs) substring scans per query;
the per-call cache (`cross_ref.py:67-82`) prevents re-fetching but not re-scanning. With
avg latency at 9.7s this is a meaningful contributor. Moving expansion to the retrieval
stage (where its passages would also be gate-checked) would be both safer and cheaper.

### B4. Mechanism 2 swallows all errors silently
**Impact: Low · Effort: Easy**

`except Exception: pass` (`cross_ref.py:138-139`) hides BM25 failures; at minimum log it
like mechanism 1 does.

---

## C. Simple vs complex path asymmetry

### C1. The "broader" complex path is actually *narrower*: vector-only, single-query
**Impact: High · Effort: Medium**

`rag_complex` calls only `_vector_search` (`src/v7/nodes/rag_complex.py:77-81`) — no
BM25, no RRF, no multi-query, unlike `rag_simple`. The irony: `_legacy_triage` escalates
on `keyword_overlap_original == 0.0` (`evaluate_triage.py:106-113`) — i.e. precisely when
a *lexical* match is missing — and the escalation target then drops the lexical channel
entirely. Recovery depends on `merge_all_passages` re-admitting simple-attempt chunks,
which is exactly where the A2 scale-mixing bug sits.

### C2. Enumeration escalation can turn a sufficient answer into an abstain
**Impact: Medium-High · Effort: Easy**

When triage says *sufficient* but the query matches an enumeration pattern,
`route_after_triage` forces `rag_complex` (`evaluate_triage.py:221-227`). But the
sufficient branch stored the passages as `final_passages`, **not** `fallback_passages`
(`evaluate_triage.py:115-120`), and `evaluate_complex`'s fallback ladder only reads
`fallback_passages` (`evaluate_complex.py:52-62`). If the complex attempt + merge fail
the (stricter) complex gates — `min_passages` 8 vs 5, `min_keyword_overlap` 0.20 vs 0.15 —
the pipeline abstains on a query the fast path had already answered. The escalation
should set `fallback_passages` (and arguably evaluate fallback with the plan that
accepted it).

### C3. Fallback passages are judged with the complex plan they never targeted
**Impact: Medium · Effort: Easy**

`evaluate_complex` checks `fallback_passages` against `plan` taken from the *complex*
attempt snapshot (`evaluate_complex.py:22,55`). A simple-path result that passed its own
gates (kw 0.15–0.20, 5–7 passages) fails the complex thresholds and is discarded →
abstain, even though triage explicitly saved it as "good enough to fall back on".

### C4. Inconsistent `retrieval_id` derivation
**Impact: Low · Effort: Easy**

Router computes `retrieval_id` from the **original** query (`router.py:95`), while
`rag_complex` recomputes it from the **glossary-expanded** `active_query`
(`rag_complex.py:68`). Dedup keys for the same logical retrieval differ between stages;
harmless today (no loops), a trap once retries are reintroduced.

---

## D. Dead or broken controls

### D1. `doc_id` is never set on any passage — diversity logic is fully degenerate
**Impact: Medium-High · Effort: Easy**

`compute_doc_diversity` and `mmr_select` read `p.get("doc_id", "unknown")`
(`src/v7/nlp_core.py:142,336`), but no producer ever writes a top-level `doc_id`
(grep: only readers exist; `bridge._doc_to_passage`, `_search`, `BM25Index.search` emit
`text/metadata/score/chunk_id`). Hence `unique_docs == 1` and `max_doc_ratio == 1.0`
always. Effects:
- `diversity_ok = (1.0 <= 0.8) = False` always → every comparison query
  (`require_multi_doc=True`, `router.py:54-56`) hard-fails triage and goes complex,
  unconditionally (`hard_gates.py:136-139`).
- MMR's diversity penalty treats the whole pool as one document → `mmr_select` is just
  relevance sort with a constant penalty.
- All logged `unique_docs`/`max_doc_ratio` metrics are wrong.

Fix: lift `metadata["source"]` (or a real doc id) into `doc_id` at passage construction.

### D2. Vector search silently ignores metadata filters; any filter empties BM25
**Impact: Medium (latent — nothing sets filters today) · Effort: Easy**

`make_vector_search_fn._search` accepts `filters` but never passes them to
`similarity_search_with_score` (`src/v7/bridge.py:172-193`). Meanwhile
`BM25Index.search` applies filters against **top-level** keys (`nlp_core.py:190-197`),
but corpus passages store everything under `metadata` (`bridge.py:398-401`,
`chroma_backend.iter_all_documents`) — so any filter like `{"doc_type": ...}` excludes
*every* BM25 passage. Net effect if a caller ever sets `state["filters"]`: vector search
ignores them, BM25 returns nothing. Both `validate_filters` plumbing and
`ALLOWED_FILTER_KEYS` suggest filters are supposed to work.

### D3. `sanitize_for_llm` is never called
**Impact: Medium (security control gap) · Effort: Easy**

`hard_gates.sanitize_for_llm` (`hard_gates.py:38-53`) and the related
`MAX_INPUT_LENGTH`/`BLOCKED_PATTERNS` config (`config.py:61-67`) have zero call sites.
Retrieved chunk text goes into the generation prompt unsanitized
(`bridge.py:352-361`), despite "anti-injection" being an advertised design property.
Either wire it in before prompt assembly or delete it to stop implying the protection
exists.

### D4. Router's per-query `mmr_lambda` is dead
**Impact: Low · Effort: Easy**

`_classify_query` picks 0.5/0.95/default (`router.py:45-58`) and the plan carries it
through `rag_complex` (`rag_complex.py:62`), but `evaluate_complex` calls
`merge_all_passages(attempts, top_k=24)` without it (`evaluate_complex.py:27`), so the
config default is always used. (Also: native Chroma-side MMR mentioned in
`mmr_select`'s docstring doesn't exist — `_search` uses plain
`similarity_search_with_score`, so the README's "top-60 + MMR" is only the broken
fallback MMR.)

### D5. `BM25_TOP_K` / `SEMANTIC_TOP_K` config knobs are unused
**Impact: Low · Effort: Easy**

`config.py:50-51` — both paths use `plan["top_k"]` for both retrievers. Delete or wire.

---

## E. API / UX correctness

### E1. API returns an empty answer for chitchat and OOS queries
**Impact: Medium · Effort: Easy**

`intent_gate` maps both noise *and* domain-gate OOS to `intent="noise"` → END
(`intent_gate.py:87,99`). `app.py:196` handles this, but `api.py:144-149` checks only
`clarify_message`/`abstain_reason`/`answer` → returns `{"answer": ""}` with an inferred
path of `"rag_simple → … → generate_answer"` — wrong on both counts. Also conflating OOS
with chitchat means an off-topic but serious question gets the chitchat brush-off in the
UI rather than an abstain with diagnostics (the README's diagram promises
`domain_gate → abstain`).

### E2. `_infer_path` keys on a state field that never exists
**Impact: Low · Effort: Easy**

`api.py:191` checks `result.get("complex_passages")` — no node ever writes that key, so
the API always reports the simple path. Use `retrieval_attempts[-1]["stage"]` (same logic
as `generate_answer._last_stage`).

### E3. `visual_enrichment` skips page 0
**Impact: Low · Effort: Easy**

`_needs_visual` requires truthy `meta.get("page_no")` (`visual_enrichment.py:42`) —
`page_no == 0` is falsy and gets skipped. Use `is not None`.

---

## F. Architecture observations / missed opportunities

1. **Crossref escalation fires on nearly all regulatory text** —
   `_count_crossref_hits >= 3` over the top-5 passages (`evaluate_triage.py:36-58`)
   counts (passage × pattern) pairs; phrases like "в соответствии с" and "пункт N" are
   ubiquitous in this corpus, so most "sufficient" queries still escalate to the slow
   path. The two-stage design's latency/cost benefit likely exists mostly on paper —
   worth measuring the actual escalation rate and either raising the threshold or
   requiring the *referenced clause to be absent* from the retrieved set before
   escalating.
2. **`fallback_passages` is advertised as a rag_complex starting point but never read
   there** (`evaluate_triage.py:198-204` comment) — only `evaluate_complex` consumes it.
3. **README/graph drift** — README shows `domain_gate` as a separate node that abstains;
   in code it lives inside `intent_gate` and emits `noise` (no abstain), and
   `DOMAIN_GATE_THRESHOLD` defaults to 0.0 (disabled), while the README headline claims
   OOS abstain 1.00 via the gate.
4. **No retrieval-stage reordering on the simple path** — the V8 light rerank
   (`rag_simple.py:112-123`) only computes metrics; passages go to generation in RRF
   order. Cheap win: apply the rerank order to the final passages when it ran anyway.
5. **Query embedding/LLM-expansion caching** — `intent_gate` embeds every query and
   multi-query expansion is an LLM call per request; an LRU on normalized query text
   would shave latency for repeated/benchmark traffic.

---

## Suggested priority

| # | Finding | Impact | Effort |
|---|---------|--------|--------|
| A1 | Raw BM25 in `score` defeats hard gates | High | Medium |
| A2 | Mixed scales in `mmr_select` crowd out reranked chunks | High | Easy-Med |
| B1/B2 | cross_ref substring false positives (bare number / single letter) | High | Easy-Med |
| C1 | Complex path drops the lexical channel | High | Medium |
| D1 | `doc_id` never set → diversity logic degenerate | Med-High | Easy |
| C2/C3 | Enumeration/fallback can abstain on already-sufficient results | Medium | Easy |
| A3 | FlashRank scores reach gates despite design note | Medium | Easy |
| D3 | `sanitize_for_llm` never wired | Medium | Easy |
| E1 | API empty answer on noise/OOS | Medium | Easy |
| D2 | Filters: dropped by vector, fatal to BM25 (latent) | Medium | Easy |
| rest | A4, B3, B4, C4, D4, D5, E2, E3, F* | Low-Med | Easy |
