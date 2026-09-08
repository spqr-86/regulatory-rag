# How triage works (`evaluate_triage`)

Triage is the central decision node in the V7 graph. After `rag_simple` it decides whether
the retrieved passages are enough to answer, need a broader search (`rag_complex`), or
should lead straight to an abstain. The decision is deterministic — numbers only, no LLM.
Threshold values live in [FACTS](../reference/FACTS.md#thresholds).

`evaluate_triage` dispatches to one of two implementations by the
`V7_V8_ENABLE_EVIDENCE_ASSESS` flag:

| flag | implementation | status |
|---|---|---|
| `true` (production) | `_evidence_assess` — reranker-score + coverage | active on the deployed instance |
| `false` | `_legacy_triage` — the three-metric hard gate | fallback / historical |

---

## Production path — `_evidence_assess` (V8)

Reads two numbers off the last retrieval attempt's `metrics`:

- `reranker_top1` — the top passage's reranker score (CrossEncoder in production,
  sigmoid-calibrated so the thresholds stay stable across backends).
- `coverage_estimate` = `keyword_overlap_active × min(passage_count / 10, 1.0)`.

Three-way verdict (thresholds in [FACTS](../reference/FACTS.md#v8-flags)):

```
reranker_top1 ≥ 0.6  AND coverage ≥ 0.6   → "answer"   → generate
reranker_top1 < 0.2  AND coverage < 0.2   → "abstain"  → rag_complex → evaluate_complex → abstain
everything else                            → "improve"  → rag_complex
```

On `answer` and `improve` the simple-path passages are stashed as `fallback_passages` so
`evaluate_complex` can recover them if a later complex attempt comes back worse. The
verdict and its inputs are saved as an `EvidenceReport` in state for per-query tracing.

---

## Fallback path — `_legacy_triage` (hard gate + structured gap)

Used when the V8 flag is off. Three independent conditions, all must pass
(`check_full_triage` in `src/v7/hard_gates.py`):

| Metric | What it measures | Threshold |
|--------|-----------------|-----------|
| `top_score` | Cosine similarity of the best passage — **vector score, not the reranker score** | `HARD_GATE_THRESHOLD` |
| `passage_count` | Passages retained | `MIN_PASSAGES` |
| `keyword_overlap` | Fraction of query keywords found in the top passages | `MIN_KEYWORD_OVERLAP_ACTIVE` |

When the hard gates pass but a soft signal fires, `_legacy_triage` runs a cascade before
deciding to escalate:

1. **Structured sufficiency gap** (issue #13, stage B2). `build_gap` inspects the
   cross-references in the top-5 passages (`_extract_refs`) and lists which п./ст. are
   *not* present anywhere in the retrieved set, deduplicated by `(doc_id, kind, num)`.
2. **Cross-reference escalation.** If the top-5 carry ≥ 3 cross-reference markers
   (`_CROSSREF_ESCALATION_THRESHOLD`) *and* the gap is open, the injected
   `_crossref_expander` pulls the referenced clauses in and `_merge_new_at_tail` appends
   them to the end of the list without reordering the originals. `check_full_triage` is
   re-run on the merged list; if the gap closed, the cascade falls through instead of
   escalating.
3. **Zero-overlap escalation** — none of the original query keywords appear in the
   passages → escalate.
4. **Enumeration escalation** — see below.

The `_merge_new_at_tail` "append, never reorder" rule exists because an earlier version
let the expanded list reorder the output and pushed a gold chunk past the top-12
(measured regression on the 133-question synthetic set). The B2 gap logic currently lives only in
`_legacy_triage`; porting it into `_evidence_assess` is not yet ticketed.

---

## Routing — two live destinations

`route_after_triage` collapses every verdict to two paths:

```
sufficient / "answer"          → generate (via visual_enrichment)
everything else                → rag_complex   (then answer or abstain)
```

One override applies to **both** paths: queries with **enumeration intent**
(`_has_enumeration_intent` — "кто проходит", "какие категории", "перечислите", …) are
forced to `rag_complex` even on a sufficient verdict, because they need complete coverage
across sections.

---

## Why the vector score, not the reranker score (legacy gate)

The legacy hard gate uses ChromaDB cosine similarity (a calibrated 0–1 metric) for
`top_score`, not the cross-encoder score. Reranker scores are not calibrated probabilities
for this domain — they cluster near the top and would inflate the gate. This was the
source of a real bug ([Design decisions §3](./design-decisions.md)): the old FlashRank
backend pushed every score to ~0.999, retrieval lost all ranking, and the exact chunk
sank. `_evidence_assess` instead reads `reranker_top1` deliberately (CrossEncoder,
sigmoid-calibrated), against V8 thresholds tuned for it.

---

## Why deterministic thresholds

- **Reproducibility** — same query + index → same path. An unexpected abstain shows in the
  log as e.g. `reranker_top1=0.14 < 0.2`. LLM routing gives no such traceability.
- **Latency / cost** — no extra LLM call in the routing decision.

Thresholds are overridable via `V7_`-prefixed env vars (`src/v7/config.py`) without code
changes.

---

## Code

- `src/v7/nodes/evaluate_triage.py` — `evaluate_triage` dispatch, `_evidence_assess`,
  `_legacy_triage`, `build_gap`, `_merge_new_at_tail`, `route_after_triage`
- `src/v7/hard_gates.py` — `check_full_triage()` and hard-gate classification
- `src/v7/cross_ref.py` — `_extract_refs`, `expand_cross_references`
- `src/v7/config.py` — thresholds and V8 flags (`V7_` env prefix)

See also: [architecture.md](./architecture.md) · [design-decisions.md](./design-decisions.md).
