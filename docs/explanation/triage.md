# How triage works (`evaluate_triage`)

Triage is the central decision node in the V7 graph. After `rag_simple` it decides whether
the retrieved passages are enough to answer, need a broader search (`rag_complex`), or
should lead straight to an abstain. The decision is deterministic — numbers only, no LLM.
Threshold values live in [FACTS](../reference/FACTS.md#thresholds).

The node returns a terminal contract rather than a loose boolean:
`route_decision` (`generate`, `complex`, or `abstain`) plus a stable `route_reason`.
An answer is generated only from `final_context` after `pack_context` has expanded,
sanitized, truncated, and budget-checked the evidence.

`evaluate_triage` is a single hard-gate path. (Before 2026-09-08 the node dispatched by a
`V7_V8_ENABLE_EVIDENCE_ASSESS` flag to an alternative `_evidence_assess` variant that
scored the reranker top-1 plus a coverage estimate; that variant and the flag were removed
— variant B, see [Design decisions §3](./design-decisions.md).)

---

## Hard gate

Three independent conditions, all must pass (`check_full_triage` in
`src/v7/hard_gates.py`):

| Metric | What it measures | Threshold |
|--------|-----------------|-----------|
| `top_score` | Cosine similarity of the best passage — **vector score, not the reranker score** | `HARD_GATE_THRESHOLD` |
| `passage_count` | Passages retained | `MIN_PASSAGES` |
| `keyword_overlap` | Fraction of query keywords found in the top passages | `MIN_KEYWORD_OVERLAP_ACTIVE` |

When the hard gates pass but a soft signal fires, the node runs a cascade before
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
(measured regression on the 133-question synthetic set).

---

## Routing — two live destinations

`route_after_triage` collapses every verdict to two paths:

```
sufficient          → generate
everything else      → rag_complex   (then answer or abstain)
```

One override applies to **both** paths: queries with **enumeration intent**
(`_has_enumeration_intent` — "кто проходит", "какие категории", "перечислите", …) are
forced to `rag_complex` even on a sufficient verdict, because they need complete coverage
across sections.

---

## Why the vector score, not the reranker score

The hard gate uses ChromaDB cosine similarity (a calibrated 0–1 metric) for `top_score`,
not the cross-encoder score. Reranker scores are not calibrated probabilities for this
domain — they cluster near the top and would inflate the gate. This was the source of a
real bug ([Design decisions §3](./design-decisions.md)): the old FlashRank backend pushed
every score to ~0.999, retrieval lost all ranking, and the exact chunk sank. The removed
`_evidence_assess` variant gated on `reranker_top1` directly, which is why it was dropped.

---

## Why deterministic thresholds

- **Reproducibility** — same query + index → same path. An unexpected abstain shows in the
  log as e.g. `top_score=0.42 < 0.5`. LLM routing gives no such traceability.
- **Latency / cost** — no extra LLM call in the routing decision.

Thresholds are overridable via `V7_`-prefixed env vars (`src/v7/config.py`) without code
changes.

---

## Code

- `src/v7/nodes/evaluate_triage.py` — `evaluate_triage`, `build_gap`,
  `_merge_new_at_tail`, `route_after_triage`
- `src/v7/hard_gates.py` — `check_full_triage()` and hard-gate classification
- `src/v7/cross_ref.py` — `_extract_refs`, `expand_cross_references`
- `src/v7/config.py` — thresholds (`V7_` env prefix)

See also: [architecture.md](./architecture.md) · [design-decisions.md](./design-decisions.md).
