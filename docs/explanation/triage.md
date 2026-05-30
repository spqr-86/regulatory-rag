# How triage works (`evaluate_triage`)

Triage is the central decision node in the V7 graph. After `rag_simple` it decides whether
the retrieved passages are sufficient to answer, or whether the search must be strengthened.
The decision is deterministic — numbers only, no LLM. Threshold values live in
[FACTS](../reference/FACTS.md#thresholds).

---

## The three metrics (hard gate)

After `rag_simple` returns its top-K passages with scores, triage checks three independent
conditions — all must pass simultaneously (`check_hard_gates` in `src/v7/hard_gates.py`):

| Metric | What it measures | Threshold (see FACTS) |
|--------|-----------------|-----------------------|
| `top_score` | Cosine similarity of the best passage — **vector score, not the reranker score** | `HARD_GATE_THRESHOLD` |
| `passage_count` | Number of passages retained | `MIN_PASSAGES` |
| `keyword_overlap` | Fraction of query keywords found in the top passages | (config) |

Additionally, `max_single_doc_ratio` — if too many passages come from one document and the
query needs multiple sources, the gate raises an escalation hint.

---

## Routing — two live destinations

`evaluate_triage` classifies the result into `sufficient` / `borderline` / `clearly_bad`,
but **routing collapses to two paths** (`route_after_triage`):

```
sufficient            → generate (via visual_enrichment)
borderline | clearly_bad → rag_complex   (deep search, top-K + MMR)
```

The `llm_verifier` / `rewriter` middle path was removed (session 61): anything not <!--freshness:ignore-->
`sufficient` now escalates straight to `rag_complex`. One special case — queries with
**enumeration intent** are forced to `rag_complex` even when triage says sufficient, because
they need complete coverage across sections.

---

## Why the vector score, not the reranker score

The gate uses ChromaDB cosine similarity (a calibrated 0–1 metric), not the cross-encoder
reranker score. Reranker scores are not calibrated probabilities for this domain — they
cluster near the top and would inflate the gate. `rag_simple` therefore takes `top_score`
from the vector results only.

This was the source of a real bug and is the subject of
[Design decisions §3](./design-decisions.md): the previous reranker (FlashRank) pushed <!--freshness:ignore-->
every score to ~0.999, so retrieval lost all ranking and the exact chunk sank — switching
to CrossEncoder and reading the vector score for the gate fixed it.

---

## Why deterministic thresholds

- **Reproducibility** — same query + index → same path. If a user reports an unexpected abstain, the log shows e.g. `top_score=0.47 < 0.50`. LLM routing gives no such traceability.
- **Latency / cost** — no extra LLM call in the routing decision.

Thresholds are calibrated on the eval dataset and overridable via `V7_`-prefixed env vars
(`src/v7/config.py`) without code changes.

---

## Code

- `src/v7/hard_gates.py` — `check_hard_gates()`, triage classification
- `src/v7/nodes/evaluate_triage.py` — thin node + `route_after_triage`
- `src/v7/config.py` — thresholds (`V7_` env prefix)

See also: [architecture.md](./architecture.md) · [design-decisions.md](./design-decisions.md).
