# How Triage Works (evaluate_triage)

Triage is the central decision node in the V7 graph. After `rag_simple` it determines whether the retrieved passages are sufficient to generate an answer, or whether the search needs to be strengthened. The decision is deterministic: numbers only, no LLM.

---

## Metrics

After `rag_simple` returns top-12 passages with scores, triage examines three independent metrics:

| Metric | What it measures | Threshold |
|--------|-----------------|-----------|
| `top_score` | Cosine similarity of the best passage (vector score, not FlashRank) | `HARD_GATE_THRESHOLD = 0.50` |
| `passage_count` | Number of passages that passed reranking | `MIN_PASSAGES = 5` |
| `keyword_overlap` | Fraction of query keywords found in the top passages | `MIN_KEYWORD_OVERLAP = 0.15` |

All three must pass simultaneously — this is the **hard gate** (`check_hard_gates`).

Additionally:
- `max_single_doc_ratio` — fraction of passages from a single document (if > 0.8 and the query requires multi-doc → `escalation_hint`)

---

## Three Outcomes

```
check_full_triage()
    │
    ├── hard_sufficient=True AND no escalation_hint
    │       → sufficient  ──→ generate_answer
    │
    ├── top_score < TRIAGE_SOFT_THRESHOLD (0.38) OR passage_count < MIN_PASSAGES
    │       → clearly_bad ──→ rag_complex
    │
    └── otherwise (hard gates partially failed, or escalation_hint)
            → borderline  ──→ llm_verifier
```

**sufficient** — all hard gates green, passages are diverse → generate the answer.

**borderline** — score in zone `[0.38, 0.50)` or diversity issue → send to `llm_verifier`, which decides: `sufficient` / `rewrite` / `escalate`.

**clearly_bad** — score < 0.38 or too few passages → go straight to `rag_complex` (top-60 + MMR), no LLM verification.

---

## Why Vector Score, Not FlashRank

FlashRank is a cross-encoder; its scores cluster near 1.0 for any relevant passages (not calibrated as probabilities). No calibration was done for this domain. Vector cosine similarity (from ChromaDB) is a calibrated metric, 0.0–1.0 with clear meaning. Therefore `top_score` = vector score of the passage with the best cosine after FlashRank reordering.

The bug was here: before the fix, `top_score` was taken from FlashRank, causing inflation (~0.95+) so all queries went to `sufficient` even when retrieval was poor. After the fix: correctness 6.86 → 7.9.

---

## Why HARD_GATE_THRESHOLD = 0.50

Calibrated on the eval dataset of 50 questions. At 0.50:
- Fast path (sufficient): ~48% of queries, ~5s
- Slow path (borderline + clearly_bad): ~52%, ~22s

At 0.45 — too many go to sufficient, false-sufficiency increases. At 0.55 — almost everything goes to complex, latency +40% with no correctness gain.

`TRIAGE_SOFT_THRESHOLD = 0.38` — lower boundary for "found anything at all". Below this — retrieval is clearly bad, `llm_verifier` won't help, need `rag_complex`.

---

## Code

- `src/v7/hard_gates.py` — `check_hard_gates()`, `check_full_triage()`
- `src/v7/nodes/evaluate_triage.py` — thin node: reads state → calls `check_full_triage()` → writes `triage`
- `src/v7/config.py` — all thresholds with `V7_` env prefix (override via `.env` without code changes)

---

## Interview FAQ

> **Q:** Why doesn't an LLM decide whether the data is sufficient?
> **A:** LLMs are non-deterministic: the same passages on different days give different decisions. Numeric thresholds are reproducible — if a user reports an abstain, I open the log and see `top_score=0.47 < 0.50`. There is no such traceability with LLM routing. Plus ~1s latency saved per query.

> **Q:** How were the thresholds 0.50 / 0.38 chosen?
> **A:** Calibrated on the eval dataset: scanned a grid of values and observed the correctness distribution by path. 0.50 is the point where the correctness gain from the slow path stops compensating for its latency cost.
