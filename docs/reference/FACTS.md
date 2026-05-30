# FACTS — canonical reference

> Single source of truth for volatile facts. Prose elsewhere links here and does not
> repeat these numbers. `scripts/check_docs.py` verifies the machine-checkable parts of
> this file (prompt versions) against the repo and greps live docs for stale terms.
>
> **CI scope is honest and partial:** prompt-version sync + stale-term grep run in CI.
> Provider/model names and the reranker backend live in `.env` (gitignored secrets) and
> are absent in CI; chunk count needs a live ChromaDB. Those are checked only locally.

## models
- simple: `openai` / `gpt-4o-mini`  (`SIMPLE_LLM_PROVIDER`, `SIMPLE_MODEL_NAME`)
- complex: `openai` / `gpt-4o`  (`COMPLEX_LLM_PROVIDER`, `COMPLEX_MODEL_NAME`)
- judge (eval only): `gpt-4o`  (`JUDGE_MODEL_NAME`)
- reranker: `crossencoder`  (`RERANKER_BACKEND`; alt: `flashrank`)
- embeddings: `openai` / `text-embedding-3-small`

## thresholds
Defined in `src/v7/config.py` (env prefix `V7_`). Values below are the **runtime** values
(config default, or `.env` override where noted).

| name | value | env | note |
|---|---|---|---|
| HARD_GATE_THRESHOLD | 0.50 | `V7_HARD_GATE_THRESHOLD` | simple-path similarity gate |
| TRIAGE_SOFT_THRESHOLD | 0.38 | `V7_TRIAGE_SOFT_THRESHOLD` | borderline floor |
| MIN_PASSAGES | 5 | `V7_MIN_PASSAGES` | simple path |
| SIMPLE_TOP_K | 12 | `V7_SIMPLE_TOP_K` | |
| COMPLEX_THRESHOLD | 0.35 | `V7_COMPLEX_THRESHOLD` | slow-path floor |
| COMPLEX_MIN_PASSAGES | 8 | `V7_COMPLEX_MIN_PASSAGES` | |
| COMPLEX_MIN_KW_OVERLAP | 0.20 | `V7_COMPLEX_MIN_KW_OVERLAP` | |
| COMPLEX_TOP_K | 60 | `V7_COMPLEX_TOP_K` | |
| DOMAIN_GATE_THRESHOLD | 0.25 | `V7_DOMAIN_GATE_THRESHOLD` | **`.env` override**; config default 0.0 (disabled) |

## prompts
- generate_answer: v8
- query_expand: v1
- applicability_retriever: v2

## corpus
- documents: 12 НТД
- chunks: 7718  (reindex 2026-05-30, session 61; chunk_id 100%, per-source int)

## nodes
Graph nodes (`src/v7/graph.py`), entry = `intent_gate`:

```
intent_gate ─(noise)→ END
            └(domain)→ router ─(ambiguous)→ clarify_respond → END
                              └(ok)→ rag_simple → evaluate_triage ─(sufficient)→ visual_enrichment
                                                                  └(insufficient)→ rag_complex → evaluate_complex ─(pass)→ visual_enrichment
                                                                                                                  └(fail)→ abstain → END
visual_enrichment → generate_answer → END
```

- The **domain gate** (cosine-to-centroid OOS filter) is a step *inside* `intent_gate`, active when `DOMAIN_GATE_THRESHOLD > 0` — not a separate node.
- `clarify_respond` returns a clarification request for short/ambiguous queries.
- `visual_enrichment` is a no-op on VPS (`visual_proof_fn` not injected).
- No `llm_verifier` / `rewriter` (removed session 61): `evaluate_triage` routes sufficient→generate, otherwise→`rag_complex`.

## metrics
Source: `benchmarks/eval_v7_2026-05-30_chunkid.jsonl` (dataset 57, valid 54, judge `gpt-4o`).

| metric | value |
|---|---|
| in-scope correctness | 7.44 / 10 |
| correctness mean | 7.39 / 10 |
| faithfulness | 0.859 |
| answer relevance | 0.872 |
| OOS rejection rate | 1.00 |
| false-sufficiency rate | 0.098 |
| complex-path rate | 0.241 |
| avg latency | 9.71 s |
| cost | $0.0102 / query (N=10, `benchmarks/cps_2026-05-22.json`) |

## deploy
- port: 8502
- process: tmux session `sia`
