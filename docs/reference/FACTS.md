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
| FINAL_MERGE_TOP_K | 24 | `V7_FINAL_MERGE_TOP_K` | complex-branch output cap — chunks the generator sees (issue #10) |
| RERANK_CANDIDATE_CAP | 100 | `V7_RERANK_CANDIDATE_CAP` | hard limit on CrossEncoder input; dominant latency/quality knob |
| RRF_K | 60 | `V7_RRF_K` | RRF constant — measured dead on {1…200}, kept at 60 |
| MMR_LAMBDA | 0.7 | `V7_MMR_LAMBDA` | |
| DOMAIN_GATE_THRESHOLD | 0.25 | `V7_DOMAIN_GATE_THRESHOLD` | **`.env` override**; config default 0.0 (disabled) |

### V8 flags (env prefix `V7_`, field name keeps `V8_`)

| name | runtime | env | note |
|---|---|---|---|
| V8_ENABLE_MULTI_QUERY | **true** | `V7_V8_ENABLE_MULTI_QUERY` | **`.env` override**; config default false |
| V8_EXPAND_N | 3 | `V7_V8_EXPAND_N` | query reformulations generated when multi-query is on |

## prompts
- generate_answer: v8
- query_expand: v1
- applicability_retriever: v2

## corpus
- documents: 12 НТД
- chunks: 7792  (reindex 2026-09-02; chunk_id 100%, per-source int)
- chunks after GT junk filter: 7276  (`eval/generate_retrieval_gt.py`, `MIN_CHUNK_CHARS=200`)
- full `index.py` run: ≈54 min, peak ≈3 GB RSS + 4.3 GB swap (docling/torch dominates;
  embeddings go over the API). The run wipes `chroma_db/` and the docling cache first,
  so a crashed run restarts from zero.
- previous: 7718 chunks (reindex 2026-05-30, session 61)

## nodes
Graph nodes (`src/v7/graph.py`), entry = `intent_gate`:

```
intent_gate ─(noise)→ END
            └(domain)→ router ─(ambiguous)→ clarify_respond → END
                              └(ok)→ rag_simple → evaluate_triage ─(generate)→ generate_answer → END
                                                                  ├(complex)→ rag_complex → evaluate_complex ─(generate)→ generate_answer → END
                                                                  │                                       └(abstain)→ abstain → END
                                                                  └(abstain)→ abstain → END
```

- The **domain gate** (cosine-to-centroid OOS filter) is a step *inside* `intent_gate`, active when `DOMAIN_GATE_THRESHOLD > 0` — not a separate node.
- `clarify_respond` returns a clarification request for short/ambiguous queries.
- Both evaluators own the internal `enrich → pack → validate → decide` pipeline;
  `visual_enrichment` and `pack_context` are helpers, not graph nodes.
- No `llm_verifier` / `rewriter` (removed session 61): `evaluate_triage` routes sufficient→generate, otherwise→`rag_complex`. <!--freshness:ignore-->
- **`evaluate_triage` is a single hard-gate path** (`check_full_triage`: `top_score` / `passage_count` / `keyword_overlap`). On a sufficient verdict it emits a structured `triage_gap` (`TriageGap`/`GapRef`, issue #13) describing which referenced п./ст. are missing from the top-5; the gap is closed in place by appending cross-referenced passages to the tail (no reorder), and escalation to `rag_complex` happens only if the gap stays open. The V8 `_evidence_assess` variant and its `V7_V8_ENABLE_EVIDENCE_ASSESS` flag were removed 2026-09-08 (variant B).

## metrics
Source: `benchmarks/eval_v7_triage_contract_final_2026-09-11.jsonl` (dataset 56,
valid 53). Pipeline: the final terminal triage contract, OpenAI `gpt-4o-mini` (simple) /
`gpt-4o` (complex), `gpt-4o` judge, CrossEncoder reranker, cap 100.

| metric | value | pre-contract baseline (2026-09-08) |
|---|---|---|
| in-scope correctness | 7.47 / 10 | 7.40 |
| correctness mean | 7.26 / 10 | 7.09 |
| faithfulness | 0.891 | 0.808 |
| answer relevance | 0.879 | 0.881 |
| OOS rejection rate | 1.00 | 1.00 |
| false-sufficiency rate | 0.114 | 0.130 |
| complex-path rate | 0.170 | 0.132 |
| latency p50 / p95 / mean | 4.51 / 15.70 / 6.83 s | 4.8 / 14.7 / 5.9 s |
| cost / query | $0.00387 ($0.205 run total) | $0.0033 |

The terminal contract improved faithfulness and false sufficiency without losing
correctness or relevance beyond judge variance. It costs about 18% more per query and
adds about 7% to p95 latency relative to the pre-contract hard-gate baseline.

> **Judge note (fix 2026-09-02).** Every run before this date was judged by
> **`gpt-4o-mini`, not `gpt-4o`** — `llm_factory` carried the resolved settings model into
> the constructor only on the gemini branch, so every OpenAI getter fell back to the
> `ChatOpenAI` default and `JUDGE_MODEL_NAME` / `COMPLEX_MODEL_NAME` were silently ignored
> (the complex path ran on mini too). The 2026-09-08 run above is the first valid
> post-fix baseline; the `gpt-4o` judge is stricter, so absolute faithfulness / relevance
> sit slightly below the historical mini-judge numbers while being better calibrated.
> Judge run-to-run variance is ~±0.25 on the correctness scale (design-decisions §8).

### Retrieval (final test: 90 practitioner questions, `eval/run_retrieval_eval.py`, 2026-09-08)

| path | HR@5 | HR@12 | MRR | p50 latency |
|---|---|---|---|---|
| hybrid (production) | 0.633 | 0.811 | 0.503 | 550 ms |
| vector-only | 0.589 | 0.822 | 0.486 | 148 ms |
| bm25-only | 0.500 | 0.667 | 0.352 | 24 ms |

Test set (`eval/data/golden_retrieval_labeled_ext.jsonl`, built 2026-09-06): 90 questions
taken verbatim from OT/PB practitioner forums (`docs/research/prompt-heldout-questions.md`),
kept only where a corpus answer was confirmed in a 3-pass model review + manual check;
the ~40 questions with no corpus answer were dropped. Built **after** the retrieval config
was frozen — not used for any tuning. Relevance labels are model-assisted then hand-checked;
candidate pooling and the full derivation are in `docs/roadmap.md` (step 4).
~19% of questions are not retrieved at all in top-12 (weakest: 29н medical exams, HR@5 0.56).

The combined set `golden_retrieval_labeled.jsonl` (133 = these 90 + 43 earlier
chunk-derived questions used during tuning) scores higher because the 43 synthetic ones
are easier: hybrid 0.692 / 0.827 / 0.542. The 90-question practitioner subset above is the
honest headline number.

### Стоимость и латентность запроса (roadmap 4a)

С 05.09.2026 `eval/run_v7_eval.py` считает цену прогона по токенам, которые пайплайн
реально потратил: usage приходит из провайдера в состояние графа (`src/v7/usage.py`,
ключ `llm_usage`), раннер умножает на прайс (`eval/pricing.py`) и печатает разбивку по
путям и p50/p95 латентности. Цифры лежат в `aggregate.cost` каждого файла прогона —
отдельной константы здесь больше нет, чтобы она не протухала.

Финальный замер 11.09.2026 (53 валидных запроса, судья gpt-4o): $0.00387 / запрос,
$0.205 весь прогон. По путям: simple $0.00107 / запрос (n=44, p50 4.31 с), complex
$0.01755 / запрос (n=9, p50 14.04 с) — разница ~16× при доле complex 17.0%.

## deploy
- port: 8502
- process: tmux session `sia`
