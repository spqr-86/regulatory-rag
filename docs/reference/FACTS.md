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
                              └(ok)→ rag_simple → evaluate_triage ─(sufficient)→ visual_enrichment
                                                                  └(insufficient)→ rag_complex → evaluate_complex ─(pass)→ visual_enrichment
                                                                                                                  └(fail)→ abstain → END
visual_enrichment → generate_answer → END
```

- The **domain gate** (cosine-to-centroid OOS filter) is a step *inside* `intent_gate`, active when `DOMAIN_GATE_THRESHOLD > 0` — not a separate node.
- `clarify_respond` returns a clarification request for short/ambiguous queries.
- `visual_enrichment` is a no-op on VPS (`visual_proof_fn` not injected).
- No `llm_verifier` / `rewriter` (removed session 61): `evaluate_triage` routes sufficient→generate, otherwise→`rag_complex`. <!--freshness:ignore-->

## metrics
Source: `benchmarks/eval_v7_2026-05-30_chunkid.jsonl` (dataset 57, valid 54).

> **Judge caveat (found 2026-09-02):** this run was judged by **`gpt-4o-mini`, not `gpt-4o`**.
> `llm_factory` carried the resolved settings model into the constructor only on the gemini
> branch, so every OpenAI getter fell back to the `ChatOpenAI` default and `JUDGE_MODEL_NAME`
> / `COMPLEX_MODEL_NAME` were silently ignored. Fixed 2026-09-02; the numbers below are not
> comparable with runs made after that fix, and the complex path ran on mini as well.

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
| cost | измеряется каждым прогоном, см. ниже |

### Стоимость и латентность запроса (roadmap 4a)

С 05.09.2026 `eval/run_v7_eval.py` считает цену прогона по токенам, которые пайплайн
реально потратил: usage приходит из провайдера в состояние графа (`src/v7/usage.py`,
ключ `llm_usage`), раннер умножает на прайс (`eval/pricing.py`) и печатает разбивку по
путям и p50/p95 латентности. Цифры лежат в `aggregate.cost` каждого файла прогона —
отдельной константы здесь больше нет, чтобы она не протухала.

Смоук 05.09.2026 (N=3, `--skip-judge`, gpt-4o-mini simple / gpt-4o complex): simple
$0.00095 за запрос, complex $0.02099 — разница в 22 раза при доле complex ~24%.
**Это смоук, а не замер:** три вопроса, один из них complex. Старая цифра $0.0102 / query
снята 22.05.2026 на N=10 и до фикса выбора моделей 02.09 — она описывала другой пайплайн.

## deploy
- port: 8502
- process: tmux session `sia`
