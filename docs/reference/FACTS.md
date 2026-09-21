# FACTS — canonical reference

> Single source of truth for volatile facts. Prose elsewhere links here and does not
> repeat these numbers. `scripts/check_docs.py` verifies the machine-checkable parts of
> this file (prompt versions) against the repo and greps live docs for stale terms.
>
> **CI scope is honest and partial:** prompt-version sync + stale-term grep run in CI.
> Provider/model names and the reranker backend live in `.env` (gitignored secrets) and
> are absent in CI; chunk count needs a live ChromaDB. Those are checked only locally.

## models
- simple: `openrouter` / `deepseek/deepseek-v4.1-flash`  (`SIMPLE_LLM_PROVIDER`, `SIMPLE_MODEL_NAME`)
  — showcase default since 17.09.2026, chosen by `eval/runs/object_profile_traps_2026-09-17/summary.md`;
  OSS/OpenAI alternative: `openai` / `gpt-4o-mini`
- complex: `openrouter` / `deepseek/deepseek-v4.1-flash`  (`COMPLEX_LLM_PROVIDER`, `COMPLEX_MODEL_NAME`)
  — same model as simple since 18.09.2026; OSS/OpenAI alternative: `openai` / `gpt-4o`
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
- department_answer: v1

`department_answer` v2 and v3 exist in the registry (neither is `active_version`); `v3` is
selected by `DEPARTMENT_QA_MODE=v2` at runtime, not by the registry — see "Q&A подразделений"
below. `v2` is kept unchanged so the 2026-09-15 paired run can be re-rendered.
`department_verify` v1 is an opt-in experimental second call; it is not enabled in the UI.

## corpus
- documents: 12 НТД
- chunks: 7792  (reindex 2026-09-02; chunk_id 100%, per-source int)
- chunks after GT junk filter: 7276  (`eval/generate_retrieval_gt.py`, `MIN_CHUNK_CHARS=200`)
- full `index.py` run: ≈54 min, peak ≈3 GB RSS + 4.3 GB swap (docling/torch dominates;
  embeddings go over the API). The run wipes `chroma_db/` and the docling cache first,
  so a crashed run restarts from zero.
- previous: 7718 chunks (reindex 2026-05-30, session 61)

## department qa
`src/department_qa/` — separate Q&A stack for units (подразделения), spec
[2026-09-14-department-qa-mvp-design](../superpowers/specs/2026-09-14-department-qa-mvp-design.md)
+ [2026-09-15-object-profile-design](../superpowers/specs/2026-09-15-object-profile-design.md).

- `DEPARTMENT_QA_MODE` (env, `config/settings.py`): `v1` or `v2` (default, decision 10). Selects one bundle
  atomically — Chroma path/collection, whether unit object sheets are loaded as profiles, and
  the `department_answer` prompt version. `wiring.ensure_store_matches` fails fast if
  `CHROMA_DB_PATH`/`CHROMA_COLLECTION_NAME` don't match the selected mode.

| mode | Chroma path | collection | profiles | prompt | structured-output schema |
|---|---|---|---|---|---|
| `v1` | `./chroma_db_dept` | `department_demo` (sheets in the index) | not loaded, `profile=None` | `department_answer` v1 | `ModelAnswerV1` |
| `v2` | `./chroma_db_dept_v2` | `department_demo_v2` (sheets excluded, `role: object_profile`) | `load_profiles` | `department_answer` v4 | `ModelAnswer` |

Scoped Department service (Issue #58) uses `department_answer` v5 and the same V2
manifest/index through the retrieval-only V7 graph. Historical v1/v2 pair artifacts
keep their recorded prompt versions. The root UI is wired to this scoped service by Issue
#59; `DEPARTMENT_QA_MODE=v1` is rejected explicitly because the scoped service requires the
v2 object-profile contract. Generic Q&A remains on `pages/2_Общий_поиск.py`.

Cutover evidence (20.09.2026): the parallel scoped service (`workers=4`, production default)
passed 9/9 contract checks on the object-profile pair set, with 0
`applied_on_unknown` and 0 forbidden-conclusion violations. The old service baseline was
8/9. The 8/9→9/9 difference includes prompt v4→v5 and the corrected q5 expectation; it is
not attributed to the retrieval refactor. See
`docs/evaluation/experiments/department-scoped-service-pr4-wiring.md`.

Two Chroma paths exist because `index.py` deletes the whole `CHROMA_DB_PATH` folder before
writing (not just the collection) — building `department_demo_v2` into `chroma_db_dept` would
wipe v1. Both stores are gitignored (`chroma_db*/`).

Launch the page: `DEPARTMENT_QA_MODE=v2 CORPUS_MANIFEST_PATH=corpus/manifest.yaml
SOURCE_DOCS_PATH=./source_docs_dept CHROMA_DB_PATH=./chroma_db_dept_v2 CHROMA_COLLECTION_NAME=department_demo_v2
.venv/bin/streamlit run app.py`. A mismatched env is shown on the page as an error.

`answered` = citations checked, not content verified: every cited id exists with the right
role, no blocking clarifying question is pending, both norm levels are present (except
`fact_only`), a cited profile carries a fill-in date, and the §2.3 lexical guard found no
normative wording in object facts. It does not mean the citation supports the claim or that
the answer is complete — see design-decisions §10. UI banner: "Ссылки сверены: законодательство
и ЛНА. Смысл ответа не проверен специалистом."

Object sheets use `typed_fields_v1` from
[2026-09-16-typed-object-sheet-design](../superpowers/specs/2026-09-16-typed-object-sheet-design.md):
nine strict fields in sections 2, 3, 4, 7 and 8. Prompt v4 renders them as a separate block
with the existing `obj_sN` evidence ids. `ModelAnswer` and `decide()` do not validate the field
semantics in this step. Six sheets are committed: `unit_office`, `unit_dispatch`, `unit_depot`,
`unit_partial` (pair run) and `unit_prod`, `unit_warehouse_v2` (trap set for cheap-model
selection, `eval/data/object_profile_traps_expectations.yaml`).

The 2026-09-16 verifier experiment used `openai/gpt-4o-mini` through OpenRouter. It gated
q1 from `answered` to `needs_review/verification_contradiction`, but failed to diagnose the
actual missing fields and left the unsafe draft text intact. Artifacts and decision:
`eval/runs/object_profile_verify_2026-09-16/summary.md`.

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
Source: `benchmarks/eval_v7_deepseek_2026-09-17.jsonl` (dataset 56, valid 53). Pipeline: the
final terminal triage contract, showcase default `deepseek/deepseek-v4.1-flash` (simple) /
`gpt-4o` (complex — moved to DeepSeek too on 18.09.2026, after this run), `gpt-4o` judge,
CrossEncoder reranker, cap 100. Cost recomputed from the run's own token usage against
`src/pricing.py::PRICE_PER_1M`, not the run's self-reported total — DeepSeek had no rate card
entry at run time, so every simple-path query originally priced at $0. Details:
[showcase-default-golden-set](../evaluation/experiments/showcase-default-golden-set.md).

| metric | value (17.09-7, showcase default) | 2026-09-11 terminal-contract baseline | pre-contract baseline (2026-09-08) |
|---|---|---|---|
| in-scope correctness | **7.91 / 10** | 7.47 | 7.40 |
| correctness mean | 7.98 / 10 | 7.26 | 7.09 |
| faithfulness | 0.926 | 0.891 | 0.808 |
| answer relevance | 0.887 | 0.879 | 0.881 |
| OOS rejection rate | 1.00 | 1.00 | 1.00 |
| false-sufficiency rate | **0.100** | 0.114 | 0.130 |
| complex-path rate | 0.245 | 0.170 | 0.132 |
| latency p50 / p95 | 24.0 / 71.3 s | 4.51 / 15.70 / 6.83 s | 4.8 / 14.7 / 5.9 s |
| cost / query | $0.00657 ($0.348 run total) | $0.00387 ($0.205 run total) | $0.0033 |

In-scope correctness clears the >7.5 target for the first time; false-sufficiency sits at
10.0%, still formally 0.1 pp outside the <10% target. The gains came with a ~5× latency
regression (p50 4.5 s → 24 s, unexplained) and a real cost about 70% above the previous
baseline, not below it — see the memo for why the run's own reported cost ($0.00430) was
wrong. The 2026-09-11 vs 2026-09-08 comparison below is unchanged: the terminal contract
improved faithfulness and false sufficiency without losing correctness or relevance beyond
judge variance, at about 18% more cost per query and 7% more p95 latency.

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
- bind: `127.0.0.1` (not public)
- deployed build: merge commit `18ad818` (PR #62, scoped Department service/UI cutover)
- deploy check 21.09.2026: Streamlit HTTP health OK; v2 collection loaded (1115 chunks),
  77 targeted tests passed; live q5 smoke returned expected
  `needs_context/applicability_unclear` (contract 1/1, 0 deterministic violations)
