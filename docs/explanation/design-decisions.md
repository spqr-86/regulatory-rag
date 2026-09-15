# Design decisions

The thesis of this system: **a hallucinated answer to a regulatory-compliance question is
a liability, not a UX bug.** Every decision below trades raw recall or fluency for
*reliability* — the system should be reproducible, should refuse when unsure, and should be
debuggable after the fact.

Each record is: **Context → Options → Choice → Why → Evidence.** Numbers are judge- and
dataset-dependent; their source benchmark is named. Current production values:
[FACTS](../reference/FACTS.md).

> **Historical evidence caveat.** Every benchmark cited below that predates
> `2026-09-02` (`eval_v7_2026-05-30_chunkid.jsonl`, `cps_2026-05-22.json`, the cap sweep)
> was scored by `gpt-4o-mini`, not `gpt-4o` — `llm_factory` silently ignored
> `JUDGE_MODEL_NAME` / `COMPLEX_MODEL_NAME` on the OpenAI branch until a fix on that date
> (see §8 and [FACTS](../reference/FACTS.md#metrics)). The *directional* conclusions
> (deterministic gates, abstain-first, CrossEncoder over FlashRank, namespaced `chunk_id`,
> dropping the verifier subgraph, the candidate cap) held up on re-measurement; treat the
> absolute figures in §2–§9 as the mini-judge baseline, and FACTS as canonical.

---

## 1. Deterministic hard gates, not LLM routing

**Context.** After retrieval the pipeline must decide: answer now, search harder, or
abstain.

**Options.** (a) An LLM reads the passages and decides. (b) Numeric thresholds on retrieval
signals decide.

**Choice.** Deterministic thresholds (`src/v7/hard_gates.py`). No LLM in any routing edge.

**Why.** Reproducibility and debuggability. The same query + index always takes the same
path. When a user reports an unexpected abstain, the log shows `top_score=0.47 < 0.50` — a
concrete, traceable cause. LLM routing gives non-deterministic decisions and no such trace,
plus an extra call of latency.

**Evidence.** Qualitative but load-bearing: every triage/abstain in the eval reports is
explainable from logged scores. See [triage.md](./triage.md).

---

## 2. Abstain over hallucinate, with a pre-retrieval domain gate

**Context.** Out-of-scope and false-premise questions are where RAG systems invent answers.

**Options.** (a) Always answer with best-effort retrieval. (b) Refuse when retrieval
confidence is low, and filter clearly out-of-domain queries before retrieving at all.

**Choice.** Explicit `abstain` node on gate failure, plus a **domain gate** (cosine of the
query embedding to the corpus centroid) inside `intent_gate` that drops OOS queries before
retrieval.

**Why.** For compliance, a wrong confident answer is worse than "I don't have this." The
domain gate also saves a retrieval+generation round-trip on junk.

**Evidence.** OOS rejection rate **1.00** (`benchmarks/eval_v7_2026-05-30_chunkid.jsonl`).
Paired with decision §7 (generation-side reliability).

---

## 3. CrossEncoder reranker, not FlashRank

**Context.** Hybrid retrieval (vector + BM25) needs reranking before the gate reads a top
score.

**Options.** (a) FlashRank cross-encoder (fast, convenient). (b) A sentence-transformers <!--freshness:ignore-->
CrossEncoder with a sigmoid wrapper.

**Choice.** CrossEncoder (`RERANKER_BACKEND=crossencoder`), and — critically — the hard
gate reads the **vector** cosine score, never the reranker score.

**Why.** FlashRank's raw scores clustered at ~0.999 for every relevant-ish passage: no <!--freshness:ignore-->
discrimination, so ranking was effectively lost and the exact answer chunk sank down the
list. Worse, when the inflated score was used for the gate, every query looked
"sufficient" and took the fast path regardless of real quality. The CrossEncoder produces
discriminating scores; reading the calibrated vector score for the gate removes the
inflation entirely.

**Evidence.** On the reranker change alone, false-sufficiency dropped **22.9% → 15%**
(`docs/archive/plans/2026-05-29-eval-failures-analysis.md`); the exact chunk moved from
~position 15 to position 0 on the diagnostic query. The full retrieval+prompt chain later
brought false-sufficiency near zero (current: 9.8%, FACTS). This was the decision that
broke a months-long correctness plateau — see §4.

---

## 4. `chunk_id` namespaced by source (a retrieval-correctness bug)

**Context.** RRF fusion and cross-reference dedup both key on a passage identity.

**Options.** (a) Identify passages by list position. (b) Assign a stable per-source
`chunk_id` and identify by `source#chunk_id`.

**Choice.** Per-source 0-based `chunk_id` assigned after dedup; identity = `source#chunk_id`
with a content-hash fallback.

**Why.** Code review found `chunk_id` was *read* in six places but **never written** at
index time. RRF was fusing by list position and `merge_all_passages` dedup was a no-op —
duplicate passages reached the LLM context, crowding out coverage. Namespacing by source is
required because a bare `chunk_id=0` collides across documents.

**Evidence.** In-scope correctness **7.08 → 7.44** after the fix
(`benchmarks/eval_v7_2026-05-30_chunkid.jsonl` vs the prior re-judge). The "plateau" was
partly a structural bug, not an algorithmic ceiling — found by reading code, not by tuning.

---

## 5. Removed the `llm_verifier` / `rewriter` subgraph <!--freshness:ignore-->

**Context.** An earlier design sent "borderline" triage results through an LLM verifier that <!--freshness:ignore-->
could answer, rewrite the query, or escalate.

**Options.** (a) Keep the verifier/rewriter middle path. (b) Collapse to: sufficient →
generate, otherwise → `rag_complex`.

**Choice.** Removed the subgraph (session 61). Anything not `sufficient` escalates straight
to deep retrieval.

**Why.** The middle path added an LLM call, two nodes, and several state fields without a
measurable correctness gain over simply escalating — complexity that wasn't paying for
itself. Deleting it simplified the graph and the state, and removed a latency hop. Knowing
when to delete a feature is a design decision.

**Evidence.** Graph and state shrank (verifier/rewriter nodes, routes, factories, and
`VerificationResult`/`VerifierVerdict`/`MAX_VERIFY_ITERATIONS` fields all removed); the unit
suite stayed green at 230 tests. Pipeline latency is currently 9.71s
(`benchmarks/eval_v7_2026-05-30_chunkid.jsonl`).

---

## 6. Two-tier LLM (cheap simple path, strong complex path)

**Context.** Most queries are answerable from the fast path; a minority need deep retrieval.

**Options.** (a) One model for everything. (b) A cheap model on the simple path, a stronger
model only on the complex path.

**Choice.** `gpt-4o-mini` on the simple path, `gpt-4o` on the complex path
([FACTS](../reference/FACTS.md#models)), wired through the LLM factory so either is swappable
per path via `.env`.

**Why.** The fast path handles the majority of queries at a fraction of the cost; the
expensive model is reserved for the harder minority that actually routes to `rag_complex`.

**Evidence.** ~$0.0102 per query (`benchmarks/cps_2026-05-22.json`, N=10), complex-path rate
0.241 (latest eval) — i.e. ~76% of queries never touch the expensive model.

---

## 7. Prompt-as-code (versioned templates, `generate_answer` v8)

**Context.** Generation can still be wrong even with perfect retrieval — sycophancy
(agreeing with a false premise) and value↔condition confusion (reporting the right number
under the wrong condition) are the dangerous failure modes for compliance.

**Options.** (a) Inline prompt strings, edited ad hoc. (b) Versioned Jinja2 templates in a
registry, switchable per env, iterated like code with A/B traces.

**Choice.** Prompt-as-code: `prompts/` + `registry.yaml`, active `generate_answer` = v8
(anti-sycophancy block + value↔condition binding). Pairs with §2 — abstain guards
retrieval, the prompt guards generation.

**Why.** Treating prompts as versioned artifacts makes regressions attributable: each
change is a diff with its own eval/trace, and a bad version is one env var to roll back.

**Evidence.** v8 fixed inverted-norm answers (e.g. "обязательна" vs "не вправе") on the
targeted false-premise traces that earlier versions got backwards; faithfulness 0.859,
in-scope 7.44 (FACTS). Prompt iteration history: `docs/archive/plans/`.

---

## 8. Eval methodology: measure judge noise before chasing deltas

**Context.** LLM-as-judge scores are themselves noisy; it is easy to "improve" a metric that
only moved within noise.

**Options.** (a) Trust aggregate score deltas run-to-run. (b) Quantify judge variance first,
then trust only changes larger than it, and read per-question traces for targeted fixes.

**Choice.** Measured judge run-to-run variance (~±0.25 on the correctness scale) and treat
sub-threshold aggregate moves as noise; for targeted fixes, trust per-question traces over
the aggregate. Added `scripts/rejudge.py` to re-score saved answers cheaply for A/B.

**Why.** Several "improvements" and "regressions" turned out to be judge noise or an
over-narrow ground truth, not real changes. Without a noise floor, tuning chases ghosts.

**Evidence.** Re-judge runs showed flat aggregates (e.g. 7.10 → 7.08) hiding real
per-question fixes; the upgrade to a stronger judge (`gpt-4o`) lowered absolute faithfulness
(0.988 old judge → 0.859) while being better calibrated — which is exactly why absolute
numbers are only compared under the same judge.

---

## 9. CrossEncoder candidate cap: more retrieval ≠ better grounding

**Context.** `rag_complex` fetches up to 60 passages by vector similarity, then calls a
CrossEncoder to rerank before the hard gate reads the top score. Section fetches (e.g.
a full ТК РФ chapter) can balloon the candidate pool to 400+ items.

**Options.** (a) No cap — pass all candidates to the CrossEncoder. (b) Pre-sort by vector
score and cap at some k before the CrossEncoder.

**Choice.** `RERANK_CANDIDATE_CAP=100` in `src/v7/config.py`, applied in `rag_complex`
before the CrossEncoder call.

**Why.** Two reasons pulled in the same direction:

1. **Latency.** CrossEncoder is O(n) on the candidate count. On an uncapped pool of 477
   candidates the reranker alone took 31s (p90). Capping at 100 brings predict time to ~4s
   while keeping the top candidates the vector ranker already identified.

2. **Grounding quality.** Counterintuitively, faithfulness *improved* as the cap shrank:
   uncapped (477): faithfulness 0.898 → cap=100: **0.936**. The tail of the 377 most
   marginal candidates was not informative — it added noise to the LLM context and produced
   weakly-grounded statements. Fewer, better-ranked passages = more reliable answers.

**Evidence.** Cap sweep, same judge (`gpt-4o`, seed=12345), same dataset:

| cap | correctness | faithfulness | mean latency | p90 |
|-----|-------------|-------------|-------------|-----|
| 477 | 7.76 | 0.898 | 16.3s | 14.9s |
| 50  | 7.52 | 0.876 | 7.2s  | 10.4s |
| **100** | **7.72** | **0.936** | **9.5s** | 17.7s |

cap=50 overshoots: it misroutes borderline queries from `rag_simple` to `rag_complex`
(correctness −0.24, routing validated per-question). cap=100 restores quality within noise
of the uncapped baseline while cutting mean latency 42%.

**Known limitation.** p90=17.7s is driven by 2 complex queries with deep cross-ref
expansion — not a regression relative to uncapped, but not improved either. The ratchet
for those is generation streaming (masks tail in UI); not worth engineering until an
interview shows it matters.

---

## 10. Unit object sheet as a structured profile, not retrieved chunks

**Context.** Department Q&A indexed each unit's "object features sheet" (headcount, fire
protection systems, duty staff) as ordinary internal chunks. A 6-question smoke run on
2026-09-15 showed facts about the unit crowded out of top-8 by company-wide documents, a
unit fact being ignored even when retrieved, and a unit-dependent question answered "yes"
with no unit selected.

**Options.** (a) Keep retrieval, boost the unit's chunks. (b) Enrich the query with unit
fields. (c) Parse the sheet into a profile and pass it whole to the prompt as a third
evidence level.

**Choice.** (c). The sheet is excluded from the index (`role: object_profile` in the
manifest), parsed into template sections `obj_s1…obj_s9` with `presence`
(`present|empty|missing`) and `as_of_date`. The answer contract gains `object_facts`
(object ids only) and `applied_conclusions` (object fact + norm) inside the existing
`ModelAnswer`; no second contract.

**Why.** A unit fact is not a norm to be ranked: it is needed in full in every answer for
that unit. Section-level granularity keeps the parser dumb (it cannot judge content, so no
known/unknown fields); typed fields are deferred until eval shows a systematic fact-application
error. Citation roles are checked deterministically: an applied conclusion without a norm goes
to review, a missing section cited as fact fails. No staleness threshold in code — the fill-in
date is shown, and a cited profile without a date goes to review.

**Guarantee boundary.** `answered` means "citations checked": every cited id exists with the
right role, no clarifying question is pending, both norm levels are present (except
`fact_only`), a cited profile carries a fill-in date, and a lexical guard found no normative
wording in object facts. The guard is a closed list of word stems (spec §2.3) and misses
forms such as «недостаточно», «требуются», «нужно»; the list grows only by a spec change. It does not mean the cited text supports the
claim or that every part of the question is answered — an external review (2026-09-15) built
`answered` outputs with irrelevant but well-formed citations. Support and completeness are
measured in eval (required sub-answers, forbidden conclusions), not enforced at runtime; the UI
says so. Revisit with a groundedness check if eval shows `answered` hiding wrong conclusions.

**Evidence.** Paired v1/v2 run, 2026-09-15 (`gpt-4o-mini`, temperature 0, 9 questions × 2
modes, expectations committed before the run). v1 matched 7/9 expected status+reasons
(7/17 required sub-answers credited, 2 forbidden conclusions found: `q4` answered "да" with
no unit selected, and `q8` named a specific duty officer the sheet does not evidence for that
unit). v2 matched 8/9 (11/17 sub-answers credited, 1 forbidden conclusion found in `q1`, where
`applied_conclusions` drew a plan-required conclusion from an 8-person headcount that meets
neither the building nor floor threshold). Full per-question breakdown and quotes:
[`eval/runs/object_profile_pair_2026-09-15/summary.md`](../../eval/runs/object_profile_pair_2026-09-15/summary.md).
Spec: [2026-09-15-object-profile-design](../superpowers/specs/2026-09-15-object-profile-design.md).

**Default mode.** `v2` (decided by Petr, 2026-09-15). v2 beat v1 on all three numbers and
produced no answered-without-unit and no norm substituted for a missing fact. **Known error,
accepted:** `q1` — `applied_conclusions` concluded an evacuation plan is required from an
8-person headcount that meets neither the building nor the floor threshold; v2 also credits
only 11/17 required sub-answers. Next task: fix threshold application and rerun the pair.
Revisit (back to `v1`) if the rerun shows more forbidden conclusions in v2 than in v1.

---

## Not separate decisions

- **Pluggable backends** (LLM factory + `VectorStoreBackend` protocol) — an architecture
  property, covered in [architecture.md](./architecture.md#pluggable-backends), not a
  contested trade-off.
- **HybridChunker** (structure-aware chunking by article/clause) — covered in
  [reference/data-pipeline.md](../reference/data-pipeline.md).
