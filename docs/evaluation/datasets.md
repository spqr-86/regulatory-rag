# Datasets

Data cards for every set used in evaluation: provenance, labels, intended use, and
limitations. Sizes are current as of 2026-09-17. Files named in **Where it lives** are
committed unless marked otherwise.

A recurring caveat across all retrieval sets: labels were chosen from the candidate pools of
the systems under test. Relative comparisons are valid; absolute scores are not, and a
retriever that never fed the pool is under-scored. See
[threats to validity](./README.md#threats-to-validity).

---

## 1. Generation golden set — `tests/dataset.csv`

**Purpose.** End-to-end answer quality: correctness, faithfulness, relevance, abstention.

**Provenance.** Grew from an early 50-question set. One question was removed 2026-09-04
(its norm, 524н, is not indexed), leaving 56.

**Size and shape.** 56 rows, columns `question`, `ground_truth`, `must_not_contain`,
`oos_type`. Composition: **43 in-scope**, **7 out-of-scope**, **6 false-premise**.

**Labels.** `ground_truth` is a reference answer written by hand; `must_not_contain` lists
forbidden phrases for false-premise questions; `oos_type` marks out-of-scope and
false-premise rows.

**Intended use.** Scored by `eval/run_v7_eval.py` with the `gpt-4o` judge. The in-scope
subset drives the headline correctness number; the OOS subset drives the abstain rate.

**Limitations.** Small (43 in-scope). Reference answers are judge-dependent in practice:
absolute scores are only comparable under one judge, and judge variance is ~±0.25 on the
correctness scale. Not a held-out set — it has been used to guide prompt iterations.

---

## 2. Retrieval headline — 90 practitioner questions

**File.** `eval/data/golden_retrieval_labeled_ext.jsonl`

**Purpose.** Honest retrieval headline: how often the right chunk is in top-k on real user
wording.

**Provenance.** 130 questions taken verbatim from OT/PB practitioner forums with the prompt
in `docs/research/prompt-heldout-questions.md`; raw set in
`eval/data/dataset_heldout_ext_2026-09-06.csv`. Labeled with the same three-pass pipeline as
the synthetic set (strict + lenient passes with a mandatory verbatim citation, then an
arbiter for disagreements). ~40 questions with no confirmed corpus answer were dropped.
**Built after the retrieval config was frozen** — not used for any tuning.

**Size and shape.** 90 questions; labels are `relevant_chunk_ids` (one or more per
question). ~19% have no relevant chunk in top-12 by any backbone.

**Intended use.** Headline retrieval table (HR@5, HR@12, MRR, latency), per backbone.

**Limitations.** Labels are model-assisted (human-check agreement ~80% on a control
sample). Pooling bias: candidates came from the systems being measured. Weakest domain:
29н medical exams (HR@5 0.56).

**Review artifact.** `eval/data/golden_retrieval_labeled_ext.review.tsv` (control sample and
disputed rows).

---

## 3. Retrieval development — 133 questions

**File.** `eval/data/golden_retrieval_labeled.jsonl`

**Purpose.** Development set for comparative measurements (tuning, escalation, backbone
ranking).

**Provenance.** 90 practitioner questions (card 2) plus 43 chunk-derived questions built
during tuning.

**Size and shape.** 133 questions. Scores higher than the 90-question set (hybrid HR@12
0.827 vs 0.811) because the 43 synthetic questions repeat the source wording and are easier
for lexical search.

**Intended use.** Any *relative* comparison of retrievers or routing configs. Explicitly a
development set, **not** an untouched final test — this is the main criticism of the
external review and is accepted.

**Limitations.** Same pooling bias as card 2, amplified: complex reached HR@12 1.000 on the
earlier 43-question held-out set, which measures the pooling construction more than retrieval
quality.

---

## 4. Synthetic retrieval set — 262 questions

**File.** `eval/data/retrieval_gt_reviewed.jsonl`

**Purpose.** Large development set for retriever tuning, where round-trip construction
(A→Q\*) gives a clean per-chunk label.

**Provenance.** `eval/generate_retrieval_gt.py` generates 3 questions per chunk for each
chunk in the collection (full raw set ~13,267 questions, `eval/data/retrieval_gt.jsonl`,
gitignored). A stratified sample of 500 was reviewed by `eval/review_retrieval_gt.py`:
strict pass (`gpt-4o-mini`) kept 282 of 500, an arbiter (`gpt-4o`) returned 184 and confirmed
34; a 30-row manual check showed both passes err (strict rejects ~⅓ valid questions; the
arbiter accepts fragments as answers in ~84% of its returns). The 262 questions where both
passes and the manual sample agree are kept; disagreements are recorded in
`retrieval_gt_sample.review.tsv` and the decision is reversible.

**Size and shape.** 262 questions covering all 8 indexed documents represented, 262 distinct
chunks. Full derivation: [roadmap](../roadmap.md) step 1.

**Intended use.** Development only — never the headline. Used for the `RRF_K` sweep and the
threshold grid.

**Limitations.** Synthetic questions repeat the source chunk's wording, so BM25 wins by
construction (BM25 HR@12 0.935 vs vector 0.813 on this set, the reverse of the practitioner
set). It measures lexical overlap, not retrieval quality, and is unfit as a tuning target
for the retriever backbone. It also exposes chunking defects (a chunk carrying another
section's heading, truncated clauses, table rows without headers).

**Related.** Generator review artifacts:
`eval/data/retrieval_gt_sample.jsonl`, `retrieval_gt_sample.review.tsv`,
`retrieval_gt_smoke60.jsonl`.

---

## 5. Routing calibration — 43 reviewed questions

**File.** `eval/data/triage_calibration_dev_annotations.json`

**Purpose.** Audit escalate/answer decisions **without an LLM judge** (issue #9).

**Provenance.** 43 in-scope questions from the golden set, annotated by hand for evidence
coverage and critical misses; used with an immutable retrieval snapshot.

**Size and shape.** 43 questions with per-question evidence annotations.

**Intended use.** `eval/triage_calibration.py` compares threshold profiles against the
current defaults on risky outcomes (unsafe generations). The 81-profile run retained the
defaults — a measured negative result.

**Limitations.** Development set; annotations reflect one reviewer's view of what evidence
is sufficient. No generation quality is measured here.

---

## 6. Department Q&A pair set — 9 questions

**File.** `eval/data/object_profile_pair_expectations.yaml`

**Purpose.** Compare the two department-Q&A modes (`v1`: object sheet inside the index;
`v2`: sheet passed whole as a profile) on the same questions, with expectations **committed
before** the paid run.

**Provenance.** Four synthetic unit sheets (`unit_office`, `unit_dispatch`, `unit_depot`,
`unit_partial`) written for issue #44, in `source_docs_dept/` (gitignored). The sheet schema
is `typed_fields_v1` (nine strict fields in sections 2, 3, 4, 7, 8).

**Size and shape.** 9 questions; each expectation lists accepted status + reason sets,
required sub-answers, and forbidden conclusions. Scoring is manual against
`required_subanswers`; `eval/score_object_profile_run.py` additionally reports deterministic
violations (applied conclusion on an `unknown` field, exact-substring forbidden phrases).

**Intended use.** Paired v1/v2 comparison and prompt/typed-sheet iterations. Runs are
committed in full under `eval/runs/object_profile_pair_*`, including the rendered prompt and
raw model outputs.

**Limitations.** N=9, one run each, `gpt-4o-mini` at temperature 0 — nine observations, not a
quality measurement and not a stable v1-vs-v2 difference. Contract status does not separate
correct from wrong content (`contract_ok` was 9/9 while the headline answer was wrong), so
semantic scoring stays manual.

---

## 7. Department Q&A trap set — 4 questions

**File.** `eval/data/object_profile_traps_expectations.yaml`

**Purpose.** Select a cheap model that does **not** apply a norm's threshold to the wrong
magnitude or over-generalize a rule. Written before the paid comparison run.

**Provenance.** Two synthetic sheets (`unit_prod`, `unit_warehouse_v2`) added to
`corpus/manifest.yaml`; norm wording verified 2026-09-17 against the extracted
`ext_ppr_1479.pdf` (п. 401 area ≤ 100 m², п. 406 distance ≤ 30 m for В1–В4, п. 17 «б»
5-year interval, appendix № 1).

**Size and shape.** 4 questions; per question: required sub-answers, accepted alternatives,
forbidden conclusions, `typed_fields`, and the norm's `applies_to` / operator.

**Intended use.** Cheap-model leaderboard; each candidate runs the same 4 questions in mode
`v2`. Forbidden conclusions match by exact substring (casefolded) — a rough signal; semantic
scoring by hand is primary.

**Limitations.** 4 questions, one run per model. A pass is not proof of stability;
`gpt-4o-mini` and `claude-haiku-4.5` fail on 2/4 here while passing other question types, and
`deepseek-v4.1-flash` has been run only once. The set is separate from the 9-question pair
baseline and does not change its `expectations_sha256`.

---

## Not datasets

- `docs/research/prompt-heldout-questions.md` — the prompt used to source practitioner
  questions (input to card 2).
- `eval/data/questions_2026-09-06_*.csv` — raw question batches by domain.
- `eval/data/pools_ext/`, `eval/data/arbitration_ext/`, `eval/data/verdicts_ext/` — candidate
  pools, arbiter jobs and verdicts behind card 2 (pools gitignored, verdicts committed).
- `benchmarks/*.json(l)` — run outputs, gitignored. Aggregates are copied into
  [FACTS](../reference/FACTS.md) and [benchmarks/README.md](../../benchmarks/README.md).