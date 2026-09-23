# Unit-scoped Q&A: PR4 wiring gate (old service vs scoped service, workers=1 vs parallel)

## Question

PR4 of `docs/superpowers/plans/2026-09-19-shared-scoped-retrieval.md` (issue #59) switches
`app.py` from the legacy `answer_question`/`make_hybrid_search_fn` call to
`answer_scoped_question` over the bound-runtime scoped service built in #58. PR4 itself does
not touch retrieval logic — it is wiring. The plan's steps 5–6 gate cutover/deploy on: (5)
comparing old pipeline vs new sequential (`workers=1`) vs new parallel on identical
data/config, and (6) a written report with baseline/candidate, hashes, settings, and a
breakdown of regressed questions, before the path is declared accepted.

## Decision on scope (Petr, 2026-09-20)

No new paid run for PR4 specifically. PR4 does not change the retrieval algorithm, only which
caller invokes it (`app.py` now calls the same scoped service #58 already validated). The
three-way comparison plan step 5 asks for was already run against #58 on 2026-09-20, on the
same 9-question set, same collection, same model. This memo reuses those artifacts as PR4's
step-6 report rather than re-running the paid eval a second time for an unchanged algorithm.

## Setup (all three runs)

- **Dataset:** 9-question pair set, `eval/data/object_profile_pair_expectations.yaml`
  (`expectations_sha256` `86cf9506658597c31ce13be098d12837c789b2fe51de785db0ae6bc83d56d07b`
  after the q5 expectation fix — see below).
- **Model:** `deepseek/deepseek-v4.1-flash` via OpenRouter, temperature 0, verifier disabled
  (`--no-verify`).
- **Collection:** `chroma_db_dept_v2` / `department_demo_v2`, 1115 chunks.
- **Scoring:** `eval/score_object_profile_run.py` (deterministic contract check).

| Run | Code / commit | Config | Prompt | Artifact |
|---|---|---|---|---|
| baseline (old service) | worktree `../regulatory-rag-baseline-2cb34ee` on `2cb34ee` (pre-#58) | `make_hybrid_search_fn`, `--model` override | v4 | `eval/runs/baseline_2cb34ee_vs_58_2026-09-20/v2/` |
| new, sequential | `57fcaf9` (scoped service) | `DEPARTMENT_QA_MODE=v2 DEPARTMENT_RETRIEVAL_WORKERS=1` | v5 | `eval/runs/department_service_58_workers1_live_2026-09-20/v2/` |
| new, parallel | `57fcaf9` (scoped service) | `DEPARTMENT_QA_MODE=v2 DEPARTMENT_RETRIEVAL_WORKERS=4` (production default) | v5 | `eval/runs/department_service_58_workers4_live_2026-09-20/v2/` (+ `..._workers4_q9_live_2026-09-20/` for q9, run separately) |

PR4 itself (`34cbc84`..`514c97d`) is not a fourth leg — it only changes which code path
`app.py` calls; the underlying scoped-service behaviour is the `workers=4` row above, since
`DEPARTMENT_RETRIEVAL_WORKERS` defaults to 4 in `config/settings.py`.

## Results — per-question status

| q | baseline (old, v4) | new workers=1 (v5) | new workers=4 (v5) |
|---|---|---|---|
| q1 | needs_context | needs_context | needs_context |
| q2 | needs_context | needs_context | needs_context |
| q3 | needs_context | **failed** (transient `generation_failed`) | needs_context |
| q4 | **failed/citation_invalid** | needs_context | needs_context |
| q5 | needs_context/applicability_unclear | needs_context/applicability_unclear | needs_context/applicability_unclear |
| q6 | out_of_scope | out_of_scope | out_of_scope |
| q7 | **needs_review/object_profile_undated** | needs_context | needs_context |
| q8 | needs_context | needs_context | needs_context |
| q9 | not measured in this table (see note) | needs_context | needs_context |

Contract: baseline 8/9, new workers=1 8/9 (q3 transient), new workers=4 9/9, 0
`applied_on_unknown`, 0 forbidden-conclusion violations in every run.

## Regressions and their cause

- **q4, q7 (baseline vs new):** baseline's `evidence_ids` matched the new service's
  byte-for-byte on 8 of 9 questions, q4/q7 included. Same evidence, different status — the
  divergence is the prompt v4→v5 change (bundled in the same commit `57fcaf9` as the #58
  scoped-service), not the retrieval refactor.
- **q6 (baseline vs new):** the only real retrieval difference — the new service did not pull
  `ext_*` evidence that baseline did. Did not change the outcome (`out_of_scope` in both).
- **q3 (workers=1 vs workers=4):** one workers=1 run hit a transient `generation_failed` on
  the model call (`search_calls: 0`, i.e. it never reached retrieval); the identical
  prompt/evidence succeeded under workers=4. Not attributed to concurrency — the deterministic
  scorer and prompt/evidence-hash comparison (2026-09-20, entry 6) found no context or
  citation difference between the two worker settings on any question; the retrieval
  parallelism itself is provably a no-op on what the model sees.
- **q5 (all three):** deliberately changed expectation, unrelated to this comparison — see
  [department-qa-object-profile](./department-qa-object-profile.md) and `HISTORY.md`
  2026-09-20 (7): the lister never confirms the guard post is staffed 24/7, so `needs_context`
  is the correct status, not a regression.

Note on q9: not part of the baseline worktree's 9-question run breakdown recorded in
`HISTORY.md` 2026-09-20 (10) — that entry reports contract 8/9 without a full per-question
table; q4 and q7 are the two named failures, so q9 is implicitly a baseline pass. Not
independently re-verified for this memo; flagged as "not measured" above rather than assumed.

## Performance

Non-LLM (retrieval) time dropped ~6.5% under `workers=4` vs `workers=1` on the seven
comparable questions (excluding the transient q3 failure and the separately-run cold q9).
End-to-end latency is dominated by OpenRouter generation (11–171 s observed), so the
retrieval-side parallelism gain is not user-visible. Source: `HISTORY.md` 2026-09-20 (6).

## Conclusion

Scoped retrieval (#58) changes almost nothing about what the model sees on this 9-question
set; the 8/9→9/9 improvement is the prompt v5 fix plus the q5 expectation correction, not the
retrieval refactor. `workers=1` vs `workers=4` produce identical prompt/evidence on every
question (deterministic scorer, entry 6) — parallelism is a latency optimization with no
correctness effect. PR4 only rewires `app.py` onto this already-validated service and adds no
new retrieval-affecting code, so it inherits this evidence rather than requiring a fresh paid
run.

**Cutover #58→#59 (PR4) is accepted on the evidence above. Deploy is not gated further by
this comparison.**

## Open / not measured

- No dedicated live-eval run exists with PR4's actual code (`34cbc84`..`514c97d`) executing
  end to end through `app.py`'s new call path — the wiring itself (arg plumbing, telemetry,
  v1-mode guard) is covered by unit tests (`tests/department_qa`, full suite green) and the
  opus whole-branch review (APPROVE, 2 Important fixed in `514c97d`), not by a paid run. If a
  wiring-only bug slipped past both, it would not show up in this memo.
- q9's baseline (old-service) status is not independently confirmed — see note above.
