# Unified eval — implementation plan

Status: architecture A approved by Petr on 2026-10-18; local implementation ready
for review. [The specification](../spec-unified-eval.html) is accepted. No paid calls.

## Implementation checkpoint (18.10)

- Shared runner, audited groundedness/correctness/citations, frozen allowlisted manifest,
  target and completed-result checkpoints, explicit denominators and separate costs are wired.
  Legacy CLI delegates; incompatible `--output` fails before execution. Historical helpers
  stay available to existing report tools.
- Rejudge defaults to the original dataset version, joins by UUID and records an assessment
  ID. Current references are explicit. Retrieval is not rescored across snapshots; reused
  pipeline cost is excluded from the new total. Source runs with repetitions are rejected
  until explicit per-repetition selection exists.
- All seven Grafana panels filter source/run_id, including feedback joins and NULL live
  rows. Defaults are ui/api/mcp. SQL predicate behavior is tested offline; live panels have
  not been verified and production was not restarted.
- Both real CLI dry-runs selected the same 82 cases: 66 regular, 9 false-premise, 7 OOS.
  All nine false-premise cases lack retrieval labels, explicitly reported as unscored
  coverage. No chunk IDs were invented. Offline contract fixtures test preservation of
  supplied labels. The worktree has no `.env`: its default snapshot is incompatible, and
  paid execution correctly requires explicit matching snapshot configuration.
- Original five failures and percentile units fixed. Existing parent-context effectiveness
  gate moved to the shared runner, retaining invalid-run diagnostics and partial artifacts.
- Full offline gate outside sandbox: 1799 passed, 3 xfailed, 5 deselected. Sandbox stalls
  even a minimal Starlette TestClient; the same health test passes outside it. Targeted
  eval/judge/dashboard/legacy checks: 272 passed. Final artifact overwrite regression passed; final eval/dashboard/parent checks: 178 passed.
  `check_docs --ci`, formatting, lint and `git diff --check` are part of the final gate.
- Embedding usage remains outside pipeline `llm_usage`; possible paid embedding cost is
  explicitly untracked/incomplete rather than silently included in a full-cost claim.
  No new dependencies or changes to production graph/PostgreSQL schema.

Next: manually align groundedness on saved real answers, review labels for the nine
false-premise cases, and only then request a paid baseline with a cost estimate. LangSmith
dataset updates, push, merge, publication and production restarts were not performed.

## Objective

Use the existing `eval/ls/` pipeline as the single normative evaluation path for generic
V7 Q&A. Preserve the course's fact-based correctness, citation judges, alignment and
pairwise tools. Add groundedness, reproducible local artifacts and shared telemetry IDs.
Separate false premises from out-of-scope requests and isolate eval traffic in Grafana.
The accepted version of the HTML specification controls scope.

## Working copy and dependencies

- Feature branch: `feat/unified-eval` in a separate worktree.
- Base: `0296bb0` (`eval/golden-expert-import`), which contains the current verified
  references and retrieval labels. This is a dependency on that branch's changes;
  rebase onto main after its PR is merged, without dropping the golden changes.
- The primary checkout has unrelated uncommitted files; leave them untouched.
- No new runtime dependencies planned. Reuse the project's existing virtual environment
  explicitly for offline checks. Never copy secrets or make a new `.env` implicitly.
- Existing authorization recorded in `2026-10-16-v8-stage1-petr-rewrite.md` permits
  assistant implementation with Petr's later rewrite for learning; retain that workflow.

## 1. Case policy and baseline contract

Start with checks that fail on today's code:

- Both entry points select only verified cases. The current fixture contains 82:
  66 regular, 9 false-premise and 7 out-of-scope. Assert behavior against fixture
  data, not permanent hard-coded production counts.
- A wrong false-premise answer containing “нет” is not credited as a correct refusal.
- A correct correction of a false premise is scored by its facts, even without a
  refusal keyword. It participates in correctness and applicable citation checks.
- A genuine OOS refusal is distinct from target failure and empty generation.
- Missing or snapshot-incompatible labels are explicit; they never silently become
  irrelevant examples or retrieval misses.

Likely paths: `eval/ls/dataset.py`, `eval/ls/evaluators.py`, `eval/ls/judge.py`,
`tests/eval_ls/test_dataset.py`, `tests/eval_ls/test_evaluators.py`,
`tests/eval_ls/test_judge.py`. Introduce a small shared case policy only if it prevents
duplicated eligibility rules. Keep stable case IDs and freeze selected reference values.
Do not change reviewed legal answers as part of refactoring their evaluator.

For the nine false-premise cases, inventory existing retrieval labels and add applicable
ones against the selected snapshot before the first complete retrieval baseline. Report
unlabelled cases and coverage explicitly until that work is finished.

## 2. Run result, artifact ownership and telemetry correlation

- Return `query_id` from the telemetry runner and expose the existing batch `run_id`
  in target outputs. Do not create a second event writer or invoke the graph twice.
- Preserve target failure details, terminal route and actual simple/complex work,
  raw accounted usage, counts, exact `final_context` and final retrieval IDs.
- Persist a manifest before execution with selected IDs, frozen references, effective
  models/settings, hashes of prompts and labels, dataset version and index identity.
  Serialize an allowlist of configuration values; exclude credentials and arbitrary env.
- Write one completed result checkpoint at a time. Include structured judge verdicts,
  diagnostic comments, coverage and errors. Keep partial results readable after interruption.
- Summarize target reliability separately from judged answer quality. For each metric
  report eligible/scored/pass/fail/unresolved counts and applicable rates. A target
  failure remains visible against assigned cases even when its judge score is absent.
- Track judge usage separately from pipeline usage; missing usage and unknown prices
  make costs incomplete. Do not represent pipeline-only cost as full experiment cost.

Likely paths: `eval/ls/target.py`, `eval/ls/run.py`, `eval/ls/evaluators.py`, plus a small
artifact/manifest module under `eval/ls/` and corresponding tests. Keep pure serialization
and aggregation separate from SDK calls. Clarify legacy `--output` semantics explicitly.

## 3. Groundedness evaluator

- Add an independent prompt and typed variables/response contract.
- Extract material answer claims and their support in the exact generator context;
  return evidence quotes and supported/unsupported/unverifiable statuses.
- Validate quoted text in code. Preserve reasonable qualified inference and the
  distinction between hallucination, missing evidence and no direct answer.
- Keep malformed/truncated judge output and provider failures out of answer-failure
  scores while recording them as unresolved evaluator errors.
- Use the existing model factory and capture judge usage. Bound output, timeout and
  retries consistently with the paid-run policy; no automatic paid retries.
- Reuse existing correctness and citation evaluators with the shared case policy.
  Preserve the agreed section/part and supplementary-fact rules.

Likely paths: a groundedness module under `eval/ls/`, `eval/prompts/judge_groundedness.md`,
`eval/ls/judge.py`, `eval/ls/run.py`, `tests/eval_ls/`.

Use mock models for deterministic contract tests. Reuse recorded real responses where
available; synthetic adversarial inputs only test contracts, not judge accuracy. A fresh
manual alignment batch is required before adopting groundedness as a quality gate.

## 4. Versioned rejudge and comparisons

- Default to the experiment's original frozen references and recorded outputs.
- Select updated references explicitly, joined by stable example ID rather than question
  text. Record the new reference/judge versions and retain the old assessments.
- Rejudge performs no RAG call and writes no extra `queries` row. Missing context in
  old runs yields an explicit unscored status for groundedness/citations.
- Keep align/pairwise tools and existing manual feedback. Comparison reports identify
  reference, dataset, judge and snapshot differences instead of silently pooling them.

Likely paths: `eval/ls/run.py`, `eval/ls/align.py`, `eval/ls/pairwise.py` only as necessary,
and tests of SDK adapters using fake clients. Consult the installed SDK and official docs
before changing method signatures or assuming dataset-version behavior.

## 5. Grafana filters

- Add source and run_id selectors to the provisioned dashboard.
- Default to ui/api/mcp. Allow eval explicitly. “All runs” must include live NULL run_id
  rows, rather than losing them through a regular-expression comparison.
- Apply both filters to every panel and to feedback joins with the correct aliases.
- Preserve time filtering, data source UID and p50/p95 calculations.
- Keep PostgreSQL schema and the journal fallback. Judge quality stays in LangSmith;
  judges are not additional user queries in Grafana.

Likely paths: `grafana/dashboards/regulatory-rag.json`,
`tests/test_grafana_dashboard.py`, monitoring how-to. Check interpolation semantics against
official Grafana/PostgreSQL documentation before writing SQL. Do not restart production.
Offline SQL checks do not replace checking panel results against a live stack later.

## 6. Single executor and legacy CLI

- Inventory imports and callers of `eval/run_v7_eval.py` before removing its helpers.
- Replace the independent old execution path with a compatibility entry into the
  common runner. Preserve mappable limit/skip-judge/output flags; fail clearly for
  incompatible legacy output contracts rather than silently changing their meaning.
- Preserve historical reports as historical, with their metric definitions. Do not
  convert scores out of ten to binary correctness or reuse the old false-sufficiency
  label with a new denominator.
- Keep narrow retrieval-only and Department/ObjectProfile scripts outside this migration.
- Update evaluation and telemetry recipes, metric definitions and documentation index.
  Register assistant-written implementation in the existing learning rewrite list.

## Validation and completion

1. Before each behavior change, run the relevant failing offline test and confirm its cause.
2. Run targeted eval/telemetry/dashboard tests, then format/lint changed Python files.
3. Run `pytest -m "not integration and not slow"`, `scripts/check_docs.py --ci` and
   `git diff --check`. Separate pre-existing failures from changes introduced here.
4. Inspect actual dry-run output, manifest, partial JSONL and summary produced by an
   offline fake run. Confirm no network/model/embedding call in dry-run.
5. Review snapshot compatibility, failure denominators, stable identity joins and
   secret-safe manifests. No new auth/payment or production RAG behavior is planned.
6. After separate approval, calibrate groundedness on manually reviewed saved responses,
   inspect false positives/negatives and score coverage. Then run the new verified baseline
   with a cost estimate and explicit paid-run authorization. Record it as a new series.
7. Live Grafana checks require a reachable monitoring stack; deployment/restarts remain
   separate. Report them as pending if unavailable, without claiming panel verification.

Local implementation readiness and semantic baseline readiness are separate milestones.
Push, publication, paid calls and production restart are not inferred from passing tests.
