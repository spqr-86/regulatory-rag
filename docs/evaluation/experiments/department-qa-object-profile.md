# Department Q&A: object profile, typed sheet, verifier, strong model

## Question

How should unit-specific facts (headcount, fire-protection systems, duty staff) reach the
prompt — and how do we stop a norm's threshold from being applied to a fact of the wrong
quantity? This is the `q1` failure that survived every prompt-side fix.

## Setup (base)

- **Mode:** `v2` (object sheet excluded from the index and passed whole as a profile).
- **Model:** `gpt-4o-mini`, temperature 0, one answer call per question; a verifier call only
  for answers the contract marked `answered` with applied conclusions.
- **Dataset:** 9 pre-registered questions (`eval/data/object_profile_pair_expectations.yaml`),
  four synthetic unit sheets.
- **Scoring:** manual against `required_subanswers`; `eval/score_object_profile_run.py` adds
  deterministic checks. A required sub-answer counts only if no forbidden conclusion is
  present.
- **Artifacts:** every run is committed under `eval/runs/object_profile_*`, with the rendered
  prompt and raw model outputs.

## Results — iteration chain

| Step (mode / prompt) | Expected status | Sub-answers | Forbidden | Note |
|---|---:|---:|---:|---|
| 2026-09-15 v1 (sheet in index) | 7/9 | 7/17 | 2 | answers without a unit; substitutes a general norm for a missing fact |
| 2026-09-15 v2 (profile, prompt v2) | 8/9 | 11/17 | 1 | `q1` threshold error |
| 2026-09-16 v2 (prompt v3, rerun) | 9/9 | 11/17 | 1 | `q1` repeats verbatim with the new rule present in the prompt |
| 2026-09-16 v2 (typed sheet, prompt v4) | 8/9 | 9/17 | 1 | `q1` fails a third time despite explicit `unknown` fields |
| 2026-09-16 verifier (separate call) | — | 9/17 | 1 | `q1` gated to `needs_review/verification_contradiction`, wrong reason, unsafe text intact |
| 2026-09-16 Sonnet 5, `q1`+`q2` | — | 4/4, 2/4 | 0 | `q1` fixed; `q2` loses to retrieval, not the model |

The failing `q1` conclusion: an evacuation-plan requirement drawn from 8 people in an office
zone, against a norm threshold of 50 in the building or 10 permanent workplaces on the floor
— a different quantity and a different part of the object, where the sheet explicitly marks
the building and floor figures `unknown`.

## Interpretation

- **`q1` is a model-side error, not retrieval or prompt.** Both thresholds were in the
  prompt from 2026-09-15; a rule telling the model to match the quantity, a typed sheet with
  explicit `unknown` fields, and a post-generation verifier each failed to change the
  behaviour. A stronger model (`claude-sonnet-5`) got `q1` right without those aids.
- **The verifier is a partial safety gate, not a diagnosis.** It moved `q1` out of
  `answered`, but classified it as a contradiction rather than missing facts and left the
  unsafe draft text in place. On the trap set it produced false positives on correct answers.
- **`q2` loss is retrieval.** п. 405 and appendix № 1's category table are split across
  chunks; the model picked a different row's rank. Parked.

## Decision (Petr, 2026-09-16)

1. `v2` stays the default mode.
2. **No norm-to-field markup or norm cards.** A map would not cover thresholds across the
   corpus.
3. Keep the deterministic `applied_on_unknown_field` gate (an applied conclusion citing a
   field whose state is `unknown`).
4. Select the model on the trap set instead of patching prompts — see
   [cheap-model-selection](./cheap-model-selection.md).
5. Accept `q1` as a documented known error in the `answered` guarantee; revisit a map only
   if a model consistently misreads one specific norm.

## Threats to validity

N=9, one run per iteration, manual semantic scoring. Contract status does not separate
correct from wrong content — it was 9/9 in the typed-sheet run whose headline answer was
wrong. The verifier's behaviour on 9 (and 4 trap) questions is itself a small sample.

## Next action

Rerun the trap set for stability and add more threshold traps (area, distance, period).
Retrieval of appendix № 1 is deferred; the applicability-agent idea is parked in
[roadmap](../../roadmap.md).