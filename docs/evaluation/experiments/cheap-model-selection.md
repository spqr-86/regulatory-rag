# Cheap model selection on the adversarial trap set

## Question

Which cheap answering model applies a norm's threshold to the **right** quantity, instead of
over-generalizing a rule or attaching a threshold to a different magnitude? Prompt-side fixes
had failed on the same error ([department-qa-object-profile](./department-qa-object-profile.md)),
so the lever moved to model choice.

## Setup

- **What changed:** the answering model, via OpenRouter (`--model`).
- **What stayed fixed:** mode `v2`, prompt `department_answer` v4, typed sheet, retrieval,
  temperature 0, base `department_demo_v2` (1115 chunks).
- **Dataset:** 4 pre-registered traps (`eval/data/object_profile_traps_expectations.yaml`) on
  п. 401 (category Д, area ≤ 100 m²), п. 406 (distance ≤ 30 m for В1–В4) and п. 17 «б»
  (5-year interval). Each question has required sub-answers and forbidden conclusions written
  before the run.
- **Runs:** one run per model, 4 questions each (plus a verify call where the contract
  required it).
- **Scoring:** semantics by hand against `required_subanswers` (primary);
  `forbidden_conclusion` is exact-substring matching and only auxiliary.
- **Artifacts:** `eval/runs/object_profile_traps_2026-09-17_*/v2/`, summary in
  `eval/runs/object_profile_traps_2026-09-17/summary.md`. Total cost ~$0.16.

## Results

| Model | q1 Д 80 m² | q2 Д 130 m² | q3 25 ≤ 30 m | q4 not expired | Passed | Cost, 4 q | Calls |
|---|:--:|:--:|:--:|:--:|---:|---:|---:|
| `deepseek/deepseek-v4.1-flash` | ✓ | ✓ | ✓ | ✓ | **4/4** | $0.022 | 8 |
| `openai/gpt-5-mini` | ✓ | ✓ | ✓ | ✓ | **4/4** | $0.049 | 8 |
| `google/gemini-3-flash-preview` | ✓ | ✓ | ✓ | ✓ | **4/4** | $0.030 | 8 |
| `openai/gpt-4o-mini` | ✓ | ✗ | ✓ | ✗ | 2/4 | $0.004 | 4 |
| `anthropic/claude-haiku-4.5` | ✓ | ✓ | ✗ | ✗ | 2/4 | $0.057 | 6 |
| `deepseek/deepseek-v4-flash` | ✓ | ✗ | ~ | ✗ | 1–1.5/4 | $0.003 | 6 |

Target failures the set was built to catch:

- **q2:** `gpt-4o-mini` inverted п. 401 ("130 m² does not require extinguishers"); a weaker
  DeepSeek conflated portable and mobile extinguisher rules.
- **q4:** three models declared an inspection overdue **while naming the correct future date**
  (15.11.2026) — a comparison against the sheet's fill-in date.
- **q3:** `claude-haiku-4.5` called 25 m "exceeds the permitted" while itself quoting the
  30 m limit.

Wiring observations:

- **Contract status did not separate models.** All six scored 4/4 on the contract check;
  `contract_ok` was blind to content.
- **The verifier produced false positives on correct answers** (three correct answers flagged
  `verification_*`). It is included in the runner only when the contract requires it.
- **`forbidden_conclusion` (exact substring) missed the split-word "inspection … overdue"**
  and once fired on a correct sentence about a neighbouring room. Semantic scoring stays
  manual.

## Interpretation

Cheap models differ by content, not by validator score, and the difference is large enough to
decide the default: three models pass all four traps, one of them at ~1/8 the per-question
cost of GPT-5 mini. Weaker models fail exactly the targeted error classes, confirming the
traps measure something real.

## Decision (Petr, 2026-09-17)

Take **`deepseek/deepseek-v4.1-flash`** as the showcase default for the simple path
(OpenRouter). Backups: `google/gemini-3-flash-preview`, `openai/gpt-5-mini`. Wired in
`config/settings.py`, `.env.example` and the README.

## Threats to validity

Four questions, one run per model. A pass is not proof of stability; the cheapest models were
run only once, and the failure detail above comes from single answers. `gpt-4o-mini` and
`claude-haiku-4.5` fail 2/4 here but pass other question types, so the result is
trap-specific, not a general model ranking.

## Next action

Repeat `q1`/`q7` and the traps for stability; add traps for area, distance and period
("below threshold but still applies"); defer retrieval of appendix № 1.