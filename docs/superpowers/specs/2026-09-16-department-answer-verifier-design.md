# Department answer verifier — experiment design

**Status:** approved for implementation and one paid v2 run on 2026-09-16.

## Goal

Measure whether a separate structured LLM call detects missing object facts before a
department answer is exposed as `answered`. The answer prompt and retrieval stay frozen.

## Contract

The verifier receives the user question, the parsed answer, and the exact evidence passed
to the answer model. It returns one verdict:

- `none` — the cited facts are sufficient for every applied conclusion;
- `missing` — a conclusion needs an object fact that is absent or explicitly unknown;
- `contradiction` — a conclusion conflicts with the supplied evidence.

It also returns `missing_fields` and a short `explanation`. The verifier does not rewrite
the answer and cannot add evidence.

## Gate

The call runs only after the existing citation/status contract succeeds and only when the
answer contains an applied conclusion. `none` preserves the existing status. `missing`
changes `answered` to `needs_context` and exposes the missing fields as clarification
questions. `contradiction` changes it to `needs_review`. A verifier/schema failure also
fails closed to `needs_review`. Existing stricter statuses are never upgraded.

## Evaluation

Run only mode v2 over the frozen nine-question ObjectProfile set with OpenRouter model
`openai/gpt-4o-mini`,
temperature 0. Record both raw structured calls, prompts, verdict and gated response.
Success criterion for this experiment: q1 is no longer `answered` and is classified as
`missing`; no previously safe answer is upgraded or rewritten. Report all nine outcomes,
latency/cost if provider usage is available, and retain artifacts under `eval/runs/`.
