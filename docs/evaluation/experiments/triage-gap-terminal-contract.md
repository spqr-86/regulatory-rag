# Structured triage gap and terminal route contract

Two changes to the escalation path, both shipped. They are combined here because the second
supersedes the first: the terminal contract makes the routing decision explicit, and the
structured gap becomes its payload.

## Part 1 — structured sufficiency gap (issue #13, stage B2)

### Question

Can triage escalate **with a reason** — name the missing clauses, try to fetch them, and
escalate only if the gap stays open — instead of escalating the whole query blindly?

### Setup

- **What changed:** `evaluate_triage` emits a typed gap (`TriageGap` / `GapRef`) naming the
  referenced п./ст. absent from the top-5; the gap is closed in place by appending
  cross-referenced passages to the tail (no reorder); escalation happens only if it stays
  open.
- **What stayed fixed:** the hard gates, retrieval config, generator.
- **Dataset:** 133-question development set (plus 43 earlier).
- **Metrics:** escalation rate simple → complex, HR@12 of the triage output, gaps found /
  closed. Deterministic, no judge.
- **Artifacts:** `benchmarks/triage_gap_*` (gitignored); [roadmap](../../roadmap.md) issue #13.

### Results

| Measure | main | after B2 | after subpara fix |
|---|---:|---:|---:|
| Escalation rate (133) | 0.248 | 0.188 | **0.128** |
| HR@12 of triage output | 0.827 | 0.827 | 0.827 |
| Gaps found / closed | — | 23 / 8 | — / 17 |

The subpara fix came out of this work: a chunk with the clause list had its numbering mangled
at indexing (`46. а)` became `6. а)`), so the matcher failed to recognize clauses that were
actually present. Allowing an optional leading number recovered 8 escalations with no hit
lost.

### Interpretation

Cheaper escalation without losing hits, and the measurement surfaced a chunking defect rather
than a ranking problem. The earlier version of the gap logic dropped a hit because a
reordering expander displaced the reference chunk; the fix keeps originals in order and
appends fetched chunks at the tail.

## Part 2 — terminal route contract

### Question

Can the routing decision be made explicit and single-valued, and can the generator be
guaranteed to read exactly the context the decision was validated on?

### Setup

- **What changed:** simple and complex emit one `route_decision` / `route_reason`; the
  generator reads only the validated, budgeted `final_context`; expansion, sanitizing,
  trimming and budgeting belong to `pack_context`.
- **Dataset:** 133-question route benchmark (deterministic) and the 56-question generation
  set with the `gpt-4o` judge.
- **Artifacts:** `benchmarks/eval_v7_triage_contract_final_2026-09-11.jsonl` (gitignored);
  [roadmap](../../roadmap.md).

### Results

Route benchmark, 133 questions: escalation **0.195** vs baseline 0.248, gaps closed 114/133,
HR@12 **0.827**, hits lost **0/0**.

Generation, 56 questions (53 valid), same judge:

| Metric | Contract | Baseline |
|---|---:|---:|
| In-scope correctness | 7.47 | 7.40 |
| Faithfulness | 0.891 | 0.808 |
| Answer relevance | 0.879 | 0.881 |
| False-sufficiency | 11.4% | 13.0% |
| Cost / query | $0.00387 | $0.0033 |
| p95 latency | 15.70 s | 14.7 s |

### Interpretation

The contract bought faithfulness and false-sufficiency at ~18% higher cost and ~7% higher
p95; correctness and relevance moved within the judge's noise. An intermediate version
regressed HR@12 (0.827 → 0.789) by actually enforcing `MAX_CHUNKS_FOR_LLM=10`; a TDD fix
restored the effective limit to 30 and put reference-closing chunks ahead of expansion noise.

## Decision

Both shipped; the portfolio MVP was closed on this baseline (2026-09-11).

## Threats to validity

The 133-question set is a development set with pooling bias, so the route numbers are
relative, not absolute. One run per configuration; judge variance ~±0.25 on correctness.

## Next action

None required. Optional post-MVP: split errors into "not enough context" vs "generation
error" (issue #14).