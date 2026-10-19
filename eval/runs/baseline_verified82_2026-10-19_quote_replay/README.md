# Offline quote replay — baseline 82, 2026-10-19

No model or LangSmith calls were made. Production code, source checkpoints and
remote feedback were not changed. This is a diagnostic alternative parsing series,
not a replacement baseline and not a judge-quality validation.

## Inputs and reproduction checks

Source: `../baseline_verified82_2026-10-19/`, code SHA
`8b5f1b88d89933a94649f7bc37257e2a22ec6d11`. The source contains 82 target
checkpoints, 28 fully assessed results and 64 raw judge replies. G059 has a
correctness assessment but no completed result; these counts must not be pooled
as if all 82 cases had been judged.

All 64 raw judge replies were parsed again with the original pure parser functions.
Their overlapping metric scores matched the saved assessments with zero differences.
Source SHA-256 values for manifest, outputs, judges, assessments, results and summary
were checked again after the diagnostic: all six unchanged. They are recorded in
[`replay.json`](replay.json).

## Candidate normalization policies

| Policy | Correctness invalid replies | Citations invalid replies | Groundedness invalid replies |
|---|---:|---:|---:|
| Original parser | 1 | 2 | 6 |
| Unicode NFC + `ё` → `е` for quote matching | 1 | 1 | 6 |
| The same + typographic double-quote folding | 1 | 1 | 6 |

Only G052/citations changed: `answer_quote="статьей 7"` matches the saved answer's
`статьёй 7` under the candidate normalization. All checks then validate, giving
`citations=1` and `citations_complete=1` instead of two unresolved metrics. Its
correctness verdict is unchanged. No previously scored verdict changed.

For the 28 fully assessed cases only, the diagnostic would change citations from
17/18 to 18/19 scored, and completeness from 13/19 to 14/20 scored. These are
partial-run denominators, not full-set performance.

Four boundary checks passed through the citation scoring path: the spelling
variation is accepted; changing article 7 to 8, 3 years to 5 years, or adding
`не` to the quoted statement is still rejected. This limited check does not
establish semantic equivalence of every Russian word containing `е` or `ё`.

## Remaining contract failures

The groundedness replies have seven invalid checks across six cases. Full evidence
and check indices are in [`groundedness_issues.json`](groundedness_issues.json).

| Case | Failure |
|---|---|
| G005 | A single context quote combines two separate provisions using a semicolon; it is not a copied continuous excerpt. |
| G008 | A quoted full sentence is not present continuously in the packed context; this is not a typography mismatch. |
| G027 | One quote combines introductions to paragraphs 393 and 394 using a semicolon. |
| G052 | A quote joins text separated by a fragment boundary and its header. |
| G057 | One quote combines the rule and an exception from separate fragments. |
| G058 | Two checks label absence-of-direct-answer statements as supported while giving null context quotes. The prompt excludes such statements from claim checks; the required evidence contract still rejects them. |

G005 also remains invalid for correctness and citations: the judge changed word
forms inside supposedly verbatim answer quotes. Folding `ё/е` and quotation
marks does not repair these differences. No morphological matching, invented
evidence or silent removal of invalid checks was used.

## Implications before another paid call

The spelling fix recovers one judge reply, not the six groundedness failures.
Groundedness remains scored for 15/21 applicable completed cases (all 15 pass).
This cannot support a claim of 100% groundedness or meet the declared 95% coverage
threshold even if every remaining applicable case is scored successfully.

Multi-fragment evidence needs an explicit contract (separate copied excerpts,
preferably with fragment IDs), followed by independent alignment if the judge
prompt/contract changes. Existing raw assessments must retain their original
version. Filtering invalid checks merely to turn a case into a pass is not a repair.

Any continuation should distinguish already valid assessments, invalid assessments
and missing assessments, preserving source run IDs. No automatic resume or new
paid execution was implemented or started here.
