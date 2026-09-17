# Experiments

Decision memos, one per question the project asked of its own system. Each follows the same
shape: Question → Setup → Results → Interpretation → Decision → Threats to validity → Next
action. Negative results are kept as first-class outcomes.

| Memo | Question | Outcome |
|---|---|---|
| [retrieval-backbones](./retrieval-backbones.md) | vector vs BM25 vs hybrid | keep hybrid; bm25 disqualified; vector a Pareto alternative |
| [rrf-k-negative-result](./rrf-k-negative-result.md) | is `RRF_K` worth tuning? | no — Hit Rate flat across {5…200} |
| [triage-threshold-calibration-negative-result](./triage-threshold-calibration-negative-result.md) | can other hard gates cut unsafe answers? | no — 81 profiles, defaults kept |
| [triage-gap-terminal-contract](./triage-gap-terminal-contract.md) | escalate with a named gap and a single route contract | shipped; escalations cut, no hit lost |
| [department-qa-object-profile](./department-qa-object-profile.md) | how do unit facts reach the prompt, and why did `q1` survive every prompt fix? | `v2` default; `q1` is a model error |
| [cheap-model-selection](./cheap-model-selection.md) | which cheap model holds the threshold traps? | DeepSeek V4.1 Flash at 1/8 the cost of GPT-5 mini |

Context, dataset cards and global threats to validity: [../README.md](../README.md),
[../datasets.md](../datasets.md). Canonical current values:
[FACTS](../../reference/FACTS.md).