# 29н: production row chunker (`src/indexing/table_rows.py`) — re-measure, 04.10.2026

Scripts: `scripts/hybrid_29n_table_context.py` (22 table questions) and `scripts/regression_29n_golden.py`
(golden 90, path `simple`). Scratch copies of the full `chroma_db`, 29н replaced by a variant built from
the saved IR (`../table_rows_29n_hybrid_2026-10-04/29n_ir.json`). Variants: `current` = chunker with the
row layer disabled (reproduces production 1:1), `explicit_ctx` = 04.10 experiment, `rows` = production
chunker. Paid: embeddings only, ≈ $0.02 total for three runs (estimate).

Run 1 (`*_report.json`): `rows` = explicit_ctx on all 22, but 13 near-empty chunks (group titles,
page-break tails: «офтальмолог», «IV. Физические факторы») — one pushed «внеплановый медосмотр» 1 → 3.
Fix: unnumbered runs < 20 tokens join a neighbour row (lowercase start or no letters → previous row,
else → next row). Run 2 (`*_v2`): the footnote «<2>:» went to row 18.1 (drivers 4 → 6) → footnotes
join the previous row. Run 3 (`*_v3`) is final: 555 chunks / 126.7K tokens.

22 table questions (final_context):

| variant | 29н chunks | hit@1 | hit@3 | hit@5 | hit@12 |
|---|---|---|---|---|---|
| current | 426 | 10 | 14 | 17 | 18 |
| explicit_ctx | 494 | 16 | 17 | 20 | 20 |
| rows (v3) | 555 | 16 | 17 | 20 | 20 |

Golden 90:

| index | HR@5 | HR@12 | MRR |
|---|---|---|---|
| production | 57/90 | 73/90 | 0.503 |
| current | 57/90 | 73/90 | 0.503 |
| rows (v3), auto mapping | 56/90 | 72/90 | 0.488 |
| rows (v3), crane fixed by hand | 56/90 | 73/90 | — |

Changed vs production: drivers miss → 5 (gain); crane operator 1 → 3 (row 8 «крановщик», counted by
hand as for explicit_ctx); «Что считать работой на высоте?» 2 → miss (weak label 29н row 6, known);
fire safety 12 → 11. Other documents unchanged.

Limits: questions written by the agent; 29н only re-chunked. Conclusion: production chunker matches
the experiment without losing table text; ready for full re-indexing (separate consent).
