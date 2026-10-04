# Golden-90 retrieval regression: 29н explicit_ctx (2026-10-04)

Script: `scripts/regression_29n_golden.py` (path `simple`, production hybrid, retrieval-only).
Indexes: production `chroma_db`; scratch copies with 29н replaced by `current` (re-chunked
from saved IR) and `explicit_ctx` (row + parent + section heading). IR: `../table_rows_29n_hybrid_2026-10-04/29n_ir.json`.
Cost: embeddings only (29н × 2 variants + 270 queries), < $0.01.

29н labels (`29н.pdf#N`) were mapped to new chunks by word containment ≥ 0.6, ignoring words
present in > 10% of chunks (section headings). `current` mapped 1:1 onto the same ids as
production — the re-chunking reproduces production exactly.

| index | HR@5 | HR@12 | MRR |
|---|---|---|---|
| production | 57/90 | 73/90 (0.811) | 0.501 |
| current | 57/90 | 73/90 | 0.503 |
| explicit_ctx (auto mapping) | 56/90 | 72/90 | 0.487 |
| explicit_ctx (1 mapping fixed by hand) | 56/90 | 73/90 | — |

Only 3 questions changed rank (all others identical):

- «Медосмотр водителей 1 раз в год или 1 раз в 2 года?» — miss → rank 4 (gain, the #319 case).
- «Машинист крана это работы на высоте?» — old label #302 held the end of row 6 (heights)
  plus the start of row 8 (crane operators). Auto mapping chose row 6.1 (#372); explicit_ctx
  returns row 8 «крановщик (машинист крана)» (#375) at rank 3. Counted as hit by hand.
- «Что считать работой на высоте?» — rank 2 → miss. The hit in production was 29н#299
  (row 6 of the table); the real definition labels (782н#10–12) are missed by both indexes.
  Real loss of a weak label.

Other documents: no question with only non-29н labels changed rank. New 29н chunks did not
push other documents' labeled chunks out of top-12.

Conclusion: no regression outside 29н; HR@12 equal after manual review, MRR −0.014.
Combined with the 22 table questions (hit@3 14 → 17): proceed to the production chunker change.
