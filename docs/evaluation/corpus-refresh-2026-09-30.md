# Corpus refresh gate — 30 September 2026

The legal-reference audit cannot be closed by editing `tests/dataset.csv` alone. This
record separates the revised key, local source files, the existing Chroma index and a
new measured baseline. It is an execution checklist, not a claim of legal certification.

## What is present locally

| Source | Local state | Verification needed |
|---|---|---|
| Rules No. 2464 | `source_docs/2464.pdf`, SHA-256 `9543c69c06487349cdd9ac05285948b374aa429222b5388b057277710c3374f5` | The PDF predates [Government Resolution No. 805](https://publication.pravo.gov.ru/document/0001202606300022). Its paragraph 60 still says annually; the amendment effective 31.08.2026 changes this to at least once in three years. Obtain and verify the complete official updated text before replacing the file. |
| MChS Order No. 1120 | Candidate in `source_docs_dept/ext_mchs_1120.docx`, SHA-256 `9af3091b90d847ebc44d8b21cfbd6523ddfe17dee84606d061afe23b308fcff4` | A parser-only run on a temporary copy produced 91 chunks and included appendix 4. The appendix 4 categories were compared with an [MChS-hosted copy](https://65.mchs.gov.ru/uploads/resource/2026-02-06/0f4a54dff13b85b7087bc6e1a76adcd4.pdf). Verify complete-document identity and publication status before placing the candidate in the active generic corpus. |
| Order No. 342н and amendment No. 392н | Absent from `source_docs/` and current generic index | Obtain the current full official text and [2025 amendment](https://publication.pravo.gov.ru/document/0001202508110001) effective in 2026 for the psychiatric examination question. |
| Law No. 125-ФЗ | Absent from `source_docs/` and current generic index | Obtain a current full text; the [SFR guidance](https://sfr.gov.ru/branches/arkhangelsk/info/~2026/05/29/10706?info_category=2) confirms the operational one-day notice rule but is not the full law. |
| Law No. 370-ФЗ and professional standard No. 696н | Absent from `source_docs/` and current generic index | Obtain full texts and check the applicability conditions behind case G025. The [Mintrud page for No. 696н](https://mintrud.gov.ru/docs/mintrud/orders/2141) links its annex but that annex has not been saved into the corpus. |

The generic Chroma collection `documents` contained **7,842** chunks at inspection,
including **310** from the stale `2464.pdf` and **50** from `test.pdf`, which is absent
from the current `source_docs/` folder. Neither `1120.docx` nor the other missing acts
had any chunks. The old index is therefore not a trustworthy candidate for a new
normative baseline. The previously documented 7,792-chunk count belongs to another
snapshot and should not be silently reused.

The official publication portal timed out from this runtime on 30.09.2026. The
[full text of the 805 amendment at Garant](https://www.garant.ru/products/ipo/prime/doc/414360397/)
supports the changed paragraphs, but it is a secondary host. Until the official
document or a verified copy is available, the affected cases retain
`reference_status=needs_clarification` and `corpus_support=known_missing`.

## Acceptance sequence

1. Save each complete current act with a source URL, publication date, effective date,
   SHA-256, and the exact article or paragraph used by each case. Verify all amendments
   effective on the evaluation date. Do not mix old and new consolidated editions in
   the same active corpus.
2. Parse the replacement documents without touching Chroma. Inspect the chunks around
   Rules No. 2464 paragraphs 13, 14, 53 and 60; Order No. 342н; Order No. 1120 appendix 4;
   Law No. 125-ФЗ article 17; and the provisions behind G025. Mark each case's
   `corpus_support` only after checking the required clause in the parsed output.
3. Build a **separate** generic index at a new Chroma path and collection. `index.py`
   deletes its target directory before embedding, so never point this trial run at
   the existing `chroma_db`. Confirm source counts and absence of stale `test.pdf`.
4. Run retrieval checks for G003, G004, G011, G020, G025, G042, G048 and G055. The
   relevant clause must reach the answer context, not merely exist in the index.
5. Independently review the revised references and label alternative correct answers.
   Only then change `reference_status` to `verified`. Rejudge the saved 23.09 answers
   using `scripts/rejudge_golden.py`; report expert–judge disagreements separately.
6. Freeze the dataset and index hashes, then run the same full evaluation on compared
   configurations and a new held-out legal set. Publish denominators, failures,
   cost and latency with the new baseline. Do not compare a 12-case partial score
   directly with the old 43-case score.

The current `tests/dataset.csv` contains the revised wording but only 12 of 43
in-scope references remain verified from the earlier audit. Its `review_note` retains
the *prior* row finding; `reviewed_at=2026-09-28` is not a verification date for
the revised wording. Literal `must_not_contain` checks were cleared. The current
judge receives `forbidden_claims` as full assertions and is told not to flag
negations or permissible conditional cases.
