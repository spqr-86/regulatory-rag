# Retrieval-разметка новых verified, снимок build-j7ceu4r4 (2026-10-18)

- `golden_retrieval_labeled.jsonl` — активные метки (`eval/ls/dataset.py`): 133 старых
  (копия `gt_remap_build-j7ceu4r4_2026-10-18/`) + 50 новых verified in-scope (G057+).
- Локальный путь без платного API (согласие Петра, TG 43662): `eval/dump_candidate_pools.py`
  (пулы, gitignored) → strict/lenient-проходы субагентами (`verdicts/`) → арбитр по спорным
  и пустым (`arbitration/`, задания — `arbiter_jobs.json`) → `eval/ingest_agent_labels.py`.
  Итог: 15 согласий, 35 решено арбитром; эталон из пула — 36 вопросов
  (`golden_retrieval_labeled_new.jsonl`, очередь человеку — `.review.tsv`).
- 14 вопросов (G058, G063, G064, G065, G070, G073, G075, G087, G092, G093, G102, G103,
  G145, G155): нормы из эталона нет в топ-30 пула — промах поиска. Чанки найдены в снимке
  по пунктам, на которые ссылается ground_truth, текст сверен; в метке `pool_miss: true`.
