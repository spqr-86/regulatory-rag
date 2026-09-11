# Бенчмарки

Актуальные baseline-метрики и локальные артефакты eval-прогонов.

**В git:** только этот README и английская версия.
**Локально:** `eval_v7_*.jsonl`, `retrieval_*_*.json`, `triage_gap_*.json`, `cps_*.json` —
артефакты прогонов, в `.gitignore`.

## Актуальный baseline — 2026-09-11

Генеративный eval (`eval/run_v7_eval.py`, судья `gpt-4o`, `tests/dataset.csv` — 56
вопросов, 53 валидных) — финальный терминальный контракт triage:

| Метрика | Значение | Baseline до контракта (08.09) |
|---------|----------|-------------------------------|
| In-scope correctness (0–10) | **7.47** | 7.40 |
| Correctness, все вопросы | **7.26** | 7.09 |
| Faithfulness (0–1) | **0.891** | 0.808 |
| Answer relevance (0–1) | **0.879** | 0.881 |
| OOS abstain rate | **1.00** | 1.00 |
| False-sufficiency rate | **11.4%** | 13.0% |
| Complex-path rate | 17.0% | 13.2% |
| Латентность p50 / p95 / средняя | 4.51 / 15.70 / 6.83 с | 4.8 / 14.7 / 5.9 с |
| Стоимость на запрос | $0.00387 ($0.205 весь прогон) | $0.0033 |

**Конфиг:** V7 LangGraph, OpenAI `text-embedding-3-small`, `gpt-4o-mini` (simple) /
`gpt-4o` (complex), единый hard-gate triage с терминальным контрактом маршрута,
CrossEncoder-реранкер (cap 100), HybridChunker (`max_tokens=400`), 12 документов,
датасет 56 вопросов.

Retrieval eval (`eval/run_retrieval_eval.py`, held-out 133, 2026-09-08):

| путь | HR@5 | HR@12 | MRR | p50 латентность |
|------|------|-------|-----|-----------------|
| hybrid (прод) | 0.692 | 0.827 | 0.542 | 502 мс |
| vector-only | — | 0.842 | 0.539 | 141 мс |
| bm25-only | — | 0.677 | 0.387 | 19 мс |

> Терминальный контракт улучшил faithfulness и false sufficiency, не сдвинув correctness
> и relevance за пределы разброса судьи ~±0.25. Сравнивать прогоны только под одним
> судьёй. Канонические значения:
> [docs/reference/FACTS.md](../docs/reference/FACTS.md).

## Запуск eval

```bash
python eval/run_v7_eval.py --skip-judge --output benchmarks/eval_v7_$(date +%F).jsonl  # только пайплайн, ~$0
python eval/run_v7_eval.py --output benchmarks/eval_v7_$(date +%F).jsonl               # полный, судья gpt-4o, ~$0.25
python eval/run_retrieval_eval.py --path hybrid                                        # IR-метрики retrieval
```

## Обновление baseline

После значимого изменения — перепрогнать и обновить таблицы выше вручную, затем закоммитить.

---

[English version](README.md)
