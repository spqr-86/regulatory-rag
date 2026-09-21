# Бенчмарки

Актуальные baseline-метрики и локальные артефакты eval-прогонов.

**В git:** только этот README и английская версия.
**Локально:** `eval_v7_*.jsonl`, `retrieval_*_*.json`, `triage_gap_*.json`, `cps_*.json` —
артефакты прогонов, в `.gitignore`.

## Актуальный baseline — 2026-09-17

Генеративный eval (`eval/run_v7_eval.py`, судья `gpt-4o`, `tests/dataset.csv` — 56
вопросов, 53 валидных) — витринный дефолт (`deepseek/deepseek-v4.1-flash` на simple-пути):

| Метрика | Значение | Терминальный контракт (11.09) | Baseline до контракта (08.09) |
|---------|----------|--------------------------------|-------------------------------|
| In-scope correctness (0–10) | **7.91** | 7.47 | 7.40 |
| Correctness, все вопросы | **7.98** | 7.26 | 7.09 |
| Faithfulness (0–1) | **0.926** | 0.891 | 0.808 |
| Answer relevance (0–1) | **0.887** | 0.879 | 0.881 |
| OOS abstain rate | **1.00** | 1.00 | 1.00 |
| False-sufficiency rate | **10.0%** | 11.4% | 13.0% |
| Complex-path rate | 24.5% | 17.0% | 13.2% |
| Латентность p50 / p95 / средняя | 24.0 / 71.3 / 30.95 с | 4.51 / 15.70 / 6.83 с | 4.8 / 14.7 / 5.9 с |
| Стоимость на запрос, реальная | $0.00657 ($0.348 весь прогон) | $0.00387 ($0.205 весь прогон) | $0.0033 |

**Конфиг:** V7 LangGraph, OpenAI `text-embedding-3-small`, `deepseek/deepseek-v4.1-flash`
(simple) / `gpt-4o` (complex — тоже перешёл на DeepSeek 18.09.2026, уже после этого прогона),
единый hard-gate triage с терминальным контрактом маршрута, CrossEncoder-реранкер (cap 100),
HybridChunker (`max_tokens=400`), 12 документов, датасет 56 вопросов. Стоимость — по реальному
расходу токенов прогона против `src/pricing.py::PRICE_PER_1M`, не по self-reported total прогона
(у DeepSeek на момент прогона не было записи в rate card). Подробности:
[showcase-default-golden-set](../docs/evaluation/experiments/showcase-default-golden-set.md).

Retrieval eval (`eval/run_retrieval_eval.py`, held-out 133, 2026-09-08):

| путь | HR@5 | HR@12 | MRR | p50 латентность |
|------|------|-------|-----|-----------------|
| hybrid (прод) | 0.692 | 0.827 | 0.542 | 502 мс |
| vector-only | — | 0.842 | 0.539 | 141 мс |
| bm25-only | — | 0.677 | 0.387 | 19 мс |

> Витринный дефолт впервые проходит цель in-scope correctness >7.5, ценой ~5-кратной
> регрессии латентности (причина не выяснена) и реальной стоимости выше, чем у
> терминального контракта, если DeepSeek оценить по факту. Сравнивать прогоны только
> под одним судьёй. Канонические значения:
> [docs/reference/FACTS.md](../docs/reference/FACTS.md).

## Запуск eval

```bash
python eval/run_v7_eval.py --skip-judge --output benchmarks/eval_v7_$(date +%F).jsonl  # только пайплайн, ~$0
python eval/run_v7_eval.py --output benchmarks/eval_v7_$(date +%F).jsonl               # полный, судья gpt-4o, ~$0.25
```

IR-метрики retrieval. `--path simple` — продовый hybrid (vector + BM25 → RRF);
`vector` и `bm25` — отдельные backbone; `complex` — vector → rerank → MMR.

```bash
# основной тестовый набор: 90 вопросов практиков (docs/reference/FACTS.md)
python eval/run_retrieval_eval.py --path simple --gt eval/data/golden_retrieval_labeled_ext.jsonl
python eval/run_retrieval_eval.py --path vector --gt eval/data/golden_retrieval_labeled_ext.jsonl
python eval/run_retrieval_eval.py --path bm25   --gt eval/data/golden_retrieval_labeled_ext.jsonl

# development-набор (по умолчанию --gt eval/data/retrieval_gt_reviewed.jsonl)
python eval/run_retrieval_eval.py --path simple

# smoke-проверка перед полным прогоном
python eval/run_retrieval_eval.py --path simple --gt eval/data/golden_retrieval_labeled_ext.jsonl --limit 3
```

Для воспроизведения нужен проиндексированный корпус: исходные документы (`source_docs/`) и
индекс ChromaDB не хранятся в git — см. [getting started](../docs/getting-started.md) и
список документов в [FACTS](../docs/reference/FACTS.md#corpus). Эмбеддинги запросов идут
через настроенного провайдера (по умолчанию OpenAI, доли цента за прогон). Raw-выходы
прогонов (`benchmarks/*.json`, `benchmarks/*.jsonl`) в .gitignore; таблицы выше перенесены
из них вручную.

## Обновление baseline

После значимого изменения — перепрогнать и обновить таблицы выше вручную, затем закоммитить.

---

[English version](README.md)
