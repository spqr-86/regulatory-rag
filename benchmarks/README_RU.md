# Бенчмарки

Актуальные baseline-метрики и локальные артефакты eval-прогонов.

**В git:** только этот README и английская версия.
**Локально:** `eval_v7_*.jsonl`, `retrieval_*_*.json`, `triage_gap_*.json`, `cps_*.json` —
артефакты прогонов, в `.gitignore`.

## Актуальный baseline — 2026-09-08

Генеративный eval (`eval/run_v7_eval.py`, судья `gpt-4o`, `tests/dataset.csv` — 56
вопросов, 53 валидных) — первый полный прогон после фикса судьи в `llm_factory` 02.09:

| Метрика | Значение | Раньше (mini-судья, 2026-05-30) |
|---------|----------|--------------------------------|
| In-scope correctness (0–10) | **7.56** | 7.44 |
| Correctness, все вопросы | **7.30** | 7.39 |
| Faithfulness (0–1) | **0.840** | 0.859 |
| Answer relevance (0–1) | **0.847** | 0.872 |
| OOS abstain rate | **1.00** | 1.00 |
| False-sufficiency rate | **4.8%** | 9.8% |
| Complex-path rate | 20.8% | 24.1% |
| Латентность p50 / p95 / средняя | 5.2 / 19.1 / 7.2 с | — / — / 9.71 |
| Стоимость на запрос | $0.0045 ($0.24 весь прогон) | — |

**Конфиг:** V7 LangGraph, OpenAI `text-embedding-3-small`, `gpt-4o-mini` (simple) /
`gpt-4o` (complex), `V7_V8_ENABLE_EVIDENCE_ASSESS=true`, CrossEncoder-реранкер (cap 100),
HybridChunker (`max_tokens=400`), 12 документов, датасет 56 вопросов.

Retrieval eval (`eval/run_retrieval_eval.py`, held-out 133, 2026-09-08):

| путь | HR@5 | HR@12 | MRR | p50 латентность |
|------|------|-------|-----|-----------------|
| hybrid (прод) | 0.692 | 0.827 | 0.542 | 502 мс |
| vector-only | — | 0.842 | 0.539 | 141 мс |
| bm25-only | — | 0.677 | 0.387 | 19 мс |

> Судья `gpt-4o` строже прежнего `gpt-4o-mini` — абсолютные faithfulness / relevance чуть
> ниже исторических, но лучше откалиброваны. Разброс судьи между прогонами ~±0.25 по
> шкале correctness. Сравнивать прогоны только под одним судьёй. Канонические значения:
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
