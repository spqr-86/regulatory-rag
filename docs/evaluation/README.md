# 📊 Evaluation & Metrics Guide

Система оценки качества V7-пайплайна.

## Компоненты

1. **Датасет** — `tests/dataset.csv`. Golden-набор, колонки `question` и `ground_truth`
   (~50 вопросов: in-scope + OOS + false-premise).
2. **Раннер** — `eval/run_v7_eval.py`. Прогоняет датасет через скомпилированный V7-граф,
   считает метрики, пишет JSON-отчёт.
3. **Метрики-судьи** — `src/advanced_generation_metrics.py` (`evaluate_faithfulness`,
   `evaluate_answer_relevance`) + `evaluate_correctness` внутри раннера. Все три —
   LLM-as-judge на Gemini (`get_gemini_llm`, та же модель что в `GEMINI_FAST_MODEL`).

## Запуск

```bash
source venv/bin/activate
python eval/run_v7_eval.py                                  # весь датасет
python eval/run_v7_eval.py --limit 5                        # быстрый smoke-тест
python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
```

CLI: `--limit N` (ограничить число вопросов), `--output PATH` (путь отчёта;
по умолчанию `benchmarks/eval_v7_{дата}.jsonl`).

## Метрики

| Метрика | Что проверяет | Цель |
| :--- | :--- | :--- |
| **faithfulness** | Обоснованность — стоит ли ответ на retrieved-контексте (LLM-судья, 0-1) | > 0.85 |
| **answer_relevance** | Соответствует ли ответ вопросу (LLM-судья, 0-1) | > 0.85 |
| **correctness_mean** | Соответствие эталону (LLM-судья, 0-10) | > 7.5 (достигнуто: 7.9) |
| **false_sufficiency_rate** | Доля simple-path ответов с correctness < 5.0 | < 10% |
| **complex_path_rate** | Доля вопросов, ушедших на complex-путь | — |
| **mean_elapsed_sec** | Средняя задержка ответа | — |

`false_sufficiency` ловит главный анти-паттерн: система пошла быстрым путём и
ответила, хотя ответ плохой.

## Формат отчёта

JSON: `{aggregate, results, dataset_size, valid_results, timestamp}`. `aggregate` —
агрегированные метрики выше; `results` — список записей по каждому вопросу
(`question`, `ground_truth`, `answer`, `path`, `*_score`, `*_reasoning`,
`elapsed_sec`). Reasoning у каждого судьи сохраняется — это резко упрощает разбор
просадок.

## Принцип LLM-as-judge

Каждая метрика — отдельный запрос к модели-судье со своим промптом. Судья обязан
вернуть не только число, но и `reasoning`. При сравнении прогонов учитывайте
run-to-run вариативность судьи: мелкие per-question дельты (особенно на OOS-вопросах)
— шум; доверяйте агрегатам и крупным изменениям.

## Дополнительно

- [Benchmarks и Baseline](./../../benchmarks/README.md)
- [Добавление вопросов в датасет](../guides/adding-questions.md)
