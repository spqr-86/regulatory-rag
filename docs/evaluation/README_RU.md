# Руководство по оценке качества

Фреймворк оценки качества пайплайна V7.

## Компоненты

1. **Датасет** — `tests/dataset.csv`. Золотой набор, столбцы `question` и `ground_truth`
   (~57 вопросов: in-scope + OOS + false-premise).
2. **Runner** — `eval/run_v7_eval.py`. Прогоняет датасет через скомпилированный граф V7,
   вычисляет метрики, записывает JSONL-отчёт.
3. **Judge метрики** — `eval/advanced_generation_metrics.py` (`evaluate_faithfulness`,
   `evaluate_answer_relevance`) + `evaluate_correctness` внутри runner. Все три —
   LLM-as-judge с использованием `gpt-4o-mini` (`JUDGE_LLM_PROVIDER=openai`, настраивается через `get_judge_llm()`).

## Запуск

```bash
source venv/bin/activate
python eval/run_v7_eval.py                                  # полный датасет
python eval/run_v7_eval.py --skip-judge                     # только пайплайн, без LLM judge (~$0)
python eval/run_v7_eval.py --limit 5                        # быстрый smoke test
python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
```

CLI флаги: `--limit N` (ограничение количества вопросов), `--skip-judge` (без LLM-оценки),
`--output PATH` (по умолчанию: `benchmarks/eval_v7_{date}.jsonl`).

## Метрики

| Метрика | Что проверяет | Цель |
| :--- | :--- | :--- |
| **faithfulness** | Ответ основан на извлечённом контексте (LLM judge, 0–1) | > 0.85 |
| **answer_relevance** | Ответ отвечает на вопрос (LLM judge, 0–1) | > 0.85 |
| **correctness_mean** | Соответствие ground truth (LLM judge, 0–10) | > 7.5 (достигнуто: 7.9) |
| **false_sufficiency_rate** | Доля ответов по быстрому пути с correctness < 5.0 | < 10% |
| **complex_path_rate** | Доля запросов, направленных на сложный путь | — |
| **mean_elapsed_sec** | Средняя задержка ответа | — |

`false_sufficiency` выявляет основной антипаттерн: система выбрала быстрый путь и дала плохой ответ.

## Формат отчёта

JSONL: каждая строка — `{aggregate, results, dataset_size, valid_results, timestamp}`. `aggregate` содержит
метрики выше; `results` — список записей по каждому вопросу (`question`, `ground_truth`, `answer`,
`path`, `*_score`, `*_reasoning`, `elapsed_sec`). Рассуждения от каждого вызова judge сохраняются —
это позволяет легко диагностировать падение scores.

## LLM-as-Judge

Каждая метрика — отдельный запрос к модели-судье с собственным промптом. Судья должен вернуть
и score, и `reasoning`. При сравнении прогонов учитывать вариативность между запусками:
небольшие per-question дельты (особенно на OOS-вопросах) — шум; доверять агрегатам и крупным изменениям.

## См. также

- [Бенчмарки и baseline](./../../benchmarks/README.md)
- [Добавление вопросов в датасет](../guides/adding-questions.md)
