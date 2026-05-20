# Benchmarks и Baseline Метрики

Эта директория содержит baseline метрики и историю результатов eval.

## Файлы

### `baseline.json`
Baseline метрики для текущей production версии системы. Используется для сравнения новых версий.

**Актуальный конфиг (май 2026, V7 pipeline):**
```json
{
  "date": "2026-05-07",
  "version": "V7 (stage 6)",
  "dataset": "golden-questions",
  "dataset_size": 41,
  "config": {
    "pipeline": "v7_langgraph",
    "llm_provider": "gemini",
    "llm_model": "gemini-2.5-flash (gemini-3-flash-preview)",
    "thinking_budget": 4096,
    "gost_rag_llm": "deepseek-chat (DeepSeek V3, openai SDK)",
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 1500,
    "chunk_overlap": 400,
    "simple_top_k": 12,
    "complex_top_k": 60,
    "corpus": "12 PDF, 1973 chunks (v2.3-noise-clean)",
    "gost_corpus": "108 DOCX, 9344 chunks (wta_gosts)"
  },
  "metrics": {
    "correctness_mean": 7.9,
    "faithfulness": 0.988,
    "note": "Eval 2026-05-16, pipeline v2.3-noise-clean. Цель correctness 7.5 достигнута."
  }
}
```

### `results_history.jsonl`
История всех запусков eval в формате JSONL (одна строка = один запуск).

Каждая запись содержит:
- timestamp
- dataset
- aggregate_metrics
- detailed_results (опционально)

## Как обновить baseline

После значительного улучшения системы:

```bash
# 1. Запустить полную оценку
python eval/run_full_evaluation.py

# 2. Если метрики улучшились - обновить baseline
cp benchmarks/baseline.json benchmarks/baseline_old.json
# Создать новый baseline.json с новыми метриками
```

## Целевые метрики

| Метрика | Целевое | Baseline | Статус |
|---------|---------|----------|--------|
| Correctness | > 7.5/10 | 7.9 | ✅ Достигнуто |
| Faithfulness | > 0.85 | 0.988 | ✅ Достигнуто |
| Answer Relevance | > 0.85 | — | — |
| False Sufficiency Rate | < 10% | 15% | 🔄 Требуется улучшение |
| P95 Latency | < 10s | — | — |

## Сравнение с baseline

```bash
# Запустить скрипт сравнения (когда будет реализован)
python scripts/compare_with_baseline.py benchmarks/results_history.jsonl
```
