# Бенчмарки

Директория содержит актуальные baseline-метрики и локальные артефакты eval-прогонов.

**В git:** только этот README и английская версия.  
**Локально:** `eval_v7_*.jsonl`, `cps_*.json` — артефакты прогонов, в `.gitignore`.

## Актуальный baseline (2026-05-15)

| Метрика | Значение | Цель | Статус |
|---------|----------|------|--------|
| Correctness | **7.9 / 10** | > 7.5 | ✅ |
| Faithfulness | **0.988** | > 0.85 | ✅ |
| Answer Relevance | **0.753** | > 0.85 | 🔄 |
| False Sufficiency Rate | **15%** | < 10% | 🔄 |
| Complex path rate | 59% | — | — |
| Mean latency | 17.4s | < 10s | 🔄 |

**Конфиг:** V7 LangGraph, OpenAI embeddings (text-embedding-3-small), Gemini 2.5 Flash (simple) + thinking (complex), 11 PDF, HybridChunker v3.0-hybrid, датасет 57 вопросов.

## Запуск eval

```bash
# Без LLM-судьи (~$0)
python eval/run_v7_eval.py --skip-judge --output benchmarks/eval_v7_$(date +%F).jsonl

# Полный eval с судьёй (OpenAI gpt-4o-mini)
python eval/run_v7_eval.py --output benchmarks/eval_v7_$(date +%F).jsonl
```

## Обновление baseline

После значимого улучшения метрик — обновить таблицу выше вручную и закоммитить.

---

[English version](README.md)
