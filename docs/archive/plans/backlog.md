# Backlog

**Обновлено:** 2026-05-26

## Активный план улучшений

| # | Улучшение | Сложность | Ожидаемый эффект | Статус |
|---|-----------|-----------|------------------|--------|
| 1 | **Contextual retrieval** — LLM генерирует 1-2 предл. контекста на чанк перед embedding | ~2 дня | +35-49% recall (Anthropic benchmark) | |
| 2 | **Hybrid retriever RRF fix** — RRF вместо concatenation в applicability_retriever.py | ~2 ч | +recall на BM25-only вопросах | |
| 3 | **Corpus gaps** — добавить методику СОУТ (классы УТ), уточнить ПП 1479 (тренировки 50+ чел) | ~1 ч | fix Q#29, Q#19 | |
| 4 | **test_sufficient fix** — pre-existing fail в test_evaluate_triage.py | ~30 мин | тесты 100% зелёные | |
| 5 | **Integration tests** с реальным ChromaDB | ~1 день | покрытие E2E | |
| 6 | **MCP-сервер поверх RAG** — search_knowledge tool для Claude | ~1 день | Claude ищет по ~/knowledge/ | |
| 7 | **eval/compare.py** — адаптировать под V7 метрики (correctness, faithfulness, false_sufficiency вместо completeness/abstain_rate) | ~1 ч | A/B сравнение прогонов из benchmarks/ | |
| 8 | **Incremental index management** — delete/upsert по источнику без полной переиндексации: `collection.delete(where={"source": X})` + add. Позволяет обновить один документ без сноса всей ChromaDB | ~2 ч | обновление документа за секунды вместо 10-15 мин | |
| 10 | **Domain gate включить в проде** — `V7_DOMAIN_GATE_THRESHOLD` сейчас 0.0 (выключен). Разобраться: измерить latency/cost overhead на embedding при включении (0.25), проверить false-positive rate на реальных запросах, решить нужен ли второй слой защиты поверх evaluate_triage keyword_overlap abstain | ~2 ч | экономия LLM-вызовов на OOS запросах |
| 9 | **Visual proof fix** — два подпункта: (а) `get_vision_llm()` хардкодит Gemini, который не доступен с VPS AEZA без прокси — переключить на OpenAI Vision (`gpt-4o` умеет vision) или добавить WARP proxy; (б) mode="show" кладёт `image_path` в passage но Streamlit UI его не рендерит — добавить отображение в app.py | ~3 ч | таблицы из PDF расшифровываются VLM; пользователь видит фрагмент PDF в ответе | |

---

## Выполнено ✅

| Улучшение | Дата | Эффект |
|-----------|------|--------|
| Retry при Gemini 503 (tenacity, 3 попытки) | 2026-05-08 | Стабы исчезли |
| Noise cleanup regex (URL, даты, watermarks) | 2026-05-16 | 1069 → 976 чанков, чище |
| FlashRank score inflation fix | 2026-05-16 | correctness 6.86 → 7.9 |
| Chunk overlap v2.4-sentence-overlap | 2026-05-25 | +continuity |
| BM25 guarantee top-3 per expanded query | 2026-05-25 | СОУТ-баг починен |
| V8 prefix fix (multi-query, evidence assess) | 2026-05-25 | env vars работают |
| HybridChunker v3.0-hybrid | 2026-05-26 | структурные чанки по пунктам НПА |
| Cross-reference expansion в bridge.py | 2026-05-26 | подтягивает связанные пункты |
| Q#11 fix: cross-ref limit 100→500, окно 24→30 | 2026-05-26 | программа В: раз в год ✓ |
| Eval 57 вопросов, OpenAI judge | 2026-05-26 | **0.80** overall |
