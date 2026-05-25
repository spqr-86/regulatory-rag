# Паспорт проекта: AI Safety Compliance Assistant

**Назначение:** Автоматизация поиска по нормативной документации (ГОСТ, СНиП, СП, ТК РФ).
**Стек:** LangGraph, Streamlit, FastAPI, ChromaDB, Docling, Gemini 2.5 Flash / Gemini 3 Flash, OpenAI embeddings.

## Особенности архитектуры
- **V7 LangGraph Pipeline (основной):** детерминированный граф состояний без LLM-роутинга —
  `intent_gate → router → rag_simple → evaluate_triage → rag_complex → generate_answer`.
  Hard gates по числовым порогам, явный `abstain` при недостатке данных.
- **Two-tier LLM:** Gemini 2.5 Flash на простом пути (~5s), Gemini 3 Flash с thinking_budget=4096 на сложном.
- **Hybrid retrieval:** векторный поиск (ChromaDB) + BM25, RRF-слияние, FlashRank reranking, MMR.
- **Pluggable backends:** LLM и vector store абстрагированы через фабрики (`src/infra/llm_factory.py`, `src/backends/`).
- **REST API:** FastAPI на порту 8503 — `POST /query`, `POST /query/gosts`, `GET /health`.

## Результаты
- Eval через `eval/run_v7_eval.py` (50 вопросов, LLM-as-judge Gemini 2.5 Flash):
  - Correctness: **7.9/10** (цель 7.5 ✅)
  - Faithfulness: **0.988**
  - False-sufficiency: 0.15
  - Cost per query: **$0.0102** (baseline May 2026)
- Corpus: 12 документов → **1 973 чанка** (после фикса bbox-бага v2.3-noise-clean)

## Вызовы (Challenges)
- False-sufficiency 15% — часть запросов некорректно классифицируется как sufficient на простом пути.
- Run-to-run вариативность LLM-судьи в eval (~±0.3 correctness).

## Моя роль (как разработчика)
- Проектирование и реализация V7-графа на LangGraph (тонкие ноды + hard gates).
- Hybrid retrieval, RRF merge, MMR, FlashRank reranking.
- Доменный глоссарий (морфологический матчинг, 0 латентности).
- Eval-фреймворк с LLM-as-judge метриками.
- Security hardening: slowapi, pickle → JSON, threading.RLock.
- src/ restructure: infra/, indexing/, backends/ (Board 1–5).
