# Обзор архитектуры

Regulatory RAG — система retrieval-augmented generation для российских нормативных документов (ГОСТ, ТК РФ, СНиП, СП). Отвечает на вопросы о соответствии нормативным требованиям: извлекает релевантные фрагменты из проиндексированного корпуса и синтезирует ответ с помощью Gemini.

См. также: [Как работает пайплайн V7](./v7-how-it-works_RU.md) · [Как работает triage](./triage-how-it-works_RU.md)

---

## Пайплайн

| Шаг | Нода | Что делает |
|------|------|--------------|
| 1 | `intent_gate` | Regex-фильтр: шумовые запросы → END, нормативные → продолжение |
| 2 | `router` | Классифицирует запрос, строит план, расширяет `active_query` через глоссарий терминов |
| 3 | `rag_simple` | Гибридное извлечение (векторный + BM25, RRF-слияние), FlashRank rerank, топ-12 фрагментов |
| 4 | `evaluate_triage` | Hard gates → `sufficient` / `borderline` / `clearly_bad` |
| 5a | `llm_verifier` | (только borderline) LLM решает: sufficient / rewrite / escalate |
| 5b | `rag_complex` | (clearly_bad / escalated) Глубокое извлечение топ-60 + MMR, объединяет все попытки |
| 6 | `evaluate_complex` | Hard gates на объединённых фрагментах; провал → abstain |
| 7 | `generate_answer` | Gemini синтезирует ответ из до 24 финальных фрагментов |

---

## Карта кодовой базы

| Путь | Содержимое |
|------|----------|
| `src/v7/` | Ноды пайплайна (`nodes/`), сборка графа (`graph.py`), типы состояния, hard gates, NLP core, bridge DI адаптер |
| `src/infra/` | LLM factory, prompt manager, semantic cache, парсеры, общие типы |
| `src/indexing/` | Процессор документов (HybridChunker), Chroma helpers, vector store, applicability retriever |
| `src/backends/` | Протокол `VectorStoreBackend` + реализация ChromaDB |
| `config/` | `settings.py` (pydantic-settings), `term_glossary.yaml` |
| `prompts/` | Jinja2 шаблоны + `registry.yaml` |
| `eval/` | `run_v7_eval.py`, модули метрик, золотые датасеты |
| `scripts/` | `trace_v7.py` (E2E smoke test), `validate_prompts.py`, `measure_cps.py` |
| `tests/` | Юнит и интеграционные тесты (`pytest -m unit`) |

---

## Добавление новой ноды

1. Создать `src/v7/nodes/<name>.py` — тонкий оркестратор: читает состояние → вызывает функцию → записывает состояние. Логику размещать в `nlp_core.py` или `hard_gates.py`, не в самой ноде.
2. Зарегистрировать ноду в `src/v7/graph.py` (`graph.add_node`, `graph.add_edge`).
3. Написать юнит-тесты в `tests/test_<name>.py`, используя `unittest.mock` для DI-зависимостей.

---

## Дополнительно

- [v7-how-it-works_RU.md](./v7-how-it-works_RU.md) — подробное описание каждой ноды, hard gates и калибровки порогов
- [triage-how-it-works_RU.md](./triage-how-it-works_RU.md) — глубокое погружение в метрики `evaluate_triage` и 3-way routing
