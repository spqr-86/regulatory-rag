# Документация

Документация системы Regulatory RAG — системы retrieval-augmented Q&A для российских нормативных документов.

## Содержание

| Файл | Описание |
|------|--------|
| [architecture/README_RU.md](./architecture/README_RU.md) | Обзор пайплайна, карта кодовой базы, как добавить ноду |
| [architecture/v7-how-it-works_RU.md](./architecture/v7-how-it-works_RU.md) | Подробное описание нод V7, hard gates, калибровка порогов |
| [architecture/triage-how-it-works_RU.md](./architecture/triage-how-it-works_RU.md) | Как работает `evaluate_triage`: метрики, 3-way routing, обоснование порогов |
| [guides/quick-start_RU.md](./guides/quick-start_RU.md) | Установка, настройка, индексация, запуск UI и API |
| [guides/testing_RU.md](./guides/testing_RU.md) | Как проверять документацию и запускать тесты |
| [guides/prompt-management_RU.md](./guides/prompt-management_RU.md) | Jinja2 реестр промптов, версионирование, A/B тестирование |
| [guides/adding-questions_RU.md](./guides/adding-questions_RU.md) | Как добавлять вопросы в датасет для eval |
| [evaluation/README_RU.md](./evaluation/README_RU.md) | Фреймворк оценки, метрики, LLM-as-judge, формат отчёта |
| [DATA_PIPELINE_RU.md](./DATA_PIPELINE_RU.md) | Как индексируются документы: Docling, HybridChunker, embeddings, ChromaDB |
| [passport.md](./passport.md) | Паспорт проекта: стек, результаты eval, архитектурные тезисы |
| [plans/backlog.md](./plans/backlog.md) | Активный бэклог и известные проблемы |
