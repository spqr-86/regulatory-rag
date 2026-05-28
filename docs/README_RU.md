# Документация

Документация системы Regulatory RAG — системы retrieval-augmented Q&A для российских нормативных документов.

## Содержание

| Файл | Описание |
|------|--------|
| [architecture/README.md](./architecture/README.md) | Обзор пайплайна, карта кодовой базы, как добавить ноду |
| [architecture/v7-how-it-works.md](./architecture/v7-how-it-works.md) | Подробное описание нод V7, hard gates, калибровка порогов |
| [architecture/triage-how-it-works.md](./architecture/triage-how-it-works.md) | Как работает `evaluate_triage`: метрики, 3-way routing, обоснование порогов |
| [guides/quick-start.md](./guides/quick-start.md) | Установка, настройка, индексация, запуск UI и API |
| [guides/testing.md](./guides/testing.md) | Как проверять документацию и запускать тесты |
| [guides/prompt-management.md](./guides/prompt-management.md) | Jinja2 реестр промптов, версионирование, A/B тестирование |
| [guides/adding-questions.md](./guides/adding-questions.md) | Как добавлять вопросы в датасет для eval |
| [evaluation/README.md](./evaluation/README.md) | Фреймворк оценки, метрики, LLM-as-judge, формат отчёта |
| [DATA_PIPELINE.md](./DATA_PIPELINE.md) | Как индексируются документы: Docling, HybridChunker, embeddings, ChromaDB |
| [passport.md](./passport.md) | Паспорт проекта: стек, результаты eval, архитектурные тезисы |
| [plans/backlog.md](./plans/backlog.md) | Активный бэклог и известные проблемы |
