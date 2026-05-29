# Передовые техники RAG для оставшихся failure modes — ресёрч 2

**Дата:** 2026-05-29 (вечер)
**Контекст:** correctness 7.3 → потолок промпта. Нужны архитектурные подходы.

---

## 4 оставшиеся типа проблем

1. **Partial enumeration** — синтез ответа из 5+ чанков (Программа А, Документы при вредных, Комиссия НС)
2. **Wrong aspect retrieval** — vocabulary mismatch (Изменение технологии: запрос ≠ ключевые слова в правильном чанке)
3. **False-premise nuance** — модель путается в инвертированных утверждениях (СОУТ можно не)
4. **Multi-step process answers** — процедура из 9 этапов (Расследование НС)

---

## Top-3 рекомендации

### #1 — Contextual Retrieval (Anthropic, 2024)
**Что:** при индексации каждому чанку добавляется LLM-сгенерированный контекст ("Этот фрагмент из Постановления 2464, раздел про внеплановый инструктаж при изменениях производства").

**Цифры:** −49% failure rate (embeddings+BM25), −67% (embeddings+BM25+reranker). Anthropic.

**Стоимость:** однократно при индексации (~$1.02 за миллион токенов с prompt caching). Runtime — ноль.

**Сложность:** средняя — новая индексация + LLM-вызов на каждый чанк.

**Закрывает:**
- Проблему 2 (vocabulary mismatch) — chunk явно содержит "о чём он" в контексте
- Часть проблемы 1 (chunks становятся самодостаточными, проще агрегировать)

**Для compliance идеально:** заголовки секций уже в metadata. Делаем мини-промпт: "опиши 1 предложением что это за фрагмент в контексте документа" → prepend к тексту чанка перед embedding.

Источник: [anthropic.com/news/contextual-retrieval](https://anthropic.com/news/contextual-retrieval)

### #2 — Multi-query expansion с conditional CoVe
**Что:** 
- Для list-вопросов (триггер: "кто", "какие", "все") → LLM генерирует 4-5 запросов с разными "якорями", параллельный retrieval, dedup в 30-чанковый бюджет
- Для утверждений ("верно ли", "можно ли не") → Chain-of-Verification: draft → verify premise → revise

**Цифры:** RQ-RAG +1.9% single-hop, больше на multi-hop. CoVe x2 precision на list-QA (0.17→0.36 на Llama 65B). CoV-RAG +3.7-4 пункта NQ/TriviaQA.

**Стоимость:** +1-3 LLM вызова на сложные запросы, ноль на простые.

**Сложность:** низкая (multi-query) — средняя (CoVe-loop).

**Закрывает:** проблему 1 (enumeration), проблему 3 (false-premise).

Источники: arXiv:2404.00610 (RQ-RAG), aclanthology.org/2024.findings-acl.212 (CoVe).

### #3 — Iterative process retrieval (для complex path)
**Что:** на complex path добавить planner-шаг — модель сначала перечисляет ожидаемые этапы процедуры по структуре документа, затем для каждого этапа отдельный fetch, финальная сборка.

**Цифры:** RSP (Retrieve-Summarize-Plan) — пишет meaningful improvement на multi-hop.

**Стоимость:** +2 LLM вызова + N параллельных retrievals.

**Сложность:** средняя.

**Закрывает:** проблему 4 (process answers).

Источник: arXiv:2407.13101 (RSP), arXiv:2501.05366 (Search-o1).

---

## Что НЕ берём из ресёрча

- **Search-o1 / Agentic RAG full** — слишком тяжело для Flash-стека (3-7 LLM вызовов в цикле)
- **HyDE classic** — арXiv:2504.14175 (2025) показал knowledge leakage; для compliance риск hallucinated terms
- **Self-RAG** — требует fine-tuning, на Gemini Flash нельзя
- **DPA-RAG** — alignment retriever ↔ LLM preferences, требует training

---

## Спорные моменты

- **HyDE vs ReDE-RF** — последний (relevance feedback) надёжнее без leakage
- **Self-RAG vs Adaptive-RAG** — Adaptive-RAG (классификатор сложности) эффективнее по cost/accuracy на frozen LLM
- **Reranker** — Anthropic показал что reranker сам по себе даёт 49→67% улучшения. Если ещё не используется в production-style — отдельный win

---

## Полный список источников

- arXiv:2403.14403 — Adaptive-RAG (NAACL 2024)
- arXiv:2401.15884 — Corrective RAG (CRAG)
- arXiv:2310.11511 — Self-RAG
- arXiv:2310.06117 — Step-Back Prompting (DeepMind, ICLR 2024)
- arXiv:2305.06983 — FLARE (EMNLP 2023)
- arXiv:2212.10509 — IRCoT (ACL 2023)
- arXiv:2404.00610 — RQ-RAG
- arXiv:2410.04343 — IterDRAG (Google DeepMind)
- arXiv:2407.13101 — Retrieve-Summarize-Plan
- arXiv:2501.05366 — Search-o1
- arXiv:2502.05078 — Adaptive Graph of Thoughts
- arXiv:2410.08815 — StructRAG
- arXiv:2406.18676 — DPA-RAG
- aclanthology.org/2024.findings-acl.212 — Chain-of-Verification
- arXiv:2504.06438 — Premise Verification via RAG (2025)
- arXiv:2504.14175 — HyDE knowledge leakage critique
- anthropic.com/news/contextual-retrieval — Contextual Retrieval (Anthropic)
- arxiv.org/html/2410.21242 — ReDE-RF

---

## Применяется в

- В работе: пока ничего из этого. Решаем приоритет с Петром.
