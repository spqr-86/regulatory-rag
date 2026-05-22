# 🛡️ Regulatory Compliance Q&A — RAG / Multi-Agent

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Test](https://github.com/spqr-86/safety-incident-analyzer/actions/workflows/evaluation.yml/badge.svg)](https://github.com/spqr-86/safety-incident-analyzer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🤔 Что это?

**Regulatory Compliance Q&A** — RAG-платформа для мгновенных ответов по нормативной документации (СНиП, ГОСТ, СП) и корпоративным регламентам.

Загружаете PDF/DOCX документы — специалисты задают вопросы на естественном языке и получают точный ответ со ссылкой на источник за секунды вместо часов поиска. Подходит для любых нормативных и корпоративных документов: охрана труда, пожарная безопасность, промышленная безопасность, внутренние регламенты.

---

## 💡 Зачем?

В крупных организациях с сетью объектов у каждого ответственного специалиста постоянный поток однотипных вопросов: «что делать в этой ситуации по регламенту?», «какие требования применяются?», «где это написано?». Поиск вручную занимает часы — особенно когда документов много и они сложно структурированы.

Система решает эту задачу: специалист задаёт вопрос в чате, агент находит релевантные фрагменты, верифицирует ответ и возвращает точную цитату с указанием источника.

**Ключевые возможности:**

*   ✅ 🔎 **Гибридный поиск**: комбинация семантического поиска (векторы) и BM25. Two-Stage Retrieval: Stage 1 — быстрый широкий поиск (top-60 кандидатов), Stage 2 — FlashRank Cross-Encoder реранкинг (оценивает каждую пару Query+Document → отбирает лучшие).
*   ✅ 🧠 **V7 LangGraph Pipeline**: модульный граф rag_simple → rag_complex → evaluate_complex → generate_answer с детерминированными hard gates.
*   ✅ ⚖️ **Нормативная точность**: строгая фильтрация — ответ только по подтверждённым фрагментам, abstain при недостаточной уверенности.
*   ✅ 🛡️ **Guardrails**: `intent_gate` (input filter — отсекает шум и OOS до retrieval) + детерминированные hard gates (output validation — score-пороги без LLM) + abstain при низкой уверенности. Нормативный вывод = высокая цена ошибки.
*   ✅ 📄 **Универсальная загрузка**: PDF/DOCX через Docling → chunking → OpenAI embeddings → ChromaDB.
*   ✅ 📖 **Доменный глоссарий**: детерминированное расширение запросов — "программа А" → официальный термин перед retrieval.
*   ✅ 📊 **Eval framework**: golden-датасет + LLM-as-judge метрики (faithfulness, correctness, answer relevance) через `eval/run_v7_eval.py`.
*   ✅ 📚 **ГОСТ RAG**: 108 нормативных документов (ГОСТы, СНиПы) проиндексированы отдельно (9 344 чанка, ChromaDB `wta_gosts`). API эндпоинт `POST /query/gosts` для поиска по нормативной базе.

---

## 🧠 Ключевые технические решения

### V7 LangGraph Pipeline (основной)

Модульный детерминированный граф без LLM-роутинга:

```
query → intent_gate → router → rag_simple → evaluate_triage ─┬─ sufficient ──→ generate_answer
                                    ↑                        ├─ borderline ──→ llm_verifier ─┬─→ generate_answer
                                    └──── rewriter ←──────────┘                              ├─→ rewriter
                                                              └─ clearly_bad ─→ rag_complex ←┘
                                                  rag_complex → evaluate_complex ─┬─ pass → generate_answer
                                                                                  └─ fail → abstain
```

1.  **`intent_gate`**: regex-классификация noise/domain (+ опциональный domain gate). noise → END.
2.  **`router`**: классификация запроса, построение `plan`, расширение `active_query` через глоссарий.
3.  **`rag_simple`** (fast path — быстрый путь): hybrid retrieval (`SIMPLE_TOP_K=12`) + FlashRank rerank. Обрабатывает большинство запросов за минимальное время.
4.  **`evaluate_triage`**: детерминированные hard gates → sufficient / borderline (→ `llm_verifier`) / clearly_bad (→ `rag_complex`).
5.  **`rag_complex`** (slow path — глубокий путь): расширенный поиск (`COMPLEX_TOP_K=60`) + rerank + MMR-диверсификация, merge всех попыток (top 24). Запускается только когда fast path не нашёл достаточно данных.
6.  **`evaluate_complex`**: hard gates по score-порогам, без LLM-вердиктов.
7.  **`generate_answer`**: синтез ответа через Gemini (thinking_budget=4096). Retry при 503, fallback — сырые чанки.
8.  **Доменный глоссарий** (`src/glossary.py` + `config/term_glossary.yaml`): расширение запросов в ноде `router`.

> V7-промпты (генерация, верификация, rewrite) — хардкод-строки в `src/v7/bridge.py`. Jinja2-реестр промптов (`prompts/`) используется только легаси-путём `multiagent_rag`.

### 📊 Целевые метрики

| Метрика | Целевое значение | Описание |
| :--- | :--- | :--- |
| **Correctness** | > 7.5/10 | Смысловое соответствие эталону |
| **Faithfulness** | > 0.85 | Отсутствие галлюцинаций |
| **Answer Relevance** | > 0.85 | Соответствие ответа вопросу |
| **False-sufficiency** | < 10% | Доля simple-path ответов с низкой correctness |

---

## 🚀 Как быстро запустить?

### Локальный запуск

**Предварительные требования:**
- Python 3.11+
- API ключи (OpenAI, Gemini)

**1. Клонирование и установка:**
```bash
git clone https://github.com/spqr-86/safety-incident-analyzer.git
cd safety-incident-analyzer
pip install -r requirements.txt
```

**2. Настройка окружения:**
Создайте файл `.env` в корне проекта (см. `.env.example`):
```env
# LLM Provider
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_gemini_key

# Embeddings
EMBEDDING_PROVIDER=openai # или hf_api, local
```

**3. Индексация документов:**
Положите документы (PDF, DOCX, MD) в папку `source_docs/` и запустите:
```bash
python index.py
```

**4. Запуск приложения:**
```bash
streamlit run app.py
```

---

## 📱 Как использовать?

1.  Откройте браузер по адресу `http://localhost:8501`.
2.  В чате введите ваш вопрос по охране труда (например, "Какие требования к высоте перил?").
3.  Система выполнит поиск, проверит релевантность и сформирует ответ со ссылками на источники.
4.  При добавлении новых файлов используйте кнопку "Переиндексировать библиотеку" в боковой панели.

---

## 📚 Куда идти дальше?

### 🚀 Для пользователей
- [**Quick Start**](./docs/guides/quick-start.md) — установка и первый запуск.

### 📊 Для аналитиков
- [**Система оценки и метрики**](./docs/evaluation/README.md) — Eval Framework (`eval/run_v7_eval.py`).
- [**Benchmarks и Baseline**](./benchmarks/README.md) — результаты тестирования производительности.

### 🛠 Для разработчиков
- [**Архитектура и анализ кодовой базы**](./docs/architecture/README.md)
- [**Data Pipeline (Обработка данных)**](./docs/DATA_PIPELINE.md) — **NEW!** Детальное описание процесса индексации.
- [**Гайд по тестированию**](./docs/guides/testing.md)
- [**Управление промптами**](./docs/guides/prompt-management.md) — руководство по работе с системой промптов.
- [**Как добавлять вопросы в датасет**](./docs/guides/adding-questions.md)

---

## 🏗 Архитектура

```mermaid
flowchart TD
    subgraph Ingestion [Индексация]
        Docs[PDF / DOCX] --> Docling[Docling Parser]
        Docling --> Split[Chunking\n1500 chars / 400 overlap]
        Split --> Embed[OpenAI Embeddings]
        Embed --> DB[(ChromaDB)]
    end

    subgraph V7 [V7 LangGraph Pipeline - основной]
        Q[Вопрос] --> Gate{intent_gate}
        Gate -->|noise| End[Конец]
        Gate -->|domain| Router[router\nplan + глоссарий]
        Router --> Simple[rag_simple\nhybrid SIMPLE_TOP_K=12 + FlashRank]
        Simple --> Triage{evaluate_triage\nhard gates}
        Triage -->|sufficient| Gen[generate_answer\nGemini]
        Triage -->|borderline| Verifier[llm_verifier]
        Triage -->|clearly_bad| Complex[rag_complex\nCOMPLEX_TOP_K=60 + MMR]
        Verifier -->|ok| Gen
        Verifier -->|rewrite| Rewriter[rewriter] --> Simple
        Verifier -->|escalate| Complex
        Complex --> Eval[evaluate_complex\nhard gates]
        Eval -->|pass| Gen
        Eval -->|fail| Abstain[abstain]
        Gen --> Answer[Ответ + источники]
    end
```

---

## 🛠 Технологии

| Категория | Технологии |
| :--- | :--- |
| **Язык** | Python 3.11+ |
| **UI** | Streamlit |
| **LLM Framework** | LangChain, LangGraph |
| **LLM Providers** | Google Gemini 2.5 Flash (generation), DeepSeek V3 (GOST RAG) |
| **Embeddings** | OpenAI text-embedding-3-small |
| **Vector Store** | ChromaDB |
| **ETL** | Docling |
| **Reranking** | FlashRank |
| **Evaluation** | Ragas, Custom metrics |

---

## 📈 Статус проекта

*   ✅ V7 LangGraph Pipeline (intent_gate → router → rag_simple → evaluate_triage → rag_complex → generate_answer)
*   ✅ Hybrid Retrieval (ChromaDB + BM25, `SIMPLE_TOP_K=12` / `COMPLEX_TOP_K=60`, FlashRank rerank)
*   ✅ Hard gates с детерминированными score-порогами (без LLM-роутинга)
*   ✅ Генерация ответов — Gemini (thinking_budget=4096), retry при 503
*   ✅ Term Glossary — расшифровка доменных аббревиатур, применяется в ноде `router`
*   ✅ Индексация PDF/DOCX через Docling, **12 нормативных документов** (добавлен полный ТК РФ), **1973 чанка** (v2.3-noise-clean)
*   ✅ Веб-интерфейс Streamlit, задеплоен на VPS (порт 8502)
*   ✅ Eval framework — `eval/run_v7_eval.py`, golden-датасет 50 вопросов, LLM-as-judge
*   ✅ Баг чанкинга исправлен (`MIN_BBOX_HEIGHT` не роняет текст, 830 → 1069 чанков) — 2026-05-15
*   ✅ Regex noise cleanup (`_clean_noise`): URL-watermarks, page markers, timestamps — PIPELINE_VERSION v2.3-noise-clean — 2026-05-16
*   ✅ FlashRank score inflation fix: `vector_score` сохраняется отдельно, MMR и threshold gates используют его — 2026-05-16
*   ✅ Eval correctness **7.9/10** (цель 7.5 достигнута) — 2026-05-16
*   ✅ **ГОСТ RAG**: 108 документов → 9 344 чанка, коллекция `wta_gosts`, API `POST /query/gosts`, генерация через DeepSeek — 2026-05-17
*   🔄 Расширение тестового датасета

---

**Автор:** Петр Балдаев (AI/ML Engineer)
[LinkedIn](https://linkedin.com/in/petr-baldaev-b1252b263/) • [GitHub](https://github.com/spqr-86)
