# Regulatory Compliance RAG

**RAG-пайплайн для российских нормативных документов (ГОСТ, СНиП, ТК РФ, правила по пожарной безопасности и охране труда) — отвечает на вопросы с указанием источника или явно отказывается от ответа при недостаточной уверенности.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![CI](https://github.com/spqr-86/regulatory-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/spqr-86/regulatory-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Качество ответов** (56-вопросный golden set, судья `gpt-4o`): in-scope correctness **7.56 / 10** · faithfulness **0.840** · answer relevance **0.847** · отказ на OOS **1.00** · false-sufficiency **4.8%** · complex-путь **20.8%** · **~$0.0045/запрос**, p50 **5.2 с**.

> Метрики зависят от судьи — канонические значения в [docs/reference/FACTS.md](./docs/reference/FACTS.md). Архитектурные решения описаны в [docs/explanation/design-decisions.md](./docs/explanation/design-decisions.md).

[English README →](./README.md)

---

## Проблема

Нормативные документы в промышленных отраслях (охрана труда, пожарная безопасность, строительство) — это сотни PDF с перекрёстными ссылками. Ручной поиск медленный и ненадёжный. Галлюцинированный ответ на вопрос по нормативам — это не UX-проблема, а прямой риск.

Проект исследует, насколько RAG + детерминированные guardrails решают задачу надёжного Q&A по нормативной базе.

---

## Как работает

```
Запрос пользователя
    ↓
intent_gate          — regex-фильтр шума + (опционально) cosine-to-centroid OOS-гейт, до retrieval
    ↓
router               — план запроса + расширение глоссарием + multi-query (RRF-слияние)
    ↓
rag_simple           — гибридный retrieval (BM25 + векторы, top-12) + CrossEncoder rerank
    ↓
evaluate_triage      — детерминированный гейт достаточности (без LLM-оценки)
    ├── sufficient    → generate_answer
    └── insufficient  → rag_complex (top-60 + MMR) → evaluate_complex
                            ├── pass  → generate_answer
                            └── fail  → abstain (явный отказ)
```

Ключевые архитектурные решения:
- **Нет LLM-роутинга** — все ветвления используют детерминированные пороги по score
- **Abstain лучше галлюцинации** — система отказывается отвечать при низкой уверенности retrieval
- **Двухэтапный retrieval** — быстрый путь обрабатывает большинство запросов; медленный активируется только при необходимости

В проде триаж работает по варианту V8 (evidence-assess: score реранкера + покрытие);
легаси hard-gate остаётся доступен за флагом. См.
[docs/explanation/triage.md](./docs/explanation/triage.md).

📖 **Документация:** [архитектура](./docs/explanation/architecture.md) · [проектные решения](./docs/explanation/design-decisions.md) · [FACTS](./docs/reference/FACTS.md) · [полная документация](./docs/README.md)

---

## Метрики

| Метрика | Значение |
|---|---|
| In-scope correctness | 7.56 / 10 |
| Correctness (все вопросы) | 7.30 / 10 |
| Faithfulness | 0.840 |
| Answer relevance | 0.847 |
| Отказ на OOS-запросах | 1.00 |
| False-sufficiency rate | 4.8% |
| Доля complex-пути | 20.8% |
| Латентность p50 / p95 / mean | 5.2 / 19.1 / 7.2 с |
| Стоимость запроса | $0.0045 ($0.24 / прогон) |
| Retrieval HR@5 / HR@12 / MRR (hybrid, 90 вопросов практиков) | 0.63 / 0.81 / 0.50 |

Eval: 56-вопросный golden dataset (`tests/dataset.csv`), `eval/run_v7_eval.py`, LLM-судья
`gpt-4o`. Числа зависят от судьи — сравнивать прогоны только под одним судьёй. Retrieval
Hit Rate / MRR меряются отдельно на наборе из 90 реальных вопросов, взятых дословно
с форумов специалистов по ОТ/ПБ, оставлены только те, на которые в корпусе есть ответ;
для тюнинга набор не использовался (`eval/run_retrieval_eval.py`), см.
[docs/roadmap.md](./docs/roadmap.md). Канонические значения:
[docs/reference/FACTS.md](./docs/reference/FACTS.md).

---

## Быстрый старт

```bash
git clone https://github.com/spqr-86/regulatory-rag.git
cd regulatory-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # добавить OPENAI_API_KEY (по умолчанию LLM + embeddings)
```

Положите PDF/DOCX нормативных документов в `source_docs/`, затем:

```bash
python index.py                              # индексировать → ChromaDB (деструктивно: сносит коллекцию)
streamlit run app.py --server.port 8502      # UI на http://localhost:8502
uvicorn api:app --port 8503                   # REST API на http://localhost:8503/docs
```

По умолчанию: OpenAI (LLM + embeddings) + ChromaDB. Раздел [Замена бэкенда](#замена-бэкенда) — переключение через `.env`.

Опциональный стек мониторинга (Postgres для событий запросов, Grafana сверху):

```bash
cp .env.example .env              # задать POSTGRES_PASSWORD и GF_SECURITY_ADMIN_PASSWORD
docker compose up -d              # оба сервиса healthy; схема применяется db/migrations на пустом томе
V7_TELEMETRY_WRITER=postgres      # в .env: писать события в стек вместо JSONL
```

Детали, дашборд и что делать, если стек не поднялся:
[docs/how-to/run-monitoring-stack.md](./docs/how-to/run-monitoring-stack.md).

---

## Архитектура

```mermaid
flowchart TD
    subgraph Ingestion [Индексация]
        Docs[PDF / DOCX] --> Docling[Docling Parser]
        Docling --> Split[HybridChunker max_tokens=400, merge_peers]
        Split --> Embed[OpenAI Embeddings]
        Embed --> DB[(ChromaDB)]
    end

    subgraph V7 [V7 LangGraph Pipeline]
        Q[Запрос] --> Gate{intent_gate + domain gate}
        Gate -->|шум / out-of-scope| End[END / abstain]
        Gate -->|in-domain| Router[router + глоссарий + multi-query]
        Router --> Simple[rag_simple hybrid top-12 + CrossEncoder]
        Simple --> Triage{evaluate_triage evidence-assess}
        Triage -->|sufficient| Gen[generate_answer]
        Triage -->|insufficient| Complex[rag_complex top-60 + MMR]
        Complex --> Eval[evaluate_complex]
        Eval -->|pass| Gen
        Eval -->|fail| Abstain[abstain]
        Gen --> Answer[Ответ + источники]
    end
```

Поставляемый индекс — 12 нормативных документов (~7,8k чанков). Это пример корпуса,
используйте свои документы. Точные числа: [docs/reference/FACTS.md](./docs/reference/FACTS.md#corpus).

---

## REST API

Запуск FastAPI-бэкенда вместе со Streamlit:

```bash
uvicorn api:app --port 8503
```

**`POST /query`** — основной RAG-пайплайн

```bash
curl -X POST http://localhost:8503/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Как часто проводится повторный инструктаж?"}'
```

```json
{
  "answer": "Повторный инструктаж проводится не реже одного раза в 6 месяцев...",
  "passages": [{"text": "...", "source": "2464.pdf", "score": 0.91}],
  "path": "rag_simple → evaluate_triage → generate_answer → END",
  "elapsed_sec": 4.2
}
```

**`POST /retrieve`** — только retrieval (гибридный поиск, без LLM); **`GET /corpus`** —
проиндексированные документы; **`GET /health`** — liveness (`{"status": "ok"}`).

Полный справочник со схемами и rate limits: [docs/reference/api.md](./docs/reference/api.md).
Интерактивная документация: `http://localhost:8503/docs`.

---

## Стек

| Слой | Технология |
|------|-----------|
| Оркестрация | LangGraph (V7 детерминированный граф) |
| LLM | OpenAI (по умолчанию), Gemini, DeepSeek — настраивается через `SIMPLE/COMPLEX_LLM_PROVIDER` в `.env` |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| Переранжирование | CrossEncoder (sentence-transformers); FlashRank выбирается через `RERANKER_BACKEND` |
| ETL | Docling (PDF/DOCX → чанки) |
| Оценка | кастомный LLM-as-judge (faithfulness, answer relevance, correctness) + IR-метрики (Hit Rate@k, MRR) |
| Мониторинг | Postgres + Grafana (docker compose), события пишутся изнутри графа |
| UI | Streamlit |

---

## Замена бэкенда

LLM и vector store доступны через фабричные слои (`src/infra/llm_factory.py`, `src/backends/`). Добавление нового провайдера — одна функция и одна запись в реестре, код пайплайна не меняется.

| Слой | Реализовано | Настройка через | Roadmap |
|------|-------------|-----------------|---------|
| LLM   | OpenAI, Gemini, DeepSeek | `SIMPLE_LLM_PROVIDER` / `COMPLEX_LLM_PROVIDER` | Anthropic |
| Vector store | Chroma | `VECTOR_STORE` | Qdrant, pgvector |
| Embeddings | OpenAI, local (sentence-transformers), hf_api | `EMBEDDING_PROVIDER` | — |

**Локальные embeddings** (LLM всё равно через API):
```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_NAME=ai-forever/sbert_large_nlu_ru
```

**Добавление нового LLM-провайдера** (пример: Anthropic):
1. Добавить `_create_anthropic_llm(**kwargs)` в `src/infra/llm_factory.py`
2. Зарегистрировать в `_LLM_PROVIDERS = {..., "anthropic": _create_anthropic_llm}`
3. Установить `SIMPLE_LLM_PROVIDER=anthropic` (и/или `COMPLEX_LLM_PROVIDER`) в `.env`

Аналогичная схема для vector store — реализовать протокол `VectorStoreBackend` в `src/backends/`, зарегистрировать в фабрике.

---

## Адаптация под свой домен

Система настроена под российские нормативные документы, но доменные знания изолированы и легко заменяются.

**Глоссарий терминов** (`config/term_glossary.yaml`) — маппинг неформальных аббревиатур на официальные полные названия, чтобы BM25 и векторный поиск находили проиндексированный текст. Расширение:

```yaml
terms:
  "ваша аббревиатура":
    official: "Полное официальное название из ваших документов"
    source: "Ссылка на норматив (необязательно)"
```

Изменений в коде не нужно — правьте YAML и перезапускайте.

**Промпты** (`prompts/`) — Jinja2-шаблоны через `PromptManager`, версионированные. Переключение активной версии через `prompts/registry.yaml`.

**Корпус** — положите PDF в `source_docs/` и запустите `python index.py`. Чанкер и embeddings языконезависимы.

---

## Статус проекта

- ✅ V7 LangGraph-пайплайн — все ноды, детерминированный роутинг (verifier/rewriter убраны — insufficient triage ведёт сразу в rag_complex)
- ✅ Гибридный retrieval — BM25 + семантический, двухэтапный (simple/complex path)
- ✅ Детерминированный гейт достаточности — по score, без LLM в роутинге; в проде V8 evidence-assess
- ✅ Структурированный triage gap — триаж отдаёт типизированный пробел и закрывает его дозапросом в хвост до эскалации (issue #13)
- ✅ Domain gate — опциональный pre-retrieval OOS-фильтр через cosine similarity к центроиду корпуса
- ✅ HybridChunker — структурно-ориентированный чанкинг по разделам/статьям документов
- ✅ Контекстное embedding — заголовок родительского раздела добавляется к вектору чанка
- ✅ Раскрытие перекрёстных ссылок — автоматически подтягивает упомянутые пункты (напр., «пункт 46») из того же источника
- ✅ Multi-query расширение — LLM генерирует варианты запроса, слияние через RRF
- ✅ Версионированные промпты — Jinja2-шаблоны, реестр сокращён до 3 активных семейств; `generate_answer` v8 (anti-sycophancy + value↔condition)
- ✅ Offline eval — golden dataset + тест-набор для retrieval из 90 вопросов практиков, цена и латентность на запрос
- ✅ Онлайн-мониторинг — каждый запрос строкой в Postgres (цена, латентность, маршрут, токены, `source`), дашборд Grafana, 👍/👎 под ответом; весь стек — один `docker compose up`
- ✅ Задеплоен на VPS (порт 8502, Streamlit)
- 🔄 Устойчивость value↔condition на запросах с несколькими значениями
- 🔄 Переписка нарезки под таблицы и заголовки разделов (модули 06/07)

---

**Автор:** Пётр Балдаев — [LinkedIn](https://linkedin.com/in/petr-baldaev-b1252b263/) · [GitHub](https://github.com/spqr-86)

[Changelog →](./CHANGELOG.md)
