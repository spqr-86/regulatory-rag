# V7 Migration Design — Modular Architecture

> **For Claude:** This is the **design document** for migrating to RAG v7. Use `superpowers:writing-plans` to create step-by-step implementation plans for each Этап. Use `superpowers:executing-plans` to implement.

**Date:** 2026-02-16
**Status:** Draft
**Source spec:** `docs/feature/migration-v7` (1729 lines, полная спецификация v7)

---

## 1. Цель

Перенести текущую реализацию (`agents/multiagent_rag.py`) на архитектуру v7 с:
- Модульной структурой (каждый модуль тестируется изолированно)
- Hard gates + soft signals + 3-way triage
- BM25Okapi с pymorphy3 лемматизацией
- RRF merge, MMR select
- Anti-injection защитой
- Externalized конфигурацией (все пороги через .env)

**Стратегия:** Поэтапная замена (incremental). Текущий граф работает параллельно, пока v7 не готов.

---

## 2. Модули

### 2.0 `state_types` — Контракты данных
**Файл:** `src/v7/state_types.py`
**Зависимости:** —

Все TypedDict/dataclass, используемые остальными модулями:

| Тип | Назначение |
|-----|-----------|
| `RAGState` | Полное состояние графа (input/internal/output секции) |
| `RetrievalPlan` | План поиска: стратегия, фильтры, top_k |
| `RetrievalAttempt` | Результат одной попытки поиска: passages, metrics, attempt_plan |
| `HardGateResult` | Результат жёстких проверок: passed, reasons, scores |
| `SufficiencyResult` | 3-way triage: sufficient / borderline / clearly_bad |

**Контракт RAGState (секции):**
```
[INPUT]      query, filters                         — immutable, set by caller
[INTERNAL]   intent, plan, retrieval_id, active_query, retrieval_attempts,
             sufficient, verify_iteration, verification
[OUTPUT]     final_passages, final_score, fallback_passages, fallback_score,
             clarify_message, abstain_reason, sufficiency_details
```

**Важно:** `active_query` — рабочий запрос (может меняться после rewrite). `query` — оригинал, immutable. Оба используются для dual keyword overlap.

**Реализация — что использовать:**
- **State:** `TypedDict` из `typing` — как в LangGraph. Reducer для списков: `Annotated[list[...], operator.add]`
- **Dataclass'ы:** `@dataclass` из `dataclasses` — для `RetrievalPlan`, `RetrievalAttempt`, `HardGateResult`, `SufficiencyResult`
- **Валидация типов:** аннотации + `pydantic` при необходимости, но для внутренних типов достаточно `dataclass`

### 2.1 `config_v7` — Конфигурация
**Файл:** `src/v7/config.py`
**Зависимости:** —

Все пороги и лимиты v7 через pydantic-settings с env-префиксом `V7_`:

```python
class V7Config(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="V7_", env_file=".env", extra="ignore")

    # Hard gates
    HARD_GATE_THRESHOLD: float = 0.65       # Мин. score для прохождения hard gate
    TRIAGE_SOFT_THRESHOLD: float = 0.40     # Мин. score для soft signal (borderline)
    COVERAGE_DROP_PCT: float = 0.30         # Макс. падение покрытия при rewrite

    # Retrieval
    RRF_K: int = 60                         # Параметр k для Reciprocal Rank Fusion
    MMR_LAMBDA: float = 0.7                 # Баланс relevance/diversity в MMR
    BM25_TOP_K: int = 20                    # Кол-во результатов BM25
    SEMANTIC_TOP_K: int = 20                # Кол-во результатов semantic search

    # Keyword overlap (dual)
    MIN_KEYWORD_OVERLAP_ACTIVE: float = 0.3   # Мин. overlap для active_query
    MIN_KEYWORD_OVERLAP_ORIGINAL: float = 0.2 # Мин. overlap для original query

    # LLM & Limits
    MAX_REWRITE_ATTEMPTS: int = 2           # Макс. итераций rewrite loop
    MAX_CHUNKS_FOR_LLM: int = 10            # Макс. чанков в LLM контекст
    VERIFIER_CONFIDENCE_ANCHOR: float = 0.7 # Якорь confidence для верификатора

    # Anti-injection
    MAX_INPUT_LENGTH: int = 2000            # Макс. длина пользовательского запроса
    BLOCKED_PATTERNS: list[str] = [         # Паттерны для блокировки
        "ignore previous", "system prompt", "you are now"
    ]
```

**Использование:** `from src.v7.config import v7_config` — синглтон, как текущий `settings`.

**Реализация — что использовать:**
- **Конфигурация:** `pydantic_settings.BaseSettings` — уже используется в проекте (`config/settings.py`), тот же паттерн
- **Env-файл:** `.env` — единый с текущим, префикс `V7_` изолирует от существующих переменных
- **Синглтон:** `v7_config = V7Config()` на уровне модуля — как текущий `settings`

### 2.2 `nlp_core` — NLP-утилиты
**Файл:** `src/v7/nlp_core.py`
**Зависимости:** `state_types`, `config_v7`

| Функция | Описание |
|---------|----------|
| `extract_keywords(text) -> list[str]` | pymorphy3 + razdel лемматизация, стоп-слова |
| `BM25Index` | Класс: `build(docs)`, `query(tokens, top_k) -> list[ScoredDoc]` на rank_bm25.BM25Okapi |
| `rrf_merge(rankings, k) -> list[ScoredDoc]` | Reciprocal Rank Fusion нескольких ранжирований |
| `mmr_select(docs, query_emb, lambda_, k) -> list[Doc]` | Maximal Marginal Relevance для diversity |

**Примечание по mmr_select:** Помечен как fallback-only. В production ChromaDB/Qdrant имеют native MMR.

**Реализация — что использовать:**
- **Токенизация:** `razdel.tokenize(text)` — разбиение на токены для русского текста
- **Лемматизация:** `pymorphy3.MorphAnalyzer()` (singleton) — `.parse(token)[0].normal_form`
- **Стоп-слова:** собственный набор русских стоп-слов (компактный `frozenset`)
- **Сохранение номеров документов:** regex `r"(?:ГОСТ|СП|СНиП|ФЗ|НПБ)[\s\-]?[\d\.\-]+"` — извлечь до лемматизации, вернуть как есть
- **BM25:** обёртка над `rank_bm25.BM25Okapi` — передать лемматизированный корпус в конструктор, `.get_scores(query_tokens)` для поиска
- **RRF:** алгоритмический (формула `1/(k + rank)`), без внешних библиотек — ~15 строк кода
- **MMR:** алгоритмический fallback (~20 строк), в production использовать native `ChromaDB` MMR через `collection.query(include_embeddings=True)`

### 2.3 `hard_gates` — Фильтрация и оценка
**Файл:** `src/v7/hard_gates.py`
**Зависимости:** `state_types`, `config_v7`, `nlp_core`

| Функция | Описание |
|---------|----------|
| `check_hard_gates(state) -> HardGateResult` | Только жёсткие проверки (keyword overlap, min score). Используется в `evaluate_complex` |
| `check_full_triage(state) -> SufficiencyResult` | Hard gates + soft signals + 3-way triage. Используется в `evaluate_triage` |
| `validate_filters(filters) -> dict` | Whitelist для фильтров (защита от NoSQL injection) |
| `sanitize_for_llm(text) -> str` | Anti-injection: очистка перед LLM-вызовами |
| `keyword_overlap_score(keywords, passage) -> float` | Score перекрытия ключевых слов |

**3-way triage логика:**
```
score >= HARD_GATE_THRESHOLD           → sufficient   → END
TRIAGE_SOFT_THRESHOLD <= score < HARD  → borderline   → llm_verifier
score < TRIAGE_SOFT_THRESHOLD          → clearly_bad  → rag_complex
```

**Реализация — что использовать:**
- **Keyword overlap:** `nlp_core.extract_keywords()` — лемматизированные множества, пересечение через `set` операции. Dual overlap: отдельно для `active_query` и `original query`
- **Validate filters:** whitelist `{"doc_type", "doc_id", "section", "category", "year"}` — чистый Python `dict comprehension`, без библиотек
- **Sanitize:** `re.sub()` — удаление паттернов из `V7Config.BLOCKED_PATTERNS`. Без внешних библиотек
- **Diversity check (soft signal):** подсчёт уникальных `doc_id` в passages — `collections.Counter`, `max_single_doc_ratio`
- **Triage:** чистая логика на пороговых сравнениях из `v7_config`. Все пороги — из конфига, ни одного хардкода

### 2.4 `graph_v7` — Граф и ноды
**Файлы:** `src/v7/nodes/*.py`, `src/v7/graph.py`
**Зависимости:** все модули выше

#### Ноды (каждая — отдельный файл):

| Нода | reads → writes | Логика |
|------|---------------|--------|
| `intent_gate` | query → intent | Шум/домен классификация. Шум → END |
| `router` | query, intent → plan | Стратегия поиска, фильтры. clarify → clarify_respond |
| `rag_simple` | plan, active_query → retrieval_attempts | Основной поиск: BM25 + semantic → RRF → chunks |
| `evaluate_triage` | retrieval_attempts → sufficient | 3-way: sufficient/borderline/clearly_bad |
| `llm_verifier` | retrieval_attempts → verification | LLM fact-check. sufficient/rewrite/escalate |
| `rewriter` | verification, active_query → active_query | Переформулировка запроса, loop back to rag_simple |
| `rag_complex` | plan → retrieval_attempts (merge) | Декомпозиция + мульти-поиск |
| `evaluate_complex` | retrieval_attempts → sufficient | Hard gates only (без soft signals) |
| `abstain` | state → abstain_reason | Формирование отказа с объяснением |

**Реализация — что использовать:**
- **Граф:** `langgraph.graph.StateGraph` + `END` — стандартный LangGraph. Reducer для `retrieval_attempts`: `Annotated[list[RetrievalAttempt], operator.add]`
- **LLM-вызовы (verifier, rewriter):** `src.llm_factory.get_gemini_llm(thinking_budget)` — уже есть в проекте, не создавать новый
- **Промпты:** Jinja2 шаблоны из `prompts/` — через существующий `PromptRegistry`, не хардкодить строки
- **Streaming:** `langgraph.get_stream_writer()` — для `status_message` в UI
- **Retrieval (rag_simple):** ChromaDB `collection.query()` для semantic + `BM25Index.search()` → `rrf_merge()` для объединения
- **Reranking (rag_complex):** `FlashRank` — уже в проекте (`src/retrieval.py`), переиспользовать
- **Doc identifiers:** regex `r"(?:ГОСТ|СП|СНиП|ФЗ|НПБ)[\s\-]?[\d\.\-]+"` — для защиты идентификаторов при rewrite
- **Retrieval ID:** `hashlib.sha256` от `json.dumps(query + filters)` — детерминированный, для дедупликации attempts

#### `build_graph(overrides)`:
```python
def build_graph(overrides: dict[str, Callable] | None = None) -> StateGraph:
    """
    Собирает граф v7. overrides позволяет подменить любую ноду для тестов.
    Пример: build_graph({"router": mock_router})
    """
```

---

## 3. Структура файлов

```
src/v7/
├── __init__.py              # re-export: v7_config, RAGState, build_graph
├── state_types.py           # Этап 0: все TypedDict/dataclass
├── config.py                # Этап 0: V7Config pydantic-settings
├── nlp_core.py              # Этап 1: BM25, RRF, MMR, лемматизация
├── hard_gates.py            # Этап 2: gates, triage, anti-injection
├── nodes/                   # Этап 3: каждая нода отдельно
│   ├── __init__.py
│   ├── intent_gate.py
│   ├── router.py
│   ├── rag_simple.py
│   ├── evaluate_triage.py
│   ├── llm_verifier.py
│   ├── rewriter.py
│   ├── rag_complex.py
│   ├── evaluate_complex.py
│   └── abstain.py
└── graph.py                 # Этап 4: build_graph(overrides)

tests/v7/                    # Зеркальная структура тестов
├── test_state_types.py
├── test_config.py
├── test_nlp_core.py
├── test_hard_gates.py
├── test_nodes/
│   ├── test_intent_gate.py
│   ├── test_router.py
│   ├── ...
└── test_graph.py
```

---

## 4. Roadmap — Этапы миграции

### Этап 0: Контракты и конфигурация ✅ Done (2026-02-16)
**Модули:** `state_types`, `config_v7`
**Файлы:** `src/v7/state_types.py`, `src/v7/config.py`, `tests/v7/test_state_types.py`, `tests/v7/test_config.py`
**Критерий готовности:** Все типы импортируются, конфиг читает `.env`, тесты проходят.
**Влияние на текущий код:** Нулевое. Новые файлы, ничего не трогаем.

### Этап 1: NLP Core
**Модули:** `nlp_core`
**Файлы:** `src/v7/nlp_core.py`, `tests/v7/test_nlp_core.py`
**Зависимости:** pymorphy3, razdel, rank_bm25 (добавить в requirements)
**Критерий готовности:** `extract_keywords` лемматизирует русский текст, BM25Index строится и ищет, rrf_merge корректно сливает ранжирования. Юнит-тесты на русских примерах.
**Влияние на текущий код:** Нулевое.

### Этап 2: Hard Gates ✅ Done (2026-02-16)
**Модули:** `hard_gates`
**Файлы:** `src/v7/hard_gates.py`, `tests/v7/test_hard_gates.py`
**Критерий готовности:** `check_hard_gates` и `check_full_triage` возвращают корректные результаты. Anti-injection блокирует вредоносные паттерны. Тесты на граничных случаях (score ровно на пороге).
**Влияние на текущий код:** Нулевое.

### Этап 3: Ноды (без графа) ✅ Done (2026-02-16)
**Модули:** `nodes/*`
**Файлы:** `src/v7/nodes/*.py`, `tests/v7/test_nodes/*.py`
**Критерий готовности:** Каждая нода тестируется изолированно с mock-state. Корректно читает/пишет в state. LLM-ноды тестируются с мок-LLM.
**Влияние на текущий код:** Нулевое. Ноды зависят только от `src/v7/`.

### Этап 4: Сборка графа ✅ Done (2026-02-16)
**Модули:** `graph_v7`
**Файлы:** `src/v7/graph.py`, `tests/v7/test_graph.py`
**Критерий готовности:** `build_graph()` собирает рабочий граф. Integration test: запрос проходит полный цикл. `build_graph(overrides)` позволяет подменять ноды.
**Влияние на текущий код:** Нулевое.

### Этап 5: Миграция и cleanup ✅ Done (2026-02-17)
**Файлы:** `agents/multiagent_rag.py`, `config/settings.py`, `app.py`
**Действия:**
1. Добавить `USE_V7_GRAPH: bool = False` в settings
2. В `app.py`: `if settings.USE_V7_GRAPH: use v7 else: use current`
3. Включить `USE_V7_GRAPH=true`, прогнать регрессию
4. Удалить старый код после стабилизации

---

## 5. Принципы

### 5.1 Обратная совместимость
Весь новый код живёт в `src/v7/`. Текущий `agents/multiagent_rag.py` НЕ трогается до Этапа 5. На любом этапе можно откатиться — переключив `USE_V7_GRAPH=false`.

### 5.2 Contracts-first
Типы (`state_types`) и конфиг (`config_v7`) создаются ПЕРВЫМИ. Все остальные модули зависят от них. Это позволяет писать сигнатуры функций (`def retrieve(plan: RetrievalPlan) -> RetrievalAttempt`) с самого начала.

### 5.3 Тестируемость
Каждый модуль тестируется изолированно. `build_graph(overrides)` позволяет подменять любую ноду мок-версией для интеграционных тестов.

### 5.4 Конфигурация через .env
Все пороги — в `V7Config`. Ни один порог не хардкодится в функциях. Env-переменные с префиксом `V7_` (пример: `V7_HARD_GATE_THRESHOLD=0.70`).

### 5.5 Не изобретать велосипед
Использовать существующие библиотеки вместо написания собственных реализаций. Не писать лишний код — если задача решается вызовом библиотечной функции, использовать её. Примеры:
- **BM25** — `rank_bm25`, не своя реализация
- **Лемматизация** — `pymorphy3` + `razdel`, не свой токенизатор
- **MMR** — `langchain` встроенный MMR или native ChromaDB/Qdrant, не своя реализация
- **Валидация** — `pydantic`, не ручные проверки
- **Шаблоны промптов** — `Jinja2` (уже используется), не конкатенация строк

Если библиотека уже есть в `requirements.txt` — использовать её. Если нужна новая — обосновать необходимость, предпочитая зрелые и поддерживаемые пакеты.

### 5.6 Строгий граф зависимостей
Импорты только сверху вниз, обратных стрелок нет:

```
state_types (0 deps) ← config ← nlp_core ← hard_gates ← nodes/* ← graph
```

`state_types.py` не импортирует ничего из проекта. `nlp_core` не знает о `hard_gates`. Ноды не импортируют друг друга. `graph.py` импортирует ноды, но ноды не знают о графе. Нарушение этого порядка — блокер на code review.

### 5.7 Тонкие ноды
Ноды (`src/v7/nodes/*.py`) — тонкие обёртки. Вся вычислительная логика (математика, парсинг, RRF, keyword overlap, triage) живёт в `nlp_core` или `hard_gates`. Нода только:
1. Читает нужные поля из `state`
2. Вызывает функцию из `nlp_core` / `hard_gates`
3. Записывает результат обратно в `state`

Это позволяет тестировать логику юнит-тестами без поднятия LangGraph. Если в ноде появляется `if/else` на 10+ строк или математика — это сигнал вынести в `nlp_core`/`hard_gates`.

---

## 6. Связь с исходной спецификацией

Полная спецификация v7 с кодом всех функций: `docs/feature/migration-v7` (1729 строк).

Маппинг модулей на секции спецификации:

| Модуль | Секции в `migration-v7` |
|--------|------------------------|
| `state_types` | «ДИЗАЙН STATE», «КОНТРАКТЫ МЕЖДУ УЗЛАМИ», TypedDict определения |
| `config_v7` | Все числовые пороги из спецификации, собранные в один класс |
| `nlp_core` | `extract_keywords`, `class BM25Index`, `rrf_merge`, `mmr_select` |
| `hard_gates` | `check_hard_gates`, `check_full_triage`, `validate_filters`, `_sanitize_for_llm` |
| `graph_v7` | Все ноды (intent_gate ... abstain), `build_graph` |

---

## 7. Инструкции для агента

### 7.1 Как работать с этим документом

1. **Перед началом работы** — прочитай этот файл целиком. Он определяет архитектуру.
2. **Перед реализацией этапа** — создай implementation plan (через `superpowers:writing-plans`) для конкретного этапа, ссылаясь на этот дизайн + исходную спецификацию `docs/feature/migration-v7`.
3. **Не меняй текущий код** до Этапа 5. Весь новый код — в `src/v7/` и `tests/v7/`.
4. **Каждый этап = отдельный PR/коммит**. Не смешивай этапы.
5. **Тесты обязательны** на каждом этапе. Критерий готовности этапа — все тесты зелёные.

### 7.2 Как обновлять этот документ

**Когда обновлять:**
- После завершения этапа — обновить его статус (добавить `✅ Done` + дату)
- При изменении архитектурного решения — обновить соответствующую секцию
- При добавлении нового модуля — добавить в таблицу модулей и roadmap
- При изменении порогов/конфигурации — обновить пример V7Config

**Формат обновления статуса этапа:**
```markdown
### Этап 0: Контракты и конфигурация ✅ Done (2026-02-XX)
```

**Что НЕ менять:**
- Секцию «Связь с исходной спецификацией» — она ссылается на зафиксированный файл
- Порядок этапов (0→5) — он определяет зависимости
- Общие принципы (секция 5) — они фиксированы

### 7.3 Порядок работы для каждого этапа

```
1. Прочитать дизайн-документ (этот файл)
2. Прочитать соответствующие секции из docs/feature/migration-v7
3. Создать implementation plan (superpowers:writing-plans)
4. Реализовать (superpowers:executing-plans / superpowers:test-driven-development)
5. Запустить тесты: pytest tests/v7/ -v
6. Запустить линтер: black src/v7/ tests/v7/ && ruff check src/v7/ tests/v7/ --fix
7. Обновить статус этапа в этом файле
8. Коммит
```

### 7.4 Важные контексты из текущей реализации

При реализации учитывать:
- **BM25 docs имеют similarity=0.00** в metadata ChromaDB — nlp_core должен работать со своим BM25Index
- **Gemini SDK** патчен для отключения AFC (Automatic Function Calling) — `src/llm_factory.py`
- **Streaming** через `get_stream_writer()` — graph_v7 должен поддерживать status_message в RAGState
- **Глоссарий** `config/term_glossary.yaml` — расширение терминов перед графом (сохранить в v7)
- **semantic_cache.json** — текущий кеш ответов, v7 может использовать или заменить
