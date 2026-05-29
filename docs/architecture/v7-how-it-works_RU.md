# Как работает пайплайн V7

> Подробное объяснение логики — для собеседований, code review, онбординга.

---

## Применяемые техники

### Retrieval
- **Гибридный поиск** — векторный (ChromaDB, cosine similarity 0–1) + BM25 (ключевые слова, бесшкальный)
- **RRF (Reciprocal Rank Fusion)** — слияние двух списков без настройки весов: `score = Σ 1/(rank + 60)`
- **FlashRank Reranking** — cross-encoder переранжирование топ-результатов после hybrid merge
- **Query Expansion** — дополнительные термины из найденных документов (в RAG Complex)
- **Multi-attempt merge** — слияние результатов нескольких попыток поиска, top_k=24

### Качество ответов
- **Hard Gates** — детерминированные пороги (score, кол-во чанков, keyword overlap) без LLM-решений
- **3-way Triage** — маршрутизация по уверенности: sufficient / borderline / clearly_bad
- **Document Diversity** — защита от ответа только из одного источника (max_doc_ratio ≤ 0.7)
- **Abstain** — явный отказ при недостатке данных вместо галлюцинации

### NLP и безопасность
- **Term Glossary** — расшифровка доменных аббревиатур (`src/glossary.py`); применяется в ноде `router`. Слова >4 букв матчатся по морфологическому стему, аббревиатуры ≤4 букв — целым словом
- **Prompt Injection фильтр** — `sanitize_for_llm()` удаляет "ignore previous instructions" и подобное
- **NoSQL Injection whitelist** — валидация фильтров ChromaDB через `ALLOWED_FILTER_KEYS`

### Чанкинг и индексация
- **Docling** — парсинг PDF/DOCX с сохранением структуры документа
- **HybridChunker(max_tokens=400)** — структурное разбиение по заголовкам/пунктам документа (docling_core); заменяет RecursiveSplitter

### LLM и оркестрация
- **Gemini (thinking_budget=4096)** — управляемая глубина рассуждений; модель из `GEMINI_FAST_MODEL`
- **Dependency Injection (bridge.py)** — LLM и vector search инжектятся в граф, не захардкожены → легко менять провайдера
- **LangGraph StateGraph** — детерминированный граф состояний

### Eval
- **`eval/run_v7_eval.py`** — прогон golden-датасета через V7-граф, LLM-as-judge метрики. См. [docs/evaluation/README_RU.md](../evaluation/README_RU.md)

---

## Почему V7?

Предыдущие версии всегда отвечали, даже когда ничего релевантного не находили.
Цель — детерминированная логика: «нашли достаточно → отвечаем; не нашли → честно говорим об этом».

V7 достигает этого через **hard gates** — числовые пороги без LLM-решений. Граф детерминирован: при одинаковом запросе и индексе путь всегда одинаковый.

---

## Поток запроса

```
Запрос
  → Intent Gate              (regex: шум → END, нормативный → Router)
  → Router                   (классификация запроса, plan, глоссарий → active_query)
  → RAG Simple               (быстрый гибридный поиск, SIMPLE_TOP_K=12)
      → Evaluate Triage      (Гейт #1)
          sufficient (score ≥ 0.50)   → Generate Answer  ← быстрый путь
          borderline (0.38–0.50)      → LLM Verifier → (ok → Generate / rewrite → Rewriter → RAG Simple / escalate → RAG Complex)
          clearly_bad (< 0.38)        → RAG Complex
  → RAG Complex              (глубокий поиск COMPLEX_TOP_K=60 + rerank + MMR)
      → Evaluate Complex     (Гейт #2)
          ПРОЙДЕН            → Generate Answer
          ПРОВАЛЕН           → Abstain  ("не нашли достаточно данных")
  → Visual Enrichment        (опционально, перед генерацией)
  → Generate Answer          (Gemini, thinking_budget=4096)
  → Финальный ответ + источники
```

---

## Hard Gates

Hard gate — это `check_hard_gates()` в `src/v7/hard_gates.py`.

Принимает список извлечённых фрагментов и план (пороги), проверяет **три условия одновременно**. Все три должны быть True, иначе `sufficient=False`.

### Три условия

| Условие | Что проверяет | Порог (rag_simple) | Порог (rag_complex) |
|---|---|---|---|
| `above_threshold` | top_score ≥ threshold | **0.50** | **0.35** |
| `enough_evidence` | кол-во чанков ≥ min_passages | **5** | **8** |
| `keyword_overlap_ok` | доля ключевых слов запроса, найденных в чанках | **0.15** | **0.20** |

```python
# Из hard_gates.py — упрощённо:
sufficient = all([
    top_score >= plan["threshold"],          # score от ChromaDB (0-1)
    len(passages) >= plan["min_passages"],    # количество чанков
    keyword_overlap >= plan["min_kw_overlap"] # доля слов запроса в тексте
])
```

### Откуда берётся score?

Score — это **cosine similarity** из ChromaDB (диапазон 0–1). BM25 scores не используются для порогов — они не калиброваны (диапазон 0–20+). В `rag_simple.py` `top_score` берётся только из `vector_results`, не из объединённого списка.

### Почему два разных порога?

`rag_simple` — быстрый путь. Порог 0.50 высокий — означает, что найдено явно релевантное.

`rag_complex` — запасной путь. Порог 0.35 ниже, но компенсируется более строгими требованиями к количеству фрагментов (8+) и keyword overlap (20%+). Если score немного ниже, объём и лексическое покрытие компенсируют это.

---

## Triage — три категории после RAG Simple

После `rag_simple` нода `evaluate_triage` классифицирует результат:

| Категория | Условие | Что происходит |
|---|---|---|
| `sufficient` | hard gates OK | → Generate Answer напрямую |
| `borderline` | score в зоне 0.38–0.50 | → LLM Verifier (решает: ответить / переформулировать / эскалировать) |
| `clearly_bad` | score < 0.38 или мало чанков | → RAG Complex |

Зона borderline (0.38–0.50) означает «возможно, можно сделать лучше»: `llm_verifier` изучает
фрагменты и решает — ответ достаточно хорош (→ генерировать), нужна переформулировка
(→ `rewriter` → RAG Simple) или эскалация (→ RAG Complex). Ниже 0.38 — явно плохо, сразу в Complex.

---

## RAG Complex — чем отличается

`src/v7/nodes/rag_complex.py` выполняет поиск иначе:

1. **Query expansion через BM25** — дополнительные термины из топ-документов
2. **Несколько попыток** — с разными параметрами (разный top_k, разные фильтры)
3. **Объединение всех попыток** — `merge_all_passages(attempts, top_k=24)` в `evaluate_complex`

`top_k=24` важен. Раньше было 12 — ответы были неполными (система находила 3 из 8 категорий). После увеличения до 24 все категории появляются в ответе.

---

## Дополнительные защитные механизмы

### Prompt Injection (в `hard_gates.py`)
```python
# Паттерны фильтруются до передачи в Gemini:
"ignore previous instructions" → "[FILTERED]"
"system:" → "[FILTERED]"
"you are now" → "[FILTERED]"
```

### NoSQL Injection (validate_filters)
Фильтры для ChromaDB проходят whitelist-валидацию — только разрешённые ключи (`ALLOWED_FILTER_KEYS`). Произвольные where-clause не проходят.

### Document Diversity
Если все 8+ чанков из одного документа, `max_doc_ratio > 0.7` и `escalation_hint = True`. Для мультидокументных запросов это делает hard gate провальным (разнообразие — жёсткое требование, не подсказка).

---

## Visual Enrichment

Нода `src/v7/nodes/visual_enrichment.py` вставляется **перед `generate_answer`** — после evaluate_triage/verifier/evaluate_complex.

Цель: добавить визуальный контекст (скриншоты таблиц, изображения страниц) к текстовым чанкам, чтобы Gemini точнее отвечал на вопросы, где важна структура документа.

### Триггеры

Нода проверяет каждый фрагмент и принимает решение на основе триггеров:

| Триггер | Условие | Действие |
|---|---|---|
| Таблица | `element_type == "Table"` | `mode=analyze` — VLM анализирует таблицу и добавляет текстовое описание |
| Короткий чанк | `len(text) < 150` | `mode=show` — передаёт image_path, пусть модель видит оригинал |
| Неполный текст | `detect_incomplete_chunk()` — обрыв на ":" или "№" | `mode=show` — текст явно обрезан, нужен оригинал |

### Ограничения

- **MAX_VISUAL_PROOFS = 3** — не более 3 визуальных доказательств на запрос (бюджет токенов)
- Нода является **no-op**, если `visual_proof_fn` не инжектирован — безопасна в окружениях без VLM
- Исключения на отдельных фрагментах не останавливают обработку остальных

### Dependency Injection

```python
# bridge.py — init_v7_pipeline()
visual_proof_fn = make_visual_proof_fn()          # returns None if agent_tools unavailable
if visual_proof_fn is not None:
    visual_enrichment_mod.set_visual_proof_fn(visual_proof_fn)
```

На VPS `agent_tools` не настроен → `make_visual_proof_fn()` возвращает None,
нода пропускается без ошибок.

### Пример трассировки

```
[visual_enrichment] passage 0: element_type=Table → analyze → добавлен visual_context
[visual_enrichment] passage 2: len=87 → show → image_path передан
[visual_enrichment] passage 5: incomplete chunk → show → image_path передан
```

---

## Generate Answer

`src/v7/nodes/generate_answer.py` вызывает `make_generate_fn()` из `bridge.py`.

Bridge инжектирует Gemini с `thinking_budget=4096`. Модель получает:
- Финальные фрагменты (до 24 чанков; `make_generate_fn` берёт `final_passages[:24]`)
- Запрос пользователя
- Промпт, загруженный из Jinja2-шаблона через PromptManager (`prompts/agents/generate_answer_v1.j2`): инструктирует модель отвечать строго по документам и цитировать источники

**Повторные попытки:** если Gemini возвращает 503, `tenacity` повторяет 3 раза с экспоненциальной задержкой (2→4→8с). Только после всех попыток происходит fallback на заглушку (сырые тексты чанков без синтеза).

---

## Настройка порогов

Все пороги находятся в `src/v7/config.py`, переопределяются через env:

```env
V7_HARD_GATE_THRESHOLD=0.50       # порог для rag_simple
V7_TRIAGE_SOFT_THRESHOLD=0.38     # нижняя граница borderline
V7_COMPLEX_THRESHOLD=0.35         # порог для rag_complex
V7_COMPLEX_MIN_PASSAGES=8         # мин. чанков для rag_complex
V7_COMPLEX_MIN_KW_OVERLAP=0.20    # мин. keyword overlap для rag_complex
```

Smoke test для проверки пайплайна целиком:
```bash
python scripts/trace_v7.py "кто должен обучаться по программе А охраны труда?"
```
