# How V7 Pipeline Works

> Detailed explanation of the logic — for interviews, code review, onboarding.

---

## Techniques Used

### Retrieval
- **Гибридный поиск** — векторный (ChromaDB, cosine similarity 0–1) + BM25 (ключевые слова, бесшкальный)
- **RRF (Reciprocal Rank Fusion)** — слияние двух списков без настройки весов: `score = Σ 1/(rank + 60)`
- **FlashRank Reranking** — cross-encoder переранжирование топ-результатов после hybrid merge
- **Query Expansion** — дополнительные термины из найденных документов (в RAG Complex)
- **Multi-attempt merge** — слияние результатов нескольких попыток поиска, top_k=24

### Answer Quality
- **Hard Gates** — детерминированные пороги (score, кол-во чанков, keyword overlap) без LLM-решений
- **3-way Triage** — маршрутизация по уверенности: sufficient / borderline / clearly_bad
- **Document Diversity** — защита от ответа только из одного источника (max_doc_ratio ≤ 0.7)
- **Abstain** — явный отказ при недостатке данных вместо галлюцинации

### NLP and Security
- **Term Glossary** — расшифровка доменных аббревиатур (`src/glossary.py`); применяется в ноде `router`. Слова >4 букв матчатся по морфологическому стему, аббревиатуры ≤4 букв — целым словом
- **Prompt Injection фильтр** — `sanitize_for_llm()` удаляет "ignore previous instructions" и подобное
- **NoSQL Injection whitelist** — валидация фильтров ChromaDB через `ALLOWED_FILTER_KEYS`

### Chunking and Indexing
- **Docling** — парсинг PDF/DOCX с сохранением структуры документа
- **HybridChunker(max_tokens=400)** — structural chunking by document headings/clauses (docling_core); replaces RecursiveSplitter

### LLM and Orchestration
- **Gemini (thinking_budget=4096)** — управляемая глубина рассуждений; модель из `GEMINI_FAST_MODEL`
- **Dependency Injection (bridge.py)** — LLM и vector search инжектятся в граф, не захардкожены → легко менять провайдера
- **LangGraph StateGraph** — детерминированный граф состояний

### Eval
- **`eval/run_v7_eval.py`** — прогон golden-датасета через V7-граф, LLM-as-judge метрики. См. [docs/evaluation/README.md](../evaluation/README.md)

---

## Why V7?

Previous versions always answered, even when nothing relevant was found.
The goal was deterministic logic: "found enough → answer; did not find → honestly say so".

V7 achieves this through **hard gates** — numeric thresholds with no LLM decisions. The graph is deterministic: given the same query and index, the path is always the same.

---

## Request Flow

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

A hard gate is `check_hard_gates()` in `src/v7/hard_gates.py`.

It takes the list of retrieved passages and a plan (thresholds) and checks **three conditions simultaneously**. All three must be True, otherwise `sufficient=False`.

### Three Conditions

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

### Where Does the Score Come From?

Score is **cosine similarity** from ChromaDB (range 0–1). BM25 scores are not used for thresholds — they are unscaled (0–20+). In `rag_simple.py`, `top_score` is taken only from `vector_results`, not from the merged list.

### Why Two Different Thresholds?

`rag_simple` is the fast path. Threshold 0.50 is high — means we found something clearly relevant.

`rag_complex` is the fallback. Threshold 0.35 is lower, but compensated by stricter requirements on passage count (8+) and keyword overlap (20%+). If the score is slightly lower, volume and lexical coverage make up for it.

---

## Triage — Three Categories After RAG Simple

After `rag_simple`, the `evaluate_triage` node classifies the result:

| Категория | Условие | Что происходит |
|---|---|---|
| `sufficient` | hard gates OK | → Generate Answer напрямую |
| `borderline` | score в зоне 0.38–0.50 | → LLM Verifier (решает: ответить / переформулировать / эскалировать) |
| `clearly_bad` | score < 0.38 или мало чанков | → RAG Complex |

The borderline zone (0.38–0.50) means "maybe we can do better": `llm_verifier` examines
the passages and decides — answer is good enough (→ generate), needs reformulation
(→ `rewriter` → RAG Simple), or escalation (→ RAG Complex). Below 0.38 — clearly bad, goes
straight to Complex.

---

## RAG Complex — What It Does Differently

`src/v7/nodes/rag_complex.py` runs search differently:

1. **Query expansion via BM25** — additional terms extracted from top documents
2. **Multiple attempts** — with different parameters (different top_k, different filters)
3. **Merge all attempts** — `merge_all_passages(attempts, top_k=24)` in `evaluate_complex`

`top_k=24` matters. Previously it was 12 — answers were incomplete (the system found 3 out of 8 categories). After increasing to 24, all categories appear in the answer.

---

## Additional Safeguards

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
If all 8+ chunks come from a single document, `max_doc_ratio > 0.7` and `escalation_hint = True`. For multi-doc queries this makes the hard gate fail (diversity is a hard requirement, not a hint).

---

## Visual Enrichment

The node `src/v7/nodes/visual_enrichment.py` is inserted **before `generate_answer`** — after evaluate_triage/verifier/evaluate_complex.

Goal: add visual context (table screenshots, page images) to text chunks, so Gemini can answer more accurately on questions where document structure matters.

### Triggers

The node inspects each passage and decides based on triggers:

| Триггер | Условие | Действие |
|---|---|---|
| Таблица | `element_type == "Table"` | `mode=analyze` — VLM анализирует таблицу и добавляет текстовое описание |
| Короткий чанк | `len(text) < 150` | `mode=show` — передаёт image_path, пусть модель видит оригинал |
| Неполный текст | `detect_incomplete_chunk()` — обрыв на ":" или "№" | `mode=show` — текст явно обрезан, нужен оригинал |

### Constraints

- **MAX_VISUAL_PROOFS = 3** — at most 3 visual proofs per query (token budget)
- Node is **no-op** if `visual_proof_fn` is not injected — safe in environments without VLM
- Exceptions on individual passages do not stop the rest

### Dependency Injection

```python
# bridge.py — init_v7_pipeline()
visual_proof_fn = make_visual_proof_fn()          # returns None if agent_tools unavailable
if visual_proof_fn is not None:
    visual_enrichment_mod.set_visual_proof_fn(visual_proof_fn)
```

On VPS, `agent_tools` is not configured → `make_visual_proof_fn()` returns None,
node is skipped silently.

### Trace Example

```
[visual_enrichment] passage 0: element_type=Table → analyze → добавлен visual_context
[visual_enrichment] passage 2: len=87 → show → image_path передан
[visual_enrichment] passage 5: incomplete chunk → show → image_path передан
```

---

## Generate Answer

`src/v7/nodes/generate_answer.py` calls `make_generate_fn()` from `bridge.py`.

Bridge injects Gemini with `thinking_budget=4096`. The model receives:
- Final passages (up to 24 chunks; `make_generate_fn` takes `final_passages[:24]`)
- The user query
- Prompt loaded from Jinja2 template via PromptManager (`prompts/agents/generate_answer_v1.j2`): instructs the model to answer strictly from documents and cite sources

**Retry:** if Gemini returns 503, `tenacity` retries 3 times with exponential backoff (2→4→8s). Only after all retries does it fall back to a stub (raw chunk texts without synthesis).

---

## Threshold Configuration

All thresholds are in `src/v7/config.py`, overridable via env:

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
