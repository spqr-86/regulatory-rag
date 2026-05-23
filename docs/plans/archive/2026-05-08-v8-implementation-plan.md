# V8 Implementation Plan

> Эволюция V7 → evidence-aware adaptive RAG.
> Цель: различать retrieval failure, coverage failure и grounding failure.
> Фокус: минимальные изменения с максимальным ROI. Не переписываем — эволюционируем.

---

## Текущее состояние V7 (точка старта)

```
intent_gate → router → rag_simple → evaluate_triage
                                       ├─ sufficient → visual_enrichment → generate_answer
                                       ├─ borderline → llm_verifier → [rewrite loop | rag_complex]
                                       └─ clearly_bad → rag_complex → evaluate_complex
                                                                        ├─ pass → visual_enrichment → generate_answer
                                                                        └─ fail → abstain
```

**Что V7 уже делает хорошо** (не трогаем):
- Enumeration detection (regex-паттерны → force rag_complex)
- Crossref escalation (>= 3 перекрёстных ссылок в чанках → escalate)
- LLM verifier для borderline zone
- Section-aware expansion в rag_complex
- Visual enrichment для таблиц и обрезанных чанков

**Главная слабость V7:** решение о качестве retrieval по 3 proxy-сигналам (top_score, passage_count, keyword_overlap). High top_score не гарантирует покрытие всех аспектов вопроса.

---

## Epic 1: Eval Suite (дни 1-2)

> Без измерения всё остальное — гадание. Eval идёт ПЕРВЫМ.

### Что делаем

Раздельная оценка retrieval и generation. Фиксируем V7 baseline, чтобы доказать что V8 лучше.

### Golden Set — уже есть

`tests/dataset.csv` — **50 пар** (question, ground_truth). Покрытие:
- Factoid: "какой срок хранения акта Н-1?", "минимальная продолжительность стажировки?"
- Enumeration: "кто проходит обучение по программе А?", "какие виды инструктажей?"
- Multi-aspect: "как организовать СУОТ?", "процедура СОУТ?"
- OOS: "как готовить борщ?"
- Short/ambiguous: "требования к вентиляции?", "нужна ли каска бухгалтеру?"

Формат CSV, не YAML. Колонка `category` отсутствует — определяем автоматически или добавляем.

**Строка 13 битая** — CSV parsing error ("Что происходит с работником" + обрыв). Исправить или исключить.

### Метрики

#### Generation (основные)

| Метрика | Как считаем | Цель |
|---------|-------------|------|
| Completeness | Ключевые фразы из ground_truth найдены в ответе (substring + лемматизация) | >= 0.7 |
| Abstain rate | Доля запросов где система отказала | информативная |
| False abstain | Abstain на domain-запросах из dataset | = 0 |
| Correct abstain | Abstain на OOS ("борщ") | = 100% |

#### Retrieval (информативные)

| Метрика | Формула | Цель |
|---------|---------|------|
| Avg top_score | Средний top_score по всем запросам | baseline |
| Avg passage_count | Среднее кол-во passages | baseline |

### Completeness — как считаем

Из каждого ground_truth извлекаем ключевые фразы автоматически:
1. Разбиваем ground_truth по пунктам (нумерация, запятые, "и")
2. Лемматизируем (pymorphy3) — переиспользуем `extract_keywords()` из nlp_core
3. Ищем каждую ключевую фразу в ответе системы (fuzzy substring match)
4. Completeness = (найдено / всего) для каждого запроса

### Файлы

| Действие | Файл | Что |
|----------|------|-----|
| CREATE | `eval/__init__.py` | |
| CREATE | `eval/metrics.py` | Completeness, abstain rates, retrieval stats |
| CREATE | `eval/run_eval.py` | Парсит tests/dataset.csv, гоняет pipeline, выводит таблицу |
| CREATE | `eval/compare.py` | A/B: два JSON-файла side-by-side |
| CREATE | `tests/eval/test_metrics.py` | Unit-тесты формул |

### Порядок

```
День 1: metrics.py + тесты + run_eval.py (парсинг CSV + pipeline integration)
День 2: V7 baseline прогон + compare.py + фиксация baseline в eval/baselines/v7_baseline.json
```

### Acceptance criteria

- [ ] `python eval/run_eval.py` парсит tests/dataset.csv (50 пар), выводит таблицу метрик
- [ ] Completeness считается по ground_truth (автоматическое извлечение ключевых фраз)
- [ ] V7 baseline зафиксирован в `eval/baselines/v7_baseline.json`
- [ ] `python eval/compare.py v7_baseline.json v8_run.json` — side-by-side diff
- [ ] Строка 13 (битая) исправлена или исключена

---

## Epic 2: Evidence Assess (дни 4-5)

### Что делаем

Расширяем evaluate_triage дополнительными сигналами. НЕ заменяем — дополняем. 3 verdict-а вместо 5: answer / improve / abstain.

### Новые сигналы (поверх существующих V7)

| Сигнал | Откуда | Зачем |
|--------|--------|-------|
| `reranker_top1` | FlashRank score top-1 | Cross-encoder точнее cosine similarity |
| `reranker_top3_mean` | Среднее FlashRank top-3 | Глубина: не только top-1 релевантен |
| `coverage_estimate` | Эвристика: `kw_overlap * min(passage_count / target, 1.0)` | Грубая оценка покрытия без LLM |

Существующие сигналы V7 остаются: `top_score`, `passage_count`, `keyword_overlap`, `doc_ratio`, `enumeration_intent`, `crossref_hits`.

### 3 verdict-а

```python
EvidenceVerdict = Literal["answer", "improve", "abstain"]

# answer:  reranker_top1 >= 0.6 AND coverage >= 0.6 → generate
# improve: что-то нашли, но недостаточно → rag_complex (с расширением)
# abstain: reranker_top1 < 0.2 AND coverage < 0.2 → отказ
```

### Ключевое отличие от V7

V7 evaluate_triage решает по top_score (cosine): >= 0.50 → sufficient, 0.38-0.50 → borderline, < 0.38 → clearly_bad.

V8 evidence_assess решает по **комбинации**: reranker score (cross-encoder) + coverage estimate + существующие V7 сигналы. Более информированное решение.

### Интеграция

**Не создаём src/v8/ директорию.** Evidence assess — это улучшение evaluate_triage, не отдельный модуль.

```python
# src/v7/nodes/evaluate_triage.py — расширяем:
def evaluate_triage(state: RAGState) -> RAGState:
    if v7_config.V8_ENABLE_EVIDENCE_ASSESS:
        return _evidence_assess(state)  # новый путь
    return _legacy_triage(state)        # текущий V7
```

### Reranker в simple path

Сейчас FlashRank только в rag_complex. Для evidence_assess нужен reranker score уже после rag_simple.

Решение: добавить **лёгкий** rerank в rag_simple (top-5 passages only, не весь результат). Это даёт reranker_top1 без значительного overhead.

### Файлы

| Действие | Файл | Что |
|----------|------|-----|
| MODIFY | `src/v7/nodes/evaluate_triage.py` | `_evidence_assess()` + feature flag switch |
| MODIFY | `src/v7/nodes/rag_simple.py` | Лёгкий rerank top-5 для reranker_top1 signal |
| MODIFY | `src/v7/state_types.py` | `EvidenceReport` TypedDict, `EvidenceVerdict` Literal |
| MODIFY | `src/v7/config.py` | `V8_ENABLE_EVIDENCE_ASSESS`, пороги |
| MODIFY | `tests/v7/test_nodes/test_evaluate_triage.py` | Тесты на новые verdict-ы |
| CREATE | `tests/v7/test_nodes/test_evidence_assess.py` | Unit-тесты _evidence_assess, 3 ветки + edge cases |

### Acceptance criteria

- [ ] `V8_ENABLE_EVIDENCE_ASSESS=false` → поведение идентично V7
- [ ] `V8_ENABLE_EVIDENCE_ASSESS=true` → 3 verdict-а: answer / improve / abstain
- [ ] reranker_top1 + coverage_estimate видны в trace
- [ ] Unit-тесты на каждый verdict (минимум 10 тестов)
- [ ] Eval прогон V8 vs V7 baseline → completeness на enumeration queries выше

---

## Epic 3: Multi-Query Expand (день 6)

### Что делаем

Одна нода: генерация 2-3 вариаций запроса для увеличения recall. Вызывается из evidence_assess при verdict="improve" вместо прямого перехода в rag_complex.

### Как работает

```python
# src/v7/nodes/multi_query_expand.py
def multi_query_expand(state: RAGState) -> RAGState:
    """Генерирует 2-3 вариации запроса → retrieval по каждой → merge."""
    original_q = state["query"]
    variations = _expand_fn(original_q)  # DI: LLM или rule-based stub

    # Retrieval по каждой вариации
    all_passages = []
    for q in [original_q] + variations:
        results = _vector_search(query=q, top_k=plan["top_k"])
        all_passages.extend(results)

    # Dedup + merge
    merged = deduplicate_by_chunk_id(all_passages)
    return {"retrieval_attempts": [...], ...}
```

### Stub (rule-based, без LLM)

```python
def _stub_expand(query: str) -> list[str]:
    """Rule-based: синонимы + переформулировка."""
    # "программа А обучения" → ["программа А охрана труда", "обучение охрана труда программа"]
    keywords = extract_keywords(query)
    return [" ".join(reversed(list(keywords)))]  # простейший вариант
```

LLM-версия инжектится через bridge.py (`make_expand_fn()`).

### Интеграция в граф

```
evidence_assess (verdict="improve")
    → multi_query_expand
    → evaluate_complex (существующий)
    → visual_enrichment / abstain
```

Заменяет прямой переход `improve → rag_complex`. Теперь: `improve → multi_query_expand → evaluate_complex`.

### Файлы

| Действие | Файл | Что |
|----------|------|-----|
| CREATE | `src/v7/nodes/multi_query_expand.py` | Нода + DI |
| MODIFY | `src/v7/bridge.py` | `make_expand_fn()` + inject |
| MODIFY | `src/v7/graph.py` | Новая нода в граф, conditional edge из evidence_assess |
| MODIFY | `src/v7/config.py` | `V8_ENABLE_MULTI_QUERY`, `V8_MAX_QUERY_VARIATIONS=3` |
| CREATE | `tests/v7/test_nodes/test_multi_query_expand.py` | |

### Acceptance criteria

- [ ] Генерирует 2-3 вариации (не больше)
- [ ] Сохраняет doc identifiers из оригинального запроса
- [ ] Stub работает без LLM
- [ ] Merged passages проходят dedup по chunk_id
- [ ] Eval: recall на enumeration queries выше чем V7

---

## Epic 4: Hardening (день 7)

### Feature flags

```python
# src/v7/config.py — итого:
V8_ENABLE_EVIDENCE_ASSESS: bool = False   # Epic 2
V8_ENABLE_MULTI_QUERY: bool = False       # Epic 3
```

Оба выключены по умолчанию. V7 behaviour при `false`.

### Error taxonomy

```python
# src/v7/state_types.py
class FailureReason(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"    # мало чанков, низкий score
    COVERAGE_FAILURE = "coverage_failure"      # чанки есть, но не все аспекты
    GROUNDING_FAILURE = "grounding_failure"    # LLM не смог обосновать ответ
    GENERATION_FAILURE = "generation_failure"  # Gemini 503 после retry
```

Добавляется в `RAGState.failure_reason: str` → заполняется в abstain/generate_answer при ошибке.

### trace_v7.py обновление

Показывает:
- evidence verdict + все сигналы (если V8 включён)
- failure_reason при abstain
- query variations (если multi_query включён)

### Acceptance criteria

- [ ] `V8_ENABLE_*=false` → V7 behaviour, все 270+ тестов зелёные
- [ ] Rollback одним env switch
- [ ] Каждый abstain содержит failure_reason в state
- [ ] trace показывает новые V8 сигналы

---

## Что НЕ делаем (и почему)

| Идея | Почему не сейчас |
|------|------------------|
| Parent-child retrieval | 749 чанков, 7 PDF — маленький корпус, ROI низкий. Упомянуть на интервью как "next step". |
| Query decomposer | Для нормативки запросы конкретные, не multi-hop. Проблема которой нет. |
| Step-back query | Текущий rewriter уже делает это через LLM. Дублирование. |
| subclaim_coverage через LLM | +1 LLM call на запрос, сложно тестировать. Эвристика достаточна. |
| 5 verdict-ов | 3 хватит: answer / improve / abstain. Overengineering. |
| Governance (ACL, provenance) | Enterprise boilerplate для одной VPS. |

---

## Порядок выполнения

```
День 1: metrics.py + тесты + run_eval.py (парсинг dataset.csv + pipeline)
День 2: V7 baseline прогон + compare.py + фиксация baseline
День 3: evidence_assess + reranker в simple path + config
День 4: Тесты evidence_assess + eval прогон V8 vs V7
День 5: multi_query_expand нода + тесты + eval
День 6: Feature flags + error taxonomy + trace + финальный A/B
```

## Зависимости

```
Epic 1 (eval) ← ни от чего — стартует первым
Epic 2 (evidence_assess) ← Epic 1 (нужен baseline для сравнения)
Epic 3 (multi_query) ← Epic 2 (verdict="improve" запускает expand)
Epic 4 (hardening) ← по ходу Epic 2-3
```

## Definition of Done (V8)

1. **Eval baseline** зафиксирован для V7 с числами
2. **V8 completeness** на enumeration queries выше V7 (измерено)
3. **False abstain** = 0 на golden set
4. **Trace**: verdict + сигналы + failure_reason видны
5. **Rollback**: `V8_ENABLE_*=false` → V7 без потерь

Формулировка для интервью:
> "Эволюционировали от threshold-gated RAG к evidence-aware adaptive RAG: добавили cross-encoder scoring и coverage estimation в triage, multi-query expansion для сложных запросов, и построили eval suite с раздельными метриками retrieval и generation. Completeness на enumeration queries выросла с X% до Y%."
