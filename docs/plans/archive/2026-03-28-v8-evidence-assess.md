# Plan: V8 Evidence Assessment

**Дата:** 2026-03-28
**Проблема:** triage классифицирует запросы до retrieval — proxy-метрика top_score не гарантирует полноту ответа для "рассеянных" вопросов.
**Паттерн:** Self-RAG / CRAG — перенести ключевое решение post-retrieval.
**Референс:** ~/knowledge/ai-engineering/evidence-assess-v8.md

---

## Этап 1: enumeration_intent в triage (30 мин)

**Файл:** `src/v7/nodes/triage.py`

**Что делать:**
1. Добавить ENUM_PATTERNS для русских вопросов-перечислений
2. Перед LLM-вызовом проверить regex-матч
3. Если матч → return `{"complexity": "complex"}` без LLM

**Паттерны:**
```python
ENUM_PATTERNS = [
    r"\bкто\s+проходит\b",
    r"\bв\s+каких\s+случаях\b",
    r"\bкакие\s+категории\b",
    r"\bкогда\s+не\s+требуется\b",
    r"\bперечислите\b",
    r"\bкаким\s+работникам\b",
]
```

**Тест:** "Кто проходит обучение по программе А" → complexity=complex → rag_complex

**Definition of Done:**
- [x] Паттерны добавлены в evaluate_triage.py (route_after_triage)
- [x] pytest: тест что enumeration_intent форсирует complex (17 тестов зелёных)
- [x] trace_v7.py "кто проходит обучение по программе А" → path включает rag_complex ✅

---

## Этап 2: crossref_signal в evaluate_simple (1 ч)

**Файл:** `src/v7/nodes/evaluate_simple.py`

**Что делать:**
1. Добавить CROSSREF_PATTERNS
2. После top_score check: если crossref_hits >= 2 → route = "expand" вместо "sufficient"
3. "expand" → повторный запуск rag_complex с исходным query

**Паттерны:**
```python
CROSSREF_PATTERNS = [
    r"\bпункт[а-я]*\s+\d+",
    r"\bза\s+исключением\b",
    r"\bв\s+соответствии\s+с\b",
    r"\bуказанн[а-я]+\s+в\b",
]
```

**Definition of Done:**
- [x] Паттерны добавлены в evaluate_triage.py (_CROSSREF_PATTERNS, _count_crossref_hits)
- [x] pytest: чанки с перекрёстными ссылками не дают `sufficient` (5 тестов, 22 всего)
- [ ] Новый route `expand` добавлен в граф (отложено до Этапа 3)

---

## Этап 3: evidence_assess узел (3–4 ч)

**Новые файлы:**
- `src/v7/nodes/evidence_assess.py`
- `src/v7/nodes/expand_local.py`

**Изменения в графе:**
- Вынести initial retrieval из rag_simple/rag_complex в общий `initial_retrieve`
- Добавить evidence_assess после initial_retrieve
- Три ветки: fast_answer / expand_local / decompose_global (= текущий rag_complex)

**Схема V8:**
```
intent_gate → initial_retrieve → evidence_assess
                                  ↓
                    fast_answer | expand_local | decompose_global
                                  ↓
                            generate_answer
```

**Definition of Done:**
- [ ] evidence_assess.py: TypedDict state, детерминированные сигналы
- [ ] expand_local.py: берёт section_id якоря из ChromaDB metadata
- [ ] Граф перестроен, старые rag_simple/rag_complex рефакторены
- [ ] pytest: coverage_ratio, dispersion_score, routing branches
- [ ] E2E: trace_v7.py "кто проходит программу А" → expand_local path

---

---

## P2 Backlog: FlashRank score inflation в evaluate_complex

**Проблема:** FlashRank cross-encoder возвращает ~0.999 для всех чанков после rag_complex.
`evaluate_complex` сравнивает FlashRank score с `COMPLEX_THRESHOLD=0.35` → всегда `sufficient`, `abstain` никогда не срабатывает.

**Фикс:**
- Хранить оригинальный vector score в passages (`vector_score`)
- threshold check через `vector_score`, FlashRank score — только для сортировки

**Влияние на метрики:** false_sufficiency_rate будет завышена до этого фикса.

---

## Метрики для eval set (после Этапа 3)

Вопросы с "рассеянными" ответами из 2464-ПП:
- fact_recall: сколько gold-атомов найдено retrieval-ом
- answer_completeness: сколько gold-атомов в ответе
- false_sufficiency_rate: evaluate "достаточно" при completeness < 0.8
