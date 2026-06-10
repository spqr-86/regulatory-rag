# План исправления пайплайна V7 (по итогам ревью 2026-06-10)

Исходный отчёт: `docs/reviews/2026-06-10-pipeline-architecture-review.md` (номера находок
A1…F1 — оттуда). План рассчитан на исполняющего агента: фазы независимы, каждая —
отдельный коммит, после каждой фазы `pytest tests/ -x` должен быть зелёным.

## Общие правила для агента

- Не менять значения порогов (`HARD_GATE_THRESHOLD` и т.д.) — чинится *семантика* score,
  а не калибровка. Перекалибровка — отдельная задача после eval-прогона.
- Если существующий тест закрепляет багованное поведение (например, копирование сырого
  BM25 в `score`) — обновить тест и явно отметить это в сообщении коммита.
- На каждый фикс — регрессионный unit-тест.
- Полный eval (`eval/run_v7_eval.py`) требует OPENAI_API_KEY и денег — НЕ запускать
  автоматически; в конце плана перечислено, что прогнать вручную.

---

## Фаза 1 — Семантика score (A1, A2, A3, A4, D1) — самая важная

Цель: ввести явный контракт каналов score и убрать сырые BM25-значения из гейтов и MMR.

**Контракт passage-словаря после фазы:**

| Поле | Кто пишет | Диапазон | Назначение |
|------|-----------|----------|------------|
| `vector_score` | vector search, сохраняется при rerank | (0,1] | пороговые гейты, MMR |
| `bm25_score` | BM25Index.search | сырой, unbounded | диагностика |
| `rerank_score` | rerank-функции | [0,1] | порядок/диагностика |
| `score` | все продюсеры | [0,1] | отображение/метки в промпте |
| `doc_id` | все продюсеры | str | diversity/MMR |

### 1.1 Продюсеры passage
- `src/v7/bridge.py::make_vector_search_fn`: в каждый passage добавить
  `vector_score = similarity` (рядом с `score`) и `doc_id = meta.get("source", "unknown")`.
- `src/v7/bridge.py::_doc_to_passage`: добавить `doc_id` из metadata source.
- `src/v7/nlp_core.py::BM25Index.search` (строки ~204-213): НЕ копировать сырой
  `bm25_score` в `score`. Вместо этого `score = round(s / (s + 5.0), 4)` (ограниченный
  монотонный сквош, только для отображения), `bm25_score` оставить сырым; добавить
  `doc_id` из `p["metadata"].get("source")`.
- `src/v7/cross_ref.py::_passage_from_doc`: добавить `doc_id`.

### 1.2 Гейты читают только vector-канал
- `src/v7/hard_gates.py::check_hard_gates` (строка 88): заменить
  `max(p.get("score"))` на `max(p.get("vector_score", 0.0))`.
  Обоснование: порог 0.50 откалиброван под L2→similarity (комментарий в
  `src/v7/config.py:24-27`) и под решение из `bridge.py:84-87` («FlashRank scores are
  not calibrated for thresholds»). BM25-only и cross-ref чанки порог не проходят —
  это и есть требуемое поведение: «есть ли хоть один векторный хит выше порога».
- Проверить, что обе rerank-фабрики (`make_rerank_fn`, `make_crossencoder_rerank_fn`)
  по-прежнему сохраняют `vector_score`; дополнительно писать rerank-результат в
  `rerank_score` (и в `score` как сейчас).

### 1.3 MMR без смешения шкал
- `src/v7/nlp_core.py::mmr_select` (строка 335): relevance =
  `p.get("vector_score") or p.get("rerank_score") or p.get("score", 0.0)` —
  все компоненты теперь в [0,1], сырые BM25 туда не попадают.

### 1.4 Тесты
- Регрессия A1: passage только из BM25 с `bm25_score=12` не должен один проходить
  `check_hard_gates` при threshold=0.5.
- Регрессия A2: в `mmr_select` чанк с `vector_score=0.9` выигрывает у BM25-чанка с
  `bm25_score=12`.
- Регрессия D1: `compute_doc_diversity` на passages из двух source возвращает
  `unique_docs=2`, `max_doc_ratio=0.5`.
- Обновить `tests/v7/test_nlp_core.py` там, где закреплено копирование сырого score.

**Acceptance:** гейты «sufficient» проходят только при наличии векторного хита ≥ порога;
`unique_docs > 1` появляется в логах/метриках; pytest зелёный.

---

## Фаза 2 — Cross-reference expansion (B1, B2, B4)

Файл: `src/v7/cross_ref.py`.

### 2.1 Типизированные ссылки вместо голых подстрок
- `_extract_refs` возвращает кортежи `(kind, value)`:
  `("clause", "46")`, `("article", "5")`, `("subpara", "а")`.
- Механизм 1 (строки 101-112) — структурное сопоставление вместо `ref in page_content`:
  - clause N: `re.search(rf"(?m)(^\s*{N}\.(?=\s))|(\bпункт\w*\s+{N}\b)", doc.page_content)`
  - article N: `re.search(rf"\bстать\w+\s+{N}\b", doc.page_content)`
  - subpara «а»: сопоставлять ТОЛЬКО как маркер подпункта
    `re.search(r"(?m)^\s*а\)", ...)` и только если в том же passage найден clause-номер
    (иначе пропустить) — одиночная буква как подстрока запрещена.
- Экранировать N через `re.escape`.

### 2.2 Капы на расширение
- Лимит на механизм 1: не более 5 доп. чанков на passage и не более 15 суммарно.
- Лимит на механизмы 3 и 4: суммарно не более 10 каждый.
- Залогировать (structlog) счётчики добавленного по каждому механизму — пригодится
  для оценки эффекта.

### 2.3 Мелочь
- Механизм 2: заменить `except Exception: pass` (строки 138-139) на
  `logger.warning("cross_ref bm25 re-search failed", error=str(exc))`.

### 2.4 Тесты
- «пункт 46» НЕ матчит чанк, содержащий только «146» / «2464» / «1946».
- «подпунктом «а»» НЕ приводит к добавлению чанка без маркера `а)`.
- Капы соблюдаются: источник из 100 чанков с буквой «а» даёт ≤ лимита.

**Acceptance:** на синтетическом корпусе из 50 чанков passage с «подпункта «а»»
добавляет ≤ 5 чанков, а не весь источник.

---

## Фаза 3 — Симметрия путей и fallback (C2, C3, C4)

### 3.1 Enumeration-эскалация не должна терять готовый ответ (C2)
- `src/v7/nodes/evaluate_triage.py::_legacy_triage`, ветка sufficient (строки 115-120):
  если `_has_enumeration_intent(state["query"])` — дополнительно записать
  `fallback_passages=passages, fallback_score=result["top_score"]`
  (перенести/реэкспортировать `_has_enumeration_intent` так, чтобы не дублировать regex).
- То же для V8-ветки `_evidence_assess` verdict=="answer".

### 3.2 Fallback оценивать планом, под который он собирался (C3)
- `src/v7/nodes/evaluate_complex.py` (строки 49-62): для fallback-гейта брать план
  simple-попытки: `next((a["attempt_plan"] for a in attempts if a.get("stage")=="simple"), plan)`.

### 3.3 Единый retrieval_id (C4)
- `src/v7/nodes/rag_complex.py:68`: использовать `state["retrieval_id"]`, если он есть,
  вместо повторного `make_retrieval_id(active_q, ...)`.

### 3.4 Тесты
- Сценарий: triage sufficient + enumeration-запрос + complex-попытка ниже порогов →
  итог НЕ abstain, ответ из fallback.

**Acceptance:** enumeration-запрос с хорошим simple-результатом не абстейнится.

---

## Фаза 4 — Мёртвые/сломанные механизмы (D2, D3, D4, D5)

### 4.1 Фильтры (D2)
- `src/v7/bridge.py::make_vector_search_fn`: пробрасывать
  `vector_store.similarity_search_with_score(query, k=top_k, filter=filters)` при
  непустых filters (проверить сигнатуру у `VectorStoreBackend` / ChromaBackend —
  расширить протокол параметром `filter: dict | None = None`).
- `src/v7/nlp_core.py::BM25Index.search` (строки 190-197): сравнивать
  `(p.get("metadata") or {}).get(k)` вместо `p.get(k)`.
- Тест: filters={"doc_type": ...} сужает BM25-результаты, а не обнуляет их.

### 4.2 sanitize_for_llm (D3)
- `src/v7/bridge.py::make_generate_fn._generate`: применять
  `sanitize_for_llm(p["text"])` к каждому passage перед сборкой `passages_text`
  (и к visual-analyze тексту, если он попал в passage).
- Тест: чанк с "ignore previous instructions" приходит в промпт как "[FILTERED]…".

### 4.3 mmr_lambda из плана (D4)
- `src/v7/nodes/evaluate_complex.py:27`:
  `merge_all_passages(attempts, top_k=24, mmr_lambda=plan.get("mmr_lambda"))`.

### 4.4 Мёртвые конфиги (D5)
- Удалить `BM25_TOP_K`, `SEMANTIC_TOP_K` из `src/v7/config.py` (поиск использований —
  их нет) либо начать использовать; по умолчанию — удалить.

---

## Фаза 5 — API/UX (E1, E2, E3)

- `api.py` (строки 144-149): добавить ветку `result.get("intent") == "noise"` →
  answer = "Задайте вопрос по нормативной документации." (паритет с `app.py:196`),
  path = `"intent_gate → END (noise/oos)"`.
- `api.py::_infer_path` (строка 191): определять путь по
  `retrieval_attempts[-1]["stage"]` (та же логика, что
  `generate_answer._last_stage`), ключ `complex_passages` убрать.
- `src/v7/nodes/visual_enrichment.py::_needs_visual` (строка 42):
  `meta.get("page_no") is not None` вместо truthy-проверки.

---

## Фаза 6 (опционально, по согласованию) — эксперименты, требующие eval

Эти пункты меняют поведение маршрутизации/ретривала — за config-флагом, по умолчанию
выключены, включать после ручного eval-прогона:

1. **Лексический канал в rag_complex (C1):** добавить BM25-поиск + RRF в
   `rag_complex` (повторить схему `rag_simple` без multi-query). Флаг
   `V7_COMPLEX_HYBRID: bool = False`.
2. **Crossref-эскалация (F1):** сперва только инструментировать — логировать
   `crossref_hits` и долю эскалаций на golden-наборе. Затем сменить триггер на
   «упомянутый пункт N отсутствует в найденных чанках» (после фазы 2 есть
   структурный матчинг — переиспользовать его). Флаг `V7_CROSSREF_SMART_ESCALATION`.
3. **Перенос cross-ref expansion из генерации в retrieval-стадию (B3):** чтобы
   расширенные чанки проходили гейты; большой рефакторинг, делать последним.
4. **Разделение intent "oos" и "noise"** + abstain с диагностикой для OOS
   (приведение в соответствие с README-диаграммой).

---

## Порядок и проверка

1. Фазы выполнять по порядку 1 → 5 (фаза 6 — только по явному запросу).
2. После каждой фазы: `pytest tests/ -x`, отдельный коммит
   (`fix: <находки>, refs docs/reviews/2026-06-10-…`).
3. После фаз 1-3 рекомендуется ручной прогон `eval/run_v7_eval.py` (нужен
   OPENAI_API_KEY): ожидаемые сдвиги — false-sufficiency ↓ (гейты снова работают),
   возможен рост abstain-rate на in-scope (если вырастет заметно — вернуться к
   калибровке порогов, НЕ откатывать семантику score).
4. README/docs: после фазы 1 обновить описание hard gates (vector-anchored top_score);
   после фазы 2 — описание cross-reference expansion.
