# Дизайн: явный контракт решения triage + единая упаковка контекста

**Дата:** 2026-09-09
**Статус:** утверждён в брейншторме (Telegram, 4 вопроса), реализация — на свежей сессии
**Связано:** #9 (калибровка порогов), #13 / B2 (структурированный gap), внешние ревью
gpt-6-astra + gemini от 2026-09-09 (`~/career/reports/regrag-external-review-*-2026-09-09.md`)

## Проблема

Два структурных дефекта пути triage, оба помечены внешним ревью как P0. Калибровку
порогов (#9) вести поверх них бессмысленно — сетка будет мерить артефакты контракта,
а не рычаги.

### P0-1: решение неявное и противоречивое

`evaluate_triage` возвращает `sufficient: bool` плюс россыпь флагов
(`fallback_passages`, `fallback_score`, `final_passages`, `triage_gap`). Маршрут
достаётся из этого булева в `route_after_triage`, который вдобавок сам заново зовёт
`_has_enumeration_intent`. Следствия:

- enumeration-запрос одновременно `sufficient=True` **и** уходит в `rag_complex` —
  «достаточно» и «недостаточно» в одном состоянии;
- сам факт наличия `fallback_passages` в `evaluate_complex` разрешает ответ, даже
  когда полнота заведомо не проверена (тот же enumeration);
- `TRIAGE_SOFT_THRESHOLD` не влияет на долю эскалаций (обе категории
  `borderline` / `clearly_bad` → `rag_complex`), но выглядит как рычаг;
- ветка fallback при `result["sufficient"]` и `triage != "sufficient"` в текущем
  коде недостижима.

### P0-2: решение принимается не на том контексте, который увидит генератор

`evaluate_triage` проверяет достаточность (и закрытие gap) на своём списке пассажей.
Но контекст генератора собирается позже и в другом месте — внутри `bridge._generate`:
там **повторный** `expand_cross_references`, затем `expanded[:30]`. Чанк, «закрывший
пробел» на этапе решения, может не дожить до генератора.

Побочно выяснено при разборе: `MAX_CHUNKS_FOR_LLM = 10` в `config.py` **нигде не
применяется** (ссылки только из `config.py` и его теста). Реальная обрезка —
`FINAL_MERGE_TOP_K = 24` (только complex-путь, в `evaluate_complex`) плюс `[:30]`
в bridge. Запись в `CLAUDE.md` про «merge 24 + обрезка 10» — неточна, правится
отдельно.

## Решение

### 1. Новый узел `pack_context` — единственная точка сборки контекста генератора

Отвечает за: взять пассажи, которые произвёл путь (simple или complex), один раз
прогнать `expand_cross_references`, обрезать до `MAX_CHUNKS_FOR_LLM`, записать
`final_context: List[dict]` в состояние.

- `MAX_CHUNKS_FOR_LLM` **проводится в код** как реальный конфиг. Дефолт — 30
  (сохраняет текущее поведение bridge `[:30]`; значение 10 из старого конфига было
  фикцией, менять его — отдельное продуктовое решение вне этого спека).
- `bridge._generate` перестаёт расширять кросс-рефы и резать список. Читает
  `final_context` как есть. Обновляются `generate_answer` и его тесты.
- После этого узла «ложное закрытие gap» невозможно by design: и решение, и
  генератор смотрят на один объект `final_context`.

### 2. Новый узел `decide_route` — явное решение на simple-пути

Вход: `final_context` (из `pack_context`), `query`, `active_query`, `triage_gap`,
результат `check_full_triage` на `final_context`.

Пишет в состояние:

```
route_decision: Literal["generate", "complex", "abstain"]
route_reason:   str          # ровно один код из таблицы ниже
fallback_sufficient: bool    # проходит ли final_context hard-gates сам по себе
```

`sufficient: bool` сохраняется как производное (`route_decision == "generate"`) —
чтобы не переписывать за один заход всех потребителей; помечается deprecated
в комментарии.

#### Таблица reason codes

| route | reason code | условие |
|---|---|---|
| generate | `context_sufficient` | hard-gates ok, soft ok, не enumeration, gap закрыт |
| complex | `enumeration_intent` | `_has_enumeration_intent(query)` — полнота требует перечисления |
| complex | `crossref_gap_open` | `triage_gap["open"]` непуст после упаковки |
| complex | `zero_overlap_original` | `keyword_overlap_original == 0.0` при ненулевом active |
| complex | `triage_borderline` | `check_full_triage` → `borderline` |
| complex | `triage_clearly_bad` | `check_full_triage` → `clearly_bad`, пул непуст |
| abstain | `empty_pool` | пассажей нет вовсе |
| abstain | `zero_overlap_both` | overlap = 0 и по `query`, и по `active_query` |

Порядок проверки — сверху вниз, первое совпавшее. `abstain`-условия проверяются
**первыми** (нет смысла платить за complex, когда пул пуст или запрос не пересёкся
с корпусом ни по одной форме — см. вопрос 3 брейншторма, вариант Б).

#### `fallback_sufficient` — статус, не разрешение

Пишется всегда, независимо от `route_decision`. `evaluate_complex` использует его
как сигнал «можно откатиться на simple-результат вместо abstain». Но:

- `fallback_sufficient == True` при `route_decision == "complex"` и
  `route_reason == "enumeration_intent"` **не даёт** генерировать ответ: полнота
  перечисления не подтверждена, откат на simple вернул бы заведомо неполный ответ.
  `evaluate_complex` обязан это учитывать — откат по `fallback_sufficient`
  запрещён, когда причина эскалации `enumeration_intent`.

### 3. Граф

```
rag_simple → evaluate_triage → pack_context → decide_route ─┬─ generate → generate_answer
                                                             ├─ complex  → rag_complex
                                                             └─ abstain  → abstain

rag_complex → evaluate_complex ─┬─ (sufficient)  → pack_context → generate_answer
                                └─ (insufficient) → abstain
```

- `evaluate_triage` ужимается до hard-gates + построения `triage_gap`. Вся логика
  эскалаций (`_count_crossref_hits`, `_has_enumeration_intent`, zero-overlap,
  ветки с `_crossref_expander`) переезжает в `decide_route`, работая на
  `final_context`.
- `pack_context` — общий узел, входы из `evaluate_triage` и из `evaluate_complex`
  (ветка sufficient). Точная топология (один узел с двумя входами vs. два
  инстанса) — на этапе плана.
- `route_after_triage` удаляется, его роль у `decide_route` (conditional edge
  читает `route_decision`).
- `evaluate_complex` дополнительно проставляет `route_decision` / `route_reason`
  (`generate` / `abstain`, коды `complex_sufficient` / `complex_insufficient`) —
  для единообразия телеметрии; своя логика fallback остаётся.

### 4. `_crossref_expander` и чистота

`decide_route` (как и нынешний `evaluate_triage`) зовёт `_crossref_expander` —
функция не чистая. Для будущей офлайн-сетки #9 это учитывается так: expander
инъектируется, в аудите его результаты кешируются, state каждого прогона
изолируется. В этот спек калибровка не входит — только фиксируем, что контракт
её не ломает.

## Что не входит

- Калибровка порогов (#9) — следующий шаг, после этого контракта.
- Разметка обязательных элементов enumeration на dev — параллельный трек #9,
  не блокируется этим спеком.
- Развязка порогов simple/complex (`max(COMPLEX_THRESHOLD, plan.threshold)`
  в `rag_complex`) — относится к #9-калибровке, не к контракту. Помянуть в roadmap,
  делать там.
- Объединение `evaluate_triage` и `evaluate_complex` в один evaluator — заманчиво,
  но это отдельный рефакторинг, не P0.

## Тестирование

- `decide_route`: таблица reason codes — по юнит-тесту на строку, TDD. Порядок
  проверки (abstain раньше complex) — отдельный тест.
- `pack_context`: идемпотентность (`expand` один раз), обрезка по
  `MAX_CHUNKS_FOR_LLM`, `final_context` совпадает с тем, что уходит в генератор.
- `generate_answer` / `bridge._generate`: тесты на повторное расширение — удалить
  или переписать под «читает `final_context` как есть».
- Регресс: `eval/measure_triage_gap.py` (без LLM) — доля эскалаций simple→complex
  и Hit Rate@12 по выходу не должны просесть против `main` (база B2:
  0.140 / 0.860 на held-out 43).
- Полный прогон тестов: сверять с предсуществующими 5 красными
  (`test_agent_tools` ×2, `test_bridge_rerank` ×3), а не считать регрессом.

## Открытые мелочи для плана

- Точное имя конфига обрезки: оживить `MAX_CHUNKS_FOR_LLM` или ввести
  `PACK_CONTEXT_TOP_K`. Склоняюсь оживить существующий.
- Нужен ли `final_context` отдельным ключом или переиспользовать `final_passages`.
  Скорее новый ключ — `final_passages` сейчас значит «до упаковки».
