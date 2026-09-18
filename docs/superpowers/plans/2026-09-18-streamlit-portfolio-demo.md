# Regulatory RAG — план реализации Streamlit Portfolio Demo

## 1. Цель

Переделать текущий Streamlit-интерфейс в portfolio demo **Regulatory Compliance Assistant для распределённой организации**.

Главный принцип реализации:

> Не перестраивать RAG/backend без необходимости. Использовать уже существующий `department_qa` vertical slice и поверх него построить понятный presentation layer.

После одного запроса hiring manager должен видеть:

`Подразделение → факты объекта → вопрос → статус доказательств → ответ → почему такой вывод → основания → источники → технические детали`.

---

## 2. Что уже есть в репозитории

Backend для этого сценария в основном уже реализован.

Есть:

- `src/department_qa/service.py`
  - два scoped retrieval: external + internal;
  - профиль подразделения;
  - structured answer;
  - citation validation;
  - verifier;
  - статусы `answered / needs_context / needs_review / out_of_scope / failed`.

- `src/department_qa/object_profile.py`
  - структурированный профиль объекта;
  - typed fields;
  - `known / unknown / not_applicable`;
  - дата профиля;
  - отдельные evidence IDs `obj_*`.

- `src/department_qa/contract.py`
  - `external_basis`;
  - `internal_basis`;
  - `object_facts`;
  - `applied_conclusions`;
  - проверка связей с evidence.

- `src/department_qa/view.py`
  - presentation helpers;
  - status banner;
  - разрешение evidence IDs в title/locator;
  - подпись профиля.

- `src/department_qa/wiring.py`
  - существующий hybrid retrieval:
    `dense + BM25 → RRF`;
  - сборка всего Department QA stack.

- `pages/1_Подразделения.py`
  - уже работающий vertical slice UI.

То есть большая часть нужной семантики уже находится в backend.

Основная проблема сейчас — UI показывает это как технический прототип:

`Законодательство | ЛНА | Сведения объекта | Применение`

вместо законченного пользовательского сценария.

---

## 3. Архитектурное решение

### Не строим второй pipeline

Новая страница должна вызывать:

```text
build_department_stack()
        ↓
answer_question()
        ↓
DepartmentResponse
        ↓
presentation helpers
        ↓
Streamlit
```

`DepartmentResponse` остаётся главным контрактом между backend и UI.

Добавляем только те поля, которых реально не хватает для отображения portfolio demo.

---

## 4. Сделать Department QA главным интерфейсом

Сейчас `app.py` — старый generic Regulatory RAG chat, а Department QA находится на отдельной странице.

Для portfolio demo это создаёт два конкурирующих продукта.

Нужно сделать наоборот.

### `app.py`

Сделать главным экраном:

**Regulatory Compliance Assistant**

Именно сюда перенести новый flow.

Старый generic chat убрать из публичного пользовательского пути.

Не должно быть ситуации, когда hiring manager сначала попадает в старый чат «спросите про ГОСТ/СНиП», а новый сценарий ищет в sidebar.

### `pages/1_Подразделения.py`

После переноса:

- удалить как отдельный дублирующий сценарий;
- либо оставить только временно во время разработки.

Финальный публичный demo должен иметь один главный flow.

### Generic Regulatory RAG — вторичная линия, не удалять

Generic-чат (текущий `app.py`) **не удаляется**: он переезжает на отдельную вторичную
страницу `pages/2_Общий_поиск.py`. Причина: README продаёт проект как две продуктовые линии,
и витринный eval 56 goldenset относится именно к generic-линии. В main-flow его быть не
должно, но как отдельная страница он остаётся доступен.

---

## 5. P0. Presentation model

Не давать Streamlit напрямую разбирать `reason_codes`.

Расширить:

`src/department_qa/view.py`

и сделать его адаптером:

```text
DepartmentResponse
       ↓
Portfolio presentation model
       ↓
Streamlit
```

Добавить helper:

```python
presentation_status(response)
```

UI-статусы:

```text
answered
    → sufficient

needs_context
    → clarification_required

out_of_scope
    → out_of_scope

needs_review + missing evidence
    → insufficient_evidence

needs_review + mismatch/contradiction
    → conflict / needs_review
```

Backend-статусы при этом не менять.

Это позволит не тащить технические `needs_review`, `verification_failed`,
`internal_evidence_missing` непосредственно в UX.

---

## 6. P0. Выбор подразделения и compact profile

Использовать уже загружаемые:

```python
config.profiles
```

Из manifest брать человекочитаемое название вместо:

```text
unit_office
unit_dispatch
unit_depot
```

Например:

```text
Офис
Диспетчерский центр
Склад и мастерская
Производственный корпус
```

Добавить helper:

```python
compact_profile(profile) -> list[str]
```

Он строит строку только из наиболее понятных typed fields.

Например:

```text
24 сотрудника · АУПТ отсутствует · план эвакуации есть
```

Ниже:

```text
Данные на 12.09.2026
```

Полный профиль остаётся в expander:

```text
Подробнее об объекте ▾
```

---

## 7. P0. Первый экран

После загрузки UI должен показывать только:

```text
Regulatory Compliance Assistant

Ответы с учётом нормативных документов,
локальных актов и данных подразделения.

Подразделение
[ Офис ▼ ]

24 сотрудника · АУПТ отсутствует · ...
Данные на ...

Что хотите проверить?

[ Какие требования применимы? ]
[ Есть ли расхождения? ]
[ Как часто требуется проверка? ]

[ Задайте вопрос........................ ] [Спросить]
```

Убрать из первого экрана:

- model/provider;
- embeddings;
- collection;
- database path;
- reindex;
- source slider;
- V7/V2;
- prompt version;
- старую историю chat bubbles.

---

## 8. P0. Progress

Нельзя просто нарисовать три spinner-сообщения — стадии должны соответствовать реальному backend flow.

`answer_question()` сейчас синхронный, поэтому добавить необязательный callback:

```python
progress_fn(stage)
```

Реальные события:

```text
retrieval_started
retrieval_completed

generation_started
generation_completed

verification_started
verification_completed
```

UI отображает их через `st.status()`.

Текущая архитектура фактически работает:

```text
retrieval
→ structured answer generation
→ deterministic/verifier checks
```

Поэтому UI лучше показывать:

```text
Поиск требований
↓
Формирование ответа по найденным основаниям
↓
Проверка доказательств
```

---

## 9. P0. Evidence status

Сразу под вопросом.

### sufficient

```text
✓ Доказательств достаточно

Ответ построен на нормативных источниках,
локальных актах и данных подразделения.
```

### clarification_required

```text
Нужно уточнение

Для проверки применимости требования
не хватает данных об объекте.

Есть ли ...?
```

### insufficient_evidence

```text
Недостаточно оснований для надёжного ответа

Не найдено подтверждение в локальном акте.
```

### out_of_scope

```text
Запрос вне области базы знаний

Доступные источники не покрывают этот вопрос.
```

Технические reason codes пользователю не показывать.

---

## 10. P0. Основной ответ

При `sufficient`:

```text
## Ответ

<response.answer>
```

Убрать нынешний label:

```text
Краткий черновик
```

Portfolio demo должен выглядеть как законченная система.

При `clarification / insufficient / OOS` не показывать уверенный generated answer.

---

## 11. P0. «Почему такой вывод»

Это главный showcase проекта.

Новые backend-сущности для этого, скорее всего, не нужны.

Уже существуют:

```python
object_facts
external_basis
internal_basis
applied_conclusions
evidence_ids
```

Добавить helper:

```python
build_reasoning_chains(response)
```

View-model:

```python
ReasoningChain(
    fact="24 сотрудника",
    requirement="...",
    conclusion="..."
)
```

Алгоритм:

```text
AppliedConclusion
        ↓
его evidence_ids
        ↓
obj_* → факт
ext_*/int_* → нормативное основание
        ↓
соответствующий statement
```

UI:

```text
Почему такой вывод

Факт
24 сотрудника · ...

↓

Требование
Проверка проводится ...

↓

Вывод
Для данного подразделения ...
```

Никакой новой LLM-генерации для этого блока.

Он должен собираться детерминированно из structured response.

---

## 12. P0. «На чём основан ответ»

Следом три категории:

```text
Внешнее требование
Локальный акт
Факты подразделения
```

Использовать уже существующие:

```python
external_basis
internal_basis
object_facts
```

и lookup через `response.evidence`.

Показывать:

```text
Название документа
locator
короткий statement/excerpt
```

Не показывать raw chunk на основном экране.

---

## 13. P0. Источники, на которые опирается ответ

Существующий expander сохранить концептуально, но переделать визуально и переименовать.

```text
Источники, на которые опирается ответ (N) ▾
```

Заголовок не «Все источники»: `DepartmentResponse.evidence` — только процитированные
фрагменты, а не вся выдача retrieval. Выдачу целиком в контракт не добавляем (отдельное
решение позже, если понадобится).

Внутри:

```text
Название
Тип: внешний / локальный / профиль объекта
Locator
Relevant excerpt
Retrieval score
```

Raw text только здесь.

Для retrieval score сделать небольшое изменение контракта:

```python
Evidence.retrieval_score: float | None
```

В `_to_evidence()` сохранить существующий `passage["score"]`.

Для object profile score будет `None`, что корректно: профиль не retrieval hit.

---

## 14. P0. Technical details

Collapsed по умолчанию:

```text
Технические детали ▾
```

Показывать только реально известные значения:

```text
LLM
Gemini ...

Retrieval
dense + BM25 → RRF

Used as evidence
5 passages

Latency
3.8 s

Trace ID
...
```

Не писать:

```text
Route: simple / complex
```

потому что текущий `department_qa` не использует такой routing.

Latency можно измерять вокруг `answer_question()` на UI/service boundary.

Cost оставить на P2, если нет надёжного usage accounting.

**Verifier в UI не включается** (`verifier_fn=None`, как и сейчас на странице). По прогону
17.09 он давал ложные `verification_*` на верных ответах — на витрине это только вредит.
`insufficient_evidence` и `conflict` строятся из reason_codes `decide()` и без него.

---

## 15. P0. Clarification

Backend уже умеет возвращать:

```python
needs_context
clarifying_questions
```

Первая версия UI:

```text
Нужно уточнение

<question>

[ Да ] [ Нет ] [ Не знаю ]
```

Но сейчас `clarifying_questions` — текст, а не структурированный ключ поля.

Поэтому полноценное:

```text
ответ пользователя
→ обновление конкретного typed field
→ повтор исходного запроса
```

потребует нового контракта вроде:

```python
Clarification(
    field_id="aupt_present",
    question="Есть ли АУПТ?",
    allowed_answers=["yes", "no", "unknown"]
)
```

Не делать fragile parsing текста вопроса.

Поэтому P0 можно разделить:

### P0a
Отображать clarification корректно.

### P0b
Добавить structured clarification и continuation flow.

Это единственное место в спеке, где требуется заметное расширение backend-контракта.

**Решение 18.09: P0b вне P0.** На витрину — только отображение (P0a). `ObjectProfile`
неизменяем: его `content_sha256` и `as_of_date` держат воспроизводимость eval, а правка
typed field из ответа пользователя ломает семантику снапшота. Когда continuation будет
делаться, ответы пользователя идут отдельным эфемерным `overrides` в `answer_question`,
лист объекта не перезаписывается.

---

## 16. P1. Conflict state

Backend уже имеет сигналы:

```text
possible_mismatch
verification_contradiction
```

Поэтому не нужен отдельный conflict detector.

Добавить presentation:

```text
Обнаружено расхождение

Внешнее требование
...

Локальный акт
...

Как обработано
Требуется проверка специалистом.
```

Не показывать этот блок без реального conflict reason.

---

## 17. P1. Следующий вопрос

Не возвращаться к обычным chat bubbles.

Session state:

```python
selected_unit
last_question
last_response
```

После результата:

```text
[ Задать следующий вопрос... ]
```

Новый вопрос заменяет структурированный result screen.

Историю пока не делать.

---

## 18. P1. Feedback

Уже существующие:

```python
render_feedback()
default_feedback_writer()
```

переиспользовать.

Перенести 👍/👎 под результат.

Backend здесь менять не нужно.

---

## 19. P2

Только после готового demo:

- история;
- developer mode;
- model override;
- streaming;
- стоимость запроса;
- расширенные telemetry metrics.

Не трогать до завершения P0/P1.

---

## 20. Изменения по файлам

### `app.py`

Сильная переработка.

Оставить:

- initialization;
- resource caching;
- logging;
- telemetry/feedback где нужно.

Удалить старый generic chat UI.

Сделать главным Department QA portfolio screen.

### `pages/1_Подразделения.py`

После миграции удалить/вывести из публичной навигации.

Не поддерживать два почти одинаковых UI.

### `pages/2_Общий_поиск.py` (новый)

Сюда переезжает текущий generic-чат из `app.py`. Вторичная линия продукта, вне main-flow.

### `src/department_qa/view.py`

Основной объём новой логики presentation layer:

```text
presentation_status()
compact_profile()
build_reasoning_chains()
evidence_cards()
technical_details()
```

Функции остаются pure — без `streamlit`.

### `src/department_qa/contract.py`

Минимально:

```python
Evidence.retrieval_score: Optional[float]
```

Позже, для continuation:

```python
Clarification
```

### `src/department_qa/service.py`

Минимально:

- сохранить score;
- optional progress callback;
- при необходимости diagnostics;
- позже structured clarification.

Retrieval/prompt/decision logic не переписывать.

### `src/department_qa/wiring.py`

Практически без изменений.

Hybrid retrieval уже подходит.

### `src/department_qa/object_profile.py`

Backend parsing не менять.

Добавить только presentation helper вне этого модуля для compact profile.

---

## 21. Тесты

Существующие unit-тесты сохранить.

Добавить:

### `test_presentation_status.py`

Все backend-status → UI-status mappings.

Отдельно — **приоритет reason_codes**: `needs_review` и `needs_context` могут нести сразу
несколько кодов, порядок маппинга фиксируется таблицей и тестируется. `needs_review` +
`external_evidence_missing`/`internal_evidence_missing` → `insufficient_evidence`;
`needs_review` + `possible_mismatch`/`verification_contradiction` → `conflict`.

### `test_reasoning_chain.py`

Проверить:

`obj evidence + norm evidence → correct chain`.

### `test_compact_profile.py`

Known/unknown/not applicable.

Расширить:

### `tests/department_qa/test_page.py`

Smoke cases:

1. initial screen;
2. answered;
3. needs_context;
4. insufficient evidence;
5. out of scope;
6. conflict.

Критический regression test:

> insufficient/OOS не должны показывать confident answer.

---

## 22. Порядок реализации

Начинаем с **pure presentation layer**, а не с UI shell: риск и новизна — в маппинге
статусов и reasoning chain, а не в Streamlit. Их можно отладить детерминированно, без
браузера и без платных вызовов.

### Этап 1 — Presentation layer (TDD, без Streamlit)

В `src/department_qa/view.py`:

```text
presentation_status()      — backend status + reason_codes → UI-статус
compact_profile()          — компактная строка из typed fields
build_reasoning_chains()   — Факт → Требование → Вывод из evidence_ids
evidence_cards()           — карточки источников с retrieval_score
technical_details()        — технический блок
```

Плюс `Evidence.retrieval_score` в `contract.py` и проброс `passage["score"]` в
`service._to_evidence()`. Unit-тесты на каждый helper.

### Этап 2 — UI shell

Переделать root screen:

```text
header
unit selector
compact profile
examples
input
```

Generic-чат переезжает в `pages/2_Общий_поиск.py`. Backend не меняется.

### Этап 3 — Result screen

Добавить:

```text
question
status
answer
bases
sources
technical details
```

### Этап 4 — Real progress

Добавить `progress_fn` callback в `answer_question`.

### Этап 5 — Clarification (P0a)

Отображение корректного вопроса; continuation (P0b) не делать.

### Этап 6 — Conflict + feedback

P1.

### Этап 7 — визуальная полировка

Whitespace, типографика, компактные блоки, единый accent.

---

## 23. Что сознательно НЕ менять

Для этой итерации не трогаем:

- Chroma;
- hybrid retrieval;
- BM25;
- RRF;
- chunking;
- indexing;
- embeddings;
- prompt architecture без выявленной проблемы;
- LangGraph/V7 generic pipeline;
- eval methodology.

Portfolio value здесь даст не ещё один слой AI-архитектуры, а то, что уже существующая архитектура станет очевидной человеку за 30–60 секунд.

---

## 24. Definition of Done

P0 можно считать готовым, когда после одного запроса без README понятно:

```text
какой объект выбран
↓
какие его факты известны
↓
хватило ли оснований
↓
каков ответ
↓
какой факт повлиял на применимость
↓
какая норма применена
↓
почему получен именно этот вывод
↓
какие документы это подтверждают
↓
может ли система отказаться/запросить данные
↓
как устроена техническая часть
```

После этого уже имеет смысл заниматься P1/P2, а не расширять scope.
