# Regulatory RAG — сведения об объекте подразделения (ObjectProfile): спецификация

Дата: 2026-09-15. Статус: **согласована в brainstorming, не реализована.**
Дополняет [спецификацию MVP Q&A для подразделений](./2026-09-14-department-qa-mvp-design.md)
(§5 модель документов, §8 контракт, §11 evaluation). Второй контракт ответа не заводится (П7).

## 1. Проблема

Лист особенностей объекта защиты подразделения сейчас индексируется как обычный внутренний
документ (`scope: unit`). Смоук 15.09 (6 вопросов, gpt-4o-mini, коллекция `department_demo`)
показал три провала, которые поиском не лечатся:

| Подразделение | Вопрос | Что пошло не так |
|---|---|---|
| `unit_dispatch` | Сколько огнетушителей и нужен ли план эвакуации? | Разделы листа вытеснены общими ЛНА из top-8; численность объекта в ответ не попала |
| `unit_office` | Тот же вопрос | Норма ППР найдена, но `external_basis` пуст → `needs_review` |
| не выбрано | Нужен ли на нашем объекте план эвакуации? | Ответ «да» вместо `needs_context` |

Факт объекта (численность, системы, дежурство) — не норма и не фрагмент для ранжирования: он
нужен в каждом ответе по этому подразделению целиком. Поэтому лист передаётся структурой,
а не поиском.

## 2. Решения

### 2.1. Гранулярность: разделы, не поля

`ObjectProfile` = разделы шаблона листа `obj_s1 … obj_s9` с исходным текстом раздела.

```python
class ObjectSection(BaseModel):
    id: str                      # obj_s1 … obj_s9
    number: int                  # номер раздела шаблона
    title: str                   # «Системы противопожарной защиты объекта»
    text: str                    # исходный текст раздела
    presence: Literal["present", "empty", "missing"]

class ObjectProfile(BaseModel):
    unit_id: str
    document_id: str
    title: str
    as_of_date: Optional[date]   # «Дата заполнения»; плейсхолдер [дата] → None
    sections: dict[str, ObjectSection]
```

- `presence`: `present` — у раздела есть текст; `empty` — заголовок есть, текста нет;
  `missing` — заголовка раздела в листе нет. Парсер содержание не понимает, поэтому
  known/unknown/not_applicable не вводятся.
- **Ограничение:** «Информация уточняется» — это `present`. Неизвестность факта внутри текста
  раздела распознаёт модель, не парсер (проверяется вопросом «факт неизвестен», §5.2).
- Раздел 6 «Контактные телефоны объекта» парсится, но в промпт не передаётся.
- **Правила разбора.** Раздел распознаётся по заголовку `## N <название>`, где N от 1 до 9;
  название не сверяется. Заголовки с другими номерами и первый заголовок листа
  («Лист особенностей…») разделами не считаются. Повтор номера → `ObjectProfileError`.
  Строка `Дата заполнения: …` ищется в любом месте листа и **вырезается** из текста раздела,
  в котором стоит (сейчас это раздел 9): дата живёт только в `as_of_date`.
- Structured-поля (численность, категории как числа/enum) — только если eval покажет
  систематическую ошибку применения факта. В этой итерации не делаются.

### 2.2. Давность

- Порога давности в коде нет.
- `as_of_date` показывается в ответе (`profile_as_of_date`) и в UI.
- Ответ процитировал хотя бы один `obj_s*`, а `as_of_date is None` →
  `needs_review` + `object_profile_undated`.
- Синтетические листы `unit_office`, `unit_dispatch`, `unit_depot` получают конкретные даты
  заполнения (данные вымышленные, обезличивать нечего). Без даты остаётся только `unit_partial`.

### 2.3. Контракт внутри `ModelAnswer`

`external_basis` / `internal_basis` не меняются. Добавляются:

```python
class ObjectFact(BaseModel):
    statement: str
    evidence_ids: list[str]          # только obj_s*

class AppliedConclusion(BaseModel):
    statement: str
    evidence_ids: list[str]          # ≥1 obj_s* И ≥1 ext_*/int_*

class ModelAnswer(BaseModel):
    ...
    object_facts: list[ObjectFact] = []
    applied_conclusions: list[AppliedConclusion] = []
```

`Level` расширяется значением `"object"`, префикс `obj_` → `object`.

Правила `check_citations` / `decide`. Порядок: `out_of_scope` → цитаты (при любой ошибке
возвращается ровно `["citation_invalid"]`) → уточнения → reason_codes.

Определения:

- `has_ext` / `has_int` — есть хотя бы один `ext_*` / `int_*` в `external_basis` /
  `internal_basis` **или** в `applied_conclusions`: норма, применённая к объекту, считается
  найденной нормой.
- `cites_object` — есть хотя бы один `obj_*` в `object_facts` или `applied_conclusions`.
- `fact_only` — `object_facts` не пуст, `applied_conclusions` пуст, `has_ext` и `has_int` ложны.

| Ситуация | Статус / reason |
|---|---|
| Пустой `evidence_ids` у любого элемента (`Basis`, `ObjectFact`, `AppliedConclusion`) | `failed` / `citation_invalid` |
| Несуществующий id, в том числе `obj_s*` при непереданном профиле | `failed` / `citation_invalid` |
| Уровень перепутан: `ext_`/`int_` в `object_facts`, `obj_` в `external_basis`/`internal_basis` | `failed` / `citation_invalid` |
| Ссылка на раздел с `presence: missing` или на раздел 6 — такие разделы не кладутся в `evidence`, это частный случай «несуществующего id» | `failed` / `citation_invalid` |
| `applied_conclusions` без `obj_*` | `failed` / `citation_invalid` |
| `applied_conclusions` с `obj_*`, но без `ext_*`/`int_*` | `needs_review` / `applied_without_norm` |
| `clarifying_questions` не пуст и не (`has_ext` и `has_int`) — в том числе при `fact_only` | `needs_context` / `applicability_unclear` (как в v1) |
| `fact_only` | `external_evidence_missing` / `internal_evidence_missing` **не** добавляются |
| Не `fact_only` (есть applied или хотя бы одна норма) | `external_evidence_missing` / `internal_evidence_missing` по `has_ext` / `has_int`, как в v1 |
| `cites_object` и `as_of_date is None` | добавляется `object_profile_undated` |
| `possible_mismatch` при `has_ext` и `has_int` | добавляется `possible_mismatch` (как в v1) |

reason_codes накапливаются в порядке строк таблицы: `applied_without_norm`,
`external_evidence_missing`, `internal_evidence_missing`, `object_profile_undated`,
`possible_mismatch`. Непустой список → `needs_review`, пустой → `answered`.
Пример: `object_facts` + `external_basis`, без applied → `needs_review` /
`internal_evidence_missing`.

Обоснование строки «только факты → answered»: вопрос «кто ночью принимает сигнал» — вопрос о
факте объекта, нормы в нём нет; v1 штрафовал его `external_evidence_missing`.

### 2.4. Поток данных

1. `corpus/manifest.yaml`: у листа `role: object_profile` (остальные документы — без `role`).
   Валидация: `role: object_profile` допустим только при `source_type: internal`, `scope: unit`;
   не больше одного листа на `unit_id`. `Manifest` получает поле
   `object_profiles: dict[unit_id, file]` — `role` в метаданные чанков не пишется.
2. `apply_manifest` отбрасывает чанки документов с `role: object_profile` — листы не попадают
   ни в Chroma, ни в BM25.
3. Новый модуль `src/department_qa/object_profile.py`: `load_profiles(manifest, source_dir)
   → dict[unit_id, ObjectProfile]`; `source_dir` передаёт `wiring.py` из
   `settings.SOURCE_DOCS_PATH`. Лист, в котором не распознан ни один раздел шаблона,
   → `ObjectProfileError` при старте (ошибка данных, не деградация). Отдельный отсутствующий
   раздел → `missing`, не падение.
4. `answer_question(..., profile: Optional[ObjectProfile])`: разделы (кроме 6) становятся
   evidence уровня `object` с id `obj_sN` и попадают в тот же словарь `evidence` для проверки.
   Поля: `title` — название листа, `locator` — «N <название раздела>», `source` — имя файла,
   `document_id` — из manifest. Разделы `missing` и раздел 6 в `evidence` не кладутся.
5. Промпт `department_answer` **v2** (v1 остаётся в registry для парного прогона):
   блок «СВЕДЕНИЯ ОБ ОБЪЕКТЕ» с датой заполнения и разделами;
   - подразделение не выбрано → «объект не выбран»; вопрос, зависящий от объекта, →
     `clarifying_questions`;
   - подразделение выбрано, листа нет → пометка «лист не предоставлен», факты не придумывать;
   - раздел `empty` показывается как «(не заполнено)».
   Правила: `object_facts` — только `obj_*`; `applied_conclusions` — факт объекта плюс норма;
   норма без факта — в `external_basis`/`internal_basis`.
   `PromptVars` расширяется полями `object_label` и `object_sections` со значениями по
   умолчанию; v1 рендерится из того же класса и новые поля игнорирует.
6. `DepartmentResponse` + `object_facts`, `applied_conclusions`, `profile_as_of_date`.
   `evidence` в ответе фильтруется по id из всех четырёх списков, включая `obj_*`.
7. Streamlit-страница: блок «Сведения объекта» (процитированные разделы и дата заполнения).

## 3. Вне этой итерации

Обогащение поискового запроса полями профиля; LLM-rewrite запроса; профиль через tool-call;
разбор произвольных DOCX листов; порог давности в коде; structured-поля (§2.1); полноценные
пользователи. Позже, в eval MVP (§11 основной спеки): ~40 случаев, gold как «норма + факты».

## 4. Тестовые данные

4-й синтетический лист `source_docs_dept/int_unit_partial_list.md` (gitignored, как остальные),
`unit_id: unit_partial`, ~50 человек одновременно, раздел 5 «Дежурный персонал» отсутствует,
«Дата заполнения» не указана. Запись в manifest с `role: object_profile`; стоп-лист прогоняется.

## 5. Проверка

### 5.1. Unit-тесты (без сети)

- Парсер: 9 разделов → `present`; пустой раздел → `empty`; нет заголовка → `missing`;
  «Информация уточняется» → `present`; `[дата]` и отсутствие строки → `as_of_date=None`;
  корректная дата → `date`; ноль разделов → `ObjectProfileError`.
- Manifest: `role: object_profile` у external или `scope: company` → `ManifestError`;
  `apply_manifest` отбрасывает чанки листа.
- `decide`: каждая строка таблицы §2.3 — отдельный кейс; регрессия v1-кейсов без профиля.
- `answer_question`: `obj_*` попадают в evidence; раздел 6 не рендерится; без профиля в промпте
  «объект не выбран» / «лист не предоставлен».
- Промпт v2: рендер через `PromptVars`, блок объекта, `empty` → «(не заполнено)».

Сверка с базой: предсуществующие красные тесты из `CLAUDE.md` проекта не считаются поломкой.

### 5.2. Парный прогон v1 vs v2 (платный, согласован 15.09)

| # | Подразделение | Вопрос |
|---|---|---|
| 1 | `unit_office` | Сколько огнетушителей должно быть на объекте и нужен ли план эвакуации? |
| 2 | `unit_dispatch` | Тот же вопрос |
| 3 | `unit_depot` | Действует ли у нас инструкция для складских помещений и когда испытывать наружную пожарную лестницу? |
| 4 | не выбрано | Нужен ли на нашем объекте план эвакуации? |
| 5 | `unit_office` | Кто принимает сигнал пожарной сигнализации ночью? |
| 6 | `unit_office` | Как оформить ежегодный отпуск? |
| 7 | `unit_partial` | Нужен ли на нашем объекте план эвакуации? (граница 50 человек) |
| 8 | `unit_partial` | Кто у нас принимает сигнал пожарной сигнализации ночью? (раздел 5 отсутствует) |

- **Порядок сборки коллекций.** После введения `role` пересборка `department_demo` потеряет
  листы. Поэтому первый шаг плана, до любых изменений кода: лист `unit_partial` добавляется
  в manifest без `role` и `department_demo` пересобирается текущим `index.py` — это база v1
  на тех же данных. После этого `department_demo` не пересобирается до конца сравнения.
  Затем в manifest у четырёх листов ставится `role: object_profile`.
- v1 — коллекция `department_demo`, промпт v1.
- v2 — новая коллекция `department_demo_v2` без листов, профили из `object_profile.py`,
  промпт v2.
- Скрипт `eval/run_object_profile_pair.py`; версия промпта выбирается через
  `PROMPT_DEPARTMENT_ANSWER_VERSION` (`PromptManager`). `active_version` в `registry.yaml`
  остаётся v1 до разбора итога; переключение — отдельное решение по результату.
- Модель `gpt-4o-mini`, 16 вызовов, оценка < $0.05.
- **Ожидания** (статус, reason_codes, ключевой вывод, какие `obj_s*` должны быть процитированы)
  записываются в `eval/data/object_profile_pair_expectations.yaml` и коммитятся **до** прогона.
- Итог: число совпадений с ожиданиями из 8 у v1 и у v2 плюс разбор каждого расхождения.
  Судья-LLM не используется, сверка ручная по ожиданиям.
- N=8 — демонстрация механизма, не замер качества; в выводах так и пишется.

## 6. Критерии готовности

- Unit-тесты §5.1 зелёные, база красных тестов не выросла.
- Листы отсутствуют в новой коллекции (проверка по `source` в Chroma).
- Файл ожиданий закоммичен раньше лога прогона.
- Результат парного прогона и разбор записаны в `docs/roadmap.md`; решение — в
  `docs/explanation/design-decisions.md` (§10).
