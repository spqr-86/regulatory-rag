# План: общий scoped retrieval без большой перестройки

Дата: 19.09.2026. Основание: [спецификация, редакция 4](../../spec-shared-retrieval-plan.html).
Проверенный исходный commit: `e9f5cb4`. Статус: план, реализация не начата.

## Результат

Department вызывает retrieval-часть существующего V7-графа для каждого выбранного
корпуса параллельно (не более двух ветвей на запрос). Каждая ветвь сама проходит simple → проверка → complex
при необходимости. Затем один Department generator получает готовые блоки и
опциональный ObjectProfile. Generic сохраняет полный граф и свои настройки.
Общая модель reranker исполняет predict последовательно; подготовка и поиск ветвей
могут перекрываться. Department переиспользует exact-text embedding внутри запроса.

Существующий `RetrievalPlan` остаётся. Новый универсальный coordinator не создаётся.
Retrieval-only wrapper не содержит собственной таблицы routing или алгоритмов поиска.

Четыре последовательных PR. Первые два дают общий и безопасный механизм; третий
подключает Department за непроизводственной проверкой; четвёртый — UI и cutover.
Не считать это «правкой одной функции»: bound dependencies и scoped pack нужны,
чтобы переиспользование узлов было корректным.

## 0. Зафиксировать исходную точку

До feature-кода прочитать актуальные проектные правила для Python и LLM-кода.
Проверить worktree и сохранить посторонние изменения; работать в отдельной ветке.
Не запускать индексацию и не менять `.env` для снятия baseline.

1. Записать commit, конфигурацию Generic/Department, версии prompts, runtime flags,
   идентификаторы коллекций, hashes профилей и expectations. Секреты не сохранять.
2. Запустить текущий unit gate, записать реально воспроизведённые красные тесты.
   Старые числа из handoff не использовать как результат нового прогона.
3. Снять/сохранить фиксированные backend fixtures для Generic: simple accepted,
   escalation accepted, simple fallback после complex, empty, retrieval error,
   clarification. Записать route, ordered final IDs, scores, fallback origin.
4. Зафиксировать отдельные benchmark-результаты старого Department V2. Старые прогоны
   можно использовать, только если corpus/model/config совпадают; иначе baseline
   снимается на исходном commit в отдельном окружении с теми же read-only данными.
5. Зафиксировать план сравнения старого pipeline, нового sequential (workers=1) и
   нового parallel на одинаковом коде/config. Измерять cold/warm модель, ожидание
   очереди/reranker и полное время ответа; не приписывать параллельности эффект
   перехода Department на другую retrieval policy.
6. Подготовить небольшой scoped-набор: один закон + один ЛНА, короткий ЛНА, only-law,
   only-internal company/unit, одна пустая ветвь, complex только в одной ветви,
   одинаковые source/chunk IDs, profile on/off, отсутствие нужного факта.

**Выход:** локальный baseline-отчёт, неизменяемые fixtures/expectations и список
известных failures. Для платных прогонов сначала зафиксировать состав и бюджет;
создание этой спеки и плана не означает их запуск.

## PR 1. Retrieval-only режим и привязанные зависимости

**Назначение:** переиспользовать весь путь поиска, не копируя его в Department.

Основные файлы:

- `src/v7/graph.py`, новый небольшой `src/v7/retrieval.py`;
- `src/v7/bridge.py`, `src/v7/nlp_core.py`;
- `src/v7/nodes/rag_simple.py`, `rag_complex.py`, `evaluate_triage.py`,
  `evaluate_complex.py`, `visual_enrichment.py`, при необходимости `generate_answer.py`;
- `src/v7/pack_context.py`, `src/v7/cross_ref.py` для передачи callback dependencies;
- `src/backends/chroma_backend.py`, `src/indexing/vector_store.py` для явной привязки store;
- соответствующие `tests/v7/` и новый `tests/v7/test_retrieval_context.py`.

### Сначала проверки

1. Retrieval-only simple/complex/fallback возвращает тот же контекст, что полный
   граф перед generation. Generator-spy падает при любом вызове — счётчик должен
   оставаться нулевым на всех терминальных маршрутах.
2. Подготовленный graph A продолжает использовать backend/BM25/callbacks A после
   создания graph B. Чередование A → B → A и конкурентный тест не смешивают данные.
3. Generic characterization из шага 0 совпадает по route, scores и порядку evidence.
4. Simple не создаёт reranker; две одновременные complex-ветви создают один общий
   экземпляр и не исполняют predict одновременно. Проверить две разные graph instances,
   включая Generic, с тем же model resource. Ошибка загрузки видна в diagnostics.
5. BM25 создаётся один раз на ресурс snapshot, а не на запрос/ветвь. Stores с разными
   collection/path не получают один и тот же no-argument singleton.

### Изменения

1. Выделить простую сборку зависимостей, например `build_v7_runtime(store, ...)`:
   bound vector search, экземпляр `BM25Index`, section fetch, cross-ref,
   reranker и существующие optional callbacks. Не писать DI framework.
   Store loader принимает явные path/collection/embedding config либо готовый backend;
   существующий no-argument load_vector_store с lru_cache(maxsize=1) не годится как
   фабрика разных bound stores. Default wrapper для старых callers можно сохранить.
   BM25 строится на snapshot один раз. Reranker — lazy resource на model/device/config
   внутри процесса с общей блокировкой загрузки/predict; блокировка принадлежит
   ресурсу, не graph. Acquire ограничен оставшимся временем запроса.
   Не создавать в Department неиспользуемые Generic generation/expand clients.
2. Узлы получают зависимости через аргументы/замыкания. Они больше не читают
   переключаемые globals в новом production graph. Учитывать скрытый BM25-вызов
   внутри `cross_ref` и callbacks visual enrichment/generation, не только rag_simple.
   Сам runtime immutable; state, recorder и pack cache принадлежат конкретной ветви.
   Проверить читаемый одновременно BM25/морфологический helper; при необходимости
   защищать только небезопасный участок, не весь retrieval.
3. `build_graph(..., runtime=..., retrieval_only=True)` использует те же узлы и
   переходы. Retrieval-only начинает с router и заканчивается перед generator;
   полный Generic сохраняет intent gate, generation, abstain и noise.
4. `retrieve_context(...)` только подготавливает fresh state, вызывает этот graph
   и переводит терминальное состояние в небольшой результат. Самостоятельного
   `if insufficient: complex` внутри wrapper быть не должно.
5. Оставить текущий `RetrievalPlan` и сериализацию attempts. Существующий интерфейс
   overrides для тестов сохранить. Временный compatibility initializer допустим
   для старых callers, но новые graph instances должны фиксировать зависимости
   при создании, а не читать их позже из globals.

**Гейт PR:** Generic offline parity, ноль generation в retrieval-only,
изоляция A/B. Department UI ещё не переключать. Если перенести globals без изменения
семантики не получается, разбирать этот дефект здесь, а не компенсировать в service.

## PR 2. Scoped поиск, reuse embedding и ограниченный добор

**Назначение:** разрешить Department использовать complex без выхода за scope.

Основные файлы: `src/v7/scope_filter.py`, `hard_gates.py`, `bridge.py`,
`cross_ref.py`, `pack_context.py`, `nlp_core.py`, evaluator/router adapters и
`src/v7/retrieval.py`, `src/backends/vector_store.py`, `chroma_backend.py`.
Для runtime options достаточно небольшого объекта/аргументов;
новый public RetrievalPlan не вводить.

### Сначала проверки

1. External; internal/company; internal/unit A; unit B не попадает ни одним путём.
2. Начальный поиск, section fetch, source fetch, BM25-добор ссылок, bbox-соседи,
   fallback и final_context соблюдают один filter. Проверять реальные переданные
   backend filters и состав результата, а не только вызов helper.
3. Один source/номер chunk у разных документов; одинаковый текст разных источников.
   Недопустимый passage не проходит даже после section expansion.
4. Потеря обязательного filter key или неподдержанный оператор даёт ошибку,
   а не поиск по более широкому набору.
5. Pack limits применяются до validate; fallback проверен с тем же scope/budget.
   Сбой expansion не выглядит как «ничего не найдено».
6. Text и by-vector API на одинаковом векторе дают те же IDs/raw distances.
   Две конкурентные ветви с одним текстом инициируют одно embedding-вычисление;
   complex не вызывает второе. Разные model/text/request не смешиваются. Исключение
   передаётся обоим ожидающим, отсутствие результата не зависает и не запускает
   бесконечные повторные попытки.
7. Section bounded fetch возвращает тот же первый N, но не читает следующие страницы;
   прежний get_by_filter продолжает возвращать полный документ для cross-ref/MCP.

### Изменения

1. Переиспользовать `build_scope_filters` и согласовать эквивалентность Chroma/BM25
   фильтров. Сохранить Generic legacy filters; новые ограничения не выводить из LLM.
2. Передать scope во все document fetch/section/cross-ref callbacks. Не ограничиться
   post-filter уже полученного global top-K. Обязательный scope не прогонять через
   helper, который молча удаляет неизвестные ключи.
3. Для manifest-корпуса fetch/dedup ключ содержит document identity; Generic legacy
   fallback не меняет публичные IDs. Не дедуплицировать Department только по тексту.
4. Известные ошибки section/cross-ref сейчас могут проглатываться как пустой список:
   передать их общему pack/evaluator как техническую деградацию. Не менять таблицу
   допустимых Generic fallback; отдельно записать поправку диагностики и её тест.
5. Ввести explicit pack limits: Department — максимум текущих 8 passages на корпус,
   переданный token budget; Generic — прежние defaults. Кеш pack остаётся локальным
   вызову и учитывает влияющие на результат scope/limits/order.
6. Разрешить caller задать `require_multi_doc=False` для dual-corpus Department,
   сохранив Generic classification. Это параметр существующей policy, не новая
   Department таблица переходов. Одиночный ЛНА не должен отклоняться только потому,
   что сравнивается с законом из другой ветви.
7. Runtime Department: multi-query и visual enrichment выключены, centroid domain
   gate не добавляется. Generic callbacks/флаги сохраняются. Для Department это
   исключает новые вспомогательные LLM-вызовы при переносе search-пути.
8. Добавить by-vector API в backend adapter с прежними distance semantics. В Department
   на запрос создать небольшой shared memo exact prepared text + embedding space.
   Вызов ленивый (после router); одновременные ожидания одного ключа используют
   один Future/эквивалент, сеть вызывается вне общей блокировки memo. Готовый вектор
   передаётся vector search обеих ветвей и complex. Никаких global caches, повторного
   использования top-K между разными filters или смены Generic domain/multi-query.
   Проверить установленную версию API перед реализацией: у Chroma by-vector метода
   с relevance_scores в названии фактический score тоже distance.
9. Добавить отдельный bounded filter-fetch для section expansion. В текущем коде
   get_by_filter(limit=50) означает размер страницы и вычитывает всё; bridge затем
   оставляет [:50]. Новая операция останавливается на прежнем префиксе и сохраняет
   порядок. Полный cross-ref fetch не ограничивать этим лимитом.

**Гейт PR:** изоляция scope во всех обходных путях, Generic parity, отсутствие
неявного расширения corpus. На локальном read-only Department corpus прогнать
scoped retrieval cases без generation; embeddings могут быть платными и учитываются.
Если полезный короткий ЛНА систематически rejected из-за старых порогов — записать
результат и решить вопрос настройки до cutover, не отключать validation.

## PR 3. Department service и контракт ответа

**Назначение:** один общий механизм retrieval, два параллельных corpus-вызова,
один structured ответ.

Основные файлы:

- `src/department_qa/contract.py`, `service.py`, `wiring.py`;
- новый `prompts/agents/department_answer_v5.j2`, `prompts/registry.yaml`;
- `eval/run_object_profile_pair.py`;
- `tests/department_qa/test_service.py`, `test_contract.py`, `test_wiring.py`,
  `test_prompt_contract.py`, существующие тесты eval runner.

### Сначала проверки

1. Табличные context cases: три corpus selection, unit/None, profile on/off,
   invalid corpora/unit, profile missing/mismatch. Invalid input вызывает ноль I/O.
2. Две ветви достигают barrier до освобождения любой из них (не flaky sleep-test);
   незапрошенная ветвь не вызвана; один corpus simple, другой complex; ни один
   retrieval-вызов не генерирует промежуточный ответ. Обратный порядок завершения
   сохраняет external → internal в prompt и стабильные citation IDs.
3. Prompt содержит оба принятых блока, только переданные profile fields и ID.
   Registry не позволяет сослаться на discarded candidate.
4. Only-external не требует internal_basis; only-internal не требует external_basis.
   Dual-corpus с одной непригодной ветвью не даёт answered, даже если модель уверена.
5. False-profile запрещает object_facts/applied_conclusions; unknown/citation
   проверки остаются. Missing/degraded статусы не скрывают citation failure.
6. Контекст не режется после verdict; oversized обязательный prompt возвращает
   ошибку. Generation вызывается один раз, schema retry ограничен существующим одним.
7. Полный bounded executor, timeout в очереди/ожидании reranker, ошибка одной ветви,
   позднее завершение другой: generation=0, pending задачи отменены, результат не
   меняет UI/чужой запрос. Worker не вызывает Streamlit. Тест не обещает остановку
   уже работающего blocking-вызова, но подтверждает лимит workers/pending tasks.

### Изменения

1. Добавить `RequestContext` в Department contract. Перевести внутреннюю service API
   на context + injected retrieve_context + model_fn. Все текущие callers перечислить
   поиском перед правкой; не оставлять старую search_fn ветку fallback в production.
2. В wiring собрать bound runtime без `init_v7_pipeline` с глобальными side effects.
   Проверить V2 manifest/index match. Исторический V1 runner не подключать к новому
   пути; завершать с явным объяснением либо воспроизводить на baseline commit.
3. До retrieval вычислить остаточный normative budget, учитывая профиль и полный
   prompt overhead. Два корпуса — равные половины; один — весь остаток. Сумма
   укладывается в общий лимит с резервом ответа/schema retry. Без dynamic allocation.
4. Один corpus — одна задача; два — две независимые retrieval-only задачи
   в общем ограниченном executor. Никаких nested pools. Ограничить число workers и
   принятых задач; на saturation вернуть явную техническую ошибку/timeout, не копить
   бесконечную очередь. Числа конфигурации зафиксировать по baseline до cutover.
   Общий deadline начинается до admission; учитывать queue и rerank wait. Проверять
   cancellation перед каждой стадией, не использовать plan.timeout_ms как якобы уже
   измеренный end-to-end лимит. Backend calls получают доступный timeout там, где API
   его поддерживает; неподдерживаемую принудительную отмену не имитировать.
   На fatal error отменить pending ветвь и запретить дальнейшую generation; late
   результат running ветви игнорируется. Не использовать request-local executor
   context manager, который при timeout всё равно ждёт завершения всех workers.
   Собирать результаты/recorders по corpus на основном потоке, в стабильном порядке.
   Progress доставлять очередью событий на UI-поток. Evidence только из ready
   final_context; successful empty/insufficient отличается от технического сбоя.
5. Создать v5 prompt с явным scope и missing-corpus состояниями. V4 не редактировать;
   V2 bundle нового stack выбирает v5 явно, global active V1 не менять без причины.
6. Расширить `decide` requested corpora/profile flag и добавить retrieval reasons
   с приоритетами из спеки. Существующий validator остаётся единственным местом
   проверки ссылок и object/normative групп.
7. Добавить минимальные timings и per-corpus route/reason в существующие логи и
   eval records: retrieval wall-clock, queue_wait_ms, embedding computations/API
   attempts/cache hits, rerank_load/wait/predict_ms, calls/candidates, llm_ms/total_ms.
   Сумму параллельных branch timings не выдавать за total. Сохранить текущие
   `search_calls`/`evidence_ids` для воспроизводимости;
   новые попытки хранить дополнительным полем, не менять смысл старых записей.
8. Обновить recorder eval: фиксировать весь retrieval trace и финально отправленные
   evidence отдельно. Основной сравнительный прогон с `--no-verify`.

**Гейт PR:** unit service/prompt/contract, scoped retrieval и overlap/timeout cases; новый service
можно проверить через eval до UI cutover. `make_hybrid_search_fn` удаляется при
миграции последнего действующего caller в следующем PR, а не превращается в второй
вечный production retriever.

## PR 4. UI, callers, регрессии и завершение

Основные файлы: `app.py`, `pages/2_Общий_поиск.py`, `api.py` только в точке сборки
Generic graph, `scripts/trace_v7.py`, `eval/run_v7_eval.py`,
`eval/run_retrieval_eval.py`, `src/department_qa/wiring.py`, затронутые тесты и доки.
Окончательный список callers подтвердить поиском. REST `/retrieve` и адресные MCP
tools не переделываются; их внешние схемы и IDs должны оставаться прежними.

1. Department UI: selector корпусов и checkbox профиля, defaults по спеке. Сброс
   сохранённого ответа при смене corpora/unit/profile flag; profile checkbox disabled
   без профиля. Передать явный RequestContext и сохранить прежние progress stages.
2. Generic page/API/eval/trace используют bound runtime; проверить hot-reload ключ
   и кеш ресурсов после появления новых типов. Не создавать новую подсистему cache.
3. Удалить отдельный Department hybrid factory и неиспользуемые production imports.
   Legacy setters допустимы для unit fixtures/явного adapter, но runtime search не
   должен зависеть от их последующего переключения.
4. Внутренние Department scope-вызовы не проходят через `run_query` как отдельные
   пользовательские события. Существующие trace_id/feedback и одна запись запроса
   сохраняются; новая БД/дашборд не нужны.
5. Прогнать проверки ниже. Сравнить новый workers=1 и parallel на одинаковых
   данных/config отдельно от старого pipeline. До latency acceptance должны совпасть
   final evidence/status на fixed fixtures. В реальном прогоне проверить Chroma и
   shared callbacks под двумя ветвями, отдельно холодный и прогретый reranker.
   Обновить FACTS/configuration/design docs только по реально
   выполненному изменению; планы и отложенные оптимизации не записывать как факты.
6. Записать итоговый отчёт с baseline/candidate, hashes, настройками и разбором
   ухудшившихся вопросов. До этого не объявлять новый путь принятым и не делать deploy.

**Гейт PR:** checklist спеки выполнен; новый UI вызывает общий поиск; отдельный
Department retriever больше не production path. Возможен rollback версии кода;
коллекции и старые prompts/артефакты не удалялись.

## Проверки и граница готовности

Команды из корня проекта, после появления перечисленных новых тестов:

```bash
.venv/bin/python -m pytest -m unit tests/v7 tests/department_qa
.venv/bin/python -m pytest -m unit
.venv/bin/python scripts/check_docs.py --ci
git diff --check
```

Black/Ruff — на изменённых Python-файлах. Интеграционные тесты запускать отдельно
от unit; unit не инициализируют живые embedding/LLM/reranker clients.

| Проверка | Критерий |
|---|---|
| Offline Generic | Exact parity route/ordered IDs/scores/fallback на фиксированных fixtures; ноль новых failures |
| Scope/provenance | Ноль попаданий чужого корпуса, unit, profile; ноль неправильных citation associations |
| Retrieval-only | Ноль generation calls на всех маршрутах; сохранены terminal reasons |
| Department scoped cases | Правильные selected branches, contexts, missing-basis статусы и поведение одного короткого ЛНА |
| Pair/traps | Ноль новых запрещённых выводов и citation нарушений; каждый прежний полезный ответ, ставший отказом, разобран до принятия |
| Generic goldenset | Baseline/candidate на тех же данных/модели; зафиксированный до запуска критерий non-regression по correctness и false-sufficiency |
| Embedding/ресурсы | Один exact-text embedding на Department request; один lazy reranker resource; predict concurrency ≤ 1; BM25 не перестраивается на запрос |
| Parallel correctness | Barrier overlap; deterministic result order; bounded workers/queue; timeout/cancel без generation и late UI writes |
| Section I/O | Тот же префикс/порядок, без лишних страниц; прежний full-fetch API совместим |
| Время | Три варианта: old / new workers=1 / new parallel; single/dual, cold/warm; queue/rerank wait и end-to-end, p50/p95 с N/повторами |

Для live eval использовать существующие `eval/run_retrieval_eval.py` (simple и
complex), `eval/run_v7_eval.py`, `eval/run_object_profile_pair.py` с pair и trap
expectations. Команды и env берутся из актуальной инструкции соответствующего
раннера; не запускать его против коллекции другого режима. Для существующих пар
сравнивать V2 baseline/candidate, не выдавать устаревший V1 за контроль новой версии.

Семантический успех не измерять одним `contract_ok` или точной подстрокой
`forbidden_conclusion`: известные ловушки требуют проверки содержания ответа.
Для stochastic quality и допустимого ожидания UI зафиксировать допуски до финального
прогона. Если их ещё нет, offline refactor можно считать проверенным, а live cutover —
ещё не принятым. Ухудшения нельзя скрывать изменением expectations задним числом.

## Что не добавлять по ходу

В текущий объём ВХОДЯТ две параллельные ветви, Department request-local embedding reuse,
один ленивый ресурс модели с последовательным predict и bounded section fetch.
Объединённый rerank двух corpus pools, межзапросный cache, перенос embedding reuse
на все Generic-вызовы, nested parallelism, dynamic quotas, новый RetrievalPlan/coordinator,
миграцию metadata Generic и большие telemetry изменения не включать «заодно». Причины возврата к ним перечислены в разделе
[«Потом» спецификации](../../spec-shared-retrieval-plan.html#later).

Если без одного из этих пунктов не проходит конкретный приёмочный case, сначала
зафиксировать воспроизводимый дефект и пересмотреть объём. Не расширять задачу
потому, что рядом оказался удобный участок кода.
