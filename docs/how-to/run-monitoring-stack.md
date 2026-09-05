# Как поднять стек мониторинга

Postgres хранит строки событий, Grafana их показывает. Оба сервиса живут в
`docker-compose.yml` в корне репозитория и слушают только localhost.

Нужен Docker Engine с plugin `compose` (v2). Проверка: `docker compose version`.

## Первый запуск

```bash
cp .env.example .env          # если .env ещё нет
# впишите POSTGRES_PASSWORD и GF_SECURITY_ADMIN_PASSWORD — любые локальные значения
docker compose up -d
docker compose ps             # оба сервиса должны быть healthy
```

Схема таблиц применяется сама: каталог `db/migrations/` смонтирован в
`/docker-entrypoint-initdb.d`, и его файлы выполняются в порядке имён при инициализации
пустого тома. Руками в psql ничего создавать не надо.

**На томе, созданном до появления файла, миграция не выполнится** — образ запускает их
только на пустой базе. Догоняющий прогон (идемпотентен, `IF NOT EXISTS`):

```bash
set -a; . ./.env; set +a
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  < db/migrations/002_feedback_one_vote.sql
```

Переменные стека в `.env`: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`,
`GF_SECURITY_ADMIN_USER`, `GF_SECURITY_ADMIN_PASSWORD`; порты хоста при конфликте —
`POSTGRES_PORT`, `GRAFANA_PORT`. В самом compose-файле и в `grafana/` значений нет,
только подстановки.

## Где что смотреть

- Grafana — <http://localhost:3000>, вход под `GF_SECURITY_ADMIN_USER` / паролем из `.env`.
  Data source `regulatory-rag` (Postgres) создан provisioning-файлом
  `grafana/provisioning/datasources/postgres.yml`, руками его заводить не нужно;
  Connections → Data sources → Save & test отвечает «Database Connection OK».
- Дашборд **Regulatory RAG — запросы** (<http://localhost:3000/d/regrag-queries>) приезжает
  тем же способом: файл `grafana/dashboards/regulatory-rag.json`, провайдер —
  `grafana/provisioning/dashboards/dashboards.yml`. Панели: итоговая цена за период,
  доля 👎, запросы по дням с разбивкой по `source`, маршруты, цена и латентность в
  перцентилях p50/p95 (не в среднем — один сложный маршрут дороже десяти простых),
  последние 👎 таблицей.
  Период задаётся стандартным пикером Grafana, все запросы фильтруются по нему.
  Панели «Доля 👎» и «Последние 👎» наполняются кнопками под ответом в Streamlit
  (issue #20): вторая показывает вопрос, маршрут, цену и комментарий к каждому промаху,
  плюс `query_id`, по которому строка запроса поднимается в базе целиком.

  Правки в UI не сохраняются в файл: `allowUiUpdates: false`, при рестарте побеждает
  репозиторий. Менять панель — значит менять JSON; после правки `docker compose up -d
  grafana` (провайдер перечитывает файл и сам, раз в 30 секунд).
- База — `psql` изнутри контейнера:

  ```bash
  docker compose exec postgres psql -U regrag -d regrag -c '\d queries'
  ```

## Таблицы

- `queries` — одна строка на завершившийся запрос. Поля повторяют событие телеметрии
  (`EVENT_FIELDS` в `src/v7/telemetry.py`, раздел спека «Что пишем на каждый запрос»):
  `query_id`, `run_id`, `ts`, `source`, `question`, `path`, длины и счётчики, токены,
  `cost_usd`, `models` (jsonb), `unpriced_models`, `error`.
- `feedback` — 👍/👎 к строке `queries`: `query_id` (FK, `ON DELETE CASCADE`), `ts`,
  `verdict` (`+1` / `-1`), `comment`. Уникальный индекс по `query_id`
  (`002_feedback_one_vote.sql`) держит одну оценку на ответ: повторное нажатие
  перезаписывает строку, а не добавляет вторую, поэтому панели считают ответы, а не клики.

## Писать события в базу

По умолчанию события идут в JSONL-журнал — он работает без стека. Переключение на Postgres
делается настройкой, ни одна точка вызова не меняется (issue #23):

```bash
# .env
V7_TELEMETRY_WRITER=postgres
```

DSN собирается из тех же `POSTGRES_*`, что читает compose; `V7_TELEMETRY_PG_DSN` нужен
только для базы вне локального стека. Если postgres выбран, а подключиться не из чего,
приложение пишет предупреждение в лог и возвращается к журналу — запросы от этого не
страдают. Упавшая на ходу база тоже не роняет ответ и **не съедает событие**: строка
уходит в тот же `logs/events.jsonl` с предупреждением `telemetry.primary_failed_using_journal`,
а после подъёма базы заливается `scripts/ingest_events.py` (идемпотентно). Это и значит
«писатель Postgres с журналом за спиной» — `FallbackWriter` в `src/v7/telemetry.py`.

Накопленный журнал заливается в базу разово:

```bash
python scripts/ingest_events.py logs/events.jsonl
```

Скрипт идемпотентен: строка узнаётся по `query_id`, повторный прогон ничего не задваивает.
Нечитаемые строки пропускаются и считаются отдельно, одна упавшая вставка не останавливает
остальные.

События, записанные до issue #22, не содержат `n_passages_found` — тогда это же число
лежало в `n_passages`. Заливка подставляет его сама, поэтому старый журнал заливается
целиком, а не по частям.

Чем смотреть журнал в файловом виде — [read-telemetry.md](./read-telemetry.md).

## Если стек не поднялся

- **`docker compose up -d` падает на порте** («address already in use») — на 5432 или 3000
  уже что-то слушает. Свои порты задаются в `.env`: `POSTGRES_PORT=5433`, `GRAFANA_PORT=3001`.
- **`postgres` не доходит до `healthy`** — `docker compose logs postgres`. Частая причина:
  том создан со старым паролем, а `.env` уже с новым; пароль лежит внутри тома, а не в
  переменной. Либо вернуть прежний пароль, либо пересоздать том (`docker compose down -v` —
  **стирает накопленные события**, перед этим залить их из журнала).
- **Grafana ждёт вечно** — она стартует только после `postgres: healthy`; чинится проблема
  выше, а не Grafana.
- **Стек не поднят вообще** — приложение работает как обычно: события идут в
  `logs/events.jsonl`, кнопки 👍/👎 под ответом не рисуются (голосу некуда лечь).
  Поднимать стек ради ответов пользователю не нужно.
- **База упала на ходу** — ответ уходит пользователю, событие ложится в журнал (см. выше),
  голос не сохраняется: в UI подпись «Оценка не сохранилась — журнал недоступен».
- **`ingest_events.py` ругается «Refusing to guess a connection»** — не прочитан `.env`
  (`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB`). Запускать из корня репозитория.
- **Панели пустые, а запросы шли** — скорее всего писатель остался журнальным: проверьте
  `V7_TELEMETRY_WRITER=postgres` в `.env` и залейте накопленное `scripts/ingest_events.py`.
  Второй кандидат — период в пикере Grafana: по умолчанию окно короче, чем возраст данных.
- **Таблицы `feedback` нет или кнопки 👍/👎 не пишут** — том создан до миграции `002`;
  догоняющий прогон описан выше, в «Первом запуске».

## Погасить

```bash
docker compose down           # контейнеры убраны, данные в томах остались
docker compose up -d          # поднимается на тех же данных, миграция не повторяется
```

Снести вместе с данными — `docker compose down -v`. Это стирает и накопленные события,
и настройки Grafana; тома называются `regulatory-rag_postgres_data` и
`regulatory-rag_grafana_data`.
