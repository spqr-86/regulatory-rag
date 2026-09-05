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
страдают. Упавшая база тоже не роняет ответ: событие теряется, ответ уходит пользователю.

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

## Погасить

```bash
docker compose down           # контейнеры убраны, данные в томах остались
docker compose up -d          # поднимается на тех же данных, миграция не повторяется
```

Снести вместе с данными — `docker compose down -v`. Это стирает и накопленные события,
и настройки Grafana; тома называются `regulatory-rag_postgres_data` и
`regulatory-rag_grafana_data`.
