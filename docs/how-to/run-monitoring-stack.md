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

Схема таблиц применяется сама: `db/migrations/001_init.sql` смонтирован в
`/docker-entrypoint-initdb.d` и выполняется при инициализации пустого тома. Руками в psql
ничего создавать не надо.

Переменные стека в `.env`: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`,
`GF_SECURITY_ADMIN_USER`, `GF_SECURITY_ADMIN_PASSWORD`; порты хоста при конфликте —
`POSTGRES_PORT`, `GRAFANA_PORT`. В самом compose-файле и в `grafana/` значений нет,
только подстановки.

## Где что смотреть

- Grafana — <http://localhost:3000>, вход под `GF_SECURITY_ADMIN_USER` / паролем из `.env`.
  Data source `regulatory-rag` (Postgres) создан provisioning-файлом
  `grafana/provisioning/datasources/postgres.yml`, руками его заводить не нужно;
  Connections → Data sources → Save & test отвечает «Database Connection OK».
  Дашборды и панели — issue #19, пока источник просто подключён.
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
  `verdict` (`+1` / `-1`), `comment`.

Запись живых событий в базу подключается в #17; пока события пишутся в JSONL-журнал —
[read-telemetry.md](./read-telemetry.md).

## Погасить

```bash
docker compose down           # контейнеры убраны, данные в томах остались
docker compose up -d          # поднимается на тех же данных, миграция не повторяется
```

Снести вместе с данными — `docker compose down -v`. Это стирает и накопленные события,
и настройки Grafana; тома называются `regulatory-rag_postgres_data` и
`regulatory-rag_grafana_data`.
