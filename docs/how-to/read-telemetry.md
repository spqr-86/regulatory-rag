# Как посмотреть, что происходит с запросами

Каждый завершившийся запрос — из Streamlit, API или eval-прогона — пишет одну строку события.
Пока Postgres не поднят (issue #15 ждёт установки Docker), строки копятся в JSONL-журнале.

## Где журнал

`logs/events.jsonl` по умолчанию; путь меняется переменной `V7_TELEMETRY_JSONL_PATH`.
Каталог в `.gitignore` — журнал не коммитится.

Выключить телеметрию целиком: `V7_TELEMETRY_ENABLED=false`. Граф при этом работает как раньше,
писатель не создаётся, файл не появляется.

## Что в строке

Поля перечислены в спеке модуля (`docs/spec-monitoring.html`, раздел «Что пишем на каждый
запрос»): `query_id`, `ts`, `source` (`ui` / `api` / `eval` / `mcp`), `question`, `path`
(`simple` / `complex` / `clarify` / `abstain`), `answer_len`, `n_passages`, `latency_ms`,
токены, `cost_usd`, `models`, `unpriced_models`, `error`.

`n_passages` — сколько пассажей ушло в промпт генерации; `n_passages_found` — сколько нашёл
поиск. Числа расходятся, когда cross-reference расширяет набор внутри узла генерации
(замер 05.09: найдено 14, в модель ушло 30). Судить о размере контекста нужно по первому
(issue #22).

`source` разделяет живой трафик и прогоны: eval пишет в тот же журнал, отличается только полем.

## Быстрые вопросы к журналу

```bash
# сколько запросов и сколько они стоили
jq -s 'length, (map(.cost_usd) | add)' logs/events.jsonl

# распределение по маршрутам
jq -r .path logs/events.jsonl | sort | uniq -c

# самые медленные запросы
jq -r '[.latency_ms, .question] | @tsv' logs/events.jsonl | sort -rn | head

# модели, для которых нет прайса (их цена считается нулём — это врёт в отчёте)
jq -r '.unpriced_models[]?' logs/events.jsonl | sort -u

# запросы, упавшие с ошибкой
jq -r 'select(.error != null) | [.ts, .error] | @tsv' logs/events.jsonl
```

## Что дальше

Postgres и Grafana — issue #15, дашборд — #19, 👍/👎 — #20. Код телеметрии:
`src/v7/telemetry.py` (событие и писатели), `src/v7/runner.py` (единственная точка вызова
графа, через которую идут все входы).
