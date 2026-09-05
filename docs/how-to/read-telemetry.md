# Как посмотреть, что происходит с запросами

Каждый завершившийся запрос — из Streamlit, API или eval-прогона — пишет одну строку события.
Куда именно, решает `V7_TELEMETRY_WRITER`: по умолчанию строки копятся в JSONL-журнале, при
`postgres` идут в базу поднятого стека ([run-monitoring-stack.md](./run-monitoring-stack.md)),
а журнал остаётся за ней страховкой на случай, если база упала на ходу.

## Где журнал

`logs/events.jsonl` по умолчанию; путь меняется переменной `V7_TELEMETRY_JSONL_PATH`.
Каталог в `.gitignore` — журнал не коммитится.

Выключить телеметрию целиком: `V7_TELEMETRY_ENABLED=false`. Граф при этом работает как раньше,
писатель не создаётся, файл не появляется.

## Что в строке

Поля перечислены в спеке модуля (`docs/spec-monitoring.html`, раздел «Что пишем на каждый
запрос»): `query_id`, `ts`, `source` (`ui` / `api` / `eval` / `mcp`), `question`, `path`
(`simple` / `complex` / `clarify` / `abstain`), `answer_len`, `n_passages`, `latency_ms`,
токены, `cost_usd`, `models`, `unpriced_models`, `error`. Плюс `run_id` — к какому
пакетному прогону относится строка (issue #18).

`n_passages` — сколько пассажей ушло в промпт генерации; `n_passages_found` — сколько нашёл
поиск. Числа расходятся, когда cross-reference расширяет набор внутри узла генерации
(замер 05.09: найдено 14, в модель ушло 30). Судить о размере контекста нужно по первому
(issue #22).

`source` разделяет живой трафик и прогоны: eval пишет в тот же журнал, отличается только полем.
`run_id` разделяет прогоны между собой: все запросы одного `run_v7_eval.py` несут один id
вида `eval-20260905T091500Z-a3f1`, он же лежит в JSON-сводке прогона — по нему строки журнала
связываются с файлом результатов. У живого запроса прогона нет, поле равно `null`.

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

# только живой трафик (без экспериментов) — так же смотрит дашборд
jq -c 'select(.source == "ui")' logs/events.jsonl

# строки одного прогона
jq -c 'select(.run_id == "eval-20260905T091500Z-a3f1")' logs/events.jsonl

# какие прогоны вообще есть в журнале и по сколько запросов в каждом
jq -r 'select(.run_id != null) | .run_id' logs/events.jsonl | sort | uniq -c

# запросы, упавшие с ошибкой
jq -r 'select(.error != null) | [.ts, .error] | @tsv' logs/events.jsonl
```

## Что дальше

Те же события пишутся в Postgres — `V7_TELEMETRY_WRITER=postgres` (issue #23), тогда
запросы к ним пишутся на SQL, а не на jq; как включить и как залить накопленный журнал —
[run-monitoring-stack.md](./run-monitoring-stack.md). Дашборд — #19, 👍/👎 — #20.

Код телеметрии: `src/v7/telemetry.py` (событие и JSONL-писатель), `src/v7/pg_writer.py`
(писатель в Postgres), `src/v7/runner.py` (единственная точка вызова графа, через которую
идут все входы).
