# v8, этап 1 — eval-контур на LangSmith

Дата: 07.10.2026. Статус: на ревью Петра. Решения: TG 41840–41849.

## Зачем

Учебный проект до 11.10: пройти на regulatory-rag полный цикл курса
[Building Reliable Agents with LangSmith](https://academy.langchain.com/courses/building-reliable-agents)
(конспекты уроков — `explainer-videos/videos/courses/building-reliable-agents/source/lessons/`).
Продуктовая цель — честная метрика вместо витринных 8,09/10, которые не подтверждают
юридическую точность (аудит golden set 30.09).

Код пишет Пётр. Ассистент даёт план с тестами и ревьюит. v7-граф на этом этапе не меняется;
`eval/run_v7_eval.py` остаётся для витринных цифр.

## Соответствие курсу

| Урок | Что делаем на regrag | Артефакт |
|---|---|---|
| 1.1–1.2 Observability, tracing | Трейсинг v7 в проект `regulatory-rag` (сейчас выключен, см. «Ограничения») | `.env`, трейсы прогонов |
| 1.3 Analyzing your agent (PRD) | Короткий PRD поведения: что отвечаем, когда отказываем, когда просим уточнить. Из него — сценарии датасета | раздел «PRD» ниже |
| 2.1 Evaluating agents | Уровни: end-to-end (ответ), step (retrieval), trajectory (маршрут `route_decision`); метрики качества + операционные (latency, cost из трейса) | таблица evaluators |
| 2.2 Datasets | Датасет `regrag-golden`, сплит `verified` сейчас, `from_traces` позже (вариант C) | `eval/ls/dataset.py` |
| 2.3 Running experiments | `client.evaluate(target, data, evaluators)`, сравнение экспериментов в UI | `eval/ls/run.py`, `target.py` |
| 2.4 Code-based eval | 4 детерминированных evaluator'а | `eval/ls/evaluators.py` |
| 2.5 LLM-as-judge + alignment | Бинарный судья correctness; Пётр размечает те же 22 ответа, согласие ≥ 85 % | `judge.py`, `align.py` |
| 2.6 Pairwise | `evaluate_comparative`: v7 effort=low vs v7 effort=medium; на этапе 2 — v7 vs v8 | эксперимент pairwise |
| 3.1–3.2 Monitoring, Insights Agent | Insights по трейсам 8502 и прогонов → кластеры провалов → кандидаты в сплит `from_traces` | сплит `from_traces` |
| 3.3 Online evals | Правило в проекте: `oos_handled` (код) на всём трафике + judge на выборке | правило в LangSmith UI |
| 3.4 Automations | 👎 из UI (`src/ui_feedback.py`) или провал online eval → annotation queue → датасет | automation в UI |

Из урока 2.3 «Running evals with pytest» берём максимум одну регрессионную проверку в CI, если
останется время.

## PRD поведения (урок 1.3)

- In-scope вопрос по корпусу: ответ с указанием документа и пункта; без запрещённых утверждений.
- Вопрос вне домена (`out_of_scope`): отказ, без попытки ответить.
- Ложная посылка (`false_premise`): посылка опровергнута, а не принята.
- Вопрос, зависящий от параметров объекта: просьба уточнить (это сценарий этапа 2, HITL; в
  датасет этапа 1 не входит).

## Датасет (урок 2.2)

Источник — `tests/dataset.csv` (golden 56) + `eval/data/golden_retrieval_labeled.jsonl`.

- Сплит `verified` — 22 примера с `reference_status=verified`: 12 in-scope (у всех есть
  `relevant_chunk_ids`, проверено 07.10) и 10 OOS (`out_of_scope`, `false_premise`).
- inputs: `question`. outputs: `ground_truth`, `relevant_chunk_ids`, `must_not_contain`,
  `forbidden_claims`. metadata: `case_id`, `oos_type`, `corpus_support`, хэш CSV.
- Загрузка идемпотентная: повторный запуск не плодит дубли; смена CSV → новая версия датасета.
- Сплит `from_traces` (позже): 20–30 плохих ответов из трейсов и 👎, эталоны дописывает Пётр.
- 34 примера `needs_clarification` не входят — их эталоны не проверены.

## Target (урок 2.3)

`eval/ls/target.py`: `question → {answer, route_decision, chunk_ids, sources}`. Внутри — тот
же путь, что в `run_v7_eval.py`: `build_full_v7_runtime` + `build_graph` + `run_query`, с
`source="eval"`. `chunk_ids` в порядке `final_passages` — нужен ранг для MRR.

## Evaluators (уроки 2.4–2.5)

| Ключ | Тип | Применяется к | Логика |
|---|---|---|---|
| `retrieval_hit` | код, step | in-scope | хотя бы один `relevant_chunk_id` в `chunk_ids`; в comment — ранг первого попадания |
| `doc_cited` | код, end-to-end | in-scope | в ответе назван номер документа из `relevant_chunk_ids` (`2464.pdf#107` → «2464») |
| `forbidden` | код, end-to-end | все | нет `must_not_contain` и `forbidden_claims` |
| `oos_handled` | код, trajectory | OOS | `route_decision=abstain` или отказ / опровержение в ответе |
| `correctness` | LLM-judge | in-scope | бинарно верно/неверно против `ground_truth` + обоснование |

Evaluator, неприменимый к примеру, возвращает `score=None`, а не 0. Сводка: доли по каждому
ключу, MRR по `retrieval_hit`.

**Alignment.** Судья — `JUDGE_*` из `.env` (сейчас OpenRouter `openai/gpt-4o`; согласие на
отправку 22 вопросов — TG 41845). Пётр размечает ответы первого эксперимента в annotation queue.
`align.py` считает согласие и матрицу ошибок; промпт судьи (`eval/prompts/judge_correctness.md`)
правится, пока согласие не станет ≥ 85 % на 12 in-scope. Перепрогон судьи — на готовых ответах,
без повторной генерации.

## Файлы

```
eval/ls/dataset.py      CSV + разметка → examples, upsert в LangSmith
eval/ls/target.py       обёртка над v7
eval/ls/evaluators.py   4 code-evaluator'а, чистые функции
eval/ls/judge.py        correctness-судья
eval/ls/run.py          CLI: --split, --skip-judge, --prefix; max_concurrency=1
eval/ls/align.py        согласие судьи с разметкой
eval/prompts/judge_correctness.md
tests/eval_ls/          unit-тесты dataset и evaluators, без сети
```

## Ошибки

Исключение в target записывается как error строки эксперимента; evaluators ставят `None`.
Платные вызовы не ретраятся. Неполный корпус (`corpus_support=unverified`, не хватает 2464+805 и
др.) даёт промахи не по вине графа — они помечаются в comment, в этом этапе не чинятся.

## Ограничения и стоимость

- **Трейсинг выключен с 05.09** (`LANGSMITH_TRACING_V2=false`): месячная квота трейсов была
  выжжена, прогоны падали с 429. Включаем только на прогонах `eval/ls/` и на 8502; массовые
  прогоны (`run_retrieval_eval.py`) по-прежнему без трейсинга. Перед первым прогоном проверить
  остаток квоты в настройках LangSmith.
- Стоимость (оценка, уточнить по первому прогону): генерация 22 × ~$0.0066 ≈ $0.15; судья
  ≈ $0.2; эксперимент ≈ $0.35–0.4; итерация alignment ≈ $0.2; весь этап ≈ $2–3. Первый прогон —
  `--skip-judge`. Каждый платный прогон — по «да» Петра.

## Вне рамок

Изменения v7-графа, переиндексация и новая нарезка, `needs_clarification`-кейсы, HITL (этап 2),
Deep Agent (этап 3), замена `run_v7_eval.py`.

## Готово, когда

1. Датасет `regrag-golden/verified` в LangSmith, 22 примера.
2. Эксперимент v7 со всеми пятью evaluators; базовые цифры записаны в
   `docs/evaluation/experiments/`.
3. Судья выровнен: согласие ≥ 85 %, история итераций промпта сохранена.
4. Pairwise-эксперимент effort=low vs medium.
5. Online eval и automation включены в проекте `regulatory-rag`.
6. Сплит `from_traces` создан (минимум 10 примеров) — если хватит времени до 11.10.
