# v8, этап 1 — план реализации eval-контура на LangSmith

> **Для исполнителя.** Учебный проект: **код реализации пишет Пётр**, ассистент пишет тесты,
> ревьюит и гоняет проверки. Поэтому в задачах тесты даны целиком, а реализация — интерфейсом,
> подсказками и ссылками на готовые функции, без листинга. Шаги отмечаются `- [ ]`.

**Цель:** датасет `regrag-golden/verified` в LangSmith, эксперимент v7 с пятью evaluators,
выровненный судья, pairwise effort=low vs medium, online eval + automation.

**Архитектура:** пакет `eval/ls/` из маленьких модулей. Всё, что можно проверить без сети
(сборка примеров, evaluators, разбор ответа судьи, согласие), — чистые функции с unit-тестами.
Сеть и платные вызовы есть только в `run.py`, `align.py`, `pairwise.py` и в `upsert`.

**Стек:** Python 3.13 (`.venv`), langsmith 0.12.1, langchain-core, pytest.

**Спек:** `docs/design/2026-10-07-v8-stage1-langsmith-eval-design.md`.

## Общие ограничения

- `eval/run_v7_eval.py` и v7-граф не меняются.
- Платные вызовы не ретраятся. Каждый платный прогон делается только после «да» Петра; первый
  прогон — `--skip-judge`.
- Трейсинг: `LANGSMITH_TRACING_V2=true` только в команде запуска `eval/ls/`, не в `.env`.
  Перед первым прогоном проверить остаток квоты трейсов в LangSmith → Settings → Usage.
- `max_concurrency=1`.
- Evaluator, неприменимый к примеру, возвращает `score=None`, а не 0.
- Unit-тесты лежат в `tests/eval_ls/`, сеть не трогают, маркер `unit`. CI-гейт:
  `pytest -m "not integration and not slow"`.
- Перед коммитом: `black` и `ruff check --fix` только по своим файлам.
- Push только с ок Петра.

## Отступления от спека (нужен ок Петра)

1. **`forbidden_claims` проверяет судья, а не код.** Во всех 22 verified `must_not_contain`
   пуст, а `forbidden_claims` — фразы вида «Утверждается, что минимум стажировки — одна смена».
   Подстрокой такое не поймать. Поэтому код-evaluator `forbidden` проверяет только
   `must_not_contain` через готовую `eval.metrics.compute_inversion_detected`; сейчас он везде
   даёт `None`. Нарушение `forbidden_claims` судья возвращает вторым ключом `forbidden_claim`,
   так же как это делает `golden_correctness.md`.
2. **`oos_type` дублируется в `outputs` примера.** Evaluator с сигнатурой
   `(outputs, reference_outputs)` остаётся чистой функцией и не лезет в `example.metadata`.
   `oos_type == ""` означает in-scope.

## Самые вероятные поломки (Review Focus)

1. Пустой ответ (domain gate отказал до графа, `route_decision` нет) — `oos_handled` должен
   засчитать отказ, а `doc_cited` и `retrieval_hit` у in-scope — дать 0, а не упасть.
   Покрыто в задаче 2.
2. Вопрос в CSV и в jsonl отличается пробелами по краям — join по `question` должен работать
   после `strip()`. Покрыто в задаче 1.
3. Номер документа с буквой (`782н.pdf#431` → «782н») и падеж/регистр в ответе («приказа
   № 782Н») — `doc_cited` сравнивает без учёта регистра. Покрыто в задаче 2.
4. Судья вернул JSON в ```json-ограде или не-bool в `correct` — `parse_verdict` чистит ограду и
   падает с `ValueError`, а не засчитывает мусор. Покрыто в задаче 4.
5. Повторный `dataset.py` при неизменном CSV не создаёт дублей, при изменённом — обновляет
   примеры с теми же id. Покрыто в задаче 1 (фейковый клиент).

## Файлы

```
eval/ls/__init__.py
eval/ls/dataset.py      сборка примеров (чистая) + upsert + CLI
eval/ls/evaluators.py   retrieval_hit, doc_cited, forbidden, oos_handled + summarize
eval/ls/target.py       state v7 → outputs (чистая) + обёртка над графом
eval/ls/judge.py        parse_verdict (чистая) + correctness-evaluator
eval/ls/run.py          CLI эксперимента
eval/ls/align.py        agreement (чистая) + CLI по фидбэку эксперимента
eval/ls/pairwise.py     parse_preference (чистая) + CLI evaluate_comparative
eval/prompts/judge_correctness.md
eval/prompts/judge_pairwise.md
tests/eval_ls/__init__.py
tests/eval_ls/test_dataset.py
tests/eval_ls/test_evaluators.py
tests/eval_ls/test_target.py
tests/eval_ls/test_judge.py
tests/eval_ls/test_align.py
tests/eval_ls/test_pairwise.py
docs/evaluation/experiments/langsmith-v8-stage1.md
```

---

### Задача 0: git в порядок

Состояние на 07.10: local `main` = `00060e5` + `421869c` (спек) поверх `c54b8cc`; origin =
`3071861` (ruff #68) поверх `c54b8cc`. Незакоммичен `docs/roadmap.md`, untracked `eval/runs/*`.

- [ ] **Шаг 1.** `git fetch origin && git stash -- docs/roadmap.md`
- [ ] **Шаг 2.** `git rebase origin/main` — ожидается без конфликтов.
- [ ] **Шаг 3.** `git stash pop && git add docs/roadmap.md && git commit -m "docs(roadmap): park RAG-as-MCP idea"`
- [ ] **Шаг 4.** `git add docs/design/2026-10-07-v8-stage1-langsmith-eval-plan.md && git commit -m "docs(design): v8 stage 1 implementation plan"`
- [ ] **Шаг 5.** `git log --oneline -5` — сверху план, roadmap, спек, `00060e5`, `3071861`.
- [ ] **Шаг 6.** Push — только по ок Петра.
- [ ] **Шаг 7.** Ветка под работу: `git switch -c feat/v8-langsmith-eval`.

---

### Задача 0.5: модуль 1 курса — пощупать LangSmith (добавлено 08.10)

Курс «Building Reliable Agents» (LangChain Academy). Модуль 1 = observability: трейсинг и разбор
агента по трейсам. Материалы: `explainer-videos/videos/courses/building-reliable-agents/source/`
(`lessons/m1-*.md`, `code/module-1/lesson-2/`). Код курса не копируется в regrag.

- [ ] **Шаг 1.** Проверить остаток квоты трейсов: LangSmith → Settings → Usage (с 05.09 трейсинг
  выключен из-за 429). Без остатка шаги 3–5 не запускать.
- [ ] **Шаг 2.** Решить модель для курсовых скриптов: ключ OpenAI (`gpt-5-nano` в коде курса) или
  OpenRouter. Платных вызовов мало, но каждый запуск — по «да» Петра.
- [ ] **Шаг 3 (урок 1.2).** `thread_agent.py` с `LANGSMITH_TRACING=true` только в команде запуска.
  В UI найти трейс и тред (`thread_id`), увидеть, что второй вызов помнит имя.
- [ ] **Шаг 4 (урок 1.2).** `third_party_agent.py` (инструмент «погода»): вложенные вызовы,
  `run_type="tool"`, вход и выход каждого шага.
- [ ] **Шаг 5 (урок 1.3, прочитать).** `lessons/m1-3-analyzing-your-agent.md`: цикл Build ↔ Test.
  В курсе три причины сбоев Emma найдены по трейсам: агент ищет несуществующую таблицу; утечка
  остатков из-за промпта (правка в Playground через Polly); база знаний не знает про офис в Чикаго.
- [ ] **Шаг 5a. PRD regrag.** Короткий список: что агент должен делать и где обязан отказать.
  Сценарии брать из 22 verified-вопросов (12 по теме, 10 вне темы).
- [ ] **Шаг 5b. Прогон и разбор.** Прогнать 3–5 сценариев с трейсингом (по «да» Петра, платно).
  В трейсе смотреть: какие чанки вернул retrieval, попал ли нужный документ, что ответила модель.
- [ ] **Шаг 5c. Список сбоев.** Таблица «вопрос, что сломалось, где в трейсе, гипотеза причины».
  Это вход для датасета (задача 1) и evaluators (задача 2).
- [ ] **Шаг 6.** Заметки Петра в 5–7 строк: что нашёл в UI и чего не хватило. Нужны для задач 3 и 7.

---

### Задача 1: датасет

**Файлы:** создать `eval/ls/__init__.py`, `eval/ls/dataset.py`, `tests/eval_ls/__init__.py`,
`tests/eval_ls/test_dataset.py`.

**Интерфейсы (produces):**

```python
DATASET_NAME = "regrag-golden"
SPLIT_VERIFIED = "verified"

def build_examples(csv_path: Path, labels_path: Path) -> list[dict]:
    """Verified-строки CSV → примеры LangSmith.

    Каждый пример:
      {"id": str (uuid5 от case_id, стабилен между запусками),
       "inputs": {"question": str},
       "outputs": {"ground_truth": str, "relevant_chunk_ids": list[str],
                   "must_not_contain": str,      # сырой, разделитель "|"
                   "forbidden_claims": str, "oos_type": str},
       "metadata": {"case_id", "oos_type", "corpus_support", "csv_sha256"},
       "split": "verified"}
    in-scope без разметки в jsonl → ValueError с case_id.
    OOS → relevant_chunk_ids = [].
    """

def upsert_examples(client, examples: list[dict], dataset_name: str = DATASET_NAME) -> dict:
    """Создать датасет при отсутствии; новые id — create_examples, изменённые — update_examples.
    Возвращает {"created": int, "updated": int, "unchanged": int}."""
```

- [ ] **Шаг 1. Тест.** `tests/eval_ls/test_dataset.py`:

```python
"""Unit tests for eval/ls/dataset.py — no network."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eval.ls.dataset import build_examples, upsert_examples

pytestmark = pytest.mark.unit

FIELDS = [
    "case_id", "question", "ground_truth", "must_not_contain", "oos_type",
    "reference_status", "reviewed_at", "review_note", "forbidden_claims",
    "corpus_support", "legal_as_of",
]


def _row(case_id, question, *, status="verified", oos_type="", forbidden=""):
    return {
        "case_id": case_id, "question": question, "ground_truth": f"gt {case_id}",
        "must_not_contain": "", "oos_type": oos_type, "reference_status": status,
        "reviewed_at": "2026-09-28", "review_note": "", "forbidden_claims": forbidden,
        "corpus_support": "unverified", "legal_as_of": "2026-09-30",
    }


@pytest.fixture
def files(tmp_path: Path):
    csv_path = tmp_path / "dataset.csv"
    labels_path = tmp_path / "labels.jsonl"
    rows = [
        _row("G001", "Вопрос один?", forbidden="Утверждается, что X."),
        _row("G002", "  Вопрос два?  "),
        _row("G003", "Не проверен?", status="needs_clarification"),
        _row("G035", "Рецепт борща?", oos_type="out_of_scope"),
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    labels = [
        {"question": "Вопрос один?", "chunk_id": "2464.pdf#107",
         "relevant_chunk_ids": ["2464.pdf#107", "782н.pdf#431"]},
        {"question": "Вопрос два?", "chunk_id": "2464.pdf#20"},
    ]
    labels_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in labels), encoding="utf-8"
    )
    return csv_path, labels_path


def test_only_verified_rows(files):
    ex = build_examples(*files)
    assert [e["metadata"]["case_id"] for e in ex] == ["G001", "G002", "G035"]


def test_shape_and_split(files):
    e = build_examples(*files)[0]
    assert e["inputs"] == {"question": "Вопрос один?"}
    assert e["outputs"]["relevant_chunk_ids"] == ["2464.pdf#107", "782н.pdf#431"]
    assert e["outputs"]["forbidden_claims"] == "Утверждается, что X."
    assert e["outputs"]["oos_type"] == ""
    assert e["split"] == "verified"
    assert set(e["metadata"]) == {"case_id", "oos_type", "corpus_support", "csv_sha256"}


def test_join_strips_whitespace_and_falls_back_to_chunk_id(files):
    e = build_examples(*files)[1]
    assert e["inputs"]["question"] == "Вопрос два?"
    assert e["outputs"]["relevant_chunk_ids"] == ["2464.pdf#20"]


def test_oos_has_no_labels(files):
    e = build_examples(*files)[2]
    assert e["outputs"]["relevant_chunk_ids"] == []
    assert e["outputs"]["oos_type"] == "out_of_scope"


def test_ids_stable_between_runs(files):
    assert [e["id"] for e in build_examples(*files)] == [e["id"] for e in build_examples(*files)]


def test_csv_hash_changes_with_csv(files):
    csv_path, labels_path = files
    before = build_examples(csv_path, labels_path)[0]["metadata"]["csv_sha256"]
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = build_examples(csv_path, labels_path)[0]["metadata"]["csv_sha256"]
    assert before != after


def test_in_scope_without_labels_fails(files):
    csv_path, labels_path = files
    labels_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="G001"):
        build_examples(csv_path, labels_path)


class FakeExample:
    def __init__(self, d):
        self.id = d["id"]
        self.inputs = d["inputs"]
        self.outputs = d["outputs"]
        self.metadata = d["metadata"]


class FakeClient:
    def __init__(self):
        self.store: dict[str, dict] = {}
        self.created = 0
        self.updated = 0

    def has_dataset(self, *, dataset_name):
        return True

    def create_dataset(self, dataset_name, **kw):
        raise AssertionError("dataset exists")

    def list_examples(self, *, dataset_name, **kw):
        return [FakeExample(d) for d in self.store.values()]

    def create_examples(self, *, dataset_name, examples, **kw):
        for d in examples:
            self.store[str(d["id"])] = d
        self.created += len(examples)

    def update_examples(self, *, dataset_name=None, updates=None, **kw):
        for d in updates:
            self.store[str(d["id"])] = d
        self.updated += len(updates)


def test_upsert_is_idempotent(files):
    client = FakeClient()
    ex = build_examples(*files)
    assert upsert_examples(client, ex) == {"created": 3, "updated": 0, "unchanged": 0}
    assert upsert_examples(client, ex) == {"created": 0, "updated": 0, "unchanged": 3}
    assert len(client.store) == 3


def test_upsert_updates_changed(files):
    client = FakeClient()
    ex = build_examples(*files)
    upsert_examples(client, ex)
    ex[0]["outputs"]["ground_truth"] = "новый эталон"
    assert upsert_examples(client, ex) == {"created": 0, "updated": 1, "unchanged": 2}
```

- [ ] **Шаг 2.** `.venv/bin/python -m pytest tests/eval_ls/test_dataset.py -v` → FAIL
  (`ModuleNotFoundError: eval.ls`).
- [ ] **Шаг 3. Реализация (Пётр).** Подсказки:
  - id: `uuid.uuid5(uuid.NAMESPACE_URL, f"regrag-golden/{case_id}")`.
  - Разметка: `{rec["question"].strip(): rec.get("relevant_chunk_ids") or [rec["chunk_id"]]}`.
  - Хэш: `hashlib.sha256(csv_path.read_bytes()).hexdigest()`.
  - «Изменён» = отличаются `inputs`, `outputs` или `metadata` у примера с тем же id. Сигнатуру
    `update_examples` в 0.12.1 проверить: `inspect.signature(Client.update_examples)`; если
    параметр называется иначе, поправить фейк в тесте.
  - CLI `python -m eval.ls.dataset [--dry-run]`: по умолчанию `tests/dataset.csv` и
    `eval/data/golden_retrieval_labeled.jsonl`. `--dry-run` печатает количество in-scope и OOS,
    сеть не трогает.
- [ ] **Шаг 4.** Тесты → PASS. `.venv/bin/python -m eval.ls.dataset --dry-run` →
  `22 examples: 12 in-scope, 10 OOS`.
- [ ] **Шаг 5.** Боевой upsert (бесплатно, без LLM): `.venv/bin/python -m eval.ls.dataset`.
  В UI: датасет `regrag-golden`, сплит `verified`, 22 примера. Повторный запуск →
  `created 0, updated 0, unchanged 22`.
- [ ] **Шаг 6.** Коммит `feat(eval): LangSmith dataset regrag-golden/verified`.

---

### Задача 2: code-evaluators

**Файлы:** создать `eval/ls/evaluators.py`, `tests/eval_ls/test_evaluators.py`.

**Интерфейсы.** Consumes: `outputs` из target (задача 3) — `{"answer": str,
"route_decision": str | None, "chunk_ids": list[str]}`; `reference_outputs` из задачи 1.
Produces:

```python
def doc_number(chunk_id: str) -> str            # "782н.pdf#431" → "782н"
def retrieval_hit(outputs, reference_outputs) -> dict   # {"key","score": 0|1|None,"comment"}
def doc_cited(outputs, reference_outputs) -> dict
def forbidden(outputs, reference_outputs) -> dict
def oos_handled(outputs, reference_outputs) -> dict
CODE_EVALUATORS = [retrieval_hit, doc_cited, forbidden, oos_handled]
def summarize(rows: list[dict[str, dict]]) -> dict
    # rows: [{key: evaluator_result}], → {"<key>": доля 1 среди не-None, "mrr": float, "n": {...}}
```

`retrieval_hit.comment` = `"rank=N"` (с 1) или `"miss"`; при `None` comment — причина
(`"oos"`). Для MRR `summarize` читает ранг из comment, промах даёт 0.

- [ ] **Шаг 1. Тест.** `tests/eval_ls/test_evaluators.py`:

```python
"""Unit tests for eval/ls/evaluators.py — pure functions."""

from __future__ import annotations

import pytest

from eval.ls.evaluators import (
    doc_cited,
    doc_number,
    forbidden,
    oos_handled,
    retrieval_hit,
    summarize,
)

pytestmark = pytest.mark.unit

IN_SCOPE = {
    "ground_truth": "не менее 2 смен",
    "relevant_chunk_ids": ["2464.pdf#107", "782н.pdf#431"],
    "must_not_contain": "",
    "forbidden_claims": "",
    "oos_type": "",
}
OOS = {**IN_SCOPE, "relevant_chunk_ids": [], "oos_type": "out_of_scope"}
FALSE_PREMISE = {**OOS, "oos_type": "false_premise"}


def out(answer="", route="generate", chunks=()):
    return {"answer": answer, "route_decision": route, "chunk_ids": list(chunks)}


@pytest.mark.parametrize(
    "cid, num",
    [("2464.pdf#107", "2464"), ("782н.pdf#431", "782н"), ("223н.pdf#298", "223н")],
)
def test_doc_number(cid, num):
    assert doc_number(cid) == num


class TestRetrievalHit:
    def test_hit_reports_first_rank(self):
        r = retrieval_hit(out(chunks=["x#1", "782н.pdf#431", "2464.pdf#107"]), IN_SCOPE)
        assert r == {"key": "retrieval_hit", "score": 1, "comment": "rank=2"}

    def test_miss(self):
        r = retrieval_hit(out(chunks=["x#1"]), IN_SCOPE)
        assert r["score"] == 0 and r["comment"] == "miss"

    def test_empty_answer_no_chunks_is_miss(self):
        assert retrieval_hit(out(route=None), IN_SCOPE)["score"] == 0

    def test_not_applicable_to_oos(self):
        assert retrieval_hit(out(chunks=["2464.pdf#107"]), OOS)["score"] is None


class TestDocCited:
    def test_number_in_answer(self):
        r = doc_cited(out("Согласно п. 53 Правил № 2464 ..."), IN_SCOPE)
        assert r["score"] == 1

    def test_case_insensitive_letter(self):
        assert doc_cited(out("по приказу № 782Н"), IN_SCOPE)["score"] == 1

    def test_not_cited(self):
        assert doc_cited(out("Стажировка не менее 2 смен."), IN_SCOPE)["score"] == 0

    def test_empty_answer(self):
        assert doc_cited(out(""), IN_SCOPE)["score"] == 0

    def test_number_inside_other_number_does_not_count(self):
        assert doc_cited(out("пункт 24640"), IN_SCOPE)["score"] == 0

    def test_not_applicable_to_oos(self):
        assert doc_cited(out("№ 2464"), OOS)["score"] is None


class TestForbidden:
    def test_empty_pattern_not_applicable(self):
        assert forbidden(out("что угодно"), IN_SCOPE)["score"] is None

    def test_pattern_found_is_zero(self):
        ref = {**IN_SCOPE, "must_not_contain": "раз в год|ежегодно"}
        assert forbidden(out("Проводится Ежегодно."), ref)["score"] == 0

    def test_pattern_absent_is_one(self):
        ref = {**IN_SCOPE, "must_not_contain": "раз в год|ежегодно"}
        assert forbidden(out("Раз в три года."), ref)["score"] == 1


class TestOosHandled:
    def test_abstain_route(self):
        assert oos_handled(out("какой-то текст", route="abstain"), OOS)["score"] == 1

    def test_empty_answer_from_domain_gate(self):
        assert oos_handled(out("", route=None), OOS)["score"] == 1

    def test_refusal_text(self):
        assert oos_handled(out("Не могу ответить: вопрос вне темы."), OOS)["score"] == 1

    def test_answered_oos_is_zero(self):
        assert oos_handled(out("Варите свёклу 40 минут."), OOS)["score"] == 0

    def test_false_premise_refuted(self):
        r = oos_handled(out("Это неверно: не реже раза в 3 года."), FALSE_PREMISE)
        assert r["score"] == 1

    def test_not_applicable_to_in_scope(self):
        assert oos_handled(out("ответ"), IN_SCOPE)["score"] is None


def test_summarize_rates_and_mrr():
    rows = [
        {"retrieval_hit": {"score": 1, "comment": "rank=1"}, "doc_cited": {"score": 1}},
        {"retrieval_hit": {"score": 1, "comment": "rank=4"}, "doc_cited": {"score": 0}},
        {"retrieval_hit": {"score": 0, "comment": "miss"}, "doc_cited": {"score": None}},
        {"retrieval_hit": {"score": None, "comment": "oos"}},
    ]
    s = summarize(rows)
    assert s["retrieval_hit"] == pytest.approx(2 / 3)
    assert s["doc_cited"] == pytest.approx(1 / 2)
    assert s["mrr"] == pytest.approx((1 + 0.25 + 0) / 3)
    assert s["n"]["retrieval_hit"] == 3
```

- [ ] **Шаг 2.** `pytest tests/eval_ls/test_evaluators.py -v` → FAIL (нет модуля).
- [ ] **Шаг 3. Реализация (Пётр).** Подсказки:
  - in-scope ⇔ `reference_outputs["oos_type"] == ""`.
  - `doc_cited`: множество `doc_number(...)` по `relevant_chunk_ids`; поиск
    `re.search(rf"(?<!\d){re.escape(num)}(?!\d)", answer, re.IGNORECASE)`.
  - `forbidden`: `eval.metrics.compute_inversion_detected(must_not_contain, answer)` — уже
    умеет `|` и регистр; пустой паттерн → `None`.
  - `oos_handled`: `route_decision == "abstain"` или пустой ответ, или отказ/опровержение в
    первых 200 знаках. Регэксп отказа как в `run_v7_eval._REFUSAL_RE` (`\bнет\b|не могу`) плюс
    `неверно`. Не импортировать `run_v7_eval`: он при импорте грузит settings, LLM-фабрику и
    логирование.
- [ ] **Шаг 4.** Тесты → PASS.
- [ ] **Шаг 5.** Коммит `feat(eval): code evaluators for LangSmith experiments`.

---

### Задача 3: target и первый эксперимент без судьи

**Файлы:** создать `eval/ls/target.py`, `eval/ls/run.py`, `tests/eval_ls/test_target.py`.

**Интерфейсы.** Consumes: `CODE_EVALUATORS`, `summarize` (задача 2), `DATASET_NAME`,
`SPLIT_VERIFIED` (задача 1). Produces:

```python
# target.py
def state_to_outputs(state: dict) -> dict
    # → {"answer": str, "route_decision": str | None, "chunk_ids": list[str], "sources": list[str]}
def make_target() -> Callable[[dict], dict]
    # строит граф один раз; inputs {"question"} → state_to_outputs(...)

# run.py — CLI
# python -m eval.ls.run --split verified --prefix v7-low [--skip-judge] [--limit N]
def run_experiment(split: str, prefix: str, skip_judge: bool, limit: int | None) -> str
    # возвращает имя эксперимента
```

- [ ] **Шаг 1. Тест.** `tests/eval_ls/test_target.py`:

```python
"""Unit tests for eval/ls/target.py — state mapping only, no graph."""

from __future__ import annotations

import pytest

from eval.ls.target import state_to_outputs

pytestmark = pytest.mark.unit


def _p(source, cid):
    return {"text": "t", "metadata": {"source": source, "chunk_id": cid}, "chunk_id": cid,
            "doc_id": source}


def test_maps_answer_route_and_ordered_chunk_ids():
    state = {
        "answer": "Ответ",
        "route_decision": "generate",
        "final_passages": [_p("782н.pdf", 431), _p("2464.pdf", 107), _p("782н.pdf", 431)],
    }
    o = state_to_outputs(state)
    assert o["answer"] == "Ответ"
    assert o["route_decision"] == "generate"
    assert o["chunk_ids"] == ["782н.pdf#431", "2464.pdf#107"]
    assert o["sources"] == ["782н.pdf", "2464.pdf"]


def test_domain_gate_state_without_route():
    o = state_to_outputs({"answer": ""})
    assert o == {"answer": "", "route_decision": None, "chunk_ids": [], "sources": []}
```

- [ ] **Шаг 2.** Тест → FAIL.
- [ ] **Шаг 3. Реализация (Пётр).** Подсказки:
  - `chunk_ids` — `eval.run_retrieval_eval.extract_chunk_ids(final_passages)`: тот же ключ
    `source#chunk_id`, что в разметке, порядок и дедуп сохранены. Если формат ключа в тесте
    разойдётся с `passage_identity`, верна функция, поправить тест.
  - `sources` — уникальные `metadata["source"]` в порядке появления.
  - `make_target`: как в `run_v7_eval.run`: `get_vector_store_backend(load_existing=True)` →
    `build_full_v7_runtime` → `build_graph(runtime=...).compile()`; вызов
    `src.v7.runner.run_query(graph, q, source="eval", run_id=...)`. Исключение не ловить:
    `client.evaluate(..., error_handling="log")` сам запишет его как error строки.
  - `run.py`: `Client().evaluate(target, data=client.list_examples(dataset_name=...,
    splits=[split]), evaluators=..., experiment_prefix=prefix, max_concurrency=1,
    metadata={"effort": settings.OPENROUTER_REASONING_EFFORT, "git_sha": ...})`.
    После прогона печатает `summarize(...)` и ссылку на эксперимент. `--limit` режет список
    примеров для смоука.
- [ ] **Шаг 4.** Тест → PASS. Полный CI-гейт: `pytest -m "not integration and not slow"` —
  новых падений нет (база — 6 известных красных из `CLAUDE.md`).
- [ ] **Шаг 5.** Коммит `feat(eval): v7 target and experiment runner`.
- [ ] **Шаг 6. Платный смоук — по «да» Петра** (≈ $0.02):
  `LANGSMITH_TRACING_V2=true .venv/bin/python -m eval.ls.run --split verified --prefix v7-smoke --skip-judge --limit 2`.
  Проверить в UI: 2 строки, 4 code-ключа, `None` у неприменимых, трейсы графа внутри строк.
- [ ] **Шаг 7. Платный полный прогон — по «да» Петра** (≈ $0.15):
  `--prefix v7-low --skip-judge` без `--limit`. Цифры записать в
  `docs/evaluation/experiments/langsmith-v8-stage1.md`: имя эксперимента, git sha, effort, доли
  по ключам, MRR, latency p50 и cost из трейса, список промахов с пометкой
  `corpus_support=unverified`. Коммит `docs(eval): v7 LangSmith baseline without judge`.

---

### Задача 4: судья correctness

**Файлы:** создать `eval/ls/judge.py`, `eval/prompts/judge_correctness.md`,
`tests/eval_ls/test_judge.py`; изменить `eval/ls/run.py` (без `--skip-judge` добавить судью).

**Интерфейсы.** Produces:

```python
def parse_verdict(raw: str) -> dict
    # → {"correct": bool, "reasoning": str, "forbidden_claims_asserted": list[str]}
def make_correctness_evaluator(llm) -> Callable
    # evaluator(inputs, outputs, reference_outputs) -> list[dict]:
    #   [{"key": "correctness", "score": 1|0|None, "comment": reasoning},
    #    {"key": "forbidden_claim", "score": 1|0|None, "comment": ...}]
    # forbidden_claim: 1 = нарушений нет; None, если forbidden_claims пуст или пример OOS.
    # OOS → оба None, LLM не вызывается.
PROMPT_PATH: Path  # eval/prompts/judge_correctness.md
```

Промпт — бинарная версия `golden_correctness.md`: те же правила (доп. верные факты не
штрафуются, `forbidden_claims` засчитываются только как утверждаемое правило), шкала заменена
на «верно = все ключевые факты эталона есть и нет фактических ошибок». Ответ — JSON
`{"correct": true|false, "reasoning": "...", "forbidden_claims_asserted": [...]}`.

- [ ] **Шаг 1. Тест.** `tests/eval_ls/test_judge.py`:

```python
"""Unit tests for eval/ls/judge.py — parsing and routing, fake LLM."""

from __future__ import annotations

import json

import pytest

from eval.ls.judge import make_correctness_evaluator, parse_verdict

pytestmark = pytest.mark.unit

FENCE = "`" * 3  # markdown-ограда; литералом сломала бы блок кода в плане


def test_parse_plain_json():
    v = parse_verdict('{"correct": true, "reasoning": "ok", "forbidden_claims_asserted": []}')
    assert v == {"correct": True, "reasoning": "ok", "forbidden_claims_asserted": []}


def test_parse_fenced_json():
    raw = FENCE + 'json\n{"correct": false, "reasoning": "нет п. 53"}\n' + FENCE
    v = parse_verdict(raw)
    assert v["correct"] is False and v["forbidden_claims_asserted"] == []


@pytest.mark.parametrize(
    "raw",
    ['{"correct": "yes", "reasoning": ""}', '{"reasoning": "x"}', "не json",
     '{"correct": true, "forbidden_claims_asserted": "строка"}'],
)
def test_parse_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_verdict(raw)


class FakeLLM:
    """Минимальный Runnable-совместимый стаб: chain = prompt | llm | StrOutputParser."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, *_a, **_kw):
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


REF = {"ground_truth": "2 смены", "relevant_chunk_ids": ["2464.pdf#1"],
       "must_not_contain": "", "forbidden_claims": "Утверждается, что одна смена.",
       "oos_type": ""}


def _by_key(results):
    return {r["key"]: r for r in results}


def test_correct_answer_scores():
    llm = FakeLLM({"correct": True, "reasoning": "ок", "forbidden_claims_asserted": []})
    ev = make_correctness_evaluator(llm)
    r = _by_key(ev({"question": "q"}, {"answer": "2 смены"}, REF))
    assert r["correctness"]["score"] == 1 and r["correctness"]["comment"] == "ок"
    assert r["forbidden_claim"]["score"] == 1


def test_forbidden_claim_asserted():
    llm = FakeLLM({"correct": False, "reasoning": "x",
                   "forbidden_claims_asserted": ["Утверждается, что одна смена."]})
    r = _by_key(make_correctness_evaluator(llm)({"question": "q"}, {"answer": "1"}, REF))
    assert r["correctness"]["score"] == 0 and r["forbidden_claim"]["score"] == 0


def test_no_forbidden_claims_is_none():
    llm = FakeLLM({"correct": True, "reasoning": "", "forbidden_claims_asserted": []})
    ref = {**REF, "forbidden_claims": ""}
    r = _by_key(make_correctness_evaluator(llm)({"question": "q"}, {"answer": "a"}, ref))
    assert r["forbidden_claim"]["score"] is None


def test_oos_skips_llm():
    llm = FakeLLM({"correct": True, "reasoning": "", "forbidden_claims_asserted": []})
    ref = {**REF, "relevant_chunk_ids": [], "oos_type": "out_of_scope"}
    r = _by_key(make_correctness_evaluator(llm)({"question": "q"}, {"answer": "a"}, ref))
    assert r["correctness"]["score"] is None and llm.calls == 0
```

- [ ] **Шаг 2.** Тест → FAIL.
- [ ] **Шаг 3. Реализация (Пётр).** Подсказки: цепочка как в
  `run_v7_eval.evaluate_correctness`: `ChatPromptTemplate.from_template(...) | llm |
  StrOutputParser()`; переменные `question, legal_as_of, ground_truth, forbidden_claims,
  answer`. `legal_as_of` взять из metadata примера: добавить его в `outputs` в задаче 1 или
  принять evaluator'ом `example`; выбор за Петром, тест поправить под выбор. Если FakeLLM не
  совместим с `|`, обернуть его в `RunnableLambda` в тесте, не в коде. Судья в `run.py` —
  `get_judge_llm(temperature=0.0, seed=12345)`. Возврат нескольких ключей — проверить в 0.12.1
  формат `{"results": [...]}` против списка (`langsmith/evaluation/evaluator.py`, разбор
  результата) и привести тест к нему.
- [ ] **Шаг 4.** Тесты → PASS. Коммит `feat(eval): binary correctness judge`.
- [ ] **Шаг 5. Платно — по «да» Петра** (≈ $0.2). Судья на готовых ответах, без генерации:
  `Client().evaluate("<имя эксперимента v7-low>", evaluators=[correctness])` — `evaluate`
  принимает имя существующего эксперимента как target. Вынести в
  `python -m eval.ls.run --rejudge <experiment>`.

---

### Задача 5: alignment судьи

**Файлы:** создать `eval/ls/align.py`, `tests/eval_ls/test_align.py`.

**Интерфейсы.** Produces:

```python
def agreement(pairs: list[tuple[bool, bool]]) -> dict
    # pairs: (human, judge). → {"n", "agreement", "tp", "tn", "fp", "fn"}
    # fp = judge True при human False (судья пропустил ошибку — худший случай).
# CLI: python -m eval.ls.align <experiment> [--human-key human_correct] [--judge-key correctness]
```

- [ ] **Шаг 1. Тест.** `tests/eval_ls/test_align.py`:

```python
"""Unit tests for eval/ls/align.py."""

from __future__ import annotations

import pytest

from eval.ls.align import agreement

pytestmark = pytest.mark.unit


def test_confusion_and_rate():
    pairs = [(True, True), (True, True), (False, False), (False, True), (True, False)]
    a = agreement(pairs)
    assert a == {"n": 5, "agreement": pytest.approx(0.6), "tp": 2, "tn": 1, "fp": 1, "fn": 1}


def test_empty_raises():
    with pytest.raises(ValueError):
        agreement([])
```

- [ ] **Шаг 2.** Тест → FAIL. **Шаг 3 (Пётр):** реализация `agreement`; CLI собирает пары
  через `client.list_runs(project_name=<experiment>, is_root=True)` +
  `client.list_feedback(run_ids=..., feedback_key=[...])`. Строки без человеческой метки
  пропускать и печатать их число. **Шаг 4:** PASS, коммит
  `feat(eval): judge alignment report`.
- [ ] **Шаг 5. Разметка (Пётр, UI).** Annotation queue `regrag-correctness`, рубрика:
  фидбэк-ключ `human_correct` (bool) + комментарий. В очередь — 12 in-scope строк эксперимента
  `v7-low` с судьёй. Пётр размечает **до** того, как смотрит оценки судьи.
- [ ] **Шаг 6. Итерации.** `python -m eval.ls.align <exp>` → если согласие < 85 % (≥ 11/12),
  разобрать fp/fn, поправить `judge_correctness.md`, `--rejudge` (каждый — по «да», ≈ $0.2).
  Каждую итерацию — отдельный коммит промпта с цифрой согласия в сообщении и строкой в
  `docs/evaluation/experiments/langsmith-v8-stage1.md` (итерация, sha промпта, agreement,
  fp/fn). Стоп-условие: ≥ 85 % или 4 итерации, после 4 — решение Петра.

---

### Задача 6: pairwise effort=low vs medium

**Файлы:** создать `eval/ls/pairwise.py`, `eval/prompts/judge_pairwise.md`,
`tests/eval_ls/test_pairwise.py`.

**Интерфейсы.** Produces:

```python
def parse_preference(raw: str) -> int        # "A"→0, "B"→1, "TIE"→-1; иначе ValueError
def make_pairwise_evaluator(llm) -> Callable  # (inputs, outputs: list[dict], reference_outputs) ->
    # {"key": "preference", "scores": {run_id_a: 1|0|0.5, run_id_b: ...}}
# CLI: python -m eval.ls.pairwise <exp_low> <exp_medium>
```

- [ ] **Шаг 1. Тест.** `tests/eval_ls/test_pairwise.py`:

```python
"""Unit tests for eval/ls/pairwise.py."""

from __future__ import annotations

import pytest

from eval.ls.pairwise import parse_preference

pytestmark = pytest.mark.unit

FENCE = "`" * 3


@pytest.mark.parametrize("raw, want", [
    ('{"winner": "A"}', 0), ('{"winner": "b"}', 1), ('{"winner": "TIE"}', -1),
    (FENCE + 'json\n{"winner": "A"}\n' + FENCE, 0),
])
def test_parse(raw, want):
    assert parse_preference(raw) == want


@pytest.mark.parametrize("raw", ['{"winner": "C"}', "{}", "A"])
def test_rejects(raw):
    with pytest.raises(ValueError):
        parse_preference(raw)
```

- [ ] **Шаг 2.** FAIL. **Шаг 3 (Пётр):** реализация. Промпт сравнивает два ответа против
  эталона по тем же правилам, что correctness; порядок A/B рандомизирует
  `evaluate_comparative(..., randomize_order=True)`. **Шаг 4:** PASS, коммит
  `feat(eval): pairwise judge`.
- [ ] **Шаг 5. Платно — по «да» Петра** (≈ $0.4 генерация + ≈ $0.2 судья). Второй эксперимент
  в отдельном процессе, потому что effort читается в settings при импорте:
  `OPENROUTER_REASONING_EFFORT=medium LANGSMITH_TRACING_V2=true python -m eval.ls.run --split verified --prefix v7-medium`.
  Затем `python -m eval.ls.pairwise v7-low-… v7-medium-…`. Итог, cost и latency обоих — в
  `langsmith-v8-stage1.md`.

---

### Задача 7: online eval и automation (UI, без кода)

- [ ] **Шаг 1.** Проект `regulatory-rag` → Rules → online evaluator `oos_handled` (код-evaluator
  UI, логика как в задаче 2; на вход `outputs.answer` и `route_decision` из корневого run) на
  100 % трафика с `metadata.source != "eval"`.
- [ ] **Шаг 2.** Правило: judge `correctness` без эталона невозможен, поэтому на выборке 10 %
  — reference-free вариант («ответ опирается на источники и не противоречит им»). Отметить в
  доке, что это другая метрика.
- [ ] **Шаг 3.** Automation: фидбэк 👎 (ключ из `src/ui_feedback.py` — проверить имя) или
  `oos_handled=0` → annotation queue `regrag-from-traces`.
- [ ] **Шаг 4.** Проверка: задать на 8502 один OOS-вопрос и один in-scope с 👎, убедиться, что
  оба попали в очередь. Трейсинг 8502 включается по согласию Петра, квоту проверить заранее.
- [ ] **Шаг 5.** Скриншоты правил и описание — в `langsmith-v8-stage1.md`, коммит.

---

### Задача 8 (если хватит времени до 11.10): сплит `from_traces`

- [ ] Из очереди `regrag-from-traces` и Insights по трейсам — 10+ примеров в датасет со сплитом
  `from_traces` (кнопка «Add to dataset» в очереди). Эталоны дописывает Пётр. `dataset.py` их
  не трогает: его upsert работает только по id из CSV.

---

## Порядок и стоимость

0 → 1 → 2 → 3 (смоук, базовый прогон) → 4 → 5 → 6 → 7 → 8. Задачи 1–2 бесплатны и не требуют
сети, кроме upsert. Платные шаги: 3.6, 3.7, 4.5, 5.6, 6.5 — суммарно ≈ $2–3, каждый по «да».
