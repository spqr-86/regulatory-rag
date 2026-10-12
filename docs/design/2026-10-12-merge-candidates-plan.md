# Склейка кандидатов complex-пути: MMR по реранкеру — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** кандидат `merged` в `evaluate_complex` перестаёт терять чанки, которые реранкер поставил высоко (G001: 2464#105/#109, G020: 29н#319).

**Architecture:** `mmr_select` берёт релевантность сначала из `rerank_score`. `_candidates` собирает `merged` так: 12 якорей из `fallback_snapshot`, затем `mmr_select` по последней complex-попытке без якорей. Simple-попытка в пул не входит. `merge_all_passages` удаляется.

**Tech Stack:** Python 3.13, pytest, `.venv` проекта. Запуск: `PYTHONPATH=$PWD .venv/bin/python -m pytest ...` из корня репо.

**Spec:** `docs/design/2026-10-12-merge-candidates-design.md`

## Global Constraints

- `SIMPLE_TOP_K = 12`, `FINAL_MERGE_TOP_K = 24` (`src/v7/config.py`) не меняются.
- Формула MMR, штраф за повтор `doc_id` и `mmr_lambda` из плана не меняются.
- Кандидаты `last_attempt` и `fallback_snapshot` не меняются.
- Платных вызовов нет. LangSmith не вызывается: фикстуры собираются из уже скачанных дампов.
- Фикстуры — без `text` и прочих внутренних полей сверх нужного (репо публичное).
- `black`/`ruff` — только по своим файлам. CI-гейт: `pytest -m "not integration and not slow"`, сверка с базой красных тестов.

## Review Focus

1. Последняя попытка не complex (только simple) → пул пуст, `merged` пуст (решение при исполнении: см. спек). Тест в Task 2.
2. Snapshot короче 12 → освободившиеся места отдаются complex-пулу, всего ≤ 24. Тест в Task 2.
3. Реранкер не подключён (нет `rerank_score`) → MMR по `vector_score`, как раньше. Тест в Task 2.
4. Один чанк дважды в complex-попытке → в `merged` один раз. Тест в Task 2.
5. Чанк из пула совпадает с якорем → не дублируется, якорь стоит на своём месте. Тест в Task 2.

`rerank_score == 0.0` по финальному ревью считается оценкой, а не отсутствием (FlashRank даёт ~1e-5 → 0.0 после округления); тест `test_zero_rerank_score_is_a_score_not_missing`.

---

### Task 1: Фикстуры G001/G020 и красный regression-тест

**Files:**
- Create: `scripts/make_merge_fixtures.py`
- Create: `tests/v7/fixtures/merge_candidates/g001_program_a.json`
- Create: `tests/v7/fixtures/merge_candidates/g020_drivers_medical.json`
- Create: `tests/v7/test_nodes/test_evaluate_complex_merge.py`

**Interfaces:**
- Consumes: `src.v7.nodes.evaluate_complex._candidates(state) -> List[Candidate]`, `src.v7.nlp_core.passage_identity(p) -> str` (`"{metadata.source}#{chunk_id}"`).
- Produces: фикстуры — урезанный `inputs` трейса (`query`, `active_query`, `plan`, `filters`, `retrieval_attempts[{stage, attempt_plan, retrieval_error, passages}]`, `fallback_snapshot`), у пассажей только `chunk_id`, `doc_id`, `metadata{source, chunk_id, page_no}` и оценки.

- [ ] **Step 0: Записать базу гейта до правок**

Run:
```bash
cd ~/projects/ai/regulatory-rag
PYTHONPATH=$PWD .venv/bin/python -m pytest -m "not integration and not slow" -q -p no:cacheprovider --ignore=tests/eval_ls 2>&1 | tail -15 > /tmp/claude-1000/merge_gate_before.txt
```
`tests/eval_ls/` — untracked, ломает сбор (известно). Сохранить список `FAILED`.

- [ ] **Step 1: Скрипт урезания дампа**

```python
"""Build trimmed evaluate_complex fixtures from a `langsmith run get --full` dump.

Usage: python scripts/make_merge_fixtures.py <run.out> <fixture.json>

Keeps only what `_candidates` reads: identities and scores. Passage text and
other internal fields are dropped — the repository is public.
"""

import json
import sys

PASSAGE_KEYS = (
    "chunk_id",
    "doc_id",
    "vector_score",
    "rerank_score",
    "bm25_score",
    "rrf_score",
    "score",
)
META_KEYS = ("source", "chunk_id", "page_no")


def trim_passage(p: dict) -> dict:
    out = {k: p[k] for k in PASSAGE_KEYS if k in p}
    meta = p.get("metadata") or {}
    out["metadata"] = {k: meta[k] for k in META_KEYS if k in meta}
    return out


def trim(inputs: dict) -> dict:
    attempts = [
        {
            "stage": a.get("stage"),
            "attempt_plan": a.get("attempt_plan") or {},
            "retrieval_error": a.get("retrieval_error", False),
            "passages": [trim_passage(p) for p in a.get("passages") or []],
        }
        for a in inputs.get("retrieval_attempts") or []
    ]
    out = {
        "query": inputs.get("query", ""),
        "active_query": inputs.get("active_query", inputs.get("query", "")),
        "plan": inputs.get("plan") or {},
        "filters": inputs.get("filters"),
        "retrieval_attempts": attempts,
    }
    snap = inputs.get("fallback_snapshot")
    if snap:
        out["fallback_snapshot"] = {
            **{k: v for k, v in snap.items() if k != "passages"},
            "passages": [trim_passage(p) for p in snap.get("passages") or []],
        }
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        run = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(trim(run["inputs"]), f, ensure_ascii=False, indent=1)
        f.write("\n")
```

- [ ] **Step 2: Собрать фикстуры**

Дампы уже скачаны: `/tmp/claude-1000/-home-petr/f68523ad-bcbe-4e6c-8d4c-2f3a7951ee84/scratchpad/ls/`. Если их нет — перекачать:
`set -a; . ./.env; langsmith run get <id> --full --project-id 04aba4b6-a9cc-4fb3-ae7c-f5f95ad2fc70 --since 2026-10-08T00:00:00Z > <id>.out`.

```bash
D=/tmp/claude-1000/-home-petr/f68523ad-bcbe-4e6c-8d4c-2f3a7951ee84/scratchpad/ls
mkdir -p tests/v7/fixtures/merge_candidates
.venv/bin/python -I scripts/make_merge_fixtures.py $D/01a11fd8-c772-7232-a031-52437c041a6d.out tests/v7/fixtures/merge_candidates/g001_program_a.json
.venv/bin/python -I scripts/make_merge_fixtures.py $D/01a11fe5-4faf-7932-9909-c48ee34d0ed5.out tests/v7/fixtures/merge_candidates/g020_drivers_medical.json
grep -c '"text"' tests/v7/fixtures/merge_candidates/*.json
```
Expected: `grep` даёт 0 для обоих файлов. Глазами проверить `plan`/`attempt_plan`/`filters` — там только параметры поиска, без текстов.

- [ ] **Step 3: Написать regression-тест**

```python
"""Merged complex candidate: rerank order + anchors (spec 2026-10-12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.v7.nlp_core import passage_identity
from src.v7.nodes.evaluate_complex import _candidates

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "merge_candidates"

# Targets the reranker ranked high but the old merge dropped (trace reviews
# docs/evaluation/experiments/langsmith-learning-2026-10-09/).
RECORDED = [
    ("g001_program_a.json", ["2464.pdf#105", "2464.pdf#109"]),
    ("g020_drivers_medical.json", ["29н.pdf#319"]),
]


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.xfail(strict=True, reason="merge ignores rerank order; fixed in Task 2")
@pytest.mark.parametrize("name,targets", RECORDED)
def test_recorded_targets_reach_merged_candidate(name, targets):
    merged = _candidates(_load(name))[0]["passages"]
    ids = {passage_identity(p) for p in merged}
    assert [t for t in targets if t not in ids] == []
```

- [ ] **Step 4: Убедиться, что тест красный на текущем коде**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/v7/test_nodes/test_evaluate_complex_merge.py -v`
Expected: оба случая `XFAIL`. Для контроля временно снять `xfail` и увидеть `FAILED` с недостающими id: G001 — оба id 2464, G020 — `29н.pdf#319`. Вернуть `xfail`.

- [ ] **Step 5: Commit**

```bash
black scripts/make_merge_fixtures.py tests/v7/test_nodes/test_evaluate_complex_merge.py
ruff check --fix scripts/make_merge_fixtures.py tests/v7/test_nodes/test_evaluate_complex_merge.py
git add scripts/make_merge_fixtures.py tests/v7/fixtures/merge_candidates tests/v7/test_nodes/test_evaluate_complex_merge.py
git commit -m "test(v7): recorded G001/G020 inputs for merged candidate regression"
```

---

### Task 2: MMR по реранкеру и новая сборка `merged`

**Files:**
- Modify: `src/v7/nlp_core.py:347-349` (relevance в `mmr_select`)
- Modify: `src/v7/nodes/evaluate_complex.py:18,33-54` (`_candidates`, импорт)
- Modify: `tests/v7/test_nlp_core.py:443-469` (`TestPhase1MMRPrefersVectorScore`)
- Modify: `tests/v7/test_nodes/test_evaluate_complex_merge.py`

**Interfaces:**
- Consumes: фикстуры и тест из Task 1; `mmr_select(passages, top_k, lambda_param=None) -> List[dict]`.
- Produces: `_merged_passages(state, plan) -> List[dict]` в `evaluate_complex.py`; `_candidates` возвращает тот же список из трёх кандидатов.

- [ ] **Step 1: Переписать тест `mmr_select`**

Заменить класс `TestPhase1MMRPrefersVectorScore` целиком:

```python
class TestPhase1MMRRelevance:
    @pytest.mark.unit
    def test_mmr_prefers_rerank_score_over_vector_score(self):
        """The cross-encoder score is the best available relevance signal."""
        passages = [
            {"chunk_id": "v", "vector_score": 0.9, "rerank_score": 0.2, "doc_id": "a.pdf"},
            {"chunk_id": "r", "vector_score": 0.3, "rerank_score": 0.95, "doc_id": "b.pdf"},
        ]
        result = mmr_select(passages, top_k=1, lambda_param=1.0)
        assert [p["chunk_id"] for p in result] == ["r"]

    @pytest.mark.unit
    def test_vector_score_beats_bm25_only_without_rerank(self):
        """Without rerank_score a vector chunk still beats a BM25-only chunk."""
        passages = [
            {
                "text": "чанк из векторного поиска",
                "vector_score": 0.9,
                "score": 0.9,
                "doc_id": "a.pdf",
                "chunk_id": "v1",
            },
            {
                "text": "чанк только из BM25",
                "bm25_score": 12.0,
                "score": round(12.0 / (12.0 + 5.0), 4),  # ~0.706
                "doc_id": "b.pdf",
                "chunk_id": "b1",
            },
        ]
        result = mmr_select(passages, top_k=1, lambda_param=1.0)
        assert [p["chunk_id"] for p in result] == ["v1"]
```

- [ ] **Step 2: Юнит-тесты правила сборки**

Дописать в `tests/v7/test_nodes/test_evaluate_complex_merge.py`:

```python
from src.v7 import pack_context as pc

PLAN = {"mmr_lambda": 0.7}


def _p(cid, rerank=None, vector=0.5, source="doc.pdf"):
    p = {"chunk_id": cid, "doc_id": source, "vector_score": vector,
         "metadata": {"source": source}}
    if rerank is not None:
        p["rerank_score"] = rerank
    return p


def _state(complex_passages, anchors=None, stages=("simple", "complex")):
    attempts = [{"stage": s, "passages": [], "attempt_plan": PLAN} for s in stages]
    if "complex" in stages:
        attempts[-1]["passages"] = complex_passages
    state = {"query": "q", "active_query": "q", "plan": PLAN,
             "retrieval_attempts": attempts}
    if anchors is not None:
        state["fallback_snapshot"] = {
            "passages": anchors, "plan": PLAN, "active_query": "q",
            "origin": "fallback_snapshot", "packed": True, "pack_status": "ok",
        }
    return state


def _ids(state):
    return [p["chunk_id"] for p in _candidates(state)[0]["passages"]]


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(pc.v7_config, "SIMPLE_TOP_K", 2)
    monkeypatch.setattr(pc.v7_config, "FINAL_MERGE_TOP_K", 4)


def test_anchors_lead_in_snapshot_order(small_limits):
    anchors = [_p("a1", vector=0.1), _p("a2", vector=0.9), _p("a3")]
    pool = [_p(f"c{i}", rerank=0.9 - i / 100) for i in range(5)]
    ids = _ids(_state(pool, anchors))
    assert ids[:2] == ["a1", "a2"]
    assert len(ids) == 4


def test_pool_chunk_equal_to_anchor_not_duplicated(small_limits):
    anchors = [_p("a1"), _p("a2")]
    pool = [_p("a1", rerank=0.99), _p("c1", rerank=0.5), _p("c2", rerank=0.4)]
    assert _ids(_state(pool, anchors)) == ["a1", "a2", "c1", "c2"]


def test_no_snapshot_fills_all_slots_from_pool(small_limits):
    pool = [_p(f"c{i}", rerank=0.9 - i / 100, source=f"d{i}.pdf") for i in range(6)]
    assert _ids(_state(pool)) == ["c0", "c1", "c2", "c3"]


def test_short_snapshot_leaves_more_room_for_pool(small_limits):
    pool = [_p(f"c{i}", rerank=0.9 - i / 100, source=f"d{i}.pdf") for i in range(6)]
    assert _ids(_state(pool, [_p("a1")])) == ["a1", "c0", "c1", "c2"]


def test_empty_complex_attempt_gives_anchors_only(small_limits):
    assert _ids(_state([], [_p("a1"), _p("a2")])) == ["a1", "a2"]


def test_without_complex_attempt_pool_is_empty(small_limits):
    state = _state([], [_p("a1")], stages=("simple",))
    state["retrieval_attempts"][0]["passages"] = [_p("s1", vector=0.99)]
    assert _ids(state) == ["a1"]


def test_simple_attempt_not_in_pool(small_limits):
    state = _state([_p("c1", rerank=0.5)], [_p("a1")])
    state["retrieval_attempts"][0]["passages"] = [_p("s1", vector=0.99)]
    assert "s1" not in _ids(state)


def test_high_rerank_low_vector_beats_inverse(small_limits):
    pool = [
        _p("hi_vec", rerank=0.10, vector=0.95, source="x.pdf"),
        _p("hi_rr", rerank=0.95, vector=0.10, source="y.pdf"),
        _p("mid", rerank=0.50, vector=0.50, source="z.pdf"),
    ]
    ids = _ids(_state(pool, [_p("a1"), _p("a2")]))
    assert ids == ["a1", "a2", "hi_rr", "mid"]


def test_without_rerank_falls_back_to_vector_order(small_limits):
    pool = [_p(f"c{i}", vector=0.1 * i, source=f"d{i}.pdf") for i in range(5)]
    assert _ids(_state(pool, [_p("a1"), _p("a2")])) == ["a1", "a2", "c4", "c3"]


def test_duplicate_in_pool_counted_once(small_limits):
    pool = [_p("c1", rerank=0.9), _p("c1", rerank=0.9), _p("c2", rerank=0.5)]
    assert _ids(_state(pool, [_p("a1"), _p("a2")])) == ["a1", "a2", "c1", "c2"]
```

- [ ] **Step 3: Убедиться, что новые тесты красные**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/v7/test_nodes/test_evaluate_complex_merge.py tests/v7/test_nlp_core.py::TestPhase1MMRRelevance -v`
Expected: FAIL как минимум `test_mmr_prefers_rerank_score_over_vector_score`, `test_high_rerank_low_vector_beats_inverse`, `test_simple_attempt_not_in_pool`, `test_without_complex_attempt_pool_is_empty`. Записать фактический список упавших.

- [ ] **Step 4: Правка `mmr_select`**

В `src/v7/nlp_core.py`:

```python
            relevance = (
                p.get("rerank_score") or p.get("vector_score") or p.get("score", 0.0)
            )
```

- [ ] **Step 5: Правка `_candidates`**

В `src/v7/nodes/evaluate_complex.py` импорт:
```python
from src.v7.nlp_core import mmr_select, passage_identity
```
Новая функция перед `_candidates` и новое начало `_candidates`:

```python
def _merged_passages(state: RAGState, plan: RetrievalPlan) -> List[dict]:
    """Simple top-k anchors, then MMR over the last complex attempt.

    Escalation is an additive search step: the simple top-k stays at the head
    so a direct norm cannot be displaced. The complex pool is ranked by the
    reranker (see mmr_select); simple scores are not comparable with it, so the
    simple attempt is not pooled.
    """
    snapshot = state.get("fallback_snapshot") or {}
    anchors = list(snapshot.get("passages") or [])[: v7_config.SIMPLE_TOP_K]
    seen = {passage_identity(p) for p in anchors}
    complex_attempts = [
        a for a in state.get("retrieval_attempts") or [] if a.get("stage") == "complex"
    ]
    last_complex = complex_attempts[-1].get("passages") if complex_attempts else None
    pool: List[dict] = []
    for p in last_complex or []:
        pid = passage_identity(p)
        if pid not in seen:
            seen.add(pid)
            pool.append(p)
    room = max(v7_config.FINAL_MERGE_TOP_K - len(anchors), 0)
    return anchors + mmr_select(pool, room, plan.get("mmr_lambda"))


def _candidates(state: RAGState) -> List[Candidate]:
    """Return merged, last-attempt, then immutable simple snapshot candidates."""
    attempts = state.get("retrieval_attempts") or []
    last = attempts[-1]
    plan = cast(RetrievalPlan, last.get("attempt_plan") or state.get("plan") or {})
    active_q = state.get("active_query", state.get("query", ""))
    snapshot = state.get("fallback_snapshot")

    merged = _merged_passages(state, plan)
    out: List[Candidate] = [
```
(остальная часть `_candidates` — без изменений; старый блок `merge_all_passages(...)` и `if snapshot: anchors ... novel ...` удалить.)

Примечание: при `room == 0` и непустом пуле `mmr_select` вернёт `[]` (цикл `range(0)`); при пустом пуле — `[]` (ветка `len <= top_k`).

- [ ] **Step 6: Снять `xfail` с regression-теста**

Удалить строку `@pytest.mark.xfail(strict=True, ...)` в `test_evaluate_complex_merge.py`.

- [ ] **Step 7: Тесты зелёные**

Run:
```bash
PYTHONPATH=$PWD .venv/bin/python -m pytest tests/v7/test_nodes/test_evaluate_complex_merge.py tests/v7/test_nodes/test_evaluate_complex.py tests/v7/test_nlp_core.py -v
```
Expected: PASS всё, включая оба случая `test_recorded_targets_reach_merged_candidate` и прежний `test_merged_candidate_keeps_simple_evidence_ahead_of_complex_noise`.

- [ ] **Step 8: Commit**

```bash
black src/v7/nlp_core.py src/v7/nodes/evaluate_complex.py tests/v7/test_nlp_core.py tests/v7/test_nodes/test_evaluate_complex_merge.py
ruff check --fix src/v7/nlp_core.py src/v7/nodes/evaluate_complex.py tests/v7/test_nlp_core.py tests/v7/test_nodes/test_evaluate_complex_merge.py
git add src/v7/nlp_core.py src/v7/nodes/evaluate_complex.py tests/v7/test_nlp_core.py tests/v7/test_nodes/test_evaluate_complex_merge.py
git commit -m "fix(v7): merge complex candidates by rerank-score MMR after simple anchors"
```
`black` может переформатировать соседние строки в `test_nlp_core.py` — в коммит только свои изменения (`git diff --stat` перед `add`).

---

### Task 3: Удалить `merge_all_passages`, поправить доки, прогнать гейт

**Files:**
- Modify: `src/v7/nlp_core.py:316-330,371-402`
- Modify: `tests/v7/test_nlp_core.py:12,302-366`
- Modify: `src/v7/config.py:55` (комментарий)
- Modify: `tests/v7/test_bridge.py:148` (докстринг)

**Interfaces:**
- Consumes: `_merged_passages` из Task 2 — единственный вызов `mmr_select` в `src/`.
- Produces: `merge_all_passages` больше нет.

- [ ] **Step 1: Убедиться, что вызовов нет**

Run: `grep -rn "merge_all_passages" --include=*.py src eval scripts tests mcp_server.py app.py pages`
Expected: только `src/v7/nlp_core.py`, `tests/v7/test_nlp_core.py`, комментарии в `src/v7/config.py:55` и `tests/v7/test_bridge.py:148`. Любой другой вызов — остановиться и сообщить.

- [ ] **Step 2: Удалить функцию и тесты**

- `src/v7/nlp_core.py`: удалить раздел `# ─── merge_all_passages ───` и функцию целиком.
- `tests/v7/test_nlp_core.py`: убрать `merge_all_passages` из импорта (строка 12) и удалить раздел `# ─── merge_all_passages ───` с классом `TestMergeAllPassages`.

- [ ] **Step 3: Докстринг `mmr_select` и комментарии**

`src/v7/nlp_core.py`, заголовок раздела и докстринг:
```python
# ─── MMR select ───────────────────────────────────────────────────────────


def mmr_select(
    passages: List[dict],
    top_k: int,
    lambda_param: float | None = None,
) -> List[dict]:
    """Maximal Marginal Relevance over already retrieved passages.

    Used by evaluate_complex to fill the merged candidate from the complex
    pool. Relevance is rerank_score, then vector_score, then score; the
    diversity penalty is based on doc_id.
    """
```
`src/v7/config.py:55`: `FINAL_MERGE_TOP_K: int = 24  # merged candidate size (evaluate_complex)`.
`tests/v7/test_bridge.py:148`: в докстринге `merge_all_passages` → `evaluate_complex merged candidate`.

- [ ] **Step 4: Тесты модулей**

Run: `PYTHONPATH=$PWD .venv/bin/python -m pytest tests/v7/test_nlp_core.py tests/v7/test_nodes/ tests/v7/test_bridge.py -q`
Expected: PASS (кроме известных красных из базы, если они в этих файлах).

- [ ] **Step 5: CI-гейт и сверка с базой**

```bash
PYTHONPATH=$PWD .venv/bin/python -m pytest -m "not integration and not slow" -q -p no:cacheprovider --ignore=tests/eval_ls 2>&1 | tail -15
```
Expected: набор `FAILED` совпадает с `/tmp/claude-1000/merge_gate_before.txt`; новых падений нет. Число passed изменится на (новые тесты − удалённые 5 `TestMergeAllPassages`).

- [ ] **Step 6: Commit**

```bash
black src/v7/nlp_core.py src/v7/config.py tests/v7/test_nlp_core.py tests/v7/test_bridge.py
ruff check --fix src/v7/nlp_core.py src/v7/config.py tests/v7/test_nlp_core.py tests/v7/test_bridge.py
git add src/v7/nlp_core.py src/v7/config.py tests/v7/test_nlp_core.py tests/v7/test_bridge.py
git commit -m "refactor(v7): drop merge_all_passages, merged candidate built in evaluate_complex"
```

---

## После плана (не входит в задачи)

- Push ветки `feat/v8-langsmith-eval` — по отдельному слову Петра.
- Переиндексация (платно) и замер 133 с подбором `FINAL_MERGE_TOP_K` — следующие шаги порядка Петра 12.10.
