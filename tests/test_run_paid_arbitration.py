"""Тесты раннера платного арбитража: выбор вопросов и сборка вердиктов.

Вызовов модели здесь нет — арбитр подменяется фейком, проверяется отбор
вопросов, формат выхода и то, что цитата вне текста чанка не проходит.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_paid_arbitration import (
    arbitrate_tasks,
    count_unverified,
    estimate_arbitration_cost,
    select_rejected,
    write_verdicts,
)


def _pool(n: int, texts: list[str]) -> dict:
    return {
        "n": n,
        "question": f"вопрос {n}",
        "status": "disputed",
        "open_indices": list(range(1, len(texts) + 1)),
        "candidates": [
            {"index": i, "chunk_id": f"doc#{i}", "source": "doc", "text": t}
            for i, t in enumerate(texts, start=1)
        ],
    }


def test_select_rejected_takes_only_empty_arbiter_with_a_cited_pass():
    arbiter = {1: {"relevant": []}, 2: {"relevant": [{"index": 1}]}, 3: {"relevant": []}}
    cited = {1: [{"index": 1, "quote": "а"}], 2: [{"index": 1, "quote": "б"}]}
    assert select_rejected(arbiter, cited) == [1]


def test_select_rejected_is_sorted_and_deduplicated():
    arbiter = {5: {"relevant": []}, 4: {"relevant": []}}
    cited = {4: [{"index": 1}], 5: [{"index": 2}]}
    assert select_rejected(arbiter, cited) == [4, 5]


def test_arbitrate_tasks_keeps_raw_verdict_and_counts_unverified_quotes():
    tasks = [_pool(7, ["норма про стажировку", "чужой текст"])]

    def fake_arbiter(question, candidates, open_indices, llm):
        labels = {
            "relevant": [
                {"index": 1, "quote": "норма про стажировку"},
                {"index": 2, "quote": "выдуманная фраза"},
            ],
            "note": "",
        }
        return labels, {"input": 10, "output": 5}

    verdicts, usages = arbitrate_tasks(tasks, llm=None, arbiter=fake_arbiter)

    # Вердикт пишется сырым — ту же проверку цитаты делает ingest_agent_labels,
    # и решает её человек, а не второй прогон той же модели.
    assert [v["n"] for v in verdicts] == [7]
    assert [r["index"] for r in verdicts[0]["relevant"]] == [1, 2]
    assert count_unverified(verdicts, {7: tasks[0]["candidates"]}) == 1
    assert usages == [{"input": 10, "output": 5}]


def test_arbitrate_tasks_returns_one_record_per_task_even_when_nothing_found():
    tasks = [_pool(1, ["текст"]), _pool(2, ["текст"])]

    def fake_arbiter(question, candidates, open_indices, llm):
        return {"relevant": [], "note": "нет ответа"}, {"input": 1, "output": 1}

    verdicts, _ = arbitrate_tasks(tasks, llm=None, arbiter=fake_arbiter)
    assert [v["n"] for v in verdicts] == [1, 2]
    assert all(v["relevant"] == [] for v in verdicts)


def test_write_verdicts_writes_arbiter_pass_file(tmp_path: Path):
    out = tmp_path / "verdicts"
    write_verdicts([{"n": 1, "relevant": [], "note": ""}], out, name="paid")
    data = json.loads((out / "paid.arbiter.json").read_text(encoding="utf-8"))
    assert data["pass"] == "arbiter"
    assert data["verdicts"][0]["n"] == 1


def test_estimate_cost_scales_with_questions_and_pool_size():
    one = estimate_arbitration_cost(1, model="gpt-5.6-sol", n_candidates=26)
    ten = estimate_arbitration_cost(10, model="gpt-5.6-sol", n_candidates=26)
    assert one > 0
    assert ten == pytest.approx(one * 10)
