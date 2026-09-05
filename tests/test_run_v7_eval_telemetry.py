"""An eval run lands in the same journal as live traffic (issue #18).

The run separates itself from other runs by its own id: ``source`` says the
rows came from eval, ``run_id`` says from which run. Nothing here touches a
real graph or a real writer.
"""

import re

from eval.run_v7_eval import new_run_id, run_query


class _FakeGraph:
    def invoke(self, _inputs):
        return {
            "answer": "ответ",
            "final_passages": [{"text": "t"}],
            "retrieval_attempts": [{"stage": "simple"}],
            "llm_usage": [],
        }


class _FakeWriter:
    def __init__(self):
        self.events = []

    def write(self, event):
        self.events.append(event)


class TestNewRunId:
    def test_is_sortable_by_time_and_marked_as_eval(self):
        assert re.fullmatch(r"eval-\d{8}T\d{6}Z-[0-9a-f]{4}", new_run_id())

    def test_two_runs_in_the_same_second_differ(self):
        assert new_run_id() != new_run_id()


class TestRunQueryTelemetry:
    def test_row_carries_source_eval_and_the_run_id(self):
        writer = _FakeWriter()

        run_query(_FakeGraph(), "вопрос", writer=writer, run_id="eval-x-0001")

        assert writer.events[0]["source"] == "eval"
        assert writer.events[0]["run_id"] == "eval-x-0001"

    def test_all_queries_of_one_run_share_the_id(self):
        writer = _FakeWriter()
        run_id = new_run_id()

        for question in ("первый", "второй", "третий"):
            run_query(_FakeGraph(), question, writer=writer, run_id=run_id)

        assert {e["run_id"] for e in writer.events} == {run_id}
        assert len({e["query_id"] for e in writer.events}) == 3
