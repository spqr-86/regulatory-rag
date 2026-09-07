"""Unit tests for eval/ingest_agent_labels.py.

The agent-judged path reuses the paid script's merge logic, so what is tested
here is only what is new: pairing two pass files against the pools, refusing a
verdict that names a question or a candidate the pool does not have, and
failing loudly on an unjudged question instead of silently labelling it
none_found — a question that quietly loses its ground truth drops out of the
metric and makes the set read better than it is.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from eval.ingest_agent_labels import (
    build_verdicts,
    load_passes,
    load_pools,
)


def _pool(n=1):
    return {
        "n": n,
        "question": "Как часто проводится повторный инструктаж?",
        "candidates": [
            {"index": 1, "chunk_id": "2464#1", "source": "2464", "text": "не реже одного раза в шесть месяцев"},
            {"index": 2, "chunk_id": "2464#2", "source": "2464", "text": "вводный инструктаж проводится при приеме"},
        ],
    }


def _pass(name, relevant, n=1):
    return {"pass": name, "verdicts": [{"n": n, "relevant": relevant, "note": ""}]}


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_pools_indexes_by_n(tmp_path):
    _write(tmp_path, "shard_01.json", [_pool(1), _pool(2)])
    pools = load_pools(tmp_path)
    assert sorted(pools) == [1, 2]
    assert pools[1]["candidates"][0]["chunk_id"] == "2464#1"


def test_load_passes_merges_shards(tmp_path):
    _write(tmp_path, "shard_01.strict.json", _pass("strict", [], n=1))
    _write(tmp_path, "shard_02.strict.json", _pass("strict", [], n=2))
    verdicts = load_passes(tmp_path, "strict")
    assert sorted(verdicts) == [1, 2]


def test_agreement_becomes_gold(tmp_path):
    pools = {1: _pool()}
    quote = "не реже одного раза в шесть месяцев"
    strict = {1: {"relevant": [{"index": 1, "quote": quote}], "note": ""}}
    lenient = {1: {"relevant": [{"index": 1, "quote": quote}], "note": ""}}
    verdicts = build_verdicts(pools, strict, lenient)
    assert verdicts[1]["status"] == "agreed"
    assert verdicts[1]["gold_chunk_ids"] == ["2464#1"]


def test_quote_absent_from_chunk_is_not_silently_gold(tmp_path):
    pools = {1: _pool()}
    made_up = [{"index": 1, "quote": "один раз в три месяца"}]
    verdicts = build_verdicts(pools, {1: {"relevant": made_up}}, {1: {"relevant": made_up}})
    assert verdicts[1]["status"] == "unverified_quote"


def test_disagreement_is_disputed(tmp_path):
    pools = {1: _pool()}
    quote = "не реже одного раза в шесть месяцев"
    strict = {1: {"relevant": [{"index": 1, "quote": quote}]}}
    lenient = {
        1: {
            "relevant": [
                {"index": 1, "quote": quote},
                {"index": 2, "quote": "вводный инструктаж проводится при приеме"},
            ]
        }
    }
    verdicts = build_verdicts(pools, strict, lenient)
    assert verdicts[1]["status"] == "disputed"
    assert verdicts[1]["disputed_indices"] == [2]


def test_missing_question_in_a_pass_is_fatal(tmp_path):
    pools = {1: _pool(1), 2: _pool(2)}
    strict = {1: {"relevant": []}, 2: {"relevant": []}}
    lenient = {1: {"relevant": []}}
    with pytest.raises(ValueError, match="lenient"):
        build_verdicts(pools, strict, lenient)


def test_verdict_for_unknown_question_is_fatal(tmp_path):
    pools = {1: _pool()}
    extra = {1: {"relevant": []}, 99: {"relevant": []}}
    with pytest.raises(ValueError, match="99"):
        build_verdicts(pools, extra, {1: {"relevant": []}})


def test_index_outside_pool_is_dropped(tmp_path):
    pools = {1: _pool()}
    bad = {1: {"relevant": [{"index": 7, "quote": "что-то"}]}}
    verdicts = build_verdicts(pools, bad, bad)
    assert verdicts[1]["status"] == "none_found"


# ─── арбитраж ────────────────────────────────────────────────────────────────


def test_arbitration_tasks_carry_only_open_indices(tmp_path):
    from eval.dump_arbitration_tasks import build_tasks

    pools = {1: _pool(1), 2: _pool(2)}
    verdicts = {
        1: {"status": "disputed", "disputed_indices": [2], "gold_chunk_ids": ["2464#1"]},
        2: {"status": "agreed", "disputed_indices": [], "gold_chunk_ids": ["2464#1"]},
    }
    tasks = build_tasks(pools, verdicts)
    assert [t["n"] for t in tasks] == [1]
    assert tasks[0]["open_indices"] == [2]


def test_none_found_sends_the_whole_pool_to_the_arbiter(tmp_path):
    from eval.dump_arbitration_tasks import build_tasks

    pools = {1: _pool(1)}
    verdicts = {1: {"status": "none_found", "disputed_indices": [], "gold_chunk_ids": []}}
    tasks = build_tasks(pools, verdicts)
    assert tasks[0]["open_indices"] == [1, 2]


def test_arbitration_from_directory_is_applied(tmp_path):
    from eval.ingest_agent_labels import apply_arbitrations, load_passes

    pools = {1: _pool()}
    quote = "не реже одного раза в шесть месяцев"
    strict = {1: {"relevant": [{"index": 1, "quote": quote}]}}
    lenient = {
        1: {
            "relevant": [
                {"index": 1, "quote": quote},
                {"index": 2, "quote": "вводный инструктаж проводится при приеме"},
            ]
        }
    }
    verdicts = build_verdicts(pools, strict, lenient)
    assert verdicts[1]["status"] == "disputed"

    _write(
        tmp_path,
        "shard_01.arbiter.json",
        _pass("arbiter", [{"index": 2, "quote": "вводный инструктаж проводится при приеме"}]),
    )
    arb = load_passes(tmp_path, "arbiter")
    out = apply_arbitrations(pools, verdicts, arb)
    assert out[1]["status"] == "arbitrated"
    assert out[1]["gold_chunk_ids"] == ["2464#1", "2464#2"]
