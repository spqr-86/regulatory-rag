"""Unit tests for eval/ls/dataset.py — no network."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eval.ls.dataset import build_examples, upsert_examples

pytestmark = pytest.mark.unit

FIELDS = [
    "case_id",
    "question",
    "ground_truth",
    "must_not_contain",
    "oos_type",
    "reference_status",
    "reviewed_at",
    "review_note",
    "forbidden_claims",
    "corpus_support",
    "legal_as_of",
]


def _row(case_id, question, *, status="verified", oos_type="", forbidden=""):
    return {
        "case_id": case_id,
        "question": question,
        "ground_truth": f"gt {case_id}",
        "must_not_contain": "",
        "oos_type": oos_type,
        "reference_status": status,
        "reviewed_at": "2026-09-28",
        "review_note": "",
        "forbidden_claims": forbidden,
        "corpus_support": "unverified",
        "legal_as_of": "2026-09-30",
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
        {
            "question": "Вопрос один?",
            "chunk_id": "2464.pdf#107",
            "relevant_chunk_ids": ["2464.pdf#107", "782н.pdf#431"],
        },
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
    # The judge checks the answer against the law as of this date.
    assert e["outputs"]["legal_as_of"] == "2026-09-30"
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
    assert [e["id"] for e in build_examples(*files)] == [
        e["id"] for e in build_examples(*files)
    ]


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

    def delete_examples(self, example_ids, **kw):
        for i in example_ids:
            del self.store[str(i)]


def test_upsert_is_idempotent(files):
    client = FakeClient()
    ex = build_examples(*files)
    assert upsert_examples(client, ex) == {
        "created": 3,
        "updated": 0,
        "unchanged": 0,
        "stale": 0,
    }
    assert upsert_examples(client, ex) == {
        "created": 0,
        "updated": 0,
        "unchanged": 3,
        "stale": 0,
    }
    assert len(client.store) == 3


def test_upsert_updates_changed(files):
    client = FakeClient()
    ex = build_examples(*files)
    upsert_examples(client, ex)
    ex[0]["outputs"]["ground_truth"] = "новый эталон"
    assert upsert_examples(client, ex) == {
        "created": 0,
        "updated": 1,
        "unchanged": 2,
        "stale": 0,
    }


def test_stale_examples_reported_not_deleted_by_default(files):
    client = FakeClient()
    ex = build_examples(*files)
    upsert_examples(client, ex)
    result = upsert_examples(client, ex[1:])
    assert result == {"created": 0, "updated": 0, "unchanged": 2, "stale": 1}
    assert ex[0]["id"] in client.store


def test_prune_deletes_stale_examples(files):
    client = FakeClient()
    ex = build_examples(*files)
    upsert_examples(client, ex)
    result = upsert_examples(client, ex[1:], prune=True)
    assert result == {
        "created": 0,
        "updated": 0,
        "unchanged": 2,
        "stale": 1,
        "deleted": 1,
    }
    assert ex[0]["id"] not in client.store


def test_prune_refuses_empty_examples(files):
    """An empty build (e.g. no verified rows left) must not wipe the dataset."""
    client = FakeClient()
    upsert_examples(client, build_examples(*files))
    with pytest.raises(ValueError, match="prune"):
        upsert_examples(client, [], prune=True)
    assert len(client.store) == 3


def test_prune_keeps_foreign_examples(files):
    """Examples not built from the CSV (e.g. added from traces) have no csv_sha256."""
    client = FakeClient()
    ex = build_examples(*files)
    upsert_examples(client, ex)
    foreign = {
        "id": "foreign-1",
        "inputs": {"question": "q"},
        "outputs": {},
        "metadata": {"source": "traces"},
    }
    client.store["foreign-1"] = foreign
    result = upsert_examples(client, ex, prune=True)
    assert result == {
        "created": 0,
        "updated": 0,
        "unchanged": 3,
        "stale": 0,
        "deleted": 0,
    }
    assert "foreign-1" in client.store
