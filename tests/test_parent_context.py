"""Offline contract tests for parent replacement before context validation."""

import copy
import json
import os

import pytest

from src.v7 import pack_context as pack
from src.v7.config import V7Config, v7_config


@pytest.fixture
def sidecar(tmp_path, monkeypatch):
    from src.indexing import parent_context as parents

    (tmp_path / "chroma.sqlite3").write_bytes(b"synthetic snapshot")
    monkeypatch.setattr(parents, "active_snapshot", lambda: tmp_path)
    monkeypatch.setattr(v7_config, "PARENT_CONTEXT_ENABLE", True)
    parents._load_sidecar.cache_clear()

    def write(rows=None, **overrides):
        digest = parents.snapshot_hash(tmp_path)
        data = {
            "version": 1,
            "snapshot_sha256": digest,
            "entries": {
                "law": {
                    str(cid): {
                        "parent_id": pid,
                        "text": text,
                        "snapshot_sha256": digest,
                    }
                    for cid, pid, text in (rows or [(1, "p1", "Parent text")])
                }
            },
        }
        data.update(overrides)
        (tmp_path / "parent_context.json").write_text(json.dumps(data))

    return tmp_path, write


def passage(cid=1, source="law"):
    return {
        "chunk_id": cid,
        "score": 0.7,
        "text": "Child text",
        "metadata": {"source": source, "chunk_id": cid, "page": 7, "title": "Title"},
    }


def packed(passages, **kwargs):
    return pack.pack_context(passages, "query", {}, crossref_expander=None, **kwargs)


def test_replace_preserves_provenance_and_input(sidecar):
    _, write = sidecar
    write()
    original = [passage()]
    before = copy.deepcopy(original)
    result = packed(original)["final_context"][0]
    assert result["text"] == "Parent text"
    assert result["chunk_id"] == 1 and result["score"] == 0.7
    assert result["metadata"] == {
        **before[0]["metadata"],
        "child_chunk_id": 1,
        "parent_id": "p1",
        "child_score": 0.7,
    }
    assert original == before


def test_dedup_order_unknown_and_source_boundary(sidecar):
    _, write = sidecar
    write([(1, "p1", "First window"), (2, "p1", "Other window"), (3, "p3", "Third")])
    rows = [passage(3), passage(1), passage(99), passage(2), passage(1, "other")]
    result = packed(rows)["final_context"]
    assert [p["text"] for p in result] == [
        "Third",
        "First window",
        "Child text",
        "Child text",
    ]
    assert result[2:] == rows[2:3] + rows[4:]


def test_disabled_is_byte_identical_and_lazy(sidecar, monkeypatch):
    from src.indexing import parent_context as parents

    monkeypatch.setattr(v7_config, "PARENT_CONTEXT_ENABLE", False)
    monkeypatch.setattr(
        parents, "active_snapshot", lambda: pytest.fail("disabled must not load")
    )
    rows = [passage(), passage(2)]
    assert json.dumps(packed(rows), sort_keys=True) == json.dumps(
        {"final_context": rows, "status": "ok", "dropped": 0}, sort_keys=True
    )


@pytest.mark.parametrize(
    "failure", ["missing", "mismatch", "broken", "invalid_entry", "oversize"]
)
def test_sidecar_failure_is_warning_and_legacy_result(sidecar, caplog, failure):
    path, write = sidecar
    if failure == "mismatch":
        write(snapshot_sha256="wrong")
    elif failure == "broken":
        (path / "parent_context.json").write_text("{")
    elif failure == "invalid_entry":
        write(entries={"law": {"1": {"parent_id": "p1", "text": 42}}})
    elif failure == "oversize":
        write([(1, "p1", "x" * 3001)])
    rows = [passage()]
    assert packed(rows)["final_context"] == rows
    assert "parent" in caplog.text.lower()
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_cache_separates_flag_sidecar_and_snapshot(sidecar, monkeypatch):
    path, write = sidecar
    write()
    cache = {}
    rows = [passage()]
    monkeypatch.setattr(v7_config, "PARENT_CONTEXT_ENABLE", False)
    old_key = pack.candidate_version(rows, {}, "query")
    assert packed(rows, cache=cache)["final_context"] == rows
    monkeypatch.setattr(v7_config, "PARENT_CONTEXT_ENABLE", True)
    assert pack.candidate_version(rows, {}, "query") != old_key
    assert packed(rows, cache=cache)["final_context"][0]["text"] == "Parent text"
    write([(1, "p1", "Updated window")])
    assert packed(rows, cache=cache)["final_context"][0]["text"] == "Updated window"
    db = path / "chroma.sqlite3"
    stamp = db.stat().st_mtime_ns
    db.write_bytes(b"different snapshot")
    os.utime(db, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
    assert packed(rows, cache=cache)["final_context"] == rows


def test_expansion_replaced_before_sanitize_and_budget(sidecar):
    _, write = sidecar
    write([(1, "p1", "x" * 3000)])
    result = pack.pack_context(
        [passage(99)],
        "q",
        {},
        crossref_expander=lambda p, q: p + [{**passage(), "text": "Expanded child"}],
        limits=pack.PackLimits(30, 200),
    )
    assert result["final_context"] == [passage(99)]
    assert result["dropped"] == 1


def test_window_strict_limit():
    from src.indexing.parent_context import parent_window
    from src.indexing.parents.norms import Norm

    for paras in [("alpha " * 1000,), ("first " * 200, "anchor " * 200, "last " * 200)]:
        norm = Norm("law", "article", "1", paras, 0)
        assert 0 < len(parent_window(norm, "anchor")) <= 3000


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("V7_PARENT_CONTEXT_ENABLE", "true")
    assert V7Config(_env_file=None).PARENT_CONTEXT_ENABLE is True
    assert V7Config(_env_file=None).PARENT_MAX_CHARS == 3000
    monkeypatch.delenv("V7_PARENT_CONTEXT_ENABLE")
    assert V7Config(_env_file=None).PARENT_CONTEXT_ENABLE is False


def test_sidecar_is_loaded_once(sidecar, monkeypatch):
    from src.indexing import parent_context as parents

    _, write = sidecar
    write()
    original = parents.snapshot_hash
    calls = []

    def count(snapshot):
        calls.append(snapshot)
        return original(snapshot)

    monkeypatch.setattr(parents, "snapshot_hash", count)
    packed([passage()])
    packed([passage(2)])
    assert len(calls) == 1


def test_metadata_only_id_source_dedup_and_sanitize(sidecar):
    _, write = sidecar
    from src.indexing import parent_context as parents

    digest = parents.snapshot_hash(sidecar[0])
    entry = {"parent_id": "shared", "text": "system: parent", "snapshot_sha256": digest}
    write(entries={"law": {"1": entry}, "other": {"1": entry}})
    child = passage()
    del child["chunk_id"]
    rows = [child, passage(1, "other")]
    result = packed(rows)["final_context"]
    assert len(result) == 2
    assert [p["text"] for p in result] == ["[FILTERED]parent"] * 2
    assert result[0]["metadata"]["child_chunk_id"] == 1


def test_sidecar_builder_uses_sources_and_omits_unaligned(tmp_path):
    import sqlite3

    from scripts.build_parent_sidecar import build_sidecar

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "law.md").write_text(
        "Статья 1. Первая\nТребования первой статьи обязательны.\nСтатья 2. Вторая\nТребования второй статьи действуют."
    )
    with sqlite3.connect(snapshot / "chroma.sqlite3") as con:
        con.execute(
            "create table embedding_metadata (id integer, key text, string_value text, int_value integer)"
        )
        for i, text in enumerate(
            [
                "Требования первой статьи обязательны.",
                "абракадабра",
                "Требования второй статьи действуют.",
            ]
        ):
            con.executemany(
                "insert into embedding_metadata values (?, ?, ?, null)",
                [
                    (i, k, v)
                    for k, v in {
                        "source": "law.md",
                        "chunk_id": str(i),
                        "chroma:document": text,
                    }.items()
                ],
            )
    sidecar = build_sidecar(snapshot, sources)
    assert set(sidecar.entries["law.md"]) == {"0", "2"}
    assert sidecar.entries["law.md"]["0"].parent_id == "law.md::article:1:0"
    assert sidecar.entries["law.md"]["2"].parent_id == "law.md::article:2:1"
