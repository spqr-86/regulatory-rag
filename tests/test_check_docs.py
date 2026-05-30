from __future__ import annotations

import pytest

from scripts.check_docs import (
    STALE_DENYLIST,
    check_prompt_versions,
    find_stale_terms,
)

pytestmark = pytest.mark.unit


def test_find_stale_terms_flags_denylisted_word(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "bad.md").write_text("The llm_verifier decides what to do.", encoding="utf-8")
    hits = find_stale_terms(d)
    assert any("llm_verifier" in h.term for h in hits)
    assert any(h.path.name == "bad.md" for h in hits)


def test_find_stale_terms_ignores_archive(tmp_path):
    d = tmp_path / "docs"
    (d / "archive").mkdir(parents=True)
    (d / "archive" / "old.md").write_text("old llm_verifier notes", encoding="utf-8")
    assert find_stale_terms(d) == []


def test_find_stale_terms_ignores_marked_line(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "x.md").write_text(
        "the llm_verifier was removed <!--freshness:ignore-->", encoding="utf-8"
    )
    assert find_stale_terms(d) == []


def test_find_stale_terms_ignores_excluded_file(tmp_path):
    d = tmp_path / "docs"
    (d / "explanation").mkdir(parents=True)
    (d / "explanation" / "design-decisions.md").write_text(
        "we removed the llm_verifier and rewriter", encoding="utf-8"
    )
    assert find_stale_terms(d) == []


def test_find_stale_terms_clean_doc(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "ok.md").write_text(
        "domain_gate filters out-of-scope queries.", encoding="utf-8"
    )
    assert find_stale_terms(d) == []


def test_check_prompt_versions_matches(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        'generate_answer:\n  active_version: "v8"\n  versions:\n    v8: x.j2\n',
        encoding="utf-8",
    )
    facts = tmp_path / "FACTS.md"
    facts.write_text("## prompts\n- generate_answer: v8\n", encoding="utf-8")
    assert check_prompt_versions(registry, facts) == []


def test_check_prompt_versions_detects_drift(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        'generate_answer:\n  active_version: "v8"\n  versions:\n    v8: x.j2\n',
        encoding="utf-8",
    )
    facts = tmp_path / "FACTS.md"
    facts.write_text("## prompts\n- generate_answer: v7\n", encoding="utf-8")
    mismatches = check_prompt_versions(registry, facts)
    assert any(m.name == "generate_answer" for m in mismatches)


def test_denylist_nonempty():
    assert "llm_verifier" in STALE_DENYLIST
