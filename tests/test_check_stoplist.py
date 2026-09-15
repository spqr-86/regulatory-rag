"""Tests for the anonymization gate: corpus text must not match the private stop-list (#36)."""

from __future__ import annotations

import pytest

from scripts.check_stoplist import find_hits, load_patterns


@pytest.fixture
def stoplist(tmp_path):
    p = tmp_path / "stop.txt"
    p.write_text("# comment\n\nООО Ромашка\nИванов\\w*\n", encoding="utf-8")
    return p


@pytest.mark.unit
def test_comments_and_blank_lines_skipped(stoplist):
    assert len(load_patterns(stoplist)) == 2


@pytest.mark.unit
def test_hit_reports_file_line_and_pattern(tmp_path, stoplist):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("чисто\nподписал ИВАНОВА И.И.\n", encoding="utf-8")
    (corpus / "b.md").write_text("ничего лишнего\n", encoding="utf-8")

    hits = find_hits(corpus, load_patterns(stoplist))

    assert [(h.path.name, h.line_no, h.pattern) for h in hits] == [
        ("a.md", 2, "Иванов\\w*")
    ]


@pytest.mark.unit
def test_clean_corpus_has_no_hits(tmp_path, stoplist):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("ООО Пример, Филиал 1\n", encoding="utf-8")
    assert find_hits(corpus, load_patterns(stoplist)) == []
