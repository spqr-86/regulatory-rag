"""Шаг 2: синтетические проверки выравнивания, контекста, причин и dry-run K."""

# ANCHOR: локальные примеры без индекса/сети → инварианты шага 2.

import pytest

from eval.chunking_ab.align import (
    AlignInput,
    align_document,
    build_children,
    parent_window,
)
from eval.chunking_ab.norms import Chunk, Norm, context_metrics
from eval.chunking_ab.step2 import loss_reason, select_context


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import socket

    def denied(*args, **kwargs):
        raise AssertionError("test attempted network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def norm(i, *paras, doc="doc"):
    return Norm(doc, "item", str(i), tuple(paras), i)


def test_sequence_resolves_identical_words_by_neighbors_and_ignores_bad_metadata():
    ns = [
        norm(1, "начало первое"),
        norm(2, "общие слова"),
        norm(3, "конец третий"),
        norm(4, "общие слова"),
    ]
    cs = [
        AlignInput("начало первое"),
        AlignInput("общие слова", article="4"),
        AlignInput("конец третий"),
    ]
    result = align_document(cs, ns, rollback=0)
    assert [r.parent for r in result] == ns[:3]


def test_rollback_is_limited_to_one_unit_and_penalized():
    ns = [norm(1, "альфа первая"), norm(2, "бета вторая"), norm(3, "гамма третья")]
    cs = [AlignInput(n.text) for n in [ns[1], ns[0], ns[2]]]
    result = align_document(cs, ns, rollback=1)
    assert [r.parent for r in result] == [ns[1], ns[0], ns[2]]
    assert result[1].rollback
    distant = align_document([AlignInput(ns[2].text), AlignInput(ns[0].text)], ns)
    assert not (distant[0].parent == ns[2] and distant[1].parent == ns[0])


def test_low_evidence_is_rejected_not_filled_by_position():
    ns = [norm(1, "альфа первая"), norm(2, "бета вторая")]
    result = align_document([AlignInput("чужие слова"), AlignInput("")], ns)
    assert [r.reason for r in result] == ["below_threshold", "empty_words"]
    assert all(r.parent is None for r in result)
    assert align_document([], ns) == []
    assert align_document([AlignInput("слово")], [])[0].reason == "no_units"


def test_window_is_strict_and_anchored_even_with_one_oversize_paragraph():
    ns = norm(1, "раннее " * 300, "якорное предложение. " * 220, "позднее " * 300)
    window = parent_window(ns, "якорное предложение.", limit=3000)
    assert len(window) <= 3000 and "якорное предложение." in window
    assert "раннее" not in window and "позднее" not in window
    n = norm(2, "первый абзац", "второй якорь", "третий абзац")
    assert parent_window(n, "второй якорь", 28) in (
        "первый абзац\nвторой якорь",
        "второй якорь\nтретий абзац",
    )


def test_bdiag_expands_whole_parents_b2_is_capped_and_deduplicates():
    n = norm(1, "вводная " * 900, "якорь " * 50)
    cs = [Chunk("doc", "якорь", "якорь", n)] * 2
    assert select_context([0, 1], cs, "B1", 3000) == []
    diag = select_context([0, 1], cs, "Bdiag", None)
    assert len(diag) == 1 and diag[0].body == n.text
    assert context_metrics(diag, n, "B")[:2] == (True, True)
    b2 = select_context([0, 1], cs, "B2", 3000)
    assert len(b2) == 1 and len(b2[0].text) <= 3000


def test_loss_diagnosis_is_exhaustive_and_missing_parent_requires_evidence():
    n = norm(1, "альфа бета гамма")
    cs = [Chunk("doc", n.text, n.text, None)]
    assert loss_reason(True, False, True, n, [0], cs) == "budget"
    assert loss_reason(True, False, False, n, [0], cs) == "missing_parent"
    mapped = [Chunk("doc", n.text, n.text, norm(2, "иная норма"))]
    assert loss_reason(True, False, False, n, [0], mapped) == "other"
    assert loss_reason(False, False, False, n, [0], cs) is None
    assert loss_reason(True, True, True, n, [0], cs) is None


def test_children_have_explicit_parent_prefix_intro_and_strict_total_length():
    ps = [
        "I. Раздел",
        "1. Работодатель обеспечивает:",
        "а) " + "Первое требование. " * 50,
        "б) Второе требование.",
        "2. " + "Длинноеслово " * 100,
    ]
    children = build_children("doc", "Документ", ps)
    assert children and all(len(c.text) <= 450 for c in children)
    assert all(c.parent_id and c.text.startswith("Документ ›") for c in children)
    sub = [c for c in children if "а)" in c.body or "б)" in c.body]
    assert sub and all("Работодатель обеспечивает:" in c.text for c in sub)
    assert len({c.id for c in children}) == len(children)
    assert all(c.parent_id.startswith("doc::") for c in children)
    assert "Второе требование." in " ".join(c.text for c in children)


def test_k_dry_run_counts_only_unique_cache_misses(tmp_path):
    import hashlib
    import pickle

    import pytest
    import tiktoken

    from eval.chunking_ab.step2 import k_stats

    try:
        tiktoken.get_encoding("cl100k_base")
    except Exception:  # tokenizer is downloaded on first use; CI has no network
        pytest.skip("cl100k_base encoding unavailable offline")

    p = tmp_path / "cache.pkl"
    p.write_bytes(pickle.dumps({hashlib.sha256(b"known").hexdigest(): [1]}))
    stats = k_stats(["known", "missing", "missing"], p)
    assert stats["unique_texts"] == 2 and stats["missing_unique_texts"] == 1
    assert stats["estimated_tokens"] > 0 and stats["estimated_usd"] > 0


def test_metadata_can_help_but_does_not_override_text():
    ns = [
        Norm("doc", "article", str(i), (text,), i)
        for i, text in [(1, "общие слова"), (2, "общие слова"), (3, "другой текст")]
    ]
    assert (
        align_document([AlignInput("общие слова", article="2")], ns)[0].parent == ns[1]
    )
    assert (
        align_document([AlignInput("другой текст", article="2")], ns)[0].parent == ns[2]
    )


def test_k_long_intro_and_unbroken_tokens_stay_bounded_and_keep_original_content():
    intro = "1. Работодатель обеспечивает " + "дополнительное условие " * 40 + ":"
    children = build_children("doc", "Название документа", [intro, "а) " + "x" * 700])
    assert all(len(c.text) <= 450 for c in children)
    assert any(c.intro_shortened for c in children)
    assert any(c.word_fallback for c in children)
    assert any("…" in c.text for c in children)


def test_missing_parent_diagnosis_accepts_original_parent_id_as_witness():
    n = norm(1, "альфа бета гамма дельта эпсилон")
    old = [Chunk("doc", "альфа", "альфа", n)]
    new = [Chunk("doc", "альфа", "альфа", None)]
    assert loss_reason(True, False, False, n, [0], new, old) == "missing_parent"


def test_b2_anchors_exact_sequence_across_paragraphs_before_word_overlap():
    n = norm(
        1,
        "дельта гамма бета альфа",
        "постороннее " * 500,
        "альфа бета",
        "гамма дельта",
        "хвостовое " * 500,
    )
    text = parent_window(n, "альфа бета\nгамма дельта", 3000)
    assert "альфа бета\nгамма дельта" in text
    assert "дельта гамма бета альфа" not in text
