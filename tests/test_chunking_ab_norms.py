"""Независимые от индекса проверки норм и фактического контекста."""

# ANCHOR: синтетические нормы → привязки, раскрытие и метрики; сети нет.

from eval.chunking_ab.norms import (
    Chunk,
    Norm,
    context_metrics,
    expand_parent,
    match_norm,
    select_context,
)


def norm(number, text, doc="doc"):
    return Norm(doc, "item", str(number), tuple(text.split("\n")), int(number))


def test_matching_is_scoped_to_document_and_uses_label_denominator():
    units = [norm(1, "альфа бета гамма дельта"), norm(2, "альфа бета", "other")]
    m = match_norm("doc", "альфа бета гамма неизвестное", units)
    assert m.norm == units[0] and m.score == 0.75 and m.reason is None


def test_matching_threshold_and_empty_input():
    n = norm(1, "альфа бета гамма")
    assert match_norm("doc", "альфа бета гамма дельта эпсилон", [n]).norm == n
    assert match_norm("doc", "альфа дельта эпсилон", [n]).reason == "below_threshold"
    assert match_norm("doc", "", [n]).reason == "empty_words"


def test_tie_prefers_smaller_but_is_disputed_and_no_parent():
    large = norm(1, "альфа бета гамма дельта")
    small = norm(2, "альфа бета гамма")
    m = match_norm("doc", "альфа бета гамма", [large, small])
    assert m.candidates[0].norm == small
    assert m.norm is None and m.reason == "ambiguous"


def test_gap_exactly_point_one_is_not_disputed():
    text = "альфа бета гамма дельта эпсилон дзета тета йота каппа лямбда"
    a = norm(1, text)
    b = norm(2, " ".join(text.split()[:-1]))
    assert match_norm("doc", text, [a, b]).norm == a


def test_whole_parent_deduplicates_and_budget_counts_expanded_text():
    n = norm(1, "вводная обязанности\nальфа бета\nгамма дельта")
    chunks = [
        Chunk("doc", "альфа бета", "альфа бета", n),
        Chunk("doc", "гамма дельта", "гамма дельта", n),
    ]
    ctx = select_context([0, 1], chunks, "B", 100)
    assert len(ctx) == 1 and ctx[0].text == n.text
    assert select_context([0], chunks, "B", len(n.text) - 1) == []
    assert len(select_context([0, 1], chunks, "A", 100)) == 2


def test_long_parent_window_has_whole_paragraphs_and_limited_sides():
    paras = tuple(f"абзац{i} " + (f"слово{i} " * 120) for i in range(12))
    n = Norm("doc", "article", "1", paras, 0)
    text = expand_parent(n, paras[6])
    assert paras[6] in text and text != n.text
    selected = text.split("\n")
    assert all(p in paras for p in selected)
    anchor = selected.index(paras[6])
    assert len("\n".join(selected[:anchor])) <= 3000
    assert len("\n".join(selected[anchor + 1 :])) <= 3000


def test_found_is_distinct_from_complete_and_union_uses_actual_text():
    n = norm(1, "вводная альфа бета гамма дельта эпсилон дзета тета йота каппа")
    c = [Chunk("doc", "альфа бета гамма", "альфа бета гамма", n)]
    a = select_context([0], c, "A", 1000)
    assert context_metrics(a, n, "A") == (True, False, 0.3)
    b = select_context([0], c, "B", 1000)
    assert context_metrics(b, n, "B") == (True, True, 1.0)
    other = [Chunk("other", n.text, n.text, None)]
    assert context_metrics(select_context([0], other, "A", 1000), n, "A") == (
        False,
        False,
        0.0,
    )


def test_complete_threshold_ninety_percent_and_duplicate_words_do_not_help():
    n = norm(1, "альфа бета гамма дельта эпсилон дзета тета йота каппа лямбда")
    cs = [
        Chunk("doc", " ".join(n.text.split()[:9]), " ".join(n.text.split()[:9]), None)
    ]
    assert context_metrics(select_context([0], cs, "A", 1000), n, "A") == (
        True,
        True,
        0.9,
    )
    cs = [Chunk("doc", "альфа " * 100, "альфа " * 100, None)]
    assert context_metrics(select_context([0], cs, "A", 1000), n, "A")[2] == 0.1


def test_parentless_falls_back_to_original_and_budget_stops_at_first_overflow():
    cs = [
        Chunk("doc", "альфа", "альфа", None),
        Chunk("doc", "длинный " * 10, "длинный " * 10, None),
        Chunk("doc", "бета", "бета", None),
    ]
    ctx = select_context([0, 1, 2], cs, "B", 20)
    assert len(ctx) == 1 and ctx[0].text == "альфа"


def test_cached_vectors_missing_stops_without_embedding(tmp_path):
    import hashlib
    import pickle

    import numpy as np
    import pytest

    from eval.chunking_ab.step1 import cached_vectors

    path = tmp_path / "embed_cache.pkl"
    path.write_bytes(
        pickle.dumps({hashlib.sha256(b"known").hexdigest(): np.array([1.0, 2.0])})
    )
    assert cached_vectors(["known"], path).shape == (1, 2)
    with pytest.raises(RuntimeError, match="missing.*1"):
        cached_vectors(["known", "missing"], path)


def test_long_parent_anchors_at_exact_chunk_across_paragraphs():
    # Ранний абзац имеет тот же словарь, но другую последовательность.
    before = ("дельта гамма бета альфа", "посторонний " * 600)
    anchor = ("альфа бета", "гамма дельта")
    after = ("хвостовой " * 600,)
    n = Norm("doc", "article", "1", before + anchor + after, 0)
    window = expand_parent(n, "альфа бета\nгамма дельта")
    assert "альфа бета\nгамма дельта" in window
    assert before[0] not in window and after[0] not in window


def test_complete_combines_chunks_but_does_not_assume_parent_text_was_sent():
    n = norm(1, "альфа бета гамма дельта эпсилон дзета тета йота каппа лямбда")
    halves = [" ".join(n.text.split()[:5]), " ".join(n.text.split()[5:])]
    cs = [Chunk("doc", t, t, None) for t in halves]
    assert context_metrics(select_context([0, 1], cs, "A", 1000), n, "A") == (
        False,
        True,
        1.0,
    )
    paras = tuple(
        " ".join(f"уникальное{i:03d}слово{j:03d}" for j in range(50)) for i in range(14)
    )
    long = Norm("doc", "article", "2", paras, 0)
    cs = [Chunk("doc", paras[6], paras[6], long)]
    ctx = select_context([0], cs, "B", 16000)
    found, complete, coverage = context_metrics(ctx, long, "B")
    assert found and not complete and 0 < coverage < 0.9
