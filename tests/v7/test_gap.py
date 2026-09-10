from src.v7.gap import build_gap, has_enumeration_intent


def _p(text, source="doc.pdf"):
    return {"text": text, "metadata": {"source": source}}


def test_open_ref_is_reported():
    gap = build_gap([_p("проводится согласно пункт 15 порядка")])
    assert gap["open"] == ["clause:15"]


def test_structural_heading_closes_ref():
    gap = build_gap(
        [_p("согласно пункт 15 порядка"), _p("15. Медосмотр проводится ежегодно.")]
    )
    assert gap["open"] == []
    assert gap["closed"] == ["clause:15"]


def test_enumeration_detector():
    assert has_enumeration_intent("кто проходит медосмотр") is True
    assert has_enumeration_intent("что такое медосмотр") is False


def test_gap_module_does_not_import_nodes():
    import src.v7.gap as g

    src = open(g.__file__, encoding="utf-8").read()
    assert "src.v7.nodes" not in src
