from src.v7.pack_context import candidate_version


def _p(chunk_id, source="doc.pdf", text="t"):
    return {"chunk_id": chunk_id, "text": text, "metadata": {"source": source}}


PLAN = {"threshold": 0.5}


def test_version_is_order_independent():
    assert candidate_version([_p(1), _p(2)], PLAN, "q") == candidate_version(
        [_p(2), _p(1)], PLAN, "q"
    )


def test_version_changes_with_passage_set():
    assert candidate_version([_p(1)], PLAN, "q") != candidate_version(
        [_p(1), _p(2)], PLAN, "q"
    )


def test_version_changes_with_text_at_same_id():
    """Enrichment меняет текст, не трогая chunk_id — версия обязана измениться."""
    assert candidate_version([_p(1, text="a")], PLAN, "q") != candidate_version(
        [_p(1, text="a\n\n[Таблица]: b")], PLAN, "q"
    )


def test_version_changes_with_plan():
    assert candidate_version([_p(1)], PLAN, "q") != candidate_version(
        [_p(1)], {"threshold": 0.35}, "q"
    )


def test_version_changes_with_query():
    assert candidate_version([_p(1)], PLAN, "q1") != candidate_version(
        [_p(1)], PLAN, "q2"
    )
