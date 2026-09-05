"""Tests for the structured triage gap (issue #13, stage B2).

Spec: docs/spec-triage-gap.html
"""

from __future__ import annotations

import pytest

from src.v7.nodes.evaluate_triage import (
    _legacy_triage,
    build_gap,
    evaluate_triage,
    set_crossref_expander,
)


def _p(text, source="SP486", score=0.8):
    return {
        "text": text,
        "score": score,
        "vector_score": score,
        "doc_id": source,
        "metadata": {"source": source},
    }


def _make_attempt(passages, plan=None):
    return {
        "retrieval_id": "rid1",
        "stage": "simple",
        "passages": passages,
        "top_score": max((p.get("score", 0) for p in passages), default=0),
        "attempt_plan": plan
        or {
            "threshold": 0.65,
            "min_passages": 2,
            "min_keyword_overlap": 0.3,
            "max_single_doc_ratio": 0.6,
            "borderline_threshold": 0.40,
            "require_multi_doc": False,
        },
    }


@pytest.fixture(autouse=True)
def _reset_expander():
    """Every test starts with no expander injected."""
    set_crossref_expander(None)
    yield
    set_crossref_expander(None)


class TestBuildGap:
    @pytest.mark.unit
    def test_gap_schema(self):
        """Gap is structured data: kind, refs, closed, open."""
        passages = [_p("в соответствии с пунктом 12 настоящих правил")]
        gap = build_gap(passages)
        assert gap["kind"] == "unresolved_ref"
        assert gap["refs"] == [{"kind": "clause", "num": "12", "doc_id": "SP486"}]
        assert gap["closed"] == []
        assert gap["open"] == ["clause:12"]

    @pytest.mark.unit
    def test_present_reference_is_closed(self):
        """A clause that is present anywhere in the same source is not a gap."""
        passages = [
            _p("в соответствии с пунктом 12 настоящих правил"),
            _p("прочий текст", score=0.7),
            _p("прочий текст 2", score=0.7),
            _p("прочий текст 3", score=0.7),
            _p("прочий текст 4", score=0.7),
            _p("12. Ограждение устанавливается по краю перепада.", score=0.4),
        ]
        gap = build_gap(passages)
        assert gap["open"] == []
        assert gap["closed"] == ["clause:12"]

    @pytest.mark.unit
    def test_reference_present_in_other_source_stays_open(self):
        """Resolution is per document: another source does not close the ref."""
        passages = [
            _p("в соответствии с пунктом 12 настоящих правил", source="SP486"),
            _p("12. Иной документ, иной пункт.", source="OTHER", score=0.4),
        ]
        gap = build_gap(passages)
        assert gap["open"] == ["clause:12"]
        assert gap["closed"] == []

    @pytest.mark.unit
    def test_self_reference_is_closed(self):
        """'пункт 12' inside the chunk that IS clause 12 needs no expansion."""
        passages = [_p("12. Настоящий пункт 12 применяется к работам на высоте.")]
        gap = build_gap(passages)
        assert gap["closed"] == ["clause:12"]
        assert gap["open"] == []

    @pytest.mark.unit
    def test_refs_deduplicated_by_doc_kind_num(self):
        """The same clause named in three chunks of one document is one ref."""
        passages = [
            _p("в соответствии с пунктом 12"),
            _p("см. пункт 12 настоящих правил", score=0.7),
            _p("согласно пункту 12", score=0.6),
        ]
        gap = build_gap(passages)
        assert gap["refs"] == [{"kind": "clause", "num": "12", "doc_id": "SP486"}]
        assert gap["open"] == ["clause:12"]

    @pytest.mark.unit
    def test_same_num_in_two_docs_gives_two_refs_one_marker(self):
        """Marker carries no doc_id: two refs collapse into one open marker."""
        passages = [
            _p("в соответствии с пунктом 12", source="SP486"),
            _p("согласно пункту 12", source="OTHER", score=0.7),
        ]
        gap = build_gap(passages)
        assert len(gap["refs"]) == 2
        assert gap["open"] == ["clause:12"]

    @pytest.mark.unit
    def test_refs_taken_from_top5_only(self):
        """Refs come from the same top-5 slice that triggers crossref escalation."""
        passages = [_p("прочий текст", score=0.9 - i / 100) for i in range(5)]
        passages.append(_p("в соответствии с пунктом 12", score=0.5))
        gap = build_gap(passages)
        assert gap["refs"] == []
        assert gap["open"] == []

    @pytest.mark.unit
    def test_no_refs_gives_empty_gap(self):
        gap = build_gap([_p("высота ограждения не менее одного метра")])
        assert gap["refs"] == []
        assert gap["open"] == []
        assert gap["closed"] == []


def _crossref_state(extra_text="", query="ограждение лестница", active_query=None):
    """State whose passages trip the crossref escalation with an open gap."""
    passages = [
        _p(
            "согласно пункту 3, за исключением лиц указанных в пункте 4, "
            "в соответствии с приложением 1 ограждение лестница",
            score=0.85,
        ),
        _p("пункт 5 настоящего документа содержит перечень ограждение", score=0.80),
        _p("ограждение лестница высота" + extra_text, score=0.75),
    ]
    return {
        "query": query,
        "active_query": active_query or query,
        "retrieval_attempts": [_make_attempt(passages)],
        "plan": {},
    }, passages


class TestGapWithoutExpander:
    @pytest.mark.unit
    def test_escalates_as_before_but_carries_gap(self):
        """No expander injected → today's behaviour, plus the gap in state."""
        state, passages = _crossref_state()
        result = _legacy_triage(state)
        assert result["sufficient"] is False
        assert result["fallback_passages"] == passages
        gap = result["triage_gap"]
        assert gap["kind"] == "unresolved_ref"
        assert "clause:3" in gap["open"]


class TestGapClosing:
    @pytest.mark.unit
    def test_closed_gap_does_not_escalate(self):
        """Gap closed by expansion → sufficient with the expanded passages."""
        state, passages = _crossref_state()
        added = [
            _p("3. Ограждение лестница устанавливается.", score=0.4),
            _p("4. За исключением ограждение временных лестница.", score=0.4),
            _p("5. Перечень ограждение лестница исключений.", score=0.4),
        ]
        set_crossref_expander(lambda ps, query: list(ps) + added)

        result = _legacy_triage(state)

        assert result["sufficient"] is True
        assert result["final_passages"] == passages + added
        assert result["triage_gap"]["open"] == []
        assert set(result["triage_gap"]["closed"]) >= {"clause:3", "clause:4", "clause:5"}

    @pytest.mark.unit
    def test_original_passage_order_preserved(self):
        state, passages = _crossref_state()
        added = [_p(f"{n}. Ограждение лестница текст.", score=0.4) for n in (3, 4, 5)]
        set_crossref_expander(lambda ps, query: list(ps) + added)

        result = _legacy_triage(state)

        assert result["final_passages"][: len(passages)] == passages

    @pytest.mark.unit
    def test_unclosed_gap_escalates_with_the_original_fallback(self):
        """Expansion that closes nothing is discarded, not handed on.

        The real expander inserts bbox siblings right after their parent, so a
        list that grew from 12 to 42 pushes relevant chunks out of the top-12 —
        measured on held-out 05.09.2026, one question lost its hit. When the
        gap does not close, rag_complex starts from the original passages.
        """
        state, passages = _crossref_state()
        added = [_p("Ничего похожего на искомые нормы.", score=0.4)]
        set_crossref_expander(lambda ps, query: list(ps) + added)

        result = _legacy_triage(state)

        assert result["sufficient"] is False
        assert result["fallback_passages"] == passages
        assert "clause:3" in result["triage_gap"]["open"]

    @pytest.mark.unit
    def test_expander_exception_behaves_as_unclosed(self):
        def _boom(ps, query):
            raise RuntimeError("backend down")

        set_crossref_expander(_boom)
        state, passages = _crossref_state()

        result = _legacy_triage(state)

        assert result["sufficient"] is False
        assert result["fallback_passages"] == passages
        assert "clause:3" in result["triage_gap"]["open"]

    @pytest.mark.unit
    def test_zero_original_overlap_still_escalates_after_closing(self):
        """Closing the gap must not silently disable the zero-overlap escalation."""
        state, passages = _crossref_state(
            query="криогенный трубопровод", active_query="ограждение лестница"
        )
        added = [
            _p("3. Ограждение лестница устанавливается.", score=0.4),
            _p("4. За исключением ограждение временных лестница.", score=0.4),
            _p("5. Перечень ограждение лестница исключений.", score=0.4),
        ]
        set_crossref_expander(lambda ps, query: list(ps) + added)

        result = _legacy_triage(state)

        assert result["sufficient"] is False
        assert result["sufficiency_details"]["keyword_overlap_original"] == 0.0

    @pytest.mark.unit
    def test_enumeration_branch_still_applies_after_closing(self):
        """Enumeration queries keep their fallback contract after expansion."""
        state, passages = _crossref_state(query="кто проходит обучение ограждение лестница")
        added = [
            _p("3. Ограждение лестница обучение проходит.", score=0.4),
            _p("4. За исключением ограждение лестница обучение.", score=0.4),
            _p("5. Перечень ограждение лестница обучение.", score=0.4),
        ]
        set_crossref_expander(lambda ps, query: list(ps) + added)

        result = _legacy_triage(state)

        assert result["sufficient"] is True
        assert result["fallback_passages"] == passages + added
        assert result["final_passages"] == passages + added


class TestNoGapNoEscalation:
    @pytest.mark.unit
    def test_crossref_escalation_dropped_when_nothing_is_missing(self):
        """Crossref hits without a gap are not a reason to pay for rag_complex.

        Every clause named in the top-5 is already in the retrieved list, so
        the escalation the counter asked for has nothing left to find.
        """
        passages = [
            _p(
                "согласно пункту 3, за исключением лиц указанных в пункте 4, "
                "в соответствии с приложением 1 ограждение лестница",
                score=0.85,
            ),
            _p("3. Ограждение лестница устанавливается по краю.", score=0.80),
            _p("4. За исключением ограждение временных лестница.", score=0.75),
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }

        result = _legacy_triage(state)

        assert result["sufficient"] is True
        assert result["final_passages"] == passages
        assert result["triage_gap"]["open"] == []
        assert set(result["triage_gap"]["closed"]) == {"clause:3", "clause:4"}


class TestV8Untouched:
    @pytest.mark.unit
    def test_evidence_assess_does_not_fill_gap(self, monkeypatch):
        from src.v7.config import v7_config

        monkeypatch.setattr(v7_config, "V8_ENABLE_EVIDENCE_ASSESS", True)
        state, _ = _crossref_state()

        result = evaluate_triage(state)

        assert "triage_gap" not in result
