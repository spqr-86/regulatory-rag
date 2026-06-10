"""Unit tests for src/v7/nodes/visual_enrichment.py"""

from __future__ import annotations

import pytest

from src.v7.nodes.visual_enrichment import (
    _needs_visual,
    set_visual_proof_fn,
    visual_enrichment,
)


def _passage(text: str, meta: dict | None = None) -> dict:
    return {"text": text, "score": 0.5, "metadata": meta or {}}


def _coords(source: str = "doc.pdf", page_no: int = 1, bbox=None) -> dict:
    return {"source": source, "page_no": page_no, "bbox": bbox or [0, 0, 100, 50]}


# ─── _needs_visual ────────────────────────────────────────────────────────


class TestNeedsVisual:
    def test_false_when_no_coords(self):
        p = _passage("Обычный текст без координат.")
        assert _needs_visual(p) is False

    def test_false_for_complete_passage_with_coords(self):
        # Long complete text — should not trigger
        text = "Работодатель обязан обеспечить безопасные условия труда. " * 5
        p = _passage(text, _coords())
        assert _needs_visual(p) is False

    def test_true_for_table_element_type(self):
        p = _passage("120 | 80 | 0.3", {**_coords(), "element_type": "TableCell"})
        assert _needs_visual(p) is True

    def test_true_for_table_in_element_type(self):
        p = _passage("данные", {**_coords(), "element_type": "Table"})
        assert _needs_visual(p) is True

    def test_true_for_short_text(self):
        p = _passage("Короткий.", _coords())  # < 150 chars
        assert _needs_visual(p) is True

    def test_true_for_page_no_zero(self):
        # page_no=0 is a valid first page and must not be treated as falsy
        p = _passage("короткий текст", _coords(page_no=0))
        assert _needs_visual(p) is True

    def test_true_for_incomplete_chunk(self):
        # No terminal punctuation → detect_incomplete_chunk → True
        text = "Требования к организации рабочих мест изложены в следующем разделе"
        p = _passage(text, _coords())
        assert _needs_visual(p) is True


# ─── visual_enrichment node ──────────────────────────────────────────────


class TestVisualEnrichmentNode:
    def setup_method(self):
        set_visual_proof_fn(None)

    def teardown_method(self):
        set_visual_proof_fn(None)

    def test_no_op_when_no_fn(self):
        state = {"final_passages": [_passage("текст", _coords())]}
        assert visual_enrichment(state) == {}

    def test_no_op_when_no_passages(self):
        set_visual_proof_fn(lambda *a: "result")
        assert visual_enrichment({}) == {}
        assert visual_enrichment({"final_passages": []}) == {}

    def test_no_op_when_no_passages_need_enrichment(self):
        set_visual_proof_fn(lambda *a: pytest.fail("should not be called"))
        # Long complete text — no enrichment needed
        long_text = "Работодатель обязан обеспечить безопасные условия труда. " * 5
        state = {"final_passages": [_passage(long_text, {})]}  # no coords either
        assert visual_enrichment(state) == {}

    def test_skips_passage_without_coords(self):
        calls = []
        set_visual_proof_fn(lambda *a: calls.append(a) or "x")
        state = {"final_passages": [_passage("короткий", {})]}  # no source/page/bbox
        visual_enrichment(state)
        assert calls == []

    def test_enriches_table_passage_analyze_mode(self):
        set_visual_proof_fn(lambda src, page, bbox, mode: "[Таблица: данные]")
        p = _passage("120", {**_coords(), "element_type": "Table"})
        result = visual_enrichment({"final_passages": [p]})
        assert "final_passages" in result
        assert "[Таблица — визуальный анализ]" in result["final_passages"][0]["text"]
        assert "[Таблица: данные]" in result["final_passages"][0]["text"]

    def test_enriches_incomplete_passage_show_mode(self):
        set_visual_proof_fn(lambda src, page, bbox, mode: "/static/proof_abc.png")
        text = "Требования изложены в следующем"  # incomplete, < 150 chars
        p = _passage(text, _coords())
        result = visual_enrichment({"final_passages": [p]})
        assert result["final_passages"][0]["image_path"] == "/static/proof_abc.png"

    def test_respects_max_limit(self):
        calls = []

        def _fn(src, page, bbox, mode):
            calls.append(mode)
            return "path"

        set_visual_proof_fn(_fn)
        # 5 short passages with coords
        passages = [_passage("x", _coords(page_no=i)) for i in range(1, 6)]
        visual_enrichment({"final_passages": passages})
        assert len(calls) <= 3

    def test_handles_fn_exception(self):
        def _bad_fn(*args):
            raise RuntimeError("VLM unavailable")

        set_visual_proof_fn(_bad_fn)
        p = _passage("данные", {**_coords(), "element_type": "Table"})
        # Should not raise, returns no-op
        result = visual_enrichment({"final_passages": [p]})
        assert result == {}

    def test_bbox_string_is_parsed(self):
        import json

        parsed_args = []

        def _fn(src, page, bbox, mode):
            parsed_args.append(bbox)
            return "ok"

        set_visual_proof_fn(_fn)
        bbox_str = json.dumps([10.0, 20.0, 100.0, 80.0])
        p = _passage("x", {**_coords(bbox=bbox_str), "element_type": "Table"})
        visual_enrichment({"final_passages": [p]})
        assert parsed_args[0] == [10.0, 20.0, 100.0, 80.0]
