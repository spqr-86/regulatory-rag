"""Юнит-тесты адресной выборки get_norm (без Chroma: backend — фейк).

get_norm — детерминированная выборка «документ + пункт» БЕЗ векторов. Надёжно работает
для обычных нумерованных пунктов (текст чанка начинается с номера: «54. ...»). Табличная
адресация (табл.1 п.18) — best-effort: текущий корпус не хранит номер таблицы в метаданных.
"""

import pytest

import mcp_server as m


class _Doc:
    def __init__(self, source, chunk_id, text, section="Document start"):
        self.metadata = {
            "source": source,
            "chunk_id": chunk_id,
            "parent_section": section,
        }
        self.page_content = text


class _FakeBackend:
    def __init__(self, docs):
        self._docs = docs

    def get_by_filter(self, where, limit=500):
        # поддерживаем и {"source": x}, и {"$and": [{"source": x}, ...]}
        if "$and" in where:
            src = where["$and"][0]["source"]
        else:
            src = where["source"]
        return [d for d in self._docs if d.metadata["source"] == src][:limit]


@pytest.mark.unit
def test_normalize_point_plain():
    assert m.normalize_point("п. 218") == "218"
    assert m.normalize_point("пункт 4.4") == "4.4"
    assert m.normalize_point("54") == "54"
    assert m.normalize_point("ст. 83") == "83"


@pytest.mark.unit
def test_normalize_point_table():
    assert m.normalize_point("табл.1 п.18") == "1.18"
    assert m.normalize_point("таблица 3, п. 48") == "3.48"


@pytest.mark.unit
def test_find_norm_by_plain_point():
    docs = [
        _Doc("ppr.docx", 0, "53. Требование A."),
        _Doc("ppr.docx", 1, "54. Руководитель организует ремонт и ТО систем."),
        _Doc("ppr.docx", 2, "При монтаже — продолжение п.54 без номера."),
    ]
    hits = m.find_norm(_FakeBackend(docs), "ppr.docx", "54")
    assert len(hits) == 1 and "Руководитель" in hits[0].page_content


@pytest.mark.unit
def test_find_norm_absent_point_returns_empty():
    docs = [_Doc("ppr.docx", 0, "53. Требование A.")]
    assert m.find_norm(_FakeBackend(docs), "ppr.docx", "999") == []


@pytest.mark.unit
def test_find_norm_does_not_match_substring_number():
    # пункт «5» не должен ловить «53. ...»
    docs = [_Doc("ppr.docx", 0, "53. Требование A.")]
    assert m.find_norm(_FakeBackend(docs), "ppr.docx", "5") == []


@pytest.mark.unit
def test_find_norm_table_row_by_section():
    docs = [
        _Doc("sp486.docx", 60, "17. строка семнадцать", section="Таблица 1"),
        _Doc(
            "sp486.docx", 61, "18. автономные извещатели до 100 м2", section="Таблица 1"
        ),
    ]
    hits = m.find_norm(_FakeBackend(docs), "sp486.docx", "1.18")
    assert len(hits) == 1 and "извещатели" in hits[0].page_content
