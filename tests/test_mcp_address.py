"""Юнит-тесты адресной выборки get_norm (без Chroma: backend — фейк).

get_norm — детерминированная выборка «документ + адрес» БЕЗ векторов. Адрес типизирован:
пункт / статья / таблица(строка). Табличная ветка включается только по явной форме
«табл.…» — точка в номере («4.4») таблицей не считается.
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
        if "$and" in where:
            src = where["$and"][0]["source"]
        else:
            src = where["source"]
        return [d for d in self._docs if d.metadata["source"] == src][:limit]


@pytest.mark.unit
def test_normalize_point_plain():
    assert m.normalize_point("п. 218") == ("point", "218", None)
    assert m.normalize_point("пункт 4.4") == ("point", "4.4", None)
    assert m.normalize_point("54") == ("point", "54", None)


@pytest.mark.unit
def test_normalize_point_article():
    assert m.normalize_point("ст. 83") == ("article", "83", None)
    assert m.normalize_point("статья 12") == ("article", "12", None)


@pytest.mark.unit
def test_normalize_point_table():
    assert m.normalize_point("табл.1 п.18") == ("table", "1", "18")
    assert m.normalize_point("таблица 3, п. 48") == ("table", "3", "48")
    # вся таблица без строки — легальный адрес (карта applicability даёт «табл.1»)
    assert m.normalize_point("табл.1") == ("table", "1", None)


@pytest.mark.unit
def test_find_norm_by_plain_point():
    docs = [
        _Doc("ppr.docx", 0, "53. Требование A."),
        _Doc("ppr.docx", 1, "54. Руководитель организует ремонт и ТО систем."),
        _Doc("ppr.docx", 2, "При монтаже — продолжение п.54 без номера."),
    ]
    hits = m.find_norm(_FakeBackend(docs), "ppr.docx", ("point", "54", None))
    assert len(hits) == 1 and "Руководитель" in hits[0].page_content


@pytest.mark.unit
def test_find_norm_absent_point_returns_empty():
    docs = [_Doc("ppr.docx", 0, "53. Требование A.")]
    assert m.find_norm(_FakeBackend(docs), "ppr.docx", ("point", "999", None)) == []


@pytest.mark.unit
def test_find_norm_does_not_match_substring_number():
    # «5» не ловит ни «53.», ни «5.1»; «54» не ловит «54.1» (ревью 18.07, major 1)
    docs = [
        _Doc("ppr.docx", 0, "53. Требование A."),
        _Doc("ppr.docx", 1, "5.1. Подпункт."),
        _Doc("ppr.docx", 2, "54.1. Подпункт пятьдесят четыре один."),
        _Doc("ppr.docx", 3, "54. Сам пункт."),
    ]
    b = _FakeBackend(docs)
    assert m.find_norm(b, "ppr.docx", ("point", "5", None)) == []
    hits = m.find_norm(b, "ppr.docx", ("point", "54", None))
    assert len(hits) == 1 and "Сам пункт" in hits[0].page_content


@pytest.mark.unit
def test_point_with_dot_is_not_table_guess():
    # «4.4» — пункт 4.4, а не строка 4 таблицы 4 (ревью 18.07, major 3)
    docs = [
        _Doc("sp.docx", 0, "4. строка четыре", section="Таблица 4"),
        _Doc("sp.docx", 1, "4.4. Обычный пункт четыре-четыре."),
    ]
    hits = m.find_norm(_FakeBackend(docs), "sp.docx", m.normalize_point("п. 4.4"))
    assert len(hits) == 1 and "Обычный пункт" in hits[0].page_content


@pytest.mark.unit
def test_find_norm_table_row_by_section():
    docs = [
        _Doc("sp486.docx", 60, "17. строка семнадцать", section="Таблица 1"),
        _Doc(
            "sp486.docx", 61, "18. автономные извещатели до 100 м2", section="Таблица 1"
        ),
    ]
    hits = m.find_norm(_FakeBackend(docs), "sp486.docx", ("table", "1", "18"))
    assert len(hits) == 1 and "извещатели" in hits[0].page_content


@pytest.mark.unit
def test_find_norm_whole_table():
    # «табл.1» без строки → все строки таблицы 1 (контракт с applicability-картой)
    docs = [
        _Doc("sp486.docx", 60, "17. строка", section="Таблица 1"),
        _Doc("sp486.docx", 61, "18. строка", section="Таблица 1"),
        _Doc("sp486.docx", 70, "48. строка", section="Таблица 3"),
    ]
    hits = m.find_norm(_FakeBackend(docs), "sp486.docx", ("table", "1", None))
    assert len(hits) == 2


@pytest.mark.unit
def test_find_norm_article():
    # статьи ФЗ начинаются со слова: «Статья 83. …» (ревью 18.07, minor 8)
    docs = [
        _Doc(
            "fz123.docx", 0, "Статья 83. Требования к системам пожарной сигнализации."
        ),
        _Doc("fz123.docx", 1, "Статья 84. Оповещение людей о пожаре."),
    ]
    hits = m.find_norm(_FakeBackend(docs), "fz123.docx", m.normalize_point("ст. 83"))
    assert len(hits) == 1 and "Статья 83" in hits[0].page_content


# ─────────── адресный слой в get_norm/get_chunk (Task 3-4) ───────────


class _FakeApp:
    def __init__(self, address, backend=None):
        self.address = address
        self.backend = backend


@pytest.mark.unit
def test_addr_id_roundtrip_table():
    assert m._addr_id("d.docx", "table", "1", "18") == "d.docx::addr:table:1.18"
    assert m._addr_id("d.docx", "point", "54", None) == "d.docx::addr:point:54"


@pytest.mark.unit
def test_address_hits_table_row():
    from src.address_layer import AddressIndex

    idx = AddressIndex(
        [
            {
                "doc": "d.docx",
                "kind": "table",
                "table": "1",
                "row": "18",
                "text": "стр 18",
            }
        ]
    )
    hits = m._address_hits(idx, "d.docx", "table", "1", "18")
    assert hits[0]["chunk_id"] == "d.docx::addr:table:1.18"
    assert hits[0]["text"] == "стр 18" and hits[0]["section"] == "Таблица 1"


@pytest.mark.unit
def test_get_address_chunk_verifiable():
    # цитата из адресного слоя достаётся обратно по синтетическому id
    from src.address_layer import AddressIndex

    idx = AddressIndex(
        [
            {
                "doc": "d.docx",
                "kind": "table",
                "table": "1",
                "row": "18",
                "text": "стр 18",
            }
        ]
    )
    app = _FakeApp(idx)
    out = m._get_address_chunk(app, "d.docx", "addr:table:1.18")
    assert out["found"] and out["text"] == "стр 18"
    miss = m._get_address_chunk(app, "d.docx", "addr:table:1.99")
    assert miss == {"found": False}


@pytest.mark.unit
def test_get_address_chunk_point():
    from src.address_layer import AddressIndex

    idx = AddressIndex(
        [{"doc": "d.docx", "kind": "point", "num": "54", "text": "54. x"}]
    )
    out = m._get_address_chunk(_FakeApp(idx), "d.docx", "addr:point:54")
    assert out["found"] and out["text"] == "54. x"
