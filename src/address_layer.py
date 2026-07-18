"""Адресный слой: детерминированный разбор нормативных docx в записи адресов.

Независим от RAG-чанкера (Docling). Обход документа в порядке блоков: подпись
«Таблица N» привязывает последующую таблицу к номеру; строки с номером в первой
ячейке — записи «таблица→строка»; нумерованные абзацы — записи «пункт». Точная
выборка текста нормы по адресу без векторов (вариант Б, развилка 18.07).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

_TABLE_LABEL = re.compile(r"^Таблица\s*(\d+)")
_POINT = re.compile(r"^(\d+(?:\.\d+)*)[.)]\s")  # «54. …», «4.4) …»
_ROW_NUM = re.compile(r"^(\d+(?:\.\d+)*)[.)\s]")  # первая ячейка строки таблицы


def _row_text(row) -> str:
    """Строка таблицы → плоский текст: непустые ячейки через ' | ' в порядке колонок.
    docx дублирует объединённые ячейки — соседние повторы убираем."""
    seen: set[str] = set()
    uniq: list[str] = []
    for cell in row.cells:
        c = cell.text.strip()
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)
    return " | ".join(uniq)


def parse_docx_records(path: str, doc_id: str) -> list[dict]:
    """Записи адресов документа: [{doc, kind: point|table, num|table+row, text}]."""
    d = docx.Document(path)
    records: list[dict] = []
    current_table: str | None = None
    for ch in d.element.body.iterchildren():
        if ch.tag == qn("w:p"):
            text = Paragraph(ch, d).text.strip()
            label = _TABLE_LABEL.match(text)
            if label:
                current_table = label.group(1)
                continue
            point = _POINT.match(text)
            if point:
                records.append(
                    {
                        "doc": doc_id,
                        "kind": "point",
                        "num": point.group(1),
                        "text": text,
                    }
                )
        elif ch.tag == qn("w:tbl"):
            table = Table(ch, d)
            # мусорные/безымянные таблицы (шапки-раскладки, футеры) — мимо
            if current_table is None or len(table.rows) < 3:
                continue
            for row in table.rows:
                rm = _ROW_NUM.match(row.cells[0].text.strip())
                if not rm:
                    continue  # шапка/пустая строка — без номера
                records.append(
                    {
                        "doc": doc_id,
                        "kind": "table",
                        "table": current_table,
                        "row": rm.group(1).rstrip("."),
                        "text": _row_text(row),
                    }
                )
    return records


class AddressIndex:
    """Индекс записей адресов в памяти. Точечный lookup по (doc, kind, num, row)."""

    def __init__(self, records: list[dict]):
        self._records = records
        self._by_key: dict[tuple, list[dict]] = {}
        self._tables: dict[tuple, list[dict]] = {}
        for r in records:
            if r["kind"] == "table":
                self._by_key.setdefault(
                    (r["doc"], "table", r["table"], r["row"]), []
                ).append(r)
                self._tables.setdefault((r["doc"], r["table"]), []).append(r)
            else:
                self._by_key.setdefault(
                    (r["doc"], r["kind"], r["num"], None), []
                ).append(r)

    def lookup(self, doc: str, kind: str, num: str, row: str | None) -> list[dict]:
        if kind == "table" and row is None:
            return self._tables.get((doc, num), [])
        return self._by_key.get((doc, kind, num, row), [])

    @classmethod
    def load(cls, path: str | Path) -> AddressIndex:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        return cls(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self._records, ensure_ascii=False, indent=1), encoding="utf-8"
        )
