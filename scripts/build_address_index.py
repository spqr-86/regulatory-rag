"""Собрать адресный индекс из нормативных docx корпуса ПБ.

doc_id = имя файла (совпадает с полем `source` в Chroma-корпусе и `source` в
applicability.yaml prombez-agent) — единый ключ между корпусом, картой и адресным слоем.

    ./venv/bin/python scripts/build_address_index.py

Пишет address_index_pb.json в корне repo. Логирует число записей на документ —
молчаливый ноль = разметку документа парсер не понял.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.address_layer import AddressIndex, parse_docx_records  # noqa: E402

SOURCE_DIR = ROOT / "source_docs_pb"
OUT = ROOT / "address_index_pb.json"


def main() -> int:
    all_records: list[dict] = []
    for docx_path in sorted(SOURCE_DIR.glob("*.docx")):
        doc_id = docx_path.name
        recs = parse_docx_records(str(docx_path), doc_id=doc_id)
        kinds = Counter(r["kind"] for r in recs)
        tables = sorted({r["table"] for r in recs if r["kind"] == "table"})
        print(
            f"{doc_id}: {len(recs)} записей "
            f"(пунктов={kinds.get('point', 0)}, строк таблиц={kinds.get('table', 0)}, "
            f"таблицы={tables})"
        )
        all_records.extend(recs)
    AddressIndex(all_records).save(OUT)
    print(f"\nИтого {len(all_records)} записей → {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
