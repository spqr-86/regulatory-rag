"""Тесты нарезок эксперимента eval/chunking_ab (без сети и без индекса)."""

import re
from pathlib import Path

from eval.chunking_ab.chunkers import (
    chunk_paragraphs,
    chunk_structural,
    chunk_window,
    paragraphs_from_text,
)

ROOT = Path(__file__).parent.parent
SUBITEM = re.compile(r"^\s*([а-яё]|\d+)\)\s")

ARTICLE_TEXT = """Статья 5. Первая
1. Работодатель обязан:
а) первое;
б) второе;
2. Отдельная часть.
Статья 6. Вторая
Единственный абзац статьи.
"""


def body(chunk_text: str) -> str:
    return chunk_text.split("\n", 1)[1] if "\n" in chunk_text else chunk_text


def test_paragraphs_join_wrapped_pdf_lines():
    text = "Работодатель обеспечивает\nпроведение обучения.\nСледующий абзац."
    assert paragraphs_from_text(text, wrapped=True) == [
        "Работодатель обеспечивает проведение обучения.",
        "Следующий абзац.",
    ]


def test_structural_keeps_unit_whole_when_fits():
    chunks = chunk_structural(
        paragraphs_from_text(ARTICLE_TEXT, wrapped=False), "Док", 1800
    )
    assert len(chunks) == 2
    assert "а) первое;" in chunks[0] and "2. Отдельная часть." in chunks[0]
    assert chunks[0].startswith("Док › Статья 5. Первая")


def test_structural_split_repeats_intro_for_every_subitem_chunk():
    items = "\n".join(f"{c}) {'пункт ' * 20}{c};" for c in "абвгде")
    text = f"16. Внеплановый инструктаж проводится в случаях:\n{items}\n17. Другое."
    chunks = chunk_structural(paragraphs_from_text(text, wrapped=False), "Док", 300)
    split = [c for c in chunks if SUBITEM.search(body(c).split("\n", 1)[-1])]
    assert len(split) >= 2
    for c in split:
        assert "Внеплановый инструктаж проводится в случаях:" in c
    # ни один чанк не начинается с подпункта без вводной
    for c in chunks:
        first = body(c).split("\n")[0]
        assert not SUBITEM.match(first)


def test_structural_never_cuts_inside_sentence():
    text = "Статья 8. Короткая\nТекст.\nСтатья 9. Длинная\n" + " ".join(
        f"Предложение номер {i} содержит слова." for i in range(80)
    )
    chunks = chunk_structural(paragraphs_from_text(text, wrapped=False), "Док", 400)
    assert len(chunks) > 3
    assert all(c.rstrip().endswith(".") for c in chunks)


def test_structural_covers_all_paragraphs():
    paras = paragraphs_from_text(ARTICLE_TEXT, wrapped=False)
    joined = "\n".join(chunk_structural(paras, "Док", 1800))
    assert all(p in joined for p in paras)


def test_item_number_with_backslash_escape_stays_item_number():
    text = (
        "15. Первый пункт.\n16\\. Внеплановый инструктаж проводится:\nа) раз;\nб) два;"
    )
    chunks = chunk_structural(paragraphs_from_text(text, wrapped=False), "Док", 1800)
    assert any("п. 16" in c for c in chunks)
    assert not any("п. 6" in c for c in chunks)


def test_window_overlap_and_size():
    words = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_window(words, size=350, stride=300)
    assert [len(c.split()) for c in chunks[:3]] == [350, 350, 350]
    assert chunks[1].split()[0] == "w300"


def test_paragraph_variant_merges_short():
    paras = ["Короткий.", "Тоже короткий.", "Длинный " * 40 + "абзац."]
    chunks = chunk_paragraphs(paras, min_chars=200, max_chars=3000)
    assert len(chunks) == 1 or all(len(c) >= 200 for c in chunks[:-1])


def test_g005_real_2464_item16_every_chunk_has_intro():
    md = (ROOT / "source_docs/2464.md").read_text(encoding="utf-8")
    paras = paragraphs_from_text(md, wrapped=False)
    chunks = chunk_structural(paras, "Правила 2464", 900)
    item16 = [c for c in chunks if "п. 16" in c.split("\n", 1)[0]]
    assert item16
    joined = "\n".join(item16)
    for letter in "абвгдежз":
        assert f"\n{letter})" in joined
    for c in item16:
        assert c.count("обусловленных") >= 1


def test_long_item_number_not_split_from_its_text():
    text = "43. Коротко.\n44. " + " ".join(
        f"Предложение {i} содержит слова." for i in range(40)
    )
    chunks = chunk_structural(paragraphs_from_text(text, wrapped=False), "Док", 300)
    assert not any(c.split("\n", 1)[1].strip() == "44." for c in chunks if "\n" in c)
    assert any("44. Предложение 0" in c for c in chunks)
