"""Lost law-article headings and article/part address in chunk metadata (426-ФЗ, 12.10)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from docling_core.transforms.chunker import HybridChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_core.documents import Document

from src.indexing.file_handler import (
    DocumentProcessor,
    annotate_article_parts,
    demote_fake_law_headings,
    is_article_heading,
    is_fake_law_heading,
    promote_lost_article_headings,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected",
    [
        ("Статья 13. Вредные и (или) опасные факторы", True),
        ("Статья  5.  Права  и  обязанности  работника", True),
        ("Статья 22.1. Особенности", True),
        ("в соответствии со статьёй 12 настоящего закона", False),
        ("Статьями 5 и 6 настоящего закона", False),
        ("1. Работник вправе:", False),
        ("Статья 13", False),
    ],
)
def test_is_article_heading(text, expected):
    assert is_article_heading(text) is expected


def _filler(tag: str) -> str:
    # Long enough that HybridChunker does not merge neighbouring items.
    return " ".join(f"{tag}слово{i}" for i in range(40))


def _law_doc() -> DoclingDocument:
    doc = DoclingDocument(name="law")
    doc.add_heading(text="Статья 12. Исследования", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="1. Работодатель " + _filler("a"))
    # Docling labelled the next article heading as plain text (426-ФЗ ст. 13, 21).
    doc.add_text(label=DocItemLabel.TEXT, text="Статья 13. Вредные факторы")
    doc.add_text(
        label=DocItemLabel.TEXT, text="1. Исследованию подлежат " + _filler("b")
    )
    doc.add_text(label=DocItemLabel.TEXT, text="продолжение части " + _filler("c"))
    return doc


@pytest.mark.unit
def test_promote_turns_lost_article_text_into_heading():
    doc = _law_doc()
    assert promote_lost_article_headings(doc) == 1
    assert doc.texts[2].label == DocItemLabel.SECTION_HEADER
    assert doc.texts[2].self_ref == "#/texts/2"


@pytest.fixture
def processor():
    with patch.object(DocumentProcessor, "__init__", lambda self: None):
        p = DocumentProcessor()
    p._chunker = HybridChunker(max_tokens=400, merge_peers=True)
    return p


@pytest.mark.unit
def test_chunks_after_lost_heading_belong_to_new_article(processor):
    chunks = processor._process_docling_document(_law_doc(), "law.pdf")
    by_tag = {
        tag: next(c for c in chunks if f"{tag}слово0" in c.page_content)
        for tag in "abc"
    }
    assert by_tag["a"].metadata["heading_path"] == "Статья 12. Исследования"
    assert by_tag["b"].metadata["heading_path"] == "Статья 13. Вредные факторы"
    assert by_tag["c"].metadata["heading_path"] == "Статья 13. Вредные факторы"
    assert (by_tag["b"].metadata["article"], by_tag["b"].metadata["part"]) == (
        "13",
        "1",
    )
    assert by_tag["b"].metadata["part_inherited"] is False
    assert (by_tag["c"].metadata["article"], by_tag["c"].metadata["part"]) == (
        "13",
        "1",
    )
    assert by_tag["c"].metadata["part_inherited"] is True


def _chunk(heading: str, body: str) -> Document:
    return Document(
        page_content=f"{heading}\n{body}",
        metadata={"heading_path": heading, "parent_section": heading},
    )


@pytest.mark.unit
def test_annotate_parts_inherits_last_marker_and_ignores_docling_list_numbers():
    h = "Статья 6. Права и обязанности"
    chunks = [
        _chunk(h, "1. Организация вправе:\n- 1) ..."),
        _chunk(h, "- 2) обжаловать предписания\n2. Организация обязана:"),
        _chunk(h, "9. а) непредоставления работодателем сведений"),
        _chunk(h, "4. международного договора."),
    ]
    annotate_article_parts(chunks)
    parts = [(c.metadata["part"], c.metadata["part_inherited"]) for c in chunks]
    assert parts == [("1", False), ("1", True), ("2", True), ("2", True)]


@pytest.mark.unit
def test_annotate_parts_skips_non_law_and_article_without_parts():
    chunks = [
        _chunk("Раздел III. Порядок", "31. Стажировка не менее 2 смен"),
        _chunk("Статья 3. Понятие", "Специальная оценка условий труда является"),
    ]
    annotate_article_parts(chunks)
    assert "article" not in chunks[0].metadata
    assert chunks[1].metadata["article"] == "3"
    assert "part" not in chunks[1].metadata


@pytest.mark.unit
def test_annotate_parts_bare_marker_cut_by_chunker_starts_next_part():
    h = "Статья 13. Вредные факторы"
    chunks = [_chunk(h, "1."), _chunk(h, "Исследованию подлежат факторы")]
    annotate_article_parts(chunks)
    assert chunks[1].metadata["part"] == "1"
    assert chunks[1].metadata["part_inherited"] is True


def _law_doc_with_fake_heading() -> DoclingDocument:
    doc = DoclingDocument(name="law")
    doc.add_heading(text="ТРУДОВОЙ КОДЕКС РОССИЙСКОЙ ФЕДЕРАЦИИ", level=1)
    doc.add_heading(text="Статья 209. Основные понятия", level=1)
    # Docling labelled amendment notes and wrapped lines as headings (ТК стр. 133, 72).
    doc.add_heading(text="(в ред. Федерального закона от 02.07.2021 N 311-ФЗ)", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="Охрана труда - " + _filler("d"))
    doc.add_heading(text="их защиты", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text="продолжение статьи " + _filler("e"))
    return doc


@pytest.mark.unit
@pytest.mark.parametrize(
    "text, expected",
    [
        ("(в ред. Федерального закона от 02.07.2021 N 311-ФЗ)", True),
        ("(введена Федеральным законом от 28.12.2013 N 421-ФЗ)", True),
        ("их защиты", True),
        ("28.11.2019 N 37-П.", True),
        ("КонсультантПлюс: примечание.", True),
        ("Статья 209. Основные понятия", False),
        ("Глава 33. ОБЩИЕ ПОЛОЖЕНИЯ", False),
        ("ЗАНЯТЫХ НА СЕЗОННЫХ РАБОТАХ Статья 293. Сезонные работы", False),
        ("О пожарной безопасности", False),
    ],
)
def test_is_fake_law_heading(text, expected):
    assert is_fake_law_heading(text) is expected


@pytest.mark.unit
def test_demote_fake_headings_only_after_first_article():
    doc = _law_doc_with_fake_heading()
    assert demote_fake_law_headings(doc) == 2
    assert doc.texts[0].label == DocItemLabel.SECTION_HEADER
    assert doc.texts[2].label == DocItemLabel.TEXT
    assert doc.texts[2].self_ref == "#/texts/2"


@pytest.mark.unit
def test_demote_skips_documents_without_articles():
    doc = DoclingDocument(name="order")
    doc.add_heading(text="(приложение к приказу)", level=1)
    doc.add_text(label=DocItemLabel.TEXT, text=_filler("f"))
    assert demote_fake_law_headings(doc) == 0


@pytest.mark.unit
def test_chunks_under_fake_heading_keep_article(processor):
    chunks = processor._process_docling_document(
        _law_doc_with_fake_heading(), "тк рф.pdf"
    )
    for tag in "de":
        ch = next(c for c in chunks if f"{tag}слово0" in c.page_content)
        assert ch.metadata["parent_section"] == "Статья 209. Основные понятия"
        assert ch.metadata["article"] == "209"
