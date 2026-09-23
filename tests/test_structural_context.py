"""Chunk context for list items and table sub-rows (issue #64).

Fixtures are real chunk texts from the generic index (2464.pdf, 29н.pdf),
without the parent_section prefix that file_handler adds later.
"""

from __future__ import annotations

import pytest

from src.indexing.structural_context import add_structural_context

STEM_16 = (
    "16. Внеплановый инструктаж по охране труда проводится для работников "
    "организации в случаях, обусловленных:"
)


@pytest.mark.unit
def test_list_items_split_from_stem_get_the_stem():
    texts = [
        "14. Повторный инструктаж по охране труда проводится не реже одного раза в 6 месяцев.\n"
        + STEM_16
        + "\n9. а) изменениями в эксплуатации оборудования;",
        "10. б) изменениями должностных (функциональных) обязанностей работников;",
        "11. в) изменениями нормативных правовых актов;",
    ]

    out = add_structural_context(texts)

    assert out[0] == texts[0]  # stem already inside the chunk
    assert out[1] == f"{STEM_16}\n{texts[1]}"
    assert out[2] == f"{STEM_16}\n{texts[2]}"


@pytest.mark.unit
def test_plain_paragraph_ends_the_list():
    texts = [
        STEM_16
        + "\n15. ж) перерывом в работе продолжительностью более 60 календарных дней;",
        "16. з) решением работодателя.\n17. Внеплановый инструктаж по охране труда "
        "проводится в объеме мероприятий.",
        "18. Целевой инструктаж проводится в случаях.",
        "19. а) пункт другого перечня без вводной фразы;",
    ]

    out = add_structural_context(texts)

    assert out[1] == f"{STEM_16}\n{texts[1]}"
    assert out[2] == texts[2]
    assert out[3] == texts[3]


@pytest.mark.unit
def test_new_stem_replaces_old_one():
    stem_53 = (
        "53. Обучению требованиям охраны труда подлежат следующие категории работников:"
    )
    texts = [
        "52. Работники федеральных органов исполнительной власти проходят обучение "
        "по следующим программам:",
        '8. а) заместитель руководителя - по программе "а";',
        '10. в) специалисты по охране труда - по программам "а" и "б".\n' + stem_53,
        '12. а) работодатель (руководитель организации) - по программе "а";',
    ]

    out = add_structural_context(texts)

    assert out[1].startswith("52. Работники федеральных органов")
    assert out[3] == f"{stem_53}\n{texts[3]}"


@pytest.mark.unit
def test_item_continued_into_next_chunk_keeps_stem():
    stem_53 = (
        "53. Обучению требованиям охраны труда подлежат следующие категории работников:"
    )
    texts = [
        stem_53
        + "\n17. е) члены комиссий по проверке знания требований охраны труда; ж)",
        "члены комитетов (комиссий) по охране труда - по программам обучения, указанным",
    ]

    out = add_structural_context(texts)

    assert out[1] == f"{stem_53}\n{texts[1]}"


@pytest.mark.unit
def test_table_subrow_gets_parent_row_label():
    head = "ПЕРИОДИЧНОСТЬ И ОБЪЕМ МЕДИЦИНСКИХ ОСМОТРОВ"
    texts = [
        f"{head}\n17, 2 = 1 раз в 2 года. 18, 1 = Управление наземными транспортными "
        "средствами. 18, 2 = . ",
        f'{head}\n18.1, <2>: = Категории "A", "B", "BE". 18.1, = 1 раз в 2 года.',
        f'{head}\nТональная пороговая аудиометрия. 18.2, <2>: = Категории "C", "C1". '
        "19, <2>: = Водолазные работы.",
    ]

    out = add_structural_context(texts)

    parent = "18. Управление наземными транспортными средствами"
    assert out[0] == texts[0]
    assert out[1] == f"{parent}\n{texts[1]}"
    assert out[2] == f"{parent}\n{texts[2]}"


@pytest.mark.unit
def test_subrow_without_known_parent_is_untouched():
    texts = ['18.1, <2>: = Категории "A". 18.1, = 1 раз в 2 года.']

    assert add_structural_context(texts) == texts


@pytest.mark.unit
def test_parent_row_without_words_is_not_a_label():
    texts = [
        "1, <2>: = 2. 1, 2 = 1 раз в год.",
        "1.35, 4 = Спирометрия Пульсоксиметрия",
    ]

    assert add_structural_context(texts) == texts


@pytest.mark.unit
def test_same_length_and_order():
    texts = ["a", "", "b"]

    assert add_structural_context(texts) == texts


@pytest.mark.unit
def test_docling_processing_prepends_stem_after_heading():
    from types import SimpleNamespace
    from unittest.mock import patch

    from src.indexing.file_handler import DocumentProcessor

    heading = "I. Общие положения"

    def chunk(text):
        return SimpleNamespace(
            text=text, meta=SimpleNamespace(headings=[heading], doc_items=[])
        )

    with patch.object(DocumentProcessor, "__init__", lambda self: None):
        p = DocumentProcessor()
    p._chunker = SimpleNamespace(
        chunk=lambda doc: [
            chunk(STEM_16 + "\n9. а) изменениями в эксплуатации оборудования;"),
            chunk("10. б) изменениями должностных обязанностей работников;"),
        ]
    )

    docs = p._process_docling_document(object(), "2464.pdf")

    assert docs[1].page_content == (
        f"{heading}\n{STEM_16}\n10. б) изменениями должностных обязанностей работников;"
    )


@pytest.mark.unit
def test_docling_processing_handles_inlined_heading():
    from types import SimpleNamespace
    from unittest.mock import patch

    from src.indexing.file_handler import DocumentProcessor

    heading = "I. Общие положения"

    def chunk(text):
        return SimpleNamespace(
            text=text, meta=SimpleNamespace(headings=[heading], doc_items=[])
        )

    with patch.object(DocumentProcessor, "__init__", lambda self: None):
        p = DocumentProcessor()
    p._chunker = SimpleNamespace(
        chunk=lambda doc: [
            chunk(f"{heading}\n{STEM_16}\n9. а) изменениями в эксплуатации;"),
            chunk(f"{heading}\n10. б) изменениями должностных обязанностей;"),
        ]
    )

    docs = p._process_docling_document(object(), "2464.pdf")

    assert (
        docs[0].page_content
        == f"{heading}\n{STEM_16}\n9. а) изменениями в эксплуатации;"
    )
    assert docs[1].page_content == (
        f"{heading}\n{STEM_16}\n10. б) изменениями должностных обязанностей;"
    )
