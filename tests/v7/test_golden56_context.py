"""Offline regression cases from golden56 error analysis (2026-10-16)."""

from src.v7.context_format import format_generation_context


def test_part_and_clause_in_context_header():
    text = format_generation_context(
        [
            {
                "text": "штраф",
                "metadata": {
                    "source": "КоАП РФ",
                    "article": "5.27.1",
                    "part": "3",
                    "clause": "2",
                },
            }
        ]
    )
    assert "Статья: 5.27.1" in text
    assert "Часть: 3" in text
    assert "Пункт: 2" in text


def test_pack_keeps_part_and_condition_and_separates_address_cache():
    from src.v7.pack_context import pack_context

    passage = {
        "text": "При повторном нарушении штраф",
        "metadata": {"source": "КоАП РФ", "article": "5.27.1", "part": "5"},
    }
    cache = {}
    first = pack_context([passage], "штраф", {}, cache=cache, crossref_expander=None)
    changed = {**passage, "metadata": {**passage["metadata"], "part": "3"}}
    second = pack_context([changed], "штраф", {}, cache=cache, crossref_expander=None)
    assert second["final_context"][0]["metadata"]["part"] == "3"
    assert "При повторном нарушении" in format_generation_context(
        first["final_context"]
    )


def test_multi_part_chunk_header_does_not_attribute_everything_to_first_part():
    from langchain_core.documents import Document

    from src.indexing.file_handler import annotate_article_parts

    chunks = [
        Document(
            page_content="1. Нарушение требований.\n2. Повторное нарушение.",
            metadata={"heading_path": "Статья 5.27.1. Ответственность"},
        )
    ]
    annotate_article_parts(chunks)
    assert chunks[0].metadata["part_end"] == "2"
    text = format_generation_context(
        [{"text": chunks[0].page_content, "metadata": chunks[0].metadata}]
    )
    assert "Часть: 1–2" in text
