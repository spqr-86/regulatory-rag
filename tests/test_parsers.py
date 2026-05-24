from src.infra.parsers import (
    parse_json_from_response,
    extract_text,
    parse_status_block,
    parse_search_results,
)


class TestParseJsonFromResponse:
    def test_parses_markdown_code_block(self):
        raw = '```json\n{"status": "approved"}\n```'
        assert parse_json_from_response(raw) == {"status": "approved"}

    def test_parses_bare_json(self):
        raw = 'Some text {"key": "value"} more text'
        assert parse_json_from_response(raw) == {"key": "value"}

    def test_returns_empty_dict_on_invalid(self):
        assert parse_json_from_response("no json here") == {}


class TestExtractText:
    def test_string_passthrough(self):
        assert extract_text("hello") == "hello"

    def test_gemini_style_blocks(self):
        content = [{"text": "part1"}, {"text": "part2"}]
        assert extract_text(content) == "part1\npart2"

    def test_mixed_list(self):
        content = [{"text": "block"}, "plain"]
        assert extract_text(content) == "block\nplain"

    def test_fallback_to_str(self):
        assert extract_text(42) == "42"


class TestParseStatusBlock:
    def test_parses_all_sections(self):
        text = (
            "===STATUS===\nFOUND\n"
            "===ANSWER===\nОтвет тут\n"
            "===UNANSWERED===\n- вопрос 1\n- вопрос 2"
        )
        status, answer, unanswered = parse_status_block(text)
        assert status.value == "FOUND"
        assert answer == "Ответ тут"
        assert unanswered == ["вопрос 1", "вопрос 2"]

    def test_defaults_when_no_markers(self):
        status, answer, unanswered = parse_status_block("Just plain text")
        assert status.value == "FOUND"
        assert answer == "Just plain text"
        assert unanswered == []


class TestParseSearchResults:
    def test_parses_structured_results(self):
        text = (
            "[Result 0] File: doc.pdf | Page: 5 | BBox: [1,2,3,4]\n"
            "Extended Context:\nSome content here\n(IDs: [10, 11])"
        )
        chunks = parse_search_results(text)
        assert len(chunks) == 1
        assert chunks[0]["source"] == "doc.pdf"
        assert chunks[0]["page_no"] == 5
        assert chunks[0]["content"] == "Some content here"

    def test_fallback_for_unstructured(self):
        chunks = parse_search_results("Some relevant text without structure")
        assert len(chunks) == 1
        assert chunks[0]["source"] == "unknown"

    def test_no_results_found(self):
        chunks = parse_search_results("No relevant documents found")
        assert chunks == []
