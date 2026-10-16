import pytest

from eval.judges.citations import parse_verdict

FENCE = "`" * 3


@pytest.mark.parametrize(
    "raw",
    ['Вот ответ: {"checks": []}', "", None, 123],
)
def test_invalid_json_raises_value_error(raw):
    with pytest.raises(ValueError, match="Judge returned invalid JSON"):
        parse_verdict(raw)


def test_valid_empty_checks():
    assert parse_verdict('{"checks": []}') == {"checks": []}


@pytest.mark.parametrize(
    "raw",
    [
        FENCE + 'json\n{"checks": []}\n' + FENCE,
        FENCE + '\n{"checks": []}\n' + FENCE,
        "  \n" + FENCE + 'json\n{"checks": []}\n' + FENCE + "\n  ",
        FENCE + 'JSON\n{"checks": []}\n' + FENCE,
    ],
)
def test_fenced_json_is_parsed(raw):
    assert parse_verdict(raw) == {"checks": []}


def test_part_field_is_accepted():
    raw = (
        '{"checks": [{"field": "part", "answer_quote": null, '
        '"context_quote": "3. Работодатель", "comparison": "missing_in_answer", '
        '"reason": "r"}]}'
    )
    assert parse_verdict(raw)["checks"][0]["field"] == "part"
