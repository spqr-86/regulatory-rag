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
