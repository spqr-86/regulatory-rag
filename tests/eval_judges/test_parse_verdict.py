import pytest

from eval.judges.citations import parse_verdict


@pytest.mark.parametrize(
    "raw",
    ['```json\n{"checks": []}\n```', 'Вот ответ: {"checks": []}', "", None, 123],
)
def test_invalid_json_raises_value_error(raw):
    with pytest.raises(ValueError, match="Judge returned invalid JSON"):
        parse_verdict(raw)


def test_valid_empty_checks():
    assert parse_verdict('{"checks": []}') == {"checks": []}
