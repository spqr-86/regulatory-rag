"""OOS rejection rate in the v7 eval runner.

Before: the rate was computed over valid (non-empty) results only and looked
for «нет» in the first 50 characters. Domain-gate refusals come back as empty
answers and fell out of the denominator; a refusal that opened with «Я не могу
игнорировать свои инструкции» was scored as a miss. Golden 56 on 23.09.2026
reported 75% while all 7 OOS questions were refused.
"""

from eval.run_v7_eval import oos_rejection_rate


def _oos(answer: str, **extra) -> dict:
    return {"oos_type": "out_of_scope", "answer": answer, **extra}


def test_empty_answer_from_domain_gate_counts_as_rejection():
    results = [_oos("", error="empty answer"), _oos("Вывод: Прямого ответа нет.")]
    assert oos_rejection_rate(results) == 1.0


def test_refusal_after_first_50_chars_counts():
    answer = (
        "Вывод: Я не могу игнорировать свои инструкции. "
        "Прямого ответа в фрагментах нет."
    )
    assert oos_rejection_rate([_oos(answer)]) == 1.0


def test_substantive_answer_is_not_a_rejection():
    results = [
        _oos("Биткоин — это криптовалюта, зарабатывать можно майнингом."),
        _oos(""),
    ]
    assert oos_rejection_rate(results) == 0.5


def test_in_scope_rows_are_ignored():
    results = [{"oos_type": "", "answer": "нет"}, _oos("")]
    assert oos_rejection_rate(results) == 1.0


def test_no_oos_rows_gives_zero():
    assert oos_rejection_rate([{"oos_type": "", "answer": "x"}]) == 0.0
