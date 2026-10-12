import json

from eval.judges.citations import judge_citations
from src.infra.llm_factory import get_citation_judge_llm


def main():
    question = "Какова минимальная продолжительность стажировки?"

    correct_response = (
        "Не менее двух смен согласно пункту 31 раздела III " "Правил №2464."
    )

    full_context = (
        "[Фрагмент 1]\n"
        "Документ: Правила обучения по охране труда и проверки "
        "знания требований охраны труда, утверждённые "
        "постановлением Правительства РФ №2464.\n"
        "Раздел III, пункт 31: продолжительность стажировки "
        "на рабочем месте должна составлять не менее двух смен."
    )
    cases = [
        {
            "name": "wrong_point",
            "expected": "fail",
            "response": correct_response.replace("пункту 31", "пункту 30"),
            "context": full_context,
        },
        {
            "name": "correct_point",
            "expected": "pass",
            "response": correct_response,
            "context": full_context,
        },
        {
            "name": "missing_point_in_context",
            "expected": "needs_review",
            "response": correct_response,
            "context": full_context.replace("Раздел III, пункт 31:", "Раздел III:"),
        },
    ]

    llm = get_citation_judge_llm(temperature=0.0)
    llm = llm.bind(response_format={"type": "json_object"})
    matched = 0
    for case in cases:
        verdict = judge_citations(llm, question, case["response"], case["context"])
        matches_expected = verdict["status"] == case["expected"]
        matched += matches_expected
        print(
            json.dumps(
                {
                    "case": case["name"],
                    "expected": case["expected"],
                    "actual": verdict,
                    "matches_expected": matches_expected,
                },
                ensure_ascii=False,
            )
        )

    print(f"Matched expectations: {matched}/{len(cases)}")


if __name__ == "__main__":
    main()
