from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.infra.llm_factory import get_llm


def check_correctness(run, example):
    """
    LangSmith-style correctness evaluator.
    """
    print("--- Running final evaluator ---")

    # 1. Extract data from 'example' (ground truth)
    input_question = example.inputs.get("question")
    reference_answer = example.outputs.get("ground_truth")

    # 2. Safely extract model answer from 'run'
    prediction = None
    if run.outputs:
        if isinstance(run.outputs, dict):
            prediction = run.outputs.get("output")

    # 3. Check that we could extract an answer
    if not prediction:
        print("EVALUATOR ERROR: Failed to extract model answer from run.outputs.")
        return {
            "score": 0,
            "comment": "Failed to extract model answer from run.outputs.",
        }

    # 4. Вызываем LLM-судью для оценки
    judge_llm = get_llm()
    evaluator_prompt = ChatPromptTemplate.from_template(
        """Вы - беспристрастный ИИ-судья. Ваша задача - оценить, насколько "Ответ Модели" соответствует "Эталонному Ответу" по шкале от 0 до 10.

# КРИТЕРИИ:
- 10: Полное смысловое совпадение.
- 7-9: В целом правильно, но есть небольшие стилистические отличия.
- 4-6: Частично правильно, но упущена важная информация.
- 1-3: Ответ нерелевантен или фактически неверен.
- 0: Полное несоответствие.

# ИНСТРУКЦИЯ ПО ФОРМАТУ ВЫВОДА:
Твой ответ ДОЛЖЕН быть только в формате JSON с двумя ключами: "score" (число от 0 до 10) и "reasoning" (короткое текстовое объяснение твоей оценки).
Пример:
{{"score": 8, "reasoning": "Ответ правильный по сути, но немного менее детальный, чем эталон."}}

# ДАННЫЕ ДЛЯ ОЦЕНКИ:
Вопрос: {question}
Эталонный Ответ: {reference}
Ответ Модели: {prediction}
"""
    )
    evaluation_chain = evaluator_prompt | judge_llm | StrOutputParser()
    response_str = evaluation_chain.invoke(
        {
            "question": input_question,
            "reference": reference_answer,
            "prediction": prediction,
        }
    )

    # 5. Parse result (strip markdown wrapper if present)
    try:
        # Strip markdown code block wrapper if present
        clean_response = response_str.strip()
        if clean_response.startswith("```"):
            # Remove ```json or ``` at start and ``` at end
            lines = clean_response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]  # remove opening ``` or ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # remove closing ```
            clean_response = "\n".join(lines)

        result = json.loads(clean_response)
        print(
            f"Score received: {result.get('score')}, Reasoning: '{result.get('reasoning')}'"
        )
        return {"score": result.get("score"), "comment": result.get("reasoning")}
    except (json.JSONDecodeError, AttributeError, TypeError):
        print(
            f"EVALUATOR ERROR: Failed to parse JSON from judge. Judge response: {response_str}"
        )
        return {
            "score": 0,
            "comment": f"Failed to parse judge response. Response: {response_str}",
        }
