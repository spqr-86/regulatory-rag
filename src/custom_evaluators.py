from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .llm_factory import get_llm


def check_correctness(run, example):
    """
    Оценщика для проверки корректности.
    """
    print("--- Запуск финального оценщика ---")

    # 1. Получаем данные из 'example' (эталон)
    input_question = example.inputs.get("question")
    reference_answer = example.outputs.get("ground_truth")

    # 2. Надежно извлекаем ответ модели из 'run'
    prediction = None
    if run.outputs:
        if isinstance(run.outputs, dict):
            prediction = run.outputs.get("output")

    # 3. Проверяем, удалось ли извлечь ответ
    if not prediction:
        print("ОШИБКА ОЦЕНЩИКА: Не удалось извлечь ответ модели из run.outputs.")
        return {
            "score": 0,
            "comment": "Не удалось извлечь ответ модели из run.outputs.",
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

    # 5. Парсим результат (убираем markdown обёртку если есть)
    try:
        # Strip markdown code block wrapper if present
        clean_response = response_str.strip()
        if clean_response.startswith("```"):
            # Remove ```json or ``` at start and ``` at end
            lines = clean_response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]  # Remove first line with ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]  # Remove last line with ```
            clean_response = "\n".join(lines)

        result = json.loads(clean_response)
        print(
            f"Оценка получена: Score - {result.get('score')}, Reasoning - '{result.get('reasoning')}'"
        )
        return {"score": result.get("score"), "comment": result.get("reasoning")}
    except (json.JSONDecodeError, AttributeError, TypeError):
        print(
            f"ОШИБКА ОЦЕНЩИКА: Не удалось распарсить JSON от судьи. Ответ судьи: {response_str}"
        )
        return {
            "score": 0,
            "comment": f"Ошибка парсинга ответа от судьи. Ответ: {response_str}",
        }
