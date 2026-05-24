"""
Расширенные метрики для оценки качества генерации ответов в RAG системе.

Метрики:
- Faithfulness: Все ли утверждения в ответе подтверждены контекстом
- Answer Relevance: Релевантен ли ответ вопросу
- Context Relevance: Релевантен ли предоставленный контекст вопросу
- Completeness: Полнота ответа
- Citation Quality: Корректность и полнота цитирования источников
"""

from __future__ import annotations

import re
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json


def clean_json_response(text: str) -> str:
    """Очищает JSON от Markdown-разметки и лишнего текста."""
    # Пытаемся найти блок кода ```json ... ``` или ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Если блоков нет, пытаемся найти просто первую и последнюю фигурные скобки
    match = re.search(r"(\{.*\})", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return text.strip()


def evaluate_faithfulness(
    question: str, context: str, answer: str, llm
) -> Dict[str, Any]:
    """
    Оценка Faithfulness (Groundedness): подтверждены ли все утверждения контекстом.

    Args:
        question: Вопрос пользователя
        context: Предоставленный контекст
        answer: Сгенерированный ответ
        llm: LLM для оценки

    Returns:
        Dict с score (0-1) и reasoning
    """
    prompt = ChatPromptTemplate.from_template(
        """Ты - строгий, но справедливый ИИ-судья. Оцени, подтвержден ли Ответ предоставленным Контекстом.

# КРИТЕРИИ:
- 1.0: Все утверждения подтверждены Контекстом. 
- 0.8-0.9: Утверждения подтверждены, допускается логический вывод (например, применение общих норм для всех работников к конкретной профессии, если в ответе это оговорено).
- 0.5-0.7: Частично подтверждено, есть небольшие неточности или недосказанности.
- 0.0-0.4: Ответ содержит выдуманные факты, ссылки на несуществующие пункты или грубые ошибки.

# ИНСТРУКЦИЯ:
Верни JSON с ключами:
- "score": число от 0.0 до 1.0
- "reasoning": краткое объяснение (какие утверждения подтверждены, какие нет)

# ДАННЫЕ:
Вопрос: {question}

Контекст:
{context}

Ответ для проверки:
{answer}

JSON:"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"question": question, "context": context, "answer": answer}
    )

    try:
        cleaned_response = clean_json_response(response)
        result = json.loads(cleaned_response)
        return {
            "faithfulness_score": float(result.get("score", 0.0)),
            "faithfulness_reasoning": result.get("reasoning", ""),
            "ungrounded_statements": result.get("ungrounded_statements", []),
        }
    except Exception as e:
        print(
            f"  ⚠️  Faithfulness JSON parse error: {e}. Raw response: {response[:100]}..."
        )
        return {
            "faithfulness_score": 0.0,
            "faithfulness_reasoning": f"Ошибка парсинга: {response}",
            "ungrounded_statements": [],
        }


def evaluate_answer_relevance(question: str, answer: str, llm) -> Dict[str, Any]:
    """
    Оценка релевантности ответа вопросу.

    Args:
        question: Вопрос пользователя
        answer: Сгенерированный ответ
        llm: LLM для оценки

    Returns:
        Dict с score и reasoning
    """
    prompt = ChatPromptTemplate.from_template(
        """Оцени, насколько Ответ релевантен Вопросу.

# КРИТЕРИИ:
- 1.0: Ответ полностью отвечает на вопрос
- 0.7-0.9: Ответ в целом релевантен, но есть лишняя информация
- 0.4-0.6: Ответ частично релевантен
- 0.1-0.3: Ответ слабо связан с вопросом
- 0.0: Ответ не релевантен вопросу

Верни JSON: {{"score": 0.0-1.0, "reasoning": "..."}}

Вопрос: {question}
Ответ: {answer}

JSON:"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question, "answer": answer})

    try:
        cleaned_response = clean_json_response(response)
        result = json.loads(cleaned_response)
        return {
            "answer_relevance_score": float(result.get("score", 0.0)),
            "answer_relevance_reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        print(
            f"  ⚠️  Relevance JSON parse error: {e}. Raw response: {response[:100]}..."
        )
        return {
            "answer_relevance_score": 0.0,
            "answer_relevance_reasoning": f"Ошибка парсинга: {response}",
        }


def evaluate_context_relevance(question: str, context: str, llm) -> Dict[str, Any]:
    """
    Оценка релевантности предоставленного контекста вопросу.

    Args:
        question: Вопрос пользователя
        context: Предоставленный контекст
        llm: LLM для оценки

    Returns:
        Dict с score и reasoning
    """
    prompt = ChatPromptTemplate.from_template(
        """Оцени, насколько Контекст релевантен для ответа на Вопрос.

# КРИТЕРИИ:
- 1.0: Контекст содержит всю нужную информацию для ответа
- 0.7-0.9: Контекст в основном релевантен, возможно есть немного лишнего
- 0.4-0.6: Контекст частично релевантен
- 0.1-0.3: Контекст слабо связан с вопросом
- 0.0: Контекст не релевантен

Верни JSON: {{"score": 0.0-1.0, "reasoning": "...", "relevant_sentences": ["..."]}}

Вопрос: {question}
Контекст: {context}

JSON:"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question, "context": context})

    try:
        cleaned_response = clean_json_response(response)
        result = json.loads(cleaned_response)
        return {
            "context_relevance_score": float(result.get("score", 0.0)),
            "context_relevance_reasoning": result.get("reasoning", ""),
            "relevant_sentences": result.get("relevant_sentences", []),
        }
    except Exception as e:
        print(
            f"  ⚠️  Context relevance JSON parse error: {e}. Raw response: {response[:100]}..."
        )
        return {
            "context_relevance_score": 0.0,
            "context_relevance_reasoning": f"Ошибка парсинга: {response}",
            "relevant_sentences": [],
        }


def evaluate_completeness(
    question: str, answer: str, reference_answer: str, llm
) -> Dict[str, Any]:
    """
    Оценка полноты ответа по сравнению с эталонным.

    Args:
        question: Вопрос
        answer: Сгенерированный ответ
        reference_answer: Эталонный ответ
        llm: LLM для оценки

    Returns:
        Dict с score и reasoning
    """
    prompt = ChatPromptTemplate.from_template(
        """Оцени, насколько полон Ответ Модели по сравнению с Эталонным Ответом.

# КРИТЕРИИ:
- 1.0: Все ключевые пункты из эталона присутствуют
- 0.7-0.9: Большинство пунктов присутствует
- 0.4-0.6: Присутствует примерно половина информации
- 0.1-0.3: Большая часть информации упущена
- 0.0: Критическая информация отсутствует

Верни JSON: {{"score": 0.0-1.0, "reasoning": "...", "missing_points": ["..."]}}

Вопрос: {question}
Эталонный Ответ: {reference}
Ответ Модели: {answer}

JSON:"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"question": question, "reference": reference_answer, "answer": answer}
    )

    try:
        cleaned_response = clean_json_response(response)
        result = json.loads(cleaned_response)
        return {
            "completeness_score": float(result.get("score", 0.0)),
            "completeness_reasoning": result.get("reasoning", ""),
            "missing_points": result.get("missing_points", []),
        }
    except Exception as e:
        print(
            f"  ⚠️  Completeness JSON parse error: {e}. Raw response: {response[:100]}..."
        )
        return {
            "completeness_score": 0.0,
            "completeness_reasoning": f"Ошибка парсинга: {response}",
            "missing_points": [],
        }


def extract_citations(text: str) -> List[str]:
    """
    Извлекает цитаты из текста. Поддерживает форматы:
    - [cite: 1, 2]
    - [Источник: Название, п. X.Y]
    - [Источник: ...]

    Args:
        text: Текст с цитатами

    Returns:
        Список строк-цитат
    """
    citations = []

    # 1. Старый формат [cite: 140, 141]
    cite_pattern = r"\[cite:\s*([0-9,\s]+)\]"
    cite_matches = re.findall(cite_pattern, text)
    for match in cite_matches:
        nums = [x.strip() for x in match.split(",") if x.strip()]
        citations.extend(nums)

    # 2. Новый экспертный формат [Источник: ...]
    source_pattern = r"\[Источник:\s*(.*?)\]"
    source_matches = re.findall(source_pattern, text)
    for match in source_matches:
        citations.append(match.strip())

    return citations


def evaluate_citation_quality(
    answer: str, context: str, source_docs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Оценка качества цитирования источников.
    """
    citations = extract_citations(answer)
    unique_citations = list(set(citations))
    has_citations = len(citations) > 0

    # Если есть хотя бы одна цитата в новом или старом формате - это уже успех для базовой метрики
    return {
        "has_citations": has_citations,
        "citation_count": len(citations),
        "unique_citation_count": len(unique_citations),
        "citation_diversity": (
            len(unique_citations) / len(citations) if citations else 0.0
        ),
        "citations": unique_citations,
    }


def evaluate_generation_comprehensive(
    question: str,
    answer: str,
    context: str,
    reference_answer: str | None,
    source_docs: List[Dict[str, Any]],
    llm,
) -> Dict[str, Any]:
    """
    Комплексная оценка качества генерации.

    Args:
        question: Вопрос
        answer: Сгенерированный ответ
        context: Контекст
        reference_answer: Эталонный ответ (опционально)
        source_docs: Список source документов
        llm: LLM для оценки

    Returns:
        Dict со всеми метриками
    """
    metrics = {}

    # Faithfulness
    metrics.update(evaluate_faithfulness(question, context, answer, llm))

    # Answer Relevance
    metrics.update(evaluate_answer_relevance(question, answer, llm))

    # Context Relevance
    metrics.update(evaluate_context_relevance(question, context, llm))

    # Completeness (если есть эталон)
    if reference_answer:
        metrics.update(evaluate_completeness(question, answer, reference_answer, llm))

    # Citation Quality
    metrics.update(evaluate_citation_quality(answer, context, source_docs))

    # Дополнительные простые метрики
    metrics["answer_length"] = len(answer)
    metrics["answer_word_count"] = len(answer.split())

    return metrics


# Пример использования
if __name__ == "__main__":
    from src.infra.llm_factory import get_llm

    llm = get_llm()

    # Пример данных
    question = "Кто проходит обучение по программе А?"
    context = "По программе А обучаются работодатели, руководители, специалисты по охране труда."
    answer = "По программе А обучаются работодатели, руководители и специалисты по охране труда [cite: 140, 141]."
    reference = "По программе А обучаются: работодатели (руководители), заместители руководителя по охране труда."

    # Оценка
    metrics = evaluate_generation_comprehensive(
        question=question,
        answer=answer,
        context=context,
        reference_answer=reference,
        source_docs=[{"metadata": {"chunk_id": 140}}, {"metadata": {"chunk_id": 141}}],
        llm=llm,
    )

    print("Результаты оценки генерации:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
