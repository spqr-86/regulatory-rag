import datetime
import os

from dotenv import load_dotenv
from langchain_classic.smith import RunEvalConfig
from langsmith import Client

from src.custom_evaluators import check_correctness
from src.advanced_generation_metrics import (
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_citation_quality,
)

# --- Импортируем цепочки кандидаты ---
from src.llm_factory import get_llm
from src.ultimate_chain import create_ultimate_chain

# Убираем предупреждения о параллелизме
os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()


# Указываем наш основной датасет для тестов (Название датасета в LangSmith)
# DATASET_NAME = "safety-docs-qa"  #
DATASET_NAME = "golden-questions"


def faithfulness_evaluator(run, example):
    """
    Evaluator для проверки faithfulness (нет ли галлюцинаций).
    """
    question = example.inputs.get("question", "")
    prediction = None
    context = ""

    if run.outputs:
        if isinstance(run.outputs, dict):
            prediction = run.outputs.get("output", "")
            context = run.outputs.get("context", "")

    if not prediction:
        return {"score": 0.0, "comment": "Не удалось извлечь ответ модели"}

    judge_llm = get_llm()
    try:
        result = evaluate_faithfulness(question, context, prediction, judge_llm)
        return {
            "score": result.get("faithfulness_score", 0.0),
            "comment": result.get("faithfulness_reasoning", ""),
        }
    except Exception as e:
        return {"score": 0.0, "comment": f"Ошибка faithfulness eval: {str(e)}"}


def answer_relevance_evaluator(run, example):
    """
    Evaluator для проверки релевантности ответа вопросу.
    """
    question = example.inputs.get("question", "")
    prediction = None

    if run.outputs:
        if isinstance(run.outputs, dict):
            prediction = run.outputs.get("output", "")

    if not prediction:
        return {"score": 0.0, "comment": "Не удалось извлечь ответ модели"}

    judge_llm = get_llm()
    try:
        result = evaluate_answer_relevance(question, prediction, judge_llm)
        return {
            "score": result.get("answer_relevance_score", 0.0),
            "comment": result.get("answer_relevance_reasoning", ""),
        }
    except Exception as e:
        return {"score": 0.0, "comment": f"Ошибка relevance eval: {str(e)}"}


def citation_quality_evaluator(run, example):
    """
    Evaluator для проверки качества цитирования.
    """
    prediction = None
    context = ""

    if run.outputs:
        if isinstance(run.outputs, dict):
            prediction = run.outputs.get("output", "")
            context = run.outputs.get("context", "")

    if not prediction:
        return {"score": 0.0, "comment": "Не удалось извлечь ответ модели"}

    try:
        result = evaluate_citation_quality(prediction, context, [])
        # Оценка: 1.0 если есть цитаты, иначе 0.0
        score = 1.0 if result.get("has_citations", False) else 0.0
        comment = f"Цитат: {result.get('citation_count', 0)}, уникальных: {result.get('unique_citation_count', 0)}"
        return {"score": score, "comment": comment}
    except Exception as e:
        return {"score": 0.0, "comment": f"Ошибка citation eval: {str(e)}"}


def main():
    """
    Основная функция для запуска A/B теста.
    """
    client = Client()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    judge_llm = get_llm()

    # Просим LangSmith автоматически оценить каждый ответ по расширенному набору критериев.
    evaluation_config = RunEvalConfig(
        custom_evaluators=[
            check_correctness,  # Correctness (0-10)
            faithfulness_evaluator,  # Faithfulness (0-1)
            answer_relevance_evaluator,  # Answer Relevance (0-1)
            citation_quality_evaluator,  # Citation Quality (0-1)
        ],
        evaluators=[
            # Судья №1: Проверяет ответ на корректность по сравнению с эталоном (Correctness)
            # Задает вопрос: "Совпадает ли ответ по смыслу с эталонным ответом из датасета?"
            "qa",
            # Судья №2: Проверяет ответ на галлюцинации (Faithfulness / Groundedness)
            # Задает вопрос: "Основан ли ответ строго на предоставленном контексте?"
            "context_qa",
        ],
        eval_llm=judge_llm,
    )

    # final_chain_for_test, _ = create_final_rag_chain()

    # Словарь с "кандидатами" на тестирование
    candidate_chains = {
        "ultimate_rag": create_ultimate_chain(),
    }

    print(f"В качестве LLM-судьи будет использоваться: {judge_llm.__class__.__name__}")

    print(
        f"Запускаем A/B тест для {len(candidate_chains)} моделей на датасете '{DATASET_NAME}'..."
    )

    for chain_name, chain in candidate_chains.items():
        print(f"--- Тестируем модель: {chain_name} ---")
        try:
            project_name = f"test-{chain_name}-{timestamp}"
            # Запускаем тест для каждой цепочки
            client.run_on_dataset(
                dataset_name=DATASET_NAME,
                llm_or_chain_factory=lambda: chain,  # Фабрика для создания объекта цепочки
                project_name=project_name,  # Создаем отдельный проект для каждого теста
                evaluation=evaluation_config,  # <-- ПЕРЕДАЕМ "СУДЕЙ"
                concurrency_level=1,  # Выполняем по одному, чтобы избежать проблем с API
            )
            print(
                f"Тестирование для '{chain_name}' успешно запущено. Результаты скоро появятся в LangSmith."
            )
        except Exception as e:
            print(f"Ошибка при тестировании '{chain_name}': {e}")


if __name__ == "__main__":
    main()
