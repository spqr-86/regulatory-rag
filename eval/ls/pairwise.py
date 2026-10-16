"""Compare stored experiments: python -m eval.ls.pairwise <exp_a> <exp_b>.

The CLI is paid: each pair calls the correctness judge's provider/model.
Importing this module and testing with a fake LLM require no network.
"""

# ANCHOR: JSON winner A/B/TIE → preference scores keyed by run UUID.
# LangSmith supplies randomized runs/example; invalid judge replies produce
# an empty scores map and a diagnostic comment, never a fabricated preference.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from eval.ls.judge import _reply_text

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "judge_pairwise.md"


class PairwiseJudgeVariables(BaseModel):
    question: str
    legal_as_of: str
    ground_truth: str
    ground_truth_extra: str
    forbidden_claims: str
    answer_a: str
    answer_b: str


def parse_preference(raw: str) -> int:
    """JSON winner A/B/TIE (case insensitive, optional JSON fence) → 0/1/-1."""
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.partition("\n")[2].removesuffix("```")
        data = json.loads(raw)
        winner = data.get("winner") if isinstance(data, dict) else None
        if not isinstance(winner, str) or winner.upper() not in ("A", "B", "TIE"):
            raise ValueError("'winner' must be A, B or TIE")
        return {"A": 0, "B": 1, "TIE": -1}[winner.upper()]
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise ValueError("Judge returned invalid JSON") from exc


def make_pairwise_evaluator(llm) -> Callable:
    """Build a LangSmith comparative evaluator; run order determines A and B."""
    from langchain_core.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_template(PROMPT_PATH.read_text(encoding="utf-8"))

    def preference(runs, example) -> dict:
        if len(runs) != 2:
            raise ValueError("Pairwise evaluation requires exactly two runs")
        result = {"key": "preference", "scores": {}}
        if any(run.error or run.outputs is None for run in runs):
            return {**result, "comment": "target_error"}
        reference = example.outputs or {}
        variables = PairwiseJudgeVariables(
            question=example.inputs["question"],
            legal_as_of=reference.get("legal_as_of") or "",
            ground_truth=reference.get("ground_truth") or "",
            ground_truth_extra=reference.get("ground_truth_extra") or "",
            forbidden_claims=reference.get("forbidden_claims") or "",
            answer_a=runs[0].outputs.get("answer") or "",
            answer_b=runs[1].outputs.get("answer") or "",
        )
        reply = llm.invoke(template.format_messages(**variables.model_dump()))
        try:
            winner = parse_preference(_reply_text(reply))
        except ValueError as exc:
            return {**result, "comment": f"judge_format_error: {exc}"}
        scores = (0.5, 0.5) if winner == -1 else ((1, 0) if winner == 0 else (0, 1))
        return {**result, "scores": {run.id: score for run, score in zip(runs, scores)}}

    return preference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise judge for two stored experiments"
    )
    parser.add_argument("exp_a")
    parser.add_argument("exp_b")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from langsmith.evaluation import evaluate_comparative

    from src.infra.llm_factory import get_judge_llm

    load_dotenv()
    llm = get_judge_llm(temperature=0.0, seed=12345)
    # The shared factory defaults to retries; disable them on this judge's client
    # as in the citation judge, without changing other judges or the pipeline.
    llm.root_client = llm.root_client.with_options(max_retries=0, timeout=300)
    llm.client = llm.root_client.chat.completions
    evaluate_comparative(
        (args.exp_a, args.exp_b),
        evaluators=[make_pairwise_evaluator(llm)],
        randomize_order=True,
        max_concurrency=1,
    )


if __name__ == "__main__":
    main()
