"""V7 node: evaluate_complex — hard gates only, merge all attempts."""

from __future__ import annotations

from typing import cast

import structlog

from src.v7.config import v7_config
from src.v7.hard_gates import check_hard_gates, make_sufficiency
from src.v7.nlp_core import merge_all_passages
from src.v7.state_types import NextAfterEvalComplex, RAGState, RetrievalPlan

logger = structlog.get_logger()


def evaluate_complex(state: RAGState) -> RAGState:
    """Final check. Hard gates only, no triage.

    Order: merged passages → last attempt → fallback → abstain.
    """
    attempts = state.get("retrieval_attempts") or []
    if not attempts:
        return {"sufficient": False}

    last = attempts[-1]
    plan = cast(RetrievalPlan, last.get("attempt_plan") or state.get("plan", {}))
    original_q = state.get("query", "")
    active_q = state.get("active_query", original_q)

    total_passages = sum(len(a.get("passages", [])) for a in attempts)
    logger.info(
        "evaluate_complex.enter", attempts=len(attempts), total_passages=total_passages
    )

    # 1. Merge passages from all attempts
    merged = merge_all_passages(
        attempts,
        top_k=v7_config.FINAL_MERGE_TOP_K,
        mmr_lambda=plan.get("mmr_lambda"),
    )
    logger.info("evaluate_complex.merged", merged=len(merged))
    if merged:
        hard_m = check_hard_gates(original_q, active_q, merged, plan)
        if hard_m["sufficient"]:
            return {
                "sufficient": True,
                "final_passages": merged,
                "final_score": hard_m["top_score"],
                "sufficiency_details": make_sufficiency(hard_m, merged),
            }

    # 2. Last attempt only
    passages = last.get("passages", [])
    hard = check_hard_gates(original_q, active_q, passages, plan)
    if hard["sufficient"]:
        return {
            "sufficient": True,
            "final_passages": passages,
            "final_score": hard["top_score"],
            "sufficiency_details": make_sufficiency(hard, passages),
        }

    # 3. Fallback (fast-path): only accept if fallback passages actually pass hard gates.
    # Without this check, OOS queries whose rag_simple fallback had non-zero score
    # would slip through as sufficient even when kw_overlap=0.
    # Use the simple-attempt plan for gate evaluation: fallback was produced on
    # simple-path thresholds (min_passages=5, keyword_overlap≥0.15), so checking
    # it against complex-plan thresholds (min_passages=8, keyword_overlap≥0.20)
    # would incorrectly reject a valid simple result.
    fallback = state.get("fallback_passages")
    fallback_score = state.get("fallback_score", 0.0)
    if fallback and fallback_score > 0:
        simple_plan = next(
            (a["attempt_plan"] for a in attempts if a.get("stage") == "simple"),
            plan,  # fall back to complex plan if no simple attempt found
        )
        fb_hard = check_hard_gates(original_q, active_q, fallback, simple_plan)
        if fb_hard["sufficient"]:
            return {
                "sufficient": True,
                "final_passages": fallback,
                "final_score": fallback_score,
                "sufficiency_details": make_sufficiency(fb_hard, fallback),
            }

    # 4. Full failure
    return {
        "sufficient": False,
        "sufficiency_details": make_sufficiency(hard, passages, triage="clearly_bad"),
    }


def route_after_eval_complex(state: RAGState) -> NextAfterEvalComplex:
    return "end" if state.get("sufficient") else "abstain"
