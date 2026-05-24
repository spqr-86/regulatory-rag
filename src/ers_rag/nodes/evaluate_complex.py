"""V7 node: evaluate_complex — hard gates only, merge all attempts."""

from __future__ import annotations

from typing import cast

from src.ers_rag.hard_gates import check_hard_gates, make_sufficiency
from src.ers_rag.nlp_core import merge_all_passages
from src.ers_rag.state_types import NextAfterEvalComplex, RAGState, RetrievalPlan


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

    # 1. Merge passages from all attempts
    merged = merge_all_passages(attempts, top_k=24)
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

    # 3. Fallback (fast-path)
    fallback = state.get("fallback_passages")
    fallback_score = state.get("fallback_score", 0.0)
    if fallback and fallback_score > 0:
        fb_hard = check_hard_gates(original_q, active_q, fallback, plan)
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
