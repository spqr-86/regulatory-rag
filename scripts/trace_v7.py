"""
Диагностика трассы v7 RAG pipeline.

Показывает какой путь прошёл запрос: какие ноды, scores, triage, passages.

Использование:
    python scripts/trace_v7.py                              # вопрос по умолчанию
    python scripts/trace_v7.py "высота ограждения лестниц"
    python scripts/trace_v7.py "сравни ГОСТ и СП по лестницам"
    python scripts/trace_v7.py --no-chroma "тест без БД"   # stub mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── ANSI colors ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

DEFAULT_QUERY = "какова минимальная высота ограждения лестничных маршей?"


def _color_score(score: float) -> str:
    if score >= 0.65:
        return f"{GREEN}{score:.3f}{RESET}"
    if score >= 0.40:
        return f"{YELLOW}{score:.3f}{RESET}"
    return f"{RED}{score:.3f}{RESET}"


def _color_triage(triage: str) -> str:
    colors = {"sufficient": GREEN, "borderline": YELLOW, "clearly_bad": RED}
    c = colors.get(triage, "")
    return f"{c}{BOLD}{triage}{RESET}"


def print_section(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'─' * 60}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")


def print_plan(plan: dict) -> None:
    print_section("PLAN (from router)")
    key_params = [
        ("threshold", "hard gate similarity"),
        ("borderline_threshold", "borderline zone"),
        ("min_passages", "min evidence count"),
        ("min_keyword_overlap", "keyword overlap"),
        ("max_single_doc_ratio", "max single-doc ratio"),
        ("require_multi_doc", "multi-doc required"),
        ("mmr_lambda", "MMR lambda"),
        ("top_k", "top_k"),
    ]
    for key, label in key_params:
        val = plan.get(key, "—")
        print(f"  {label:<30} {val}")


def print_attempt(idx: int, attempt: dict) -> None:
    stage = attempt.get("stage", "?")
    passages = attempt.get("passages", [])
    top_score = attempt.get("top_score", 0.0)
    metrics = attempt.get("metrics", {})

    print_section(f"RETRIEVAL ATTEMPT #{idx + 1}  stage={stage.upper()}")
    print(f"  passages found:   {len(passages)}")
    print(f"  top score:        {_color_score(top_score)}")
    print(f"  kw overlap active:{metrics.get('keyword_overlap_active', '—')}")
    print(f"  unique docs:      {metrics.get('unique_docs', '—')}")
    print(f"  max_doc_ratio:    {metrics.get('max_doc_ratio', '—')}")

    if passages:
        print(f"\n  {DIM}Top-3 passages:{RESET}")
        for i, p in enumerate(passages[:3]):
            score = _color_score(p.get("score", 0.0))
            doc_id = p.get("metadata", {}).get("doc_id") or p.get("doc_id", "—")
            text = p.get("text", "")[:80].replace("\n", " ")
            print(f"    [{i + 1}] score={score} doc={doc_id}")
            print(f"        {DIM}{text}…{RESET}")


def print_triage(details: dict) -> None:
    print_section("TRIAGE RESULT (evaluate_triage)")
    triage = details.get("triage", "?")
    print(f"  triage:           {_color_triage(triage)}")
    print(f"  sufficient:       {details.get('sufficient')}")
    print(f"  top_score:        {_color_score(details.get('top_score', 0.0))}")
    print(f"  above_threshold:  {details.get('above_threshold')}")
    print(f"  enough_evidence:  {details.get('enough_evidence')}")
    print(f"  keyword_overlap:  {details.get('keyword_overlap_ok')}")
    print(f"  diversity_ok:     {details.get('diversity_ok')}")
    print(f"  escalation_hint:  {details.get('escalation_hint')}")
    print(f"  unique_docs:      {details.get('unique_docs', '—')}")
    print(f"  max_doc_ratio:    {details.get('max_doc_ratio', '—')}")


def print_verification(verif: dict) -> None:
    print_section("LLM VERIFIER RESULT")
    verdict = verif.get("verdict", "?")
    color = (
        GREEN if verdict == "sufficient" else (YELLOW if verdict == "rewrite" else RED)
    )
    print(f"  verdict:          {color}{BOLD}{verdict}{RESET}")
    print(f"  confidence:       {verif.get('confidence', '—')}")
    print(f"  reason:           {verif.get('reason', '—')}")
    hint = verif.get("rewrite_hint")
    if hint:
        print(f"  rewrite_hint:     {hint}")
    aspects = verif.get("missing_aspects", [])
    if aspects:
        print(f"  missing_aspects:  {', '.join(aspects)}")


def print_final(state: dict) -> None:
    print_section("FINAL STATE")
    sufficient = state.get("sufficient")
    color = GREEN if sufficient else RED
    print(f"  sufficient:       {color}{BOLD}{sufficient}{RESET}")

    final_passages = state.get("final_passages") or []
    fallback = state.get("fallback_passages") or []
    print(f"  final_passages:   {len(final_passages)}")
    if fallback:
        print(
            f"  fallback_passages:{len(fallback)} {YELLOW}(используется fallback){RESET}"
        )
    if state.get("abstain_reason"):
        print(f"  abstain_reason:   {RED}{state['abstain_reason']}{RESET}")
    if state.get("clarify_message"):
        print(f"  clarify_message:  {YELLOW}{state['clarify_message']}{RESET}")

    answer = state.get("answer")
    if answer:
        print(f"\n{BOLD}ANSWER:{RESET}")
        print(f"  {answer[:600]}{'…' if len(answer) > 600 else ''}")


def infer_path(state: dict) -> str:
    """Reconstruct the node path from state."""
    attempts = state.get("retrieval_attempts") or []
    stages = [a.get("stage") for a in attempts]

    parts = ["intent_gate → router"]

    if state.get("clarify_message"):
        return parts[0] + " → clarify_respond → END"

    parts.append("rag_simple")

    if not attempts:
        return " → ".join(parts) + " → [no retrieval]"

    details = state.get("sufficiency_details") or {}
    triage = details.get("triage", "?")

    details = state.get("sufficiency_details") or {}
    triage_after_simple = details.get("triage", "?")
    verif = state.get("verification") or {}
    verdict = verif.get("verdict")

    if "complex" not in stages and state.get("sufficient"):
        parts.append("evaluate_triage[sufficient] → generate_answer → END")
        return " → ".join(parts)

    if verdict:
        parts.append(f"evaluate_triage[borderline] → llm_verifier[{verdict}]")
    elif triage_after_simple != "?" and triage_after_simple != "sufficient":
        parts.append(f"evaluate_triage[{triage_after_simple}]")

    if "complex" in stages:
        parts.append("rag_complex → evaluate_complex")
        if state.get("sufficient"):
            parts.append("generate_answer → END")
        else:
            parts.append("abstain → END")

    return " → ".join(parts)


def trace(query: str, use_chroma: bool = True) -> None:
    print(f"\n{BOLD}QUERY:{RESET} {query}")

    from src.v7.graph import build_graph

    if use_chroma:
        from src.backends.vector_store import get_vector_store_backend
        from src.v7.bridge import init_v7_from_chroma

        print(f"{DIM}Загружаю vector store…{RESET}", end="", flush=True)
        vs = get_vector_store_backend(load_existing=True)
        init_v7_from_chroma(vs)
        print(f" {GREEN}готово{RESET}")
    else:
        print(f"{YELLOW}Stub mode (нет ChromaDB){RESET}")

    app = build_graph().compile()
    state: dict[str, Any] = app.invoke({"query": query})

    # ── Plan ──
    plan = state.get("plan") or {}
    if plan:
        print_plan(plan)

    # ── Retrieval attempts ──
    attempts = state.get("retrieval_attempts") or []
    for i, attempt in enumerate(attempts):
        print_attempt(i, attempt)

    # ── Triage ──
    triage_details = state.get("sufficiency_details")
    if triage_details:
        print_triage(triage_details)

    # ── Verifier ──
    verif = state.get("verification")
    if verif:
        print_verification(verif)

    # ── Final ──
    print_final(state)

    # ── Inferred path ──
    print_section("INFERRED PATH")
    print(f"  {BOLD}{infer_path(state)}{RESET}")

    print(f"\n{CYAN}{'─' * 60}{RESET}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace v7 RAG pipeline")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument(
        "--no-chroma",
        action="store_true",
        help="Run without ChromaDB (stub mode, для проверки логики маршрутизации)",
    )
    args = parser.parse_args()
    trace(args.query, use_chroma=not args.no_chroma)


if __name__ == "__main__":
    main()
