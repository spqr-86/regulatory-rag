"""Minimal eval: gosts_pipeline (baseline, DeepSeek) vs ers_rag (new, via /query/gosts).

Метрики (deterministic, без LLM-judge):
  - exact_match: ключевая фраза/число из gold_answer присутствует в answer (lowercase substring)
  - chunk_hint_hit: chunk_hint встречается в metadata/source/text топ-N passages
  - source_doc_hit: source (например "ГОСТ 32601-2022") встречается в text passages
  - answered: непустой ответ, не abstain
  - elapsed_sec
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLD = Path(__file__).parent / "gosts_gold_v2.json"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)
ERS_URL = "http://localhost:8503/query/gosts"


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def extract_key_tokens(gold: str) -> list[str]:
    """Из gold_answer выделяем числа+единицы и значимые токены для substring-match."""
    s = gold.lower()
    nums = re.findall(r"\d+[.,]?\d*\s*(?:мм|м|%|мин|минут|дюйм|дюймов|°c|мпа)?", s)
    nums = [n.strip() for n in nums if n.strip()]
    # plus standalone numbers
    plain_nums = re.findall(r"\b\d+[.,]?\d*\b", s)
    return list(set(nums + plain_nums))


def exact_match_score(gold: str, answer: str) -> float:
    """Доля ключевых числовых токенов из gold, найденных в answer."""
    tokens = extract_key_tokens(gold)
    if not tokens:
        # fallback: substring проверка первых 30 символов gold
        snippet = normalize(gold)[:30]
        return 1.0 if snippet in normalize(answer) else 0.0
    ans = normalize(answer)
    hits = sum(1 for t in tokens if t in ans)
    return hits / len(tokens)


def chunk_hint_in_passages(hint: str, passages: list[dict]) -> bool:
    if not hint or not passages:
        return False
    h = hint.lower()
    for p in passages:
        text = (p.get("text") or p.get("content") or "").lower()
        meta = json.dumps(p.get("metadata") or {}, ensure_ascii=False).lower()
        if h in text or h in meta:
            return True
    return False


def source_doc_in_passages(source: str, passages: list[dict]) -> bool:
    if not source or not passages:
        return False
    # Extract GOST number e.g. "32601"
    nums = re.findall(r"\d{4,6}", source)
    if not nums:
        return False
    for p in passages:
        text = (p.get("text") or p.get("content") or "").lower()
        meta = json.dumps(p.get("metadata") or {}, ensure_ascii=False).lower()
        blob = text + " " + meta
        if any(n in blob for n in nums):
            return True
    return False


def run_baseline(question: str) -> dict:
    from src.gosts_pipeline import query as gp_query

    t0 = time.perf_counter()
    try:
        r = gp_query(question)
        return {
            "answer": r.get("answer", ""),
            "passages": r.get("passages", []),
            "elapsed_sec": r.get("elapsed_sec", round(time.perf_counter() - t0, 2)),
            "error": None,
        }
    except Exception as e:
        return {"answer": "", "passages": [], "elapsed_sec": 0.0, "error": str(e)}


def run_ers_rag(question: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = requests.post(ERS_URL, json={"question": question}, timeout=180)
        resp.raise_for_status()
        d = resp.json()
        return {
            "answer": d.get("answer", ""),
            "passages": d.get("passages", []),
            "elapsed_sec": d.get("elapsed_sec", round(time.perf_counter() - t0, 2)),
            "error": None,
        }
    except Exception as e:
        return {"answer": "", "passages": [], "elapsed_sec": 0.0, "error": str(e)}


def score(item: dict, run: dict) -> dict:
    answer = run.get("answer") or ""
    passages = run.get("passages") or []
    return {
        "answered": bool(answer.strip()) and "уточните" not in answer.lower()[:80],
        "exact_match": round(exact_match_score(item["gold_answer"], answer), 3),
        "chunk_hint_hit": chunk_hint_in_passages(item.get("chunk_hint", ""), passages),
        "source_doc_hit": source_doc_in_passages(item.get("source", ""), passages),
        "elapsed_sec": run.get("elapsed_sec", 0.0),
        "passage_count": len(passages),
        "error": run.get("error"),
    }


def aggregate(records: list[dict], key: str) -> dict:
    n = len(records)
    if n == 0:
        return {}
    scored = [r[key] for r in records]
    return {
        "answered_rate": round(sum(1 for s in scored if s["answered"]) / n, 3),
        "exact_match_mean": round(sum(s["exact_match"] for s in scored) / n, 3),
        "chunk_hint_hit_rate": round(
            sum(1 for s in scored if s["chunk_hint_hit"]) / n, 3
        ),
        "source_doc_hit_rate": round(
            sum(1 for s in scored if s["source_doc_hit"]) / n, 3
        ),
        "avg_elapsed_sec": round(sum(s["elapsed_sec"] for s in scored) / n, 2),
        "avg_passage_count": round(sum(s["passage_count"] for s in scored) / n, 1),
        "errors": sum(1 for s in scored if s["error"]),
    }


def main() -> None:
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    print(f"Loaded {len(gold)} gold questions from {GOLD.name}\n")

    records = []
    for i, item in enumerate(gold, 1):
        q = item["question"]
        print(f"[{i}/{len(gold)}] {q[:80]}...")
        baseline = run_baseline(q)
        print(
            f"   baseline: {baseline['elapsed_sec']}s | passages={len(baseline['passages'])}"
        )
        ers = run_ers_rag(q)
        print(f"   ers_rag:  {ers['elapsed_sec']}s | passages={len(ers['passages'])}")
        rec = {
            "question": q,
            "gold_answer": item["gold_answer"],
            "source": item.get("source", ""),
            "chunk_hint": item.get("chunk_hint", ""),
            "baseline": {**baseline, "score": score(item, baseline)},
            "ers_rag": {**ers, "score": score(item, ers)},
        }
        # promote scores
        rec["baseline_score"] = rec["baseline"].pop("score")
        rec["ers_rag_score"] = rec["ers_rag"].pop("score")
        records.append(rec)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "gold": str(GOLD),
        "n": len(records),
        "baseline_gosts_pipeline": aggregate(records, "baseline_score"),
        "ers_rag_via_endpoint": aggregate(records, "ers_rag_score"),
        "results": records,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_DIR / f"ers_rag_vs_gosts_pipeline_{ts}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {out}")

    # Print compare table
    b = summary["baseline_gosts_pipeline"]
    e = summary["ers_rag_via_endpoint"]
    print(f"\n{'Metric':<25} {'baseline':>12} {'ers_rag':>12} {'delta':>10}")
    print("-" * 62)
    for k in [
        "answered_rate",
        "exact_match_mean",
        "chunk_hint_hit_rate",
        "source_doc_hit_rate",
        "avg_elapsed_sec",
        "avg_passage_count",
        "errors",
    ]:
        bv = b.get(k, 0)
        ev = e.get(k, 0)
        delta = round(ev - bv, 3)
        sign = "+" if delta > 0 else ""
        print(f"{k:<25} {bv:>12} {ev:>12} {sign}{delta:>9}")


if __name__ == "__main__":
    main()
