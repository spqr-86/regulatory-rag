"""Re-run G001/G006/G019/G020 through the full V7 graph, capture the exact
generator context, then judge each answer with judge_citations.

Run from the repo root with CHROMA_DB_PATH pointing at the snapshot.
Usage: python rerun_judge_cases.py <out_dir>
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import src.v7.bridge as bridge  # noqa: E402
from eval.judges.citations import judge_citations  # noqa: E402
from src.backends.vector_store import get_vector_store_backend  # noqa: E402
from src.infra.llm_factory import get_judge_llm  # noqa: E402
from src.v7.graph import build_graph  # noqa: E402

CASES = ["G001", "G006", "G019", "G020"]

captured = {}
_orig_render = bridge._pm.render


def _render(name, **kw):
    if name == "generate_answer":
        captured["context"] = kw.get("context", "")
    return _orig_render(name, **kw)


bridge._pm.render = _render


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        r["case_id"]: r for r in csv.DictReader(open("tests/dataset.csv", encoding="utf-8"))
    }
    runtime = bridge.build_full_v7_runtime(get_vector_store_backend(load_existing=True))
    graph = build_graph(runtime=runtime).compile()
    judge = get_judge_llm(thinking_budget=0)
    results = []
    for cid in CASES:
        q = rows[cid]["question"]
        captured.clear()
        state = graph.invoke({"query": q, "filters": None})
        answer = state.get("answer", "")
        context = captured.get("context", "")
        rec = {
            "case_id": cid,
            "question": q,
            "response": answer,
            "context": context,
            "route_reason": state.get("route_reason"),
            "n_passages": len(state.get("final_passages") or []),
        }
        try:
            rec["verdict"] = judge_citations(judge, q, answer, context)
        except Exception as exc:  # judge format errors are results too
            rec["judge_error"] = f"{type(exc).__name__}: {exc}"
        (out_dir / f"{cid}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(
            cid,
            rec.get("route_reason"),
            rec.get("verdict", {}).get("status"),
            rec.get("judge_error", ""),
        )
        results.append(rec)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
