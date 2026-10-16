"""Step 2: fresh verified golden cases through the full V7 graph on the active snapshot,
then the citation judge on DeepSeek with reasoning (effort from settings).

Combines rerun_judge_cases.py (context capture) and judge_thinking.py (judge setup).
Judge model is passed explicitly: .env JUDGE_MODEL_NAME also drives the golden-eval judge.
Run from the repo root: python eval/runs/citation_judge_fresh_2026-10-12/fresh_cases.py [ids]
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

import src.v7.bridge as bridge  # noqa: E402
from eval.judges.citations import judge_citations  # noqa: E402
from src.backends.vector_store import get_vector_store_backend  # noqa: E402
from src.infra.llm_factory import get_judge_llm  # noqa: E402
from src.v7.graph import build_graph  # noqa: E402

OUT = Path("eval/runs/citation_judge_fresh_2026-10-12")
CASES = sys.argv[1:] or ["G002", "G008", "G009", "G010", "G013", "G015", "G022", "G028"]
MODEL = "deepseek/deepseek-v4.1-flash"
# Reasoning counts toward max_tokens. Uncapped, G002 looped to 131072 reasoning tokens
# (882 s, empty answer); step 1 peaked at ~5k.
JUDGE_MAX_TOKENS = 16000

captured = {}
_orig_render = bridge._pm.render


def _render(name, **kw):
    if name == "generate_answer":
        captured["context"] = kw.get("context", "")
    return _orig_render(name, **kw)


bridge._pm.render = _render


class RecordingLLM:
    def __init__(self, llm):
        self.llm = llm
        self.reply = None

    def invoke(self, messages):
        self.reply = self.llm.invoke(messages)
        return self.reply


rows = {
    r["case_id"]: r for r in csv.DictReader(open("tests/dataset.csv", encoding="utf-8"))
}
runtime = bridge.build_full_v7_runtime(get_vector_store_backend(load_existing=True))
graph = build_graph(runtime=runtime).compile()

llm = get_judge_llm(model_name=MODEL, max_tokens=JUDGE_MAX_TOKENS)
# langchain sends it as max_completion_tokens; also send OpenRouter's native max_tokens.
llm.extra_body = {**(llm.extra_body or {}), "max_tokens": JUDGE_MAX_TOKENS}
# Factory hardcodes max_retries=3: a timed-out reasoning call is billed and retried.
llm.root_client = llm.root_client.with_options(max_retries=0, timeout=300)
llm.client = llm.root_client.chat.completions
judge = RecordingLLM(llm)

for cid in CASES:
    row = rows[cid]
    q = row["question"]
    captured.clear()
    t0 = time.monotonic()
    state = graph.invoke({"query": q, "filters": None})
    rec = {
        "case_id": cid,
        "question": q,
        "ground_truth": row["ground_truth"],
        "review_note": row["review_note"],
        "response": state.get("answer", ""),
        "context": captured.get("context", ""),
        "route_reason": state.get("route_reason"),
        "n_passages": len(state.get("final_passages") or []),
        "rag_seconds": round(time.monotonic() - t0, 1),
        "judge_model": MODEL,
    }
    judge.reply = None
    t1 = time.monotonic()
    try:
        rec["verdict"] = judge_citations(judge, q, rec["response"], rec["context"])
    except Exception as exc:  # judge format errors are results too
        rec["judge_error"] = f"{type(exc).__name__}: {exc}"
    rec["judge_seconds"] = round(time.monotonic() - t1, 1)
    if judge.reply is not None:
        rec["raw"] = judge.reply.content
        rec["usage"] = getattr(judge.reply, "usage_metadata", None)
        rec["finish_reason"] = (judge.reply.response_metadata or {}).get(
            "finish_reason"
        )
    (OUT / f"{cid}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        cid,
        rec["route_reason"],
        rec["rag_seconds"],
        rec.get("verdict", {}).get("status"),
        rec["judge_seconds"],
        rec.get("finish_reason"),
        rec.get("usage"),
        rec.get("judge_error", ""),
        flush=True,
    )
