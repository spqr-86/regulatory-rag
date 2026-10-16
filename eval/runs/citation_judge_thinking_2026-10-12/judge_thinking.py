"""Judge saved G001/G006/G019/G020 cases with DeepSeek reasoning on (effort from settings).

Model is passed explicitly: .env JUDGE_MODEL_NAME also drives the golden-eval judge.
Saves raw reply, latency and usage next to the verdict.
Run from the repo root: python eval/runs/citation_judge_thinking_2026-10-12/judge_thinking.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from eval.judges.citations import judge_citations  # noqa: E402
from src.infra.llm_factory import get_judge_llm  # noqa: E402

SRC = Path("eval/runs/citation_judge_rerun_2026-10-12")
OUT = Path("eval/runs/citation_judge_thinking_2026-10-12")
CASES = sys.argv[1:] or ["G001", "G006", "G019", "G020"]
MODEL = "deepseek/deepseek-v4.1-flash"


class RecordingLLM:
    def __init__(self, llm):
        self.llm = llm
        self.reply = None

    def invoke(self, messages):
        self.reply = self.llm.invoke(messages)
        return self.reply


llm = get_judge_llm(model_name=MODEL)
# Factory hardcodes max_retries=3: a timed-out reasoning call is billed and retried.
llm.root_client = llm.root_client.with_options(max_retries=0, timeout=300)
llm.client = llm.root_client.chat.completions
judge = RecordingLLM(llm)
for cid in CASES:
    case = json.loads((SRC / f"{cid}.json").read_text(encoding="utf-8"))
    rec = {"case_id": cid, "model": MODEL}
    judge.reply = None
    t0 = time.monotonic()
    try:
        rec["verdict"] = judge_citations(
            judge, case["question"], case["response"], case["context"]
        )
    except Exception as exc:
        rec["judge_error"] = f"{type(exc).__name__}: {exc}"
    rec["seconds"] = round(time.monotonic() - t0, 1)
    if judge.reply is not None:
        rec["raw"] = judge.reply.content
        rec["usage"] = getattr(judge.reply, "usage_metadata", None)
        rec["finish_reason"] = (judge.reply.response_metadata or {}).get("finish_reason")
    (OUT / f"{cid}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(cid, rec.get("verdict", {}).get("status"), rec["seconds"], rec.get("finish_reason"),
          rec.get("usage"), rec.get("judge_error", ""), flush=True)
