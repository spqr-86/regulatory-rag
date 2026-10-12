"""Citation judge with optional `part` (c6adcd7) on the saved answers and contexts of
citation_judge_fresh_2026-10-12: same inputs, only the judge prompt changed.
Run from the repo root: python eval/runs/citation_judge_part_2026-10-12/judge_part.py [ids]
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from eval.judges.citations import judge_citations  # noqa: E402
from src.infra.llm_factory import get_judge_llm  # noqa: E402

SRC = Path("eval/runs/citation_judge_fresh_2026-10-12")
OUT = Path("eval/runs/citation_judge_part_2026-10-12")
CASES = sys.argv[1:] or ["G002", "G008", "G009", "G010", "G013", "G015", "G022", "G028"]
MODEL = "deepseek/deepseek-v4.1-flash"
# G022 hit 16000 reasoning tokens twice; rerun alone with JUDGE_MAX_TOKENS=32000.
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", 16000))

llm = get_judge_llm(model_name=MODEL, max_tokens=JUDGE_MAX_TOKENS)
llm.extra_body = {**(llm.extra_body or {}), "max_tokens": JUDGE_MAX_TOKENS}
llm.root_client = llm.root_client.with_options(max_retries=0, timeout=300)
llm.client = llm.root_client.chat.completions


class RecordingLLM:
    def __init__(self, llm):
        self.llm = llm
        self.reply = None

    def invoke(self, messages):
        self.reply = self.llm.invoke(messages)
        return self.reply


judge = RecordingLLM(llm)

for cid in CASES:
    prev = json.loads((SRC / f"{cid}.json").read_text(encoding="utf-8"))
    rec = {k: prev[k] for k in ("case_id", "question", "response", "context")}
    rec["judge_model"] = MODEL
    rec["prev_status"] = prev.get("verdict", {}).get("status")
    judge.reply = None
    t1 = time.monotonic()
    try:
        rec["verdict"] = judge_citations(judge, rec["question"], rec["response"], rec["context"])
    except Exception as exc:
        rec["judge_error"] = f"{type(exc).__name__}: {exc}"
    rec["judge_seconds"] = round(time.monotonic() - t1, 1)
    if judge.reply is not None:
        rec["raw"] = judge.reply.content
        rec["usage"] = getattr(judge.reply, "usage_metadata", None)
        rec["finish_reason"] = (judge.reply.response_metadata or {}).get("finish_reason")
    (OUT / f"{cid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(cid, rec["prev_status"], "->", rec.get("verdict", {}).get("status"),
          rec["judge_seconds"], rec.get("finish_reason"), rec.get("usage"),
          rec.get("judge_error", ""), flush=True)
