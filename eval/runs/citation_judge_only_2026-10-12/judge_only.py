"""Re-judge saved G001/G006 cases (question/response/context) without re-running RAG.

Saves the judge's raw reply next to the verdict.
Run from the repo root: python eval/runs/citation_judge_only_2026-10-12/judge_only.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from eval.judges.citations import judge_citations  # noqa: E402
from src.infra.llm_factory import get_judge_llm  # noqa: E402

SRC = Path("eval/runs/citation_judge_rerun_2026-10-12")
OUT = Path("eval/runs/citation_judge_only_2026-10-12")
CASES = ["G001", "G006"]


class RecordingLLM:
    def __init__(self, llm):
        self.llm = llm
        self.raw = None

    def invoke(self, messages):
        reply = self.llm.invoke(messages)
        self.raw = reply.content
        return reply


judge = RecordingLLM(get_judge_llm(thinking_budget=0))
for cid in CASES:
    case = json.loads((SRC / f"{cid}.json").read_text(encoding="utf-8"))
    rec = {"case_id": cid}
    try:
        rec["verdict"] = judge_citations(
            judge, case["question"], case["response"], case["context"]
        )
    except Exception as exc:
        rec["judge_error"] = f"{type(exc).__name__}: {exc}"
    rec["raw"] = judge.raw
    (OUT / f"{cid}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(cid, rec.get("verdict", {}).get("status"), rec.get("judge_error", ""))
