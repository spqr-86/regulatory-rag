import json,sys
from dotenv import load_dotenv; load_dotenv(".env")
from langsmith import Client
case, score, comment = sys.argv[1], int(sys.argv[2]), sys.argv[3]
key = sys.argv[4] if len(sys.argv) > 4 else "human_correct"
o={x["case_id"]:x for x in json.load(open("eval/runs/align_batch2_2026-10-16/judged.json"))}[case]
c=Client(); c.create_feedback(o["run_id"], key=key, score=score, comment=comment)
print(case, o["run_id"], key, score, [ (f.key,f.score) for f in c.list_feedback(run_ids=[o["run_id"]], feedback_key=[key])])
