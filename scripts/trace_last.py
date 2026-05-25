"""Show the last LangSmith trace for the regulatory-rag project."""

import sys
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client

client = Client()
project = "regulatory-rag"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1

runs = list(client.list_runs(project_name=project, is_root=True, limit=n))
if not runs:
    print("No runs found.")
    sys.exit(0)

root = runs[0]
dur = (root.end_time - root.start_time).total_seconds() if root.end_time else 0
print(f"Query   : {root.inputs.get('query', '?')}")
print(f"Status  : {root.status} | {dur:.1f}s total")
print(f"TraceID : {root.trace_id}")

children = list(client.list_runs(project_name=project, trace_id=root.trace_id))
children.sort(key=lambda r: r.start_time)
print(f"\nSpans ({len(children)}):")
for r in children:
    d = (r.end_time - r.start_time).total_seconds() if r.end_time else 0
    print(f"  {r.run_type:8} | {r.name:35} | {d:.2f}s | {r.status}")

# Show answer
output = root.outputs or {}
answer = output.get("answer") or output.get("final_answer", "")
if answer:
    print(f"\nAnswer:\n{answer[:600]}")
