"""Read completed on3 runs and per-run feedback; never invoke models."""

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv('.env')
root = Path('eval/runs/parent_context_verified_2026-10-17_on3')
manifest = json.loads((root / 'manifest.json').read_text())
log = (root / 'run.log').read_text()
names = re.findall(r'experiment: (v8-parent-full-on3-[\w-]+)', log)
partial = '--partial' in sys.argv
if partial and not names:
    names = ['v8-parent-full-on3-0bb8da21']
assert names, 'runner has not printed its final experiment name'
client = Client()
project = client.read_project(project_name=names[-1])
examples = {e['id']: e for e in manifest['examples']}
runs = list(client.list_runs(project_id=project.id, is_root=True))
assert len(runs) == 26 and len({str(r.id) for r in runs}) == 26
assert {str(r.reference_example_id) for r in runs} == set(examples)
rows = []
for run in runs:
    feedback = {str(f.id): f for f in client.list_feedback(run_ids=[run.id])}
    assert all(str(f.run_id) == str(run.id) for f in feedback.values())
    items = [dict(key=f.key, score=f.score, comment=f.comment,
                  created_at=str(f.created_at), modified_at=str(f.modified_at),
                  feedback_id=str(f.id)) for f in feedback.values()]
    latest = {}
    for item in sorted(items, key=lambda f: (f['modified_at'], f['created_at'])):
        latest[item['key']] = item
    rows.append(dict(run_id=str(run.id), example_id=str(run.reference_example_id),
                     case_id=examples[str(run.reference_example_id)]['case_id'],
                     inputs=run.inputs, outputs=run.outputs or {}, error=run.error,
                     feedback=items, latest_feedback=latest))
    print(rows[-1]['case_id'], len(latest), 'error', bool(run.error), flush=True)
data = dict(name=project.name, id=str(project.id), metadata=project.metadata, rows=rows)
(root / ('on3.partial.json' if partial else 'on3.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str)+'\n')
if not partial:
    assert all(len(r['latest_feedback']) == 8 for r in rows), 'missing feedback'
assert project.metadata['dataset_version'] == '2026-10-17T18:34:35.480208+00:00'
print('exported', project.name, len(rows), flush=True)
