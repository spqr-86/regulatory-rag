"""Offline paired summary of saved parent-context experiments."""

import json
import re
import statistics
from pathlib import Path

root = Path('eval/runs/parent_context_verified_2026-10-17_on3')
old = root.parent / 'parent_context_verified_2026-10-17_on2'
arms = {name: json.loads(path.read_text()) for name, path in {
    'off': old/'off.json', 'on2': old/'on2.json', 'on3': root/'on3.json'
}.items()}
keys = ['correctness', 'citations', 'citations_complete', 'retrieval_hit', 'oos_handled']
summary = {'arms': {}, 'paired': {}}
for name, data in arms.items():
    rows = data['rows']
    outputs = [r['outputs'] for r in rows]
    metrics = {}
    for key in keys:
        values = [r['latest_feedback'].get(key, {}).get('score') for r in rows]
        values = [v for v in values if v is not None]
        metrics[key] = {'successes': sum(values), 'n': len(values)}
    ranks = [r['latest_feedback']['retrieval_hit'] for r in rows
             if r['latest_feedback'].get('retrieval_hit', {}).get('score') is not None]
    rr = [1/int(m.group(1)) if (m := re.search(r'rank=(\d+)', f['comment'] or '')) else 0
          for f in ranks]
    summary['arms'][name] = {
        'name': data['name'], 'id': data['id'], 'metadata': data['metadata'],
        'n': len(rows), 'errors': [r['case_id'] for r in rows if r['error']],
        'metrics': metrics, 'mrr': statistics.mean(rr) if rr else None,
        'pipeline_cost_usd': sum(o.get('cost_usd', 0) for o in outputs),
        'formatted_context_chars': sum(len(o.get('context', '')) for o in outputs),
        'context_fragments': sum(len(re.findall(r'^\[\d+\] \((?:HIGH|MED|LOW)\)',
                                                o.get('context', ''), re.M)) for o in outputs),
        'mean_elapsed_sec': statistics.mean(o.get('elapsed_sec', 0) for o in outputs),
        'parent_rows': sum(bool(o.get('parent_replaced')) for o in outputs),
        'nonempty_context_rows': sum(bool(o.get('context')) for o in outputs),
        'judge_errors': {r['case_id']: {k: f['comment'] for k, f in r['latest_feedback'].items()
                                      if k in keys and f.get('score') is None
                                      and not (f.get('comment') or '').startswith(('oos', 'no_', 'in_scope'))}
                         for r in rows
                         if any(f.get('score') is None and ('error' in (f.get('comment') or '').lower()
                                                           or 'truncat' in (f.get('comment') or '').lower())
                                for f in r['latest_feedback'].values())},
    }
    for key in ['parent_replaced', 'parent_deduplicated', 'parent_children_kept',
                'parent_truncated', 'parent_slots_lost']:
        summary['arms'][name][key] = sum(o.get(key, 0) for o in outputs)
for baseline in ['off', 'on2']:
    before = {r['case_id']: r for r in arms[baseline]['rows']}
    after = {r['case_id']: r for r in arms['on3']['rows']}
    assert before.keys() == after.keys()
    summary['paired'][baseline+'->on3'] = {}
    for key in keys:
        pairs, unpaired = [], []
        for cid in sorted(before):
            a = before[cid]['latest_feedback'].get(key, {}).get('score')
            b = after[cid]['latest_feedback'].get(key, {}).get('score')
            if a is not None and b is not None:
                pairs.append((cid,a,b))
            elif a != b:
                unpaired.append({'case_id':cid, 'before':a, 'after':b})
        summary['paired'][baseline+'->on3'][key] = {
            'n':len(pairs), 'before_successes':sum(a for _,a,_ in pairs),
            'after_successes':sum(b for _,_,b in pairs),
            'improved':[cid for cid,a,b in pairs if b>a],
            'regressed':[cid for cid,a,b in pairs if b<a], 'unpaired':unpaired}
(root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
