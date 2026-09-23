#!/bin/bash
# A/B: default reasoning vs effort=low + provider sort=latency (deepseek-v4.1-flash, OpenRouter)
set -u
cd /home/petr/projects/ai/regulatory-rag
D=eval/runs/openrouter_reasoning_ab_2026-09-23
P=.venv/bin/python
DEPT="DEPARTMENT_QA_MODE=v2 CHROMA_DB_PATH=./chroma_db_dept_v2 CHROMA_COLLECTION_NAME=department_demo_v2 CORPUS_MANIFEST_PATH=corpus/manifest.yaml SOURCE_DOCS_PATH=./source_docs_dept"
run_cfg () {
  name=$1; shift
  echo "=== $name golden $(date +%T)"
  env "$@" $P eval/run_v7_eval.py --limit 20 --output $D/golden_$name.jsonl > $D/golden_$name.log 2>&1; echo "exit $?"
  echo "=== $name traps $(date +%T)"
  env "$@" $DEPT $P eval/run_object_profile_pair.py --out $D/traps_$name \
    --expectations eval/data/object_profile_traps_expectations.yaml --only 1,2,3,4 \
    --model deepseek/deepseek-v4.1-flash > $D/traps_$name.log 2>&1; echo "exit $?"
  $P eval/score_object_profile_run.py --run $D/traps_$name/v2 \
    --expectations eval/data/object_profile_traps_expectations.yaml > $D/traps_${name}_score.txt 2>&1; echo "score exit $?"
}
run_cfg default DUMMY=1
run_cfg low_latency OPENROUTER_REASONING_EFFORT=low OPENROUTER_PROVIDER_SORT=latency
echo "=== done $(date +%T)"
