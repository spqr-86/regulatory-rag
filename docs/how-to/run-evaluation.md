# How to run an evaluation

Runs the golden dataset through the V7 graph and scores it. Metric definitions and the
report format are in [reference/evaluation](../reference/evaluation.md).

```bash
source .venv/bin/activate

python eval/run_v7_eval.py                              # full dataset, with LLM judge
python eval/run_v7_eval.py --skip-judge                 # pipeline only, no judge (~$0)
python eval/run_v7_eval.py --limit 5                    # quick smoke test
python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
```

**Flags:** `--limit N` (cap questions), `--skip-judge` (no LLM scoring),
`--output PATH` (default `benchmarks/eval_v7_{date}.jsonl`).

**Cost:** the judge issues separate LLM calls per metric per question. A full run over the
56-question dataset with the default `gpt-4o` judge costs ≈ $0.25 (pipeline + judge
combined; measured 2026-09-08). Use `--skip-judge` for a free pipeline-only smoke run.

**Output:** a JSONL report under `benchmarks/`. The judge model is set by
`JUDGE_MODEL_NAME` (see [FACTS](../reference/FACTS.md#models)).

**Re-judging without re-running the pipeline:** `scripts/rejudge.py` re-scores saved
answers with a judge — cheap A/B of judge prompts or models without paying for retrieval
and generation again.

For a single-question trace instead of a full run:

```bash
python scripts/trace_v7.py "your question"
```

For deterministic triage calibration (no judge), use the reviewed annotations and an
immutable retrieval snapshot as described in
[`2026-09-11-triage-threshold-calibration.md`](../superpowers/plans/2026-09-11-triage-threshold-calibration.md).
The completed 81-profile run retained the current defaults.

## Парный прогон ObjectProfile (department Q&A, v1 vs v2)

Сравнивает режимы `DEPARTMENT_QA_MODE` (см. [FACTS](../reference/FACTS.md#department-qa)) на
одном наборе вопросов: `v1` — листы объекта в индексе, `v2` — листы исключены и передаются как
`ObjectProfile`. Спека:
[2026-09-15-object-profile-design §5.2](../superpowers/specs/2026-09-15-object-profile-design.md).

Сборка двух баз (`index.py` удаляет всю папку `CHROMA_DB_PATH`, поэтому у режимов разные пути).
**Важно:** committed `corpus/manifest.yaml` уже несёт `role: object_profile` на всех четырёх
листах (issue #44) — сборка v1 **по нему** индексирует v1 без листов объекта и молча превращает
сравнение в «v2-ретрив против v2». v1 нужно собирать из отдельной копии манифеста без строк
`role: object_profile`, в scratch-каталоге вне репозитория (не коммитится). После сборки **ни
`chroma_db_dept`, ни `chroma_db_dept_v2` больше не пересобирать до конца сравнения** — повторный
`index.py` по любой из них удалит и перезапишет базу.

```bash
export SOURCE_DOCS_PATH=./source_docs_dept
RUN=eval/runs/object_profile_pair_$(date +%F) && mkdir -p $RUN
SCRATCH=$(mktemp -d)

# v1 — листы в индексе: манифест без role, копия в scratch (не коммитится)
sed '/^[[:space:]]*role: object_profile$/d' corpus/manifest.yaml > "$SCRATCH/manifest_v1.yaml"
sha256sum "$SCRATCH/manifest_v1.yaml" | tee $RUN/manifest_v1.sha256
CORPUS_MANIFEST_PATH="$SCRATCH/manifest_v1.yaml" \
  CHROMA_DB_PATH=./chroma_db_dept CHROMA_COLLECTION_NAME=department_demo .venv/bin/python index.py
CORPUS_MANIFEST_PATH="$SCRATCH/manifest_v1.yaml" \
  CHROMA_DB_PATH=./chroma_db_dept CHROMA_COLLECTION_NAME=department_demo \
  .venv/bin/python eval/object_profile_collections.py --expect present | tee $RUN/collection_v1.json

# v2 — листы исключены (role: object_profile в реальном manifest), отдельная папка Chroma
CORPUS_MANIFEST_PATH=corpus/manifest.yaml \
  CHROMA_DB_PATH=./chroma_db_dept_v2 CHROMA_COLLECTION_NAME=department_demo_v2 .venv/bin/python index.py
CORPUS_MANIFEST_PATH=corpus/manifest.yaml \
  CHROMA_DB_PATH=./chroma_db_dept_v2 CHROMA_COLLECTION_NAME=department_demo_v2 \
  .venv/bin/python eval/object_profile_collections.py --expect absent | tee $RUN/collection_v2.json
sha256sum corpus/manifest.yaml | tee $RUN/manifest_final.sha256
```

Сам парный прогон (`eval/run_object_profile_pair.py`, один процесс на режим — Chroma-стор
процесс-синглтон):

```bash
DEPARTMENT_QA_MODE=v1 CHROMA_DB_PATH=./chroma_db_dept CHROMA_COLLECTION_NAME=department_demo \
  CORPUS_MANIFEST_PATH=corpus/manifest.yaml SOURCE_DOCS_PATH=./source_docs_dept \
  .venv/bin/python eval/run_object_profile_pair.py --out $RUN
DEPARTMENT_QA_MODE=v2 CHROMA_DB_PATH=./chroma_db_dept_v2 CHROMA_COLLECTION_NAME=department_demo_v2 \
  CORPUS_MANIFEST_PATH=corpus/manifest.yaml SOURCE_DOCS_PATH=./source_docs_dept \
  .venv/bin/python eval/run_object_profile_pair.py --out $RUN
```

Пишет `$RUN/<mode>/config.json` + `$RUN/<mode>/q<N>.json` (prompt, сырой ответ модели со всеми
попытками, `search_calls` — сырые вызовы поиска, `evidence_ids` — id evidence в порядке промпта,
`DepartmentResponse`). В прогоне 15.09 сырые вызовы поиска лежат под старым ключом `evidence_passed`. **Платно:** `gpt-4o-mini`, temperature 0, 9 вопросов
× 2 режима = 18 вызовов (+ретраи схемы), согласованный бюджет < $0.05 (согласовано 15.09) —
запускать по прямому согласию, не автоматически. `--dry-run` печатает `config.json` (режим,
число чанков, профили, вопросы) без единого вызова модели — им можно проверить обвязку без
затрат.

**Перепрогон одного режима.** После правки промпта режима достаточно прогнать только его: `--out` указывает на новый каталог прогона (`check_paid_run` отказывается дописывать в каталог, где уже лежат `q*.json`). Перепрогон 16.09 — только `v2` (промпт `department_answer` v3, 9 вызовов, согласовано Петром).
