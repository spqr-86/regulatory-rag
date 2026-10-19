# How to run an evaluation

The generic normative evaluator is `eval/ls/run.py`, built on the existing LangSmith
executor. It selects only `reference_status=verified` from `tests/dataset.csv`;
`needs_clarification` and `expert_candidate` never enter the benchmark. Counts are computed
on each launch. The [legal reference audit](../evaluation/golden-set-audit-2026-09-28.md)
explains why the original key cannot establish legal accuracy. Metric definitions and
the new artifact format are in [reference/evaluation](../reference/evaluation.md).

```bash
source .venv/bin/activate

python -m eval.ls.run --dry-run --output-dir /tmp/regrag-plan  # local files only
python -m eval.ls.run --paid --prefix unified --output-dir eval/runs/my-run  # paid
python -m eval.ls.run --paid --prefix smoke --cases G001,G005 --limit 1   # paid; select before limit
python -m eval.ls.run --paid --prefix pipeline --skip-judge             # paid RAG, no judges
```

Run from the worktree root. `--dry-run` constructs no LangSmith client, model or vector
store; it writes a manifest with selected cases, hashes, configuration and label coverage.
Each `--output-dir` must be new: an existing manifest is never overwritten.

Before paid execution, configure `CHROMA_DB_PATH` to the reviewed snapshot matching the
labels. The default labels target `build-j7ceu4r4`; another set needs matching `--labels`
and `--labels-snapshot`. An incompatible snapshot or a missing snapshot report stops the
run before graph construction. The selected LangSmith examples must match the local
questions and references; review and upsert separately with `python -m eval.ls.dataset`
if they are stale. The runner does not change the dataset. It does not copy `.env`
between worktrees.

**Costs and readiness:** ordinary runs pay for generation/embeddings and three separate
judges: correctness, citations and groundedness. `--skip-judge` still pays for RAG.
Judge usage is reported separately, with bounded output, 300s timeout and no automatic
paid retries. Missing usage or unknown rates make the cost incomplete. Embedding usage
is not in the pipeline's `llm_usage`: with a paid embedding provider, accounted cost is
explicitly a lower bound. Groundedness requires manual alignment on saved real answers
before a paid baseline or quality decisions; offline tests establish its contract only.
The current groundedness contract is v2 (per-fragment evidence lists), which has not
yet been aligned with human labels. The prior 19/19 alignment applies to v1 and
does not validate v2. Retain prior assessments when comparing versions; do not merge
the two versions into one metric series.

**Artifacts:** `manifest.json` freezes references, versions and allowlisted configuration;
`outputs.jsonl` checkpoints the target before judges; `judges.jsonl` stores verdicts/usage;
`assessments.jsonl` stores per-evaluator results; `results.jsonl` checkpoints completed
examples; `summary.json` updates after each result. Ctrl+C preserves these files and
stops subsequent target/judge calls; an already running provider call can still finish.
Automatic resumption is outside this release.

**Rejudge (paid judges, no RAG or telemetry queries):**

```bash
python -m eval.ls.run --paid --rejudge EXPERIMENT --output-dir eval/runs/rejudge-original
python -m eval.ls.run --paid --rejudge EXPERIMENT --current-references --output-dir eval/runs/rejudge-updated
```

Default references come from the experiment's original `dataset_version`. The explicit
flag selects current references by stable example UUID, even if the question changed.
Each assessment has a new manifest/assessment ID and retains the source run IDs. Missing
original versions, missing IDs and repeated examples fail clearly. Old runs without
generator context remain unscored for citations/groundedness. Retrieval is not rejudged
against a different snapshot. Source pipeline costs are shown as reused, excluded from
the new evaluation's total.

`eval/run_v7_eval.py` delegates to this executor (`--limit`, `--skip-judge`, `--dry-run`,
`--output-dir`). Its old `--output` report contract is rejected before execution with a
migration hint. Historical helpers and reports remain readable; old scripts below are
for historical report formats, not a second end-to-end normative runner.

Align and pairwise remain `eval.ls.align` and `eval.ls.pairwise`. Comparison must identify
dataset/reference hashes, judge prompts/models and index snapshot changes rather than
pooling incompatible metric series.

For the revised legal key, use `scripts/rejudge_golden.py INPUT --dry-run` to inspect
eligible saved cases, then `scripts/rejudge_golden.py INPUT --output OUTPUT` after
source review and authorization for sending saved questions, references and answers
to the configured judge provider. It checkpoints each score. Its partial reused-answer
mean is not an end-to-end baseline.

For a single-question trace instead of a full run:

```bash
python scripts/trace_v7.py "your question"
```

For deterministic triage calibration (no judge), use the reviewed annotations and an
immutable retrieval snapshot as described in `2026-09-11-triage-threshold-calibration.md`
(removed in the 21.09.2026 portfolio cleanup, see git history at `cf61c1c`).
The completed 81-profile run retained the current defaults.

## Парный прогон ObjectProfile (department Q&A, v1 vs v2)

Сравнивает режимы `DEPARTMENT_QA_MODE` (см. [FACTS](../reference/FACTS.md#department-qa)) на
одном наборе вопросов: `v1` — листы объекта в индексе, `v2` — листы исключены и передаются как
`ObjectProfile`. Спека:
[2026-09-15-object-profile-design §5.2](../design/2026-09-15-object-profile-design.md).

Сборка двух баз (`index.py` удаляет всю папку `CHROMA_DB_PATH`, поэтому у режимов разные пути).
**Важно:** committed `corpus/manifest.yaml` уже несёт `role: object_profile` на всех листах
объекта (шесть: четыре смоук-листа issue #44 плюс `unit_prod` и `unit_warehouse_v2` из набора
ловушек) — сборка v1 **по нему** индексирует v1 без листов объекта и молча превращает
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

**Набор ловушек (выбор дешёвой модели).** Отдельный набор
`eval/data/object_profile_traps_expectations.yaml` (4 вопроса, синтетические листы `unit_prod` и
`unit_warehouse_v2`) проверяет подмену величины порога и обобщение правила: п. 401 (категория Д,
площадь ≤ 100 м²), п. 406 (30 м для В1–В4), п. 17 «б» (испытания лестницы не реже 1 раза в 5 лет).
Набор не входит в парный прогон 9 вопросов и не меняет его baseline. Запуск — режим `v2`,
частичный прогон:

```bash
RUN=eval/runs/object_profile_traps_$(date +%F)
DEPARTMENT_QA_MODE=v2 CHROMA_DB_PATH=./chroma_db_dept_v2 CHROMA_COLLECTION_NAME=department_demo_v2 \
  CORPUS_MANIFEST_PATH=corpus/manifest.yaml SOURCE_DOCS_PATH=./source_docs_dept \
  .venv/bin/python eval/run_object_profile_pair.py --out $RUN \
  --expectations eval/data/object_profile_traps_expectations.yaml --only 1,2,3,4 --model <model>
.venv/bin/python eval/score_object_profile_run.py --run $RUN/v2 \
  --expectations eval/data/object_profile_traps_expectations.yaml
```

`forbidden_conclusion` в этом наборе — точное совпадение подстроки; смысловая оценка по
`required_subanswers` остаётся ручной.
