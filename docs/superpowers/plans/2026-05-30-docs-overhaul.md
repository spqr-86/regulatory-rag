# Docs Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `docs/` to Diátaxis-lite, de-stale all content against current code, remove `_RU` duplication, and add a `FACTS.md` single-source-of-truth guarded by a `check_docs.py` CI check.

**Architecture:** Four buckets (Explanation / How-to / Reference / top-level Getting Started). All volatile facts centralized in `reference/FACTS.md`; prose links to it instead of repeating numbers. A `scripts/check_docs.py` enforces freshness: value-sync vs committed config + a forbidden-stale-term grep over live docs.

**Tech Stack:** Markdown, Python 3.11 (`check_docs.py`: stdlib + PyYAML, already a dep), pytest, GitHub Actions.

**Branch:** `docs/overhaul` (already created; spec committed at `ae88493`).

**Reference spec:** `docs/superpowers/specs/2026-05-30-docs-overhaul-design.md`

---

## Ground rules for every content task

- **Verify, never recall.** Before writing any number (threshold, chunk count, metric, version), open the source: `src/v7/config.py`, `prompts/registry.yaml`, `.env`, `benchmarks/*.jsonl`, `src/v7/graph.py`. If a fact is volatile, do not repeat it in prose — link to `reference/FACTS.md`.
- **Stale deny-list** (must NOT appear in any live doc outside `docs/archive/`): `llm_verifier`, `rewriter`, `FlashRank rerank`, `generate_answer_v1`, `borderline → llm_verifier`, `Gemini 2.5 Flash (simple`, `Gemini 3 Flash (complex`.
- **Current truth** (from this session, re-verify at write time): providers OpenAI (`gpt-4o-mini` simple / `gpt-4o` complex), judge `gpt-4o`, reranker `crossencoder`, prompt `generate_answer` v8, nodes `intent_gate → domain_gate → router → rag_simple → evaluate_triage → rag_complex → evaluate_complex → visual_enrichment → generate_answer / abstain`, 12 НТД / 7718 chunks, port 8502.
- Use `git mv` for moves so history is preserved.

---

## Task 1: Scaffold directory tree and move files (no content edits yet)

**Files:**
- Create dirs: `docs/explanation/`, `docs/how-to/`, `docs/reference/`, `docs/archive/`
- Move via `git mv` (see steps)
- Delete: all `docs/**/*_RU.md` except none-kept (hub RU lives inline), `docs/passport.md`

- [ ] **Step 1: Create the bucket directories**

```bash
cd /home/petr/projects/ai/regulatory-rag
mkdir -p docs/explanation docs/how-to docs/reference docs/archive/plans
```

- [ ] **Step 2: Move files to new homes (history-preserving), EN only**

```bash
git mv docs/guides/quick-start.md            docs/getting-started.md
git mv docs/architecture/triage-how-it-works.md docs/explanation/triage.md
git mv docs/architecture/v7-how-it-works.md  docs/explanation/architecture.md
git mv docs/guides/prompt-management.md       docs/how-to/manage-prompts.md
git mv docs/guides/adding-questions.md        docs/how-to/add-eval-questions.md
git mv docs/guides/testing.md                 docs/how-to/run-tests.md
git mv docs/DATA_PIPELINE.md                  docs/reference/data-pipeline.md
git mv docs/evaluation/README.md              docs/reference/evaluation.md
git mv docs/feature/migration-v7              docs/archive/migration-v7
git mv docs/plans/* docs/archive/plans/
```

Note: `docs/architecture/README.md` is split (not a clean move) — handled in Task 3. Leave it in place for now.

- [ ] **Step 3: Delete RU duplicates and passport**

```bash
git rm docs/**/*_RU.md docs/*_RU.md docs/passport.md
# if the glob misses nested ones, also:
find docs -name '*_RU.md' -exec git rm {} +
rmdir docs/guides docs/evaluation docs/feature docs/plans docs/architecture 2>/dev/null || true
```

- [ ] **Step 4: Verify tree shape**

Run: `find docs -type f -name '*.md' | sort`
Expected: no `*_RU.md` outside archive; files present at `docs/explanation/{architecture,triage}.md`, `docs/how-to/{manage-prompts,add-eval-questions,run-tests}.md`, `docs/reference/{data-pipeline,evaluation}.md`, `docs/getting-started.md`, `docs/architecture/README.md` (still, pending split).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: scaffold Diátaxis-lite tree, move files, drop _RU duplicates and passport"
```

---

## Task 2: Write `reference/FACTS.md` (single source of truth)

**Files:**
- Create: `docs/reference/FACTS.md`

- [ ] **Step 1: Extract current values from sources**

```bash
grep -nE "THRESHOLD|TOP_K|MIN_PASSAGES|MIN_KW_OVERLAP" src/v7/config.py
grep -A2 -E "active_version" prompts/registry.yaml
grep -iE "MODEL_NAME|LLM_PROVIDER|RERANKER_BACKEND" .env | grep -v KEY
ls -t benchmarks/*.jsonl | head -1   # newest eval run for metrics
```

- [ ] **Step 2: Write FACTS.md with anchored, greppable sections**

Use this exact skeleton; fill values from Step 1 (do NOT trust the spec's numbers if config disagrees — config wins):

```markdown
# FACTS — canonical reference

> Single source of truth for volatile facts. Prose elsewhere links here and does not
> repeat these numbers. `scripts/check_docs.py` verifies this file against code/config.

## models
- simple: openai / gpt-4o-mini
- complex: openai / gpt-4o
- judge: gpt-4o
- reranker: crossencoder
- embeddings: openai / text-embedding-3-small

## thresholds
| name (env: `V7_<NAME>`) | value | source |
|---|---|---|
| HARD_GATE_THRESHOLD | 0.50 | src/v7/config.py |
| TRIAGE_SOFT_THRESHOLD | 0.38 | src/v7/config.py |
| COMPLEX_THRESHOLD | 0.35 | src/v7/config.py |
| COMPLEX_MIN_PASSAGES | 8 | src/v7/config.py |
| COMPLEX_MIN_KW_OVERLAP | 0.20 | src/v7/config.py |
| DOMAIN_GATE_THRESHOLD | 0.25 | src/v7/config.py |
| SIMPLE_TOP_K | 12 | src/v7/config.py |
| COMPLEX_TOP_K | 60 | src/v7/config.py |

## prompts
- generate_answer: v8
- query_expand: v1
- applicability_retriever: v2

## corpus
- documents: 12 НТД
- chunks: 7718

## nodes
intent_gate → domain_gate → router → rag_simple → evaluate_triage → rag_complex → evaluate_complex → visual_enrichment → generate_answer / abstain

## metrics
- overall: 0.80
- in-scope correctness: 7.4 / 10
- faithfulness: 0.988
- OOS abstain rate: 1.00
- avg latency: 9.7s
- cost: $0.0102 / query

## deploy
- port: 8502
- process: tmux session `sia`
```

- [ ] **Step 3: Verify every value against its source** (re-run Step 1 greps, reconcile any mismatch — config/registry/.env win over this plan's text).

- [ ] **Step 4: Commit**

```bash
git add docs/reference/FACTS.md
git commit -m "docs(reference): add FACTS single-source-of-truth"
```

---

## Task 3: Build `explanation/architecture.md` (merge + de-stale) and split out `how-to/add-a-node.md`

**Files:**
- Modify/rewrite: `docs/explanation/architecture.md` (currently the moved v7-how-it-works content — heavily stale)
- Create: `docs/how-to/add-a-node.md` (from `docs/architecture/README.md`)
- Read for merge: `docs/architecture/README.md`, `src/v7/graph.py`, `src/v7/nodes/evaluate_triage.py:221`
- Delete after merge: `docs/architecture/README.md`

- [ ] **Step 1: Read current graph to capture the real flow**

```bash
sed -n '1,120p' src/v7/graph.py
grep -n "add_node\|add_edge\|add_conditional_edges" src/v7/graph.py
```

- [ ] **Step 2: Rewrite `docs/explanation/architecture.md`** to contain:
  - Pipeline overview prose (the "why deterministic" framing — 1 short para, link to design-decisions.md).
  - The **current** request-flow diagram: `intent_gate → domain_gate → router → rag_simple → evaluate_triage →(sufficient) generate_answer / (insufficient) rag_complex → evaluate_complex →(pass) generate_answer / (fail) abstain`. NO verifier/rewriter/borderline branch.
  - Node-by-node table (one row per node, one-line purpose, source file path). Pull node list from graph.py.
  - Codebase map (`src/` layout — copy the accurate tree from CLAUDE.md `### src/ layout`).
  - Pluggable-backends note (LLM factory + VectorStore Protocol — 1 paragraph; this is where it lives, not an ADR).
  - All thresholds/models/versions referenced as links to `../reference/FACTS.md`, not inline numbers.

- [ ] **Step 3: Create `docs/how-to/add-a-node.md`** from the "how to add a node" section of `docs/architecture/README.md` (steps: write node fn in `src/v7/nodes/`, register in `graph.py`, add edge, test). Keep recipe-style, imperative.

- [ ] **Step 4: Delete the old architecture README**

```bash
git rm docs/architecture/README.md
rmdir docs/architecture 2>/dev/null || true
```

- [ ] **Step 5: Verify no stale terms**

Run: `grep -nE "llm_verifier|rewriter|FlashRank rerank|borderline → llm" docs/explanation/architecture.md docs/how-to/add-a-node.md`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs(explanation): rewrite architecture (current graph, no verifier) + split add-a-node how-to"
```

---

## Task 4: De-stale `explanation/triage.md`

**Files:**
- Modify: `docs/explanation/triage.md`
- Read: `src/v7/nodes/evaluate_triage.py`, `src/v7/hard_gates.py`

- [ ] **Step 1: Read the real triage logic** — `route_after_triage` returns `"end"` (sufficient) or `"rag_complex"`; there is no verifier path.

```bash
sed -n '60,233p' src/v7/nodes/evaluate_triage.py
```

- [ ] **Step 2: Rewrite triage.md** so the 3-way description reflects: `sufficient → generate`, `borderline/insufficient → rag_complex` (no llm_verifier, no rewriter). Keep the hard-gate three-conditions table but reference thresholds via `../reference/FACTS.md`. Note enumeration-intent forcing rag_complex (from `route_after_triage`).

- [ ] **Step 3: Verify**

Run: `grep -nE "llm_verifier|rewriter|Verifier" docs/explanation/triage.md`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add docs/explanation/triage.md
git commit -m "docs(explanation): de-stale triage — verifier path removed, link thresholds to FACTS"
```

---

## Task 5: De-stale Reference docs (`data-pipeline.md`, `evaluation.md`) and split `how-to/run-evaluation.md`

**Files:**
- Modify: `docs/reference/data-pipeline.md`, `docs/reference/evaluation.md`
- Create: `docs/how-to/run-evaluation.md`
- Read: `src/indexing/file_handler.py` (chunker), `eval/run_v7_eval.py`

- [ ] **Step 1: De-stale `data-pipeline.md`** — confirm HybridChunker(max_tokens=400), contextual embedding (parent-section prepended), element_type from Docling, chunk_id namespaced per source. Add a short **HybridChunker decision note** (why structure-aware chunking for legal docs). Embeddings model + chunk count → link to FACTS.

- [ ] **Step 2: Split `evaluation.md`** — keep metric definitions + report format as reference; move the "how to run a run" commands into a new `docs/how-to/run-evaluation.md` (`python eval/run_v7_eval.py [--skip-judge]`, where output lands, judge model via FACTS link).

- [ ] **Step 3: Verify** — `grep -niE "flashrank|gemini.*simple|generate_answer_v1" docs/reference/data-pipeline.md docs/reference/evaluation.md docs/how-to/run-evaluation.md` → no matches.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(reference): de-stale data-pipeline + evaluation, split run-evaluation how-to"
```

---

## Task 6: De-stale remaining how-to docs (`manage-prompts.md`, `add-eval-questions.md`, `run-tests.md`)

**Files:**
- Modify: `docs/how-to/manage-prompts.md`, `docs/how-to/add-eval-questions.md`, `docs/how-to/run-tests.md`
- Read: `prompts/registry.yaml`, `src/infra/prompt_manager.py`

- [ ] **Step 1: `manage-prompts.md`** — registry now has 3 live families (`applicability_retriever`, `query_expand`, `generate_answer`); active `generate_answer` = v8. Remove references to deleted templates and to `scripts/validate_prompts.py` (does not exist). Env override pattern: `PROMPT_GENERATE_ANSWER_VERSION=v7`.

- [ ] **Step 2: `add-eval-questions.md`** — confirm dataset paths (`tests/dataset_original.csv` etc. per CLAUDE.md Eval Datasets), format `question, ground_truth`. Fix any stale path.

- [ ] **Step 3: `run-tests.md`** — `pytest -m unit` (230 tests), markers, `black . && ruff check .`. Remove any doc-validation step referencing nonexistent scripts.

- [ ] **Step 4: Verify** — `grep -rniE "validate_prompts|generate_answer_v[1-6]|verifier_v2|router_v2" docs/how-to/` → no matches.

- [ ] **Step 5: Commit**

```bash
git add docs/how-to/
git commit -m "docs(how-to): de-stale prompt/eval/test guides, drop dead validate_prompts refs"
```

---

## Task 7: Write `explanation/design-decisions.md` (8 ADRs — the showcase)

**Files:**
- Create: `docs/explanation/design-decisions.md`
- Read for evidence: `docs/archive/plans/2026-05-29-eval-failures-*.md`, `benchmarks/*.jsonl`, CLAUDE.md Session Log, `~/assistant-core/handoffs/last_handoff.md`

- [ ] **Step 1: Verify the evidence numbers** for each ADR against `benchmarks/` and archived analyses. Record the exact figures used. If a number can't be substantiated, soften the claim — never invent.

```bash
grep -rniE "in.?scope|faith|latency|false.?suff|7\.[0-9]|0\.9[0-9]" benchmarks/*.jsonl docs/archive/plans/2026-05-29-eval-failures-analysis.md | head -40
```

- [ ] **Step 2: Write the doc** — intro paragraph (compliance reliability thesis), then 8 ADR records, each with **Context → Options → Choice → Why → Evidence**:
  1. Deterministic hard gates, no LLM routing
  2. Abstain > hallucinate + domain_gate
  3. CrossEncoder over FlashRank
  4. chunk_id namespaced by source
  5. Removed verifier/rewriter subgraph
  6. Two-tier LLM (cost)
  7. Prompt-as-code v8 (paired with #2)
  8. Eval methodology (judge-noise ±0.25, trace-over-aggregate)

  Volatile numbers (current metrics) link to FACTS; historical deltas (e.g. 7.08→7.4) are stated inline with their benchmark source named.

- [ ] **Step 3: Verify** — `grep -nE "llm_verifier|FlashRank rerank" docs/explanation/design-decisions.md` → only allowed inside ADR #3/#5 as the *rejected/removed* option (that's intentional context, not a stale claim). Confirm each ADR has an Evidence line.

- [ ] **Step 4: Commit**

```bash
git add docs/explanation/design-decisions.md
git commit -m "docs(explanation): add design-decisions (8 ADRs) — portfolio showcase"
```

---

## Task 8: Reference `api.md` + rewrite hub `README.md`

**Files:**
- Create: `docs/reference/api.md`
- Rewrite: `docs/README.md`
- Read: root `README.md` (REST API section), `api.py`

- [ ] **Step 1: `docs/reference/api.md`** — extract REST endpoints from root README + `api.py` (`POST /query`, `POST /query/gosts`, `GET /health`), curl examples, port via FACTS link.

- [ ] **Step 2: Rewrite `docs/README.md`** as the hub: 1-paragraph intro + a 4-bucket navigation table (Getting Started / Explanation / How-to / Reference) linking every doc. Append a short **RU section** at the bottom (the only bilingual content) — 3-4 sentences pointing Russian readers to the same docs.

- [ ] **Step 3: Verify all hub links resolve**

Run: `grep -oE '\]\(\./[^)]+\)' docs/README.md | sed 's/](\.\///;s/)//' | while read f; do test -f "docs/$f" || echo "MISSING: $f"; done`
Expected: no MISSING lines.

- [ ] **Step 4: Commit**

```bash
git add docs/README.md docs/reference/api.md
git commit -m "docs: add api reference + rewrite hub README (4-bucket nav, RU section)"
```

---

## Task 9: `scripts/check_docs.py` — freshness check (TDD)

**Files:**
- Create: `scripts/check_docs.py`
- Create: `tests/test_check_docs.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_check_docs.py
from __future__ import annotations

import pytest

from scripts.check_docs import (
    find_stale_terms,
    check_prompt_versions,
    STALE_DENYLIST,
)


def test_find_stale_terms_flags_denylisted_word(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "bad.md").write_text("The llm_verifier decides what to do.", encoding="utf-8")
    hits = find_stale_terms(d, archive_dirname="archive")
    assert any("llm_verifier" in h.term for h in hits)
    assert any(h.path.name == "bad.md" for h in hits)


def test_find_stale_terms_ignores_archive(tmp_path):
    d = tmp_path / "docs"
    (d / "archive").mkdir(parents=True)
    (d / "archive" / "old.md").write_text("old llm_verifier notes", encoding="utf-8")
    hits = find_stale_terms(d, archive_dirname="archive")
    assert hits == []


def test_find_stale_terms_clean_doc(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "ok.md").write_text("domain_gate filters out-of-scope queries.", encoding="utf-8")
    assert find_stale_terms(d, archive_dirname="archive") == []


def test_check_prompt_versions_matches(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "generate_answer:\n  active_version: \"v8\"\n  versions:\n    v8: x.j2\n",
        encoding="utf-8",
    )
    facts = tmp_path / "FACTS.md"
    facts.write_text("## prompts\n- generate_answer: v8\n", encoding="utf-8")
    mismatches = check_prompt_versions(registry, facts)
    assert mismatches == []


def test_check_prompt_versions_detects_drift(tmp_path):
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "generate_answer:\n  active_version: \"v8\"\n  versions:\n    v8: x.j2\n",
        encoding="utf-8",
    )
    facts = tmp_path / "FACTS.md"
    facts.write_text("## prompts\n- generate_answer: v7\n", encoding="utf-8")
    mismatches = check_prompt_versions(registry, facts)
    assert any(m.name == "generate_answer" for m in mismatches)


def test_denylist_nonempty():
    assert "llm_verifier" in STALE_DENYLIST
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `venv/bin/python -m pytest tests/test_check_docs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.check_docs'`

- [ ] **Step 3: Implement `scripts/check_docs.py`**

```python
# scripts/check_docs.py
"""Docs freshness check: forbidden stale terms + FACTS-vs-source value sync.

CI scope (committed sources only): stale-term grep, prompt versions, thresholds.
Local-only (needs .env / chroma): provider/model names, chunk count. Skipped with a
notice when those sources are absent — CI does NOT verify everything, by design.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

STALE_DENYLIST: list[str] = [
    "llm_verifier",
    "rewriter",
    "FlashRank rerank",
    "generate_answer_v1",
    "borderline → llm_verifier",
    "Gemini 2.5 Flash (simple",
    "Gemini 3 Flash (complex",
]


@dataclass
class StaleHit:
    path: Path
    line: int
    term: str


@dataclass
class VersionMismatch:
    name: str
    registry_version: str
    facts_version: str


def find_stale_terms(docs_dir: Path, archive_dirname: str = "archive") -> list[StaleHit]:
    hits: list[StaleHit] = []
    for md in sorted(docs_dir.rglob("*.md")):
        if archive_dirname in md.relative_to(docs_dir).parts:
            continue
        for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for term in STALE_DENYLIST:
                if term in line:
                    hits.append(StaleHit(path=md, line=i, term=term))
    return hits


def _facts_prompt_versions(facts: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    in_prompts = False
    for line in facts.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("## prompts"):
            in_prompts = True
            continue
        if in_prompts and line.startswith("## "):
            break
        m = re.match(r"\s*-\s*([\w_]+):\s*(v\d+)", line)
        if in_prompts and m:
            out[m.group(1)] = m.group(2)
    return out


def check_prompt_versions(registry: Path, facts: Path) -> list[VersionMismatch]:
    reg = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    facts_versions = _facts_prompt_versions(facts)
    mismatches: list[VersionMismatch] = []
    for name, fv in facts_versions.items():
        rv = (reg.get(name) or {}).get("active_version")
        if rv is not None and rv != fv:
            mismatches.append(VersionMismatch(name=name, registry_version=rv, facts_version=fv))
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ci", action="store_true", help="committed-source checks only")
    parser.add_argument("--root", default=".", type=Path)
    args = parser.parse_args(argv)

    root: Path = args.root
    docs_dir = root / "docs"
    registry = root / "prompts" / "registry.yaml"
    facts = docs_dir / "reference" / "FACTS.md"

    failures = 0

    stale = find_stale_terms(docs_dir)
    if stale:
        failures += len(stale)
        print("STALE TERMS:")
        for h in stale:
            print(f"  {h.path}:{h.line}  '{h.term}'")

    mism = check_prompt_versions(registry, facts)
    if mism:
        failures += len(mism)
        print("PROMPT VERSION DRIFT:")
        for m in mism:
            print(f"  {m.name}: FACTS={m.facts_version} registry={m.registry_version}")

    if not args.ci:
        env = root / ".env"
        if not env.exists():
            print("NOTE: .env absent — skipping provider/model/chunk checks (local-only).")

    if failures:
        print(f"\ncheck_docs: FAIL ({failures} issue(s))")
        return 1
    print("check_docs: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `venv/bin/python -m pytest tests/test_check_docs.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Run the check against the real repo**

Run: `venv/bin/python scripts/check_docs.py`
Expected: `check_docs: OK` (if any stale term remains, fix the doc, not the denylist).

- [ ] **Step 6: Commit**

```bash
git add scripts/check_docs.py tests/test_check_docs.py
git commit -m "feat(docs): add check_docs.py freshness check (stale-term grep + prompt version sync)"
```

---

## Task 10: Wire CI + root README deep-links

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: root `README.md`

- [ ] **Step 1: Add the check_docs step to CI** — after the pytest step in `.github/workflows/ci.yml`:

```yaml
      - name: Docs freshness check
        run: python scripts/check_docs.py --ci
```

(Match the existing job's Python invocation style — the repo uses `python -m pytest`; mirror that environment.)

- [ ] **Step 2: Add 1-click showcase links to root README** — in the existing root `README.md`, ensure the docs section links directly to `docs/explanation/architecture.md` and `docs/explanation/design-decisions.md` (the showcase must be reachable in one click from the front door). Add a short "Design decisions" callout near the architecture section.

- [ ] **Step 3: Verify CI yaml parses**

Run: `venv/bin/python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/ci.yml')); print('yaml ok')"`
Expected: `yaml ok`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "ci: run check_docs in CI; link showcase docs from root README"
```

---

## Task 11: Final verification

- [ ] **Step 1: Full freshness check**

Run: `venv/bin/python scripts/check_docs.py`
Expected: `check_docs: OK`

- [ ] **Step 2: Unit tests still green**

Run: `venv/bin/python -m pytest -m unit -q`
Expected: all pass (≈230 + new check_docs tests).

- [ ] **Step 3: No stale terms anywhere live**

Run: `grep -rniE "llm_verifier|rewriter|flashrank rerank|generate_answer_v1|validate_prompts" docs --include=*.md | grep -v '/archive/'`
Expected: no output (matches inside design-decisions.md ADR #3/#5 describing the *removed* option are acceptable — review manually).

- [ ] **Step 4: No `_RU` deep docs remain**

Run: `find docs -name '*_RU.md' -not -path '*/archive/*'`
Expected: no output.

- [ ] **Step 5: All hub links resolve** (re-run Task 8 Step 3 link checker). Expected: no MISSING.

- [ ] **Step 6: Commit any final fixes, then summarize for merge**

```bash
git add -A && git commit -m "docs: final verification fixes" || echo "nothing to fix"
git log --oneline main..docs/overhaul
```

Report the commit list; leave the ff-merge to main + prod restart decision to Petr (per project deploy convention).

---

## Self-review notes

- **Spec coverage:** §3 tree → Task 1; §4 mapping → Tasks 1,3,5,8; §5 FACTS → Task 2; §6 check_docs → Tasks 9,10; §7 ADRs → Task 7; root README links → Task 10; non-goals respected (no autodoc/site). ✓
- **Placeholders:** none — every content task names exact source files and the specific stale items to fix; check_docs.py has full test + impl code. ✓
- **Type consistency:** `find_stale_terms`, `check_prompt_versions`, `STALE_DENYLIST`, `StaleHit.term/.path`, `VersionMismatch.name` used identically in tests and impl. ✓
