# Docs Overhaul — Design Spec

**Date:** 2026-05-30
**Status:** Approved (design); pending implementation plan
**Scope:** `docs/` of regulatory-rag — accuracy + structure + a freshness mechanism

---

## 1. Goal

Make `docs/` (a) accurate to the current code, and (b) genuinely useful, convenient,
detailed, modern documentation. Today docs are *systemically* stale (not point fixes)
and every file is duplicated EN + `_RU`, which is the root cause of drift: a code change
must be synced across two language copies in six places, so it isn't.

## 2. Decisions (agreed)

| # | Decision | Choice |
|---|----------|--------|
| Audience | All of portfolio / self-onboarding / contributor, **priority = portfolio (showcase for GitHub + interviews)** | D, priority A |
| Bilingual | EN everywhere; **RU only in the hub README** (as a bottom section). All `_RU.md` deep docs removed | A |
| Freshness | **Centralize volatile facts in one reference file** + a lightweight `check_docs.py` in CI | B |
| Structure | **Diátaxis-lite** (Explanation / How-to / Reference / top-level Getting Started) | 1 |

## 3. Target structure

```
docs/
├ README.md                 — hub: 1 paragraph + 4-bucket nav. ONLY bilingual file (RU block at bottom).
├ getting-started.md        — install, configure, index, run UI + API (was guides/quick-start). Top-level, no tutorial/ folder for a single file.
│
├ explanation/              — the "WHY" (showcase)
│  ├ architecture.md        — pipeline overview + V7 graph (current: domain_gate, NO verifier, OpenAI, CrossEncoder, prompt v8) + codebase map + pluggable-backends note. Merge of architecture/README + v7-how-it-works, de-staled.
│  ├ triage.md              — evaluate_triage deep dive (from triage-how-it-works, de-staled)
│  └ design-decisions.md    — NEW. Portfolio centerpiece. 8 ADR-style records (see §6).
│
├ how-to/                   — recipes
│  ├ add-a-node.md          — from architecture/README
│  ├ manage-prompts.md      — from guides/prompt-management
│  ├ add-eval-questions.md  — from guides/adding-questions
│  ├ run-evaluation.md      — from evaluation/README (how-to part)
│  └ run-tests.md           — from guides/testing
│
├ reference/                — facts, single source of truth
│  ├ FACTS.md               — NEW. Canonical volatile facts (see §5). check_docs.py verifies against this.
│  ├ data-pipeline.md       — from DATA_PIPELINE.md (+ HybridChunker decision note)
│  ├ evaluation.md          — from evaluation/README (reference part: metrics, report format)
│  └ api.md                 — REST endpoints (from root README API section)
│
└ archive/                  — historical, excluded from nav and from check_docs.py stale-term grep
   ├ plans/                 — docs/plans/* moved here (mined for design-decisions first)
   └ migration-v7          — docs/feature/migration-v7
```

**Removed:** all `*_RU.md` (13 files, except the RU block inside the hub README);
`docs/passport.md` (content folded into README + FACTS).

**Repo-level additions:** `scripts/check_docs.py` + a step in `.github/workflows/ci.yml`.

**Root README.md (repo front door):** add 1-click deep links into
`docs/explanation/architecture.md` and `docs/explanation/design-decisions.md` —
the showcase must be reachable in one click from the front page.

## 4. File mapping (old → new)

| Old | New | Action |
|-----|-----|--------|
| `docs/README.md` (+`_RU`) | `docs/README.md` | rewrite as 4-bucket hub; keep RU block; drop `_RU` file |
| `docs/guides/quick-start.md` (+`_RU`) | `docs/getting-started.md` | move, de-stale, drop `_RU` |
| `docs/architecture/README.md` (+`_RU`) | `docs/explanation/architecture.md` + `docs/how-to/add-a-node.md` | split, de-stale, drop `_RU` |
| `docs/architecture/v7-how-it-works.md` (+`_RU`) | merge into `docs/explanation/architecture.md` | merge, heavy de-stale, drop `_RU` |
| `docs/architecture/triage-how-it-works.md` (+`_RU`) | `docs/explanation/triage.md` | move, de-stale, drop `_RU` |
| `docs/DATA_PIPELINE.md` (+`_RU`) | `docs/reference/data-pipeline.md` | move, de-stale, add chunker note, drop `_RU` |
| `docs/evaluation/README.md` (+`_RU`) | `docs/reference/evaluation.md` + `docs/how-to/run-evaluation.md` | split, drop `_RU` |
| `docs/guides/prompt-management.md` (+`_RU`) | `docs/how-to/manage-prompts.md` | move, de-stale, drop `_RU` |
| `docs/guides/adding-questions.md` (+`_RU`) | `docs/how-to/add-eval-questions.md` | move, drop `_RU` |
| `docs/guides/testing.md` (+`_RU`) | `docs/how-to/run-tests.md` | move, drop `_RU` |
| `docs/passport.md` | — | delete (content → README + FACTS) |
| `docs/plans/*` | `docs/archive/plans/*` | move (mine for design-decisions first) |
| `docs/feature/migration-v7` | `docs/archive/migration-v7` | move |
| — | `docs/reference/FACTS.md` | new |
| — | `docs/reference/api.md` | new (from root README) |
| — | `docs/explanation/design-decisions.md` | new |
| — | `scripts/check_docs.py` | new |

## 5. `reference/FACTS.md` — single source of truth

Canonical, anchor-marked (greppable) list of volatile facts. Prose elsewhere **links
here and does not repeat numbers**. Contents:

- **Providers/models:** SIMPLE `openai`/`gpt-4o-mini`, COMPLEX `openai`/`gpt-4o`, JUDGE `gpt-4o`, reranker `crossencoder`, embeddings `text-embedding-3-small`
- **Thresholds (with env names):** `HARD_GATE_THRESHOLD=0.50`, `TRIAGE_SOFT_THRESHOLD=0.38`, `COMPLEX_THRESHOLD=0.35`, `COMPLEX_MIN_PASSAGES=8`, `COMPLEX_MIN_KW_OVERLAP=0.20`, `DOMAIN_GATE_THRESHOLD=0.25`, `SIMPLE_TOP_K=12`, `COMPLEX_TOP_K=60`
- **Corpus:** 12 НТД, 7718 chunks
- **Prompts:** `generate_answer` v8, `query_expand` v1, `applicability_retriever` v2
- **Nodes (order):** intent_gate → domain_gate → router → rag_simple → evaluate_triage → rag_complex → evaluate_complex → visual_enrichment → generate_answer / abstain
- **Metrics:** overall 0.80, in-scope 7.4/10, faithfulness 0.988, OOS abstain 1.0, latency 9.7s, cost $0.0102/query
- **Deploy:** port 8502, tmux `sia`

All metric numbers verified against `benchmarks/` before being written — never asserted from memory.

## 6. `scripts/check_docs.py` — freshness check

Two checks. **Honest CI scope** stated in the doc itself:

1. **Value sync vs committed sources** (works in CI — files are in repo):
   - thresholds in FACTS ↔ `src/v7/config.py` defaults
   - prompt active versions in FACTS ↔ `prompts/registry.yaml`
2. **Forbidden stale-term grep** over `docs/` excluding `docs/archive/` (works in CI):
   - fails if any of: `llm_verifier`, `rewriter`, `FlashRank rerank`, `generate_answer_v1`, `Gemini … (simple path)` (and a maintained deny-list) appears in a live doc.

**Out of CI scope (local-only, needs `.env` / chroma):** provider/model names and
reranker backend live in `.env` (gitignored secrets), absent in CI; chunk count needs
a live ChromaDB. `check_docs.py` checks these only when `.env`/chroma are present, and
says so — no illusion that CI verifies everything.

- Output: a table of mismatches; exit 1 on any failure.
- CI: `python scripts/check_docs.py --ci` step in `.github/workflows/ci.yml` after pytest.

## 7. `explanation/design-decisions.md` — the showcase

Lightweight ADR records. Each: **Context → Options → Choice → Why → Evidence (numbers)**.
Narrative spine: *reliable RAG for compliance — a hallucinated legal answer is a liability.*

1. **Deterministic hard gates, no LLM routing** — reproducibility (same query+index → same path), no LLM in branching, cheaper.
2. **Abstain > hallucinate + domain_gate** — pre-retrieval OOS filter (cosine to corpus centroid). Evidence: OOS abstain 1.0.
3. **CrossEncoder over FlashRank** — FlashRank inflated all scores to ~0.999 → no ranking → exact chunk sank to pos 15; CrossEncoder → pos 0. Evidence: false-sufficiency 22.9%→0%, in-scope 7.08→7.4.
4. **chunk_id namespaced by source** — chunk_id was never written → RRF fused by list position, dedup was a no-op → duplicate context. Found via code review. Evidence: in-scope +0.3, latency −45%.
5. **Removed verifier/rewriter subgraph** — complexity + latency without measurable gain; insufficient → rag_complex directly. (Maturity: knowing when to delete code.)
6. **Two-tier LLM** (gpt-4o-mini simple / gpt-4o complex) — cost. Evidence: $0.0102/query.
7. **Prompt-as-code** (versioned Jinja2 + registry, v8 anti-sycophancy + value↔condition) — paired thematically with #2: abstain guards retrieval, anti-sycophancy guards generation. Evidence: inverted-norm bugs B3/B4 fixed.
8. **Eval methodology** — measured judge noise (±0.25) *before* chasing deltas; trust per-question traces over a noisy aggregate. Differentiator: eval rigor.

Pluggable backends (LLM factory + VectorStore Protocol) and HybridChunker
(structure-aware chunking by article/clause) are **not** ADRs — covered as notes in
`architecture.md` and `reference/data-pipeline.md` respectively, to keep the ADR set tight (≤8).

All evidence numbers verified against `benchmarks/` and session logs before writing — never from memory.

## 8. Non-goals (YAGNI)

- No generated/autodoc docs (Sphinx-style) — rejected as too heavy (option C).
- No docs site / static-site generator — Markdown on GitHub is the target surface.
- No translation of deep docs back to RU.
- No unrelated refactoring of code outside what de-staling requires.

## 9. Verification

- `python scripts/check_docs.py` passes (both checks).
- Every doc cross-checked against current code (graph.py, config.py, registry.yaml, .env, bridge.py) — no stale node/provider/version/threshold.
- All deep `_RU.md` removed; only the hub README is bilingual.
- Root README links reach `explanation/` in one click.
- `pytest -m unit` still green (no code regressions from any touched scripts).
