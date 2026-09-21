# Regulatory RAG — evidence-gated compliance Q&A

In a distributed organisation the same compliance questions come back every week: is this
briefing still mandatory, how often, and does it apply to *this* particular site. The answer
usually sits in three places at once — an external regulation (ГОСТ, СНиП, ТК РФ, fire- and
labour-safety rules), the unit's own local act, and the recorded facts of the object itself.
Manual lookup across hundreds of PDFs with dense cross-references is slow, and in compliance
a confident unsupported answer is worse than an explicit "I don't know".

**This system answers such questions with citations — or refuses.** Every routing decision is
deterministic (score thresholds, no LLM in the loop), the answer is released only when the
evidence passes a sufficiency gate, and retrieval and generation are measured separately so
that a wrong answer can be attributed to a missing chunk or to a bad decision over a good one.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![CI](https://github.com/spqr-86/regulatory-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/spqr-86/regulatory-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Retrieval** (90 questions taken verbatim from OT/PB practitioner forums, never used for
tuning): HR@5 **0.63** · HR@12 **0.81** · MRR **0.50**.
**Generation** (56-question golden set, `gpt-4o` judge, showcase default
`deepseek/deepseek-v4.1-flash`): in-scope correctness **7.91 / 10** ·
faithfulness **0.926** · answer relevance **0.887** · **~$0.0066/query**, p50 **24.0 s**.

One number still misses its target and is reported anyway: false-sufficiency 10.0% against a
<10% target (0.1 pp outside). The showcase default also runs ~5× slower than the prior
GPT-4o-mini baseline (p50 4.5 s → 24 s), unexplained so far. Sample sizes and what each metric
actually denominates are in [Metrics](#metrics); the pricing fix behind the cost number is in
[the memo](./docs/evaluation/experiments/showcase-default-golden-set.md).

> Canonical values live in [docs/reference/FACTS.md](./docs/reference/FACTS.md). Design
> reasoning: [docs/explanation/design-decisions.md](./docs/explanation/design-decisions.md).
> Full methodology, per-experiment evidence and threats to validity:
> [evaluation report](./docs/evaluation/README.md).

[Russian README →](./README_RU.md)

---

## How it works

```mermaid
flowchart TD
    subgraph Ingestion
        Docs[PDF / DOCX] --> Docling[Docling Parser]
        Docling --> Split[HybridChunker max_tokens=400, merge_peers]
        Split --> Embed[OpenAI Embeddings]
        Embed --> DB[(ChromaDB)]
    end

    subgraph V7 [V7 LangGraph Pipeline]
        Q[Query] --> Gate{intent_gate + domain gate}
        Gate -->|noise / out-of-scope| End[END / abstain]
        Gate -->|in-domain| Router[router + glossary + multi-query]
        Router --> Simple[rag_simple hybrid top-12 + CrossEncoder]
        Simple --> Triage{evaluate_triage hard gate + gap}
        Triage -->|sufficient| Gen[generate_answer]
        Triage -->|insufficient| Complex[rag_complex top-60 + MMR]
        Complex --> Eval[evaluate_complex]
        Eval -->|pass| Gen
        Eval -->|fail| Abstain[abstain]
        Gen --> Answer[Answer + sources]
    end
```

Key design decisions:
- **No LLM routing** — all branching uses deterministic score thresholds, so the same query
  takes the same path twice.
- **Abstain > hallucinate** — the system refuses when retrieval confidence is low; triage is a
  three-metric hard gate plus a structured sufficiency gap that pulls in cross-referenced
  clauses before escalating ([triage](./docs/explanation/triage.md)).
- **Two-stage retrieval** — a fast path handles most queries; the slow path (top-60 + MMR)
  activates only when the gate says the evidence is thin.

The shipped index holds 12 regulatory documents (~7.8k chunks) as an example corpus — bring
your own. Exact counts: [FACTS § corpus](./docs/reference/FACTS.md#corpus).

📖 **Docs:** [architecture](./docs/explanation/architecture.md) · [design decisions](./docs/explanation/design-decisions.md) · [evaluation report](./docs/evaluation/README.md) · [FACTS](./docs/reference/FACTS.md) · [full documentation](./docs/README.md)

---

## Metrics

| Metric | Value | Measured on |
|---|---|---|
| Retrieval HR@5 / HR@12 / MRR (hybrid) | 0.63 / 0.81 / 0.50 | 90 practitioner questions |
| In-scope correctness | 7.91 / 10 | 43 in-scope questions (target >7.5, met) |
| Correctness, all questions | 7.98 / 10 | 56-question golden set |
| Faithfulness | 0.926 | 56-question golden set |
| Answer relevance | 0.887 | 56-question golden set |
| OOS abstain rate | 1.00 | out-of-scope subset only — 7 questions, a small sample |
| False-sufficiency rate | 10.0% | share of simple-path answers the judge scored < 5/10 (target <10%, 0.1 pp outside) |
| Complex-path rate | 24.5% | 56-question golden set |
| Latency p50 / p95 / mean | 24.0 / 71.3 / 30.95 s | per query, end to end |
| Cost / query | $0.00657 ($0.348 / run) | provider token usage, `src/pricing.py` rate card |

**How to read these.** The golden set is 56 questions — 43 in-scope, 7 out-of-scope, 6 with a
false premise; 53 of 56 answers were valid in the reported run. *False-sufficiency* is not a
hallucination rate and not a gate error rate: it is the share of answers released on the fast
path that the judge then scored below 5/10 (`eval/run_v7_eval.py`) — the question it answers
is "how often did the fast path release something weak". *OOS abstain* is measured on the
out-of-scope subset alone, so 1.00 rests on 7 questions and should be read as a sanity check,
not as a guarantee. All generation numbers are judge-dependent: compare runs only under the
same judge. Retrieval is measured independently (`eval/run_retrieval_eval.py`) on questions
frozen before any tuning.

---

## What the measurements showed

Retrieval and generation are measured **separately** — a single end-to-end score hides
whether a wrong answer came from a missing chunk or from a bad decision over a good chunk.
Full methodology, per-experiment memos and threats to validity:
[docs/evaluation/](./docs/evaluation/README.md).

**Retrieval backbones** (90 practitioner questions; hybrid is production):

| Backbone | HR@5 | HR@12 | MRR | p50 |
|---|---:|---:|---:|---:|
| hybrid (production) | 0.633 | 0.811 | 0.503 | 550 ms |
| vector-only | 0.589 | 0.822 | 0.486 | 148 ms |
| bm25-only | 0.500 | 0.667 | 0.352 | 24 ms |

BM25-only is disqualified; vector vs hybrid is a near-tie (hybrid wins the top-5 and MRR that
feed reranking, vector wins HR@12 and runs ~3.5× faster). Hybrid kept on measured grounds,
not on "best practice" ([memo](./docs/evaluation/experiments/retrieval-backbones.md)).

**Department Q&A — object sheet as a profile (`v2`) vs in the index (`v1`)** (9 questions,
expectations committed before the run, `gpt-4o-mini`):

| Metric | v1 | v2 |
|---|---:|---:|
| Expected status + reasons | 7/9 | 8/9 |
| Required sub-answers credited | 7/17 | 11/17 |
| Forbidden conclusions | 2 | 1 |

Hypothesis, pre-committed expectations, controlled comparison, failure analysis, architecture
change. The remaining `v2` error is a norm threshold applied to a fact of the wrong quantity;
it survived prompt iterations, a typed object sheet and a post-generation verifier, and was
only fixed by the model
([memo](./docs/evaluation/experiments/department-qa-object-profile.md)).

The current scoped Department service was then checked on the same 9-question set with the
showcase model: the parallel configuration passed **9/9** contract checks with no
unknown-field or forbidden-conclusion violations. This is the accepted pre-deploy cutover
result, not a claim about the older paired run above
([cutover memo](./docs/evaluation/experiments/department-scoped-service-pr4-wiring.md)).

**Cheap-model selection on 4 adversarial threshold traps** (mode `v2`, one run per model):

| Model | Traps passed | Cost, 4 q |
|---|---:|---:|
| `deepseek/deepseek-v4.1-flash` | **4/4** | $0.022 |
| `openai/gpt-5-mini` | 4/4 | $0.049 |
| `google/gemini-3-flash-preview` | 4/4 | $0.030 |
| `openai/gpt-4o-mini` | 2/4 | $0.004 |
| `anthropic/claude-haiku-4.5` | 2/4 | $0.057 |
| `deepseek/deepseek-v4-flash` | 1–1.5/4 | $0.003 |

DeepSeek V4.1 Flash is the cheapest model that passed all traps and is the showcase default.
The contract status did not separate correct from wrong answers — models were compared on
semantics ([memo](./docs/evaluation/experiments/cheap-model-selection.md)).

Rejected after measurement: tuning `RRF_K` (dead knob), a different hard-gate threshold
profile (81 profiles, none safer), norm-to-field markup. Negative results are documented, not
hidden: [experiments/](./docs/evaluation/experiments/).

---

## Department Q&A

A second product line on the same retrieval core: questions asked **by a specific unit**
about its own object, answered over two norm levels — company-wide legislation (`external`)
and the unit's own local acts (`internal`) — plus a third evidence level, the unit's object
sheet. Vertical slice on a synthetic corpus, limits documented.

The sheet is not a norm to be ranked: it is parsed into a structured **object profile** and
passed to the prompt whole (`DEPARTMENT_QA_MODE=v2`, default), instead of being indexed as
ordinary chunks (`v1`). The answer schema adds `object_facts` (facts quoted from the sheet)
and `applied_conclusions` (an object fact plus the norm applied to it) to the existing answer
contract.

This is the main Streamlit screen. The user selects a unit and the corpus scope (law, local
acts, or both); the service runs external and internal scoped retrieval through the shared V7
runtime, then generates one answer over the retrieved norms and the full object profile.
Generic regulatory search is the fourth option in the same source selector; it hides object
controls and runs the full Generic V7 graph without claiming applicability to a site.

`answered` means **"citations checked"**, not "content verified": every cited id exists with
the right role, no clarifying question is pending, both norm levels are present, a cited
profile carries a fill-in date, and a deterministic gate rejects an applied conclusion that
cites a field the sheet marks `unknown`. Whether the cited text actually supports the claim is
measured in eval, not enforced at runtime — and the UI says so.

- Specs: [department Q&A MVP](./docs/design/2026-09-14-department-qa-mvp-design.md) · [object profile](./docs/design/2026-09-15-object-profile-design.md)
- Mode, env and guarantee boundary: [FACTS § department qa](./docs/reference/FACTS.md#department-qa)
- Results and known error: [evaluation report](./docs/evaluation/README.md)

---

## Quick start

```bash
git clone https://github.com/spqr-86/regulatory-rag.git
cd regulatory-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill OPENAI_API_KEY (embeddings, complex path, judge) + OPENROUTER_API_KEY (simple path)
```

For the generic regulatory corpus, drop PDF/DOCX files into `source_docs/`, then:

```bash
python index.py                 # index documents → ChromaDB (rebuilds the collection; the old index is removed only after chunking succeeds)
streamlit run app.py --server.port 8502   # UI at http://localhost:8502
uvicorn api:app --port 8503                # REST API at http://localhost:8503/docs
```

The root UI is Department Q&A and therefore also needs its manifest, object-profile source
directory, and dedicated Chroma collection. The complete Department launch command and mode
contract are in [Quick Start](./docs/getting-started.md#run-the-ui). The same screen also
offers `Общая нормативная база`, backed by the separate Generic index.

Defaults: ChromaDB, OpenAI embeddings, and a two-provider LLM split (OpenRouter on the simple
path, OpenAI on the complex path). Every layer — LLM, embeddings, reranker, vector store — is
swappable via `.env`, and the glossary, prompts and corpus are what you change to move the
system to another domain: [configuration reference](./docs/reference/configuration.md).

**Monitoring** (optional): `docker compose up -d` brings up Postgres + Grafana; with
`V7_TELEMETRY_WRITER=postgres` every query becomes a row (cost, latency, route, tokens,
`source`) and the dashboard *Regulatory RAG — запросы* at `http://localhost:3000/d/regrag-queries`
shows cost, 👍/👎 rate, routes and p50/p95. Without the stack nothing breaks: events go to
`logs/events.jsonl` and can be replayed later. Setup, ports and troubleshooting:
[how-to/run-monitoring-stack.md](./docs/how-to/run-monitoring-stack.md).

**REST API:** `POST /query` (full pipeline), `POST /retrieve` (retrieval only, no LLM),
`GET /corpus`, `GET /health`. Request/response shapes, examples and rate limits:
[reference/api.md](./docs/reference/api.md); interactive docs at `http://localhost:8503/docs`.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (V7 deterministic graph) |
| LLM | OpenRouter `deepseek/deepseek-v4.1-flash` (both paths); OpenAI, Gemini, DeepSeek, OpenRouter configurable per path |
| Embeddings | OpenAI text-embedding-3-small (local sentence-transformers optional) |
| Vector store | ChromaDB |
| Reranking | CrossEncoder (sentence-transformers); FlashRank selectable |
| ETL | Docling (PDF/DOCX → chunks), HybridChunker |
| Evaluation | custom LLM-as-judge (faithfulness, relevance, correctness) + IR metrics (HR@k, MRR) |
| Monitoring | Postgres + Grafana (docker compose), events written from inside the graph |
| UI | Streamlit |

---

## Project status

**Portfolio MVP complete (2026-09-21).** Core pipeline and eval closed 2026-09-11; the
Department Q&A / object-profile mode below landed later and closed 2026-09-21. Built:

- **Evidence-gated LangGraph pipeline** — deterministic routing, three-metric sufficiency
  gate, structured triage gap, explicit abstention
- **Hybrid retrieval** — BM25 + vectors, CrossEncoder rerank, two-stage, with
  cross-reference expansion and glossary/multi-query expansion
- **Object-aware compliance mode** — Department Q&A over external + internal norms and a
  typed object profile, with deterministic citation and unknown-field gates
- **Offline evaluation** — golden set + a 90-question practitioner retrieval set, separate
  retrieval and generation measurement, per-query cost and latency, negative results kept
- **Online telemetry** — every query a row in Postgres, Grafana dashboard, 👍/👎 feedback;
  the whole stack is one `docker compose up`

The scoped Department build is deployed on a VPS as a localhost-only Streamlit service
(port 8502). HTTP health, configuration loading, local tests and a post-deploy live
generation smoke pass. The full shipped-capability checklist and the
optional post-MVP backlog (independent judge validation, error attribution between retrieval
and generation, a chunking experiment for tables and headers) are in
[docs/roadmap.md](./docs/roadmap.md); results, rejected variants and threats to validity are
in the [evaluation report](./docs/evaluation/README.md).

---

**Author:** Petr Baldaev — [LinkedIn](https://linkedin.com/in/petr-baldaev-b1252b263/) · [GitHub](https://github.com/spqr-86)

[Changelog →](./CHANGELOG.md)
