# Quick Start

## Requirements

- Python 3.11+ (developed on 3.13)
- OpenAI API key — used for embeddings (`text-embedding-3-small`) and the eval judge
  (`gpt-4o`)
- OpenRouter API key — used for generation on both paths
  (`deepseek/deepseek-v4.1-flash` by default)
- Gemini / DeepSeek API key — only if you switch `SIMPLE/COMPLEX_LLM_PROVIDER`

## Install

```bash
git clone https://github.com/spqr-86/regulatory-rag.git
cd regulatory-rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure .env

Copy `.env.example` to `.env` and fill in your keys:

```env
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
```

Optional overrides (defaults are in `config/settings.py` and `src/v7/config.py`):

```env
# LLM providers (default: openrouter/deepseek-v4.1-flash on both paths)
# SIMPLE_LLM_PROVIDER=openrouter
# SIMPLE_MODEL_NAME=deepseek/deepseek-v4.1-flash
# COMPLEX_LLM_PROVIDER=openrouter
# COMPLEX_MODEL_NAME=deepseek/deepseek-v4.1-flash

# ChromaDB path (default: ./chroma_db)
# CHROMA_DB_PATH=./chroma_db

# V7/V8 pipeline flags (prefix V7_)
# V7_V8_ENABLE_MULTI_QUERY=true

# LangSmith tracing (optional)
# LANGSMITH_API_KEY=your_key
# LANGSMITH_TRACING_V2=true
```

## Index Documents

Place PDF/DOCX files in `source_docs/` and run:

```bash
python index.py
```

> WARNING: `index.py` replaces the entire ChromaDB collection. It first checks that
> `source_docs/` has supported files and that they produce chunks; if not, it exits with
> code 1 and leaves the existing index untouched.

The UI reindex button is hidden unless `ENABLE_UI_REINDEX=true` is set in `.env` — keep it
off for public deployments.

The indexer uses docling `HybridChunker` (`max_tokens=400`, `merge_peers`) to chunk
documents by structural headings and clauses. The shipped index holds 12 regulatory
documents (see [FACTS](reference/FACTS.md#corpus)).

## Run the UI

The root screen is Department Q&A. It requires the v2 Department index, corpus manifest and
object-profile source directory:

```bash
DEPARTMENT_QA_MODE=v2 \
CORPUS_MANIFEST_PATH=corpus/manifest.yaml \
SOURCE_DOCS_PATH=./source_docs_dept \
CHROMA_DB_PATH=./chroma_db_dept_v2 \
CHROMA_COLLECTION_NAME=department_demo_v2 \
streamlit run app.py --server.port 8502
```

Open `http://localhost:8502`. Select where to search: `Закон для объекта`,
`ЛНА для объекта`, `Закон + ЛНА для объекта`, or `Общая нормативная база`. The first three
options select a unit. The scoped service runs the shared V7 graph in retrieval-only mode for
the selected external/internal corpora, combines that evidence with the unit's object profile,
and generates one Department answer. External and internal retrieval branches run in
parallel by default (`DEPARTMENT_RETRIEVAL_WORKERS=4`).

`Общая нормативная база` hides unit/profile controls and uses
`GENERIC_CHROMA_DB_PATH` / `GENERIC_CHROMA_COLLECTION` (defaults:
`./chroma_db` / `documents`). Its query goes through the full V7 graph:
`intent_gate → router → rag_simple → evaluate_triage → [rag_complex] → generate_answer`
(insufficient results escalate to `rag_complex`, then answer or abstain).

## Run the API

```bash
uvicorn api:app --port 8503
```

Endpoints: `POST /query` (body `{"question": "..."}`), `POST /retrieve`, `GET /corpus`,
`GET /health`. Full reference: [reference/api](reference/api.md).

## Eval

```bash
python eval/run_v7_eval.py --skip-judge    # pipeline only, no LLM judge (~$0)
python eval/run_v7_eval.py                 # full eval with the gpt-4o judge
python eval/run_v7_eval.py --limit 5       # smoke test
```

Results written to `benchmarks/eval_v7_{date}.jsonl`. See [eval/README.md](../eval/README.md).

## Tests

```bash
pytest -m unit          # ~520 unit tests (CI gate)
pytest                  # full suite (~900 tests; 5 pre-existing failures —
                        # test_agent_tools, v7/test_bridge_rerank)
```
