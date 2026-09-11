# Quick Start

## Requirements

- Python 3.11+ (developed on 3.13)
- OpenAI API key — used for embeddings (`text-embedding-3-small`), generation
  (`gpt-4o-mini` / `gpt-4o` by default) and the eval judge (`gpt-4o`)
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
```

Optional overrides (defaults are in `config/settings.py` and `src/v7/config.py`):

```env
# LLM providers (default: openai)
# SIMPLE_LLM_PROVIDER=openai
# SIMPLE_MODEL_NAME=gpt-4o-mini
# COMPLEX_LLM_PROVIDER=openai
# COMPLEX_MODEL_NAME=gpt-4o

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

> WARNING: `index.py` is destructive — it drops the entire ChromaDB collection before reindexing.

The indexer uses docling `HybridChunker` (`max_tokens=400`, `merge_peers`) to chunk
documents by structural headings and clauses. The shipped index holds 12 regulatory
documents (see [FACTS](reference/FACTS.md#corpus)).

## Run the UI

```bash
streamlit run app.py --server.port 8502
```

Open `http://localhost:8502`. The query goes through the V7 graph:
`intent_gate → router → rag_simple → evaluate_triage → [rag_complex] → pack_context → generate_answer`
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
