# Quick Start

## Requirements

- Python 3.11+
- OpenAI API key (embeddings via `text-embedding-3-small` + LLM judge `gpt-4o-mini`)
- Gemini API key (generation: Gemini 2.5 Flash / Gemini 3 Flash)

## Install

```bash
git clone https://github.com/spqr-86/regulatory-rag.git
cd regulatory-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure .env

Copy `.env.example` to `.env` and fill in your keys:

```env
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

Optional overrides (defaults are set in `config/settings.py` and `src/v7/config.py`):

```env
# LLM providers (default: gemini)
# SIMPLE_LLM_PROVIDER=gemini
# COMPLEX_LLM_PROVIDER=gemini

# ChromaDB path (default: ./chroma_db)
# CHROMA_DB_PATH=./chroma_db

# LangSmith tracing (optional)
# LANGSMITH_API_KEY=your_key
# LANGSMITH_TRACING_V2=true
# LANGSMITH_PROJECT=regulatory-rag
```

## Index Documents

Place PDF/DOCX files in `source_docs/` and run:

```bash
python index.py
```

> WARNING: `index.py` is destructive — it drops the entire ChromaDB collection before reindexing.

The indexer uses HybridChunker (docling_core, max_tokens=400) to chunk documents by structural headings and clauses. Current corpus: 11 PDFs, index not yet rebuilt after v3.0-hybrid.

## Run the UI

```bash
streamlit run app.py --server.port 8502
```

Open `http://localhost:8502`. The query goes through the V7 graph:
`intent_gate → router → rag_simple → evaluate_triage → [rag_complex] → generate_answer` (insufficient results escalate to `rag_complex`, then answer or abstain)

## Run the API

```bash
uvicorn api:app --port 8503
```

Endpoints:
- `POST /query` — V7 pipeline, body: `{"query": "..."}`
- `GET /health` — health check

## Eval

```bash
python eval/run_v7_eval.py --skip-judge    # pipeline only, no LLM judge (~$0)
python eval/run_v7_eval.py                 # full eval with gpt-4o-mini judge
python eval/run_v7_eval.py --limit 5       # smoke test
```

Results written to `benchmarks/eval_v7_{date}.jsonl`. See [evaluation/README.md](../evaluation/README.md).

## Tests

```bash
pytest -m unit          # 237 unit tests
pytest                  # all tests (3 pre-existing failures in test_evaluate_triage.py)
```
