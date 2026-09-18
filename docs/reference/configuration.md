# Configuration — backends and domain adaptation

Everything on this page is `.env` and YAML configuration; the pipeline code does not change.
Canonical current values (models, thresholds): [FACTS.md](./FACTS.md).

## Backend abstraction

LLM and vector store are accessed through factory layers (`src/infra/llm_factory.py`,
`src/backends/`). Adding a new provider is one function plus one registry entry.

| Layer | Shipped | Configurable via | Roadmap |
|-------|---------|------------------|---------|
| LLM   | OpenAI, Gemini, DeepSeek, OpenRouter | `SIMPLE_LLM_PROVIDER` / `COMPLEX_LLM_PROVIDER` | Anthropic |
| Vector store | Chroma | `VECTOR_STORE` | Qdrant, pgvector |
| Embeddings | OpenAI, local (sentence-transformers), hf_api | `EMBEDDING_PROVIDER` | — |

The showcase default runs OpenRouter `deepseek/deepseek-v4.1-flash` on both paths since
18.09.2026 (simple path picked it first, complex followed); embeddings and the eval judge
stay on OpenAI.

**Fully local embeddings** (LLM still over an API):

```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_NAME=ai-forever/sbert_large_nlu_ru
```

**Adding a new LLM provider** (example: Anthropic):

1. Add `_create_anthropic_llm(**kwargs)` to `src/infra/llm_factory.py`
2. Register it in `_LLM_PROVIDERS = {..., "anthropic": _create_anthropic_llm}`
3. Set `SIMPLE_LLM_PROVIDER=anthropic` (and/or `COMPLEX_LLM_PROVIDER`) in `.env`

Same pattern for vector stores — implement the `VectorStoreBackend` protocol in
`src/backends/`, register it in the factory.

## Adapting to your domain

The system ships tuned for Russian regulatory documents, but domain-specific knowledge is
isolated and easy to swap.

**Term glossary** (`config/term_glossary.yaml`) — maps informal abbreviations to their
official full names so BM25 and vector search can match indexed text. To extend:

```yaml
terms:
  "your abbreviation":
    official: "Full official term from your documents"
    source: "Regulation / standard reference (optional)"
```

No code changes needed — edit the YAML and restart.

**Prompts** (`prompts/`) — Jinja2 templates via `PromptManager`, versioned. Switch the
active version via `prompts/registry.yaml`.

**Corpus** — drop your PDFs into `source_docs/` and run `python index.py`. The chunker and
embeddings are language-agnostic.
