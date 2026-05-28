# Docs

Documentation for the Regulatory RAG system — a retrieval-augmented Q&A system for Russian normative documents.

## Table of Contents

| File | Covers |
|------|--------|
| [architecture/README.md](./architecture/README.md) | Pipeline overview, codebase map, how to add a node |
| [architecture/v7-how-it-works.md](./architecture/v7-how-it-works.md) | Detailed walkthrough of V7 nodes, hard gates, threshold calibration |
| [architecture/triage-how-it-works.md](./architecture/triage-how-it-works.md) | How `evaluate_triage` works: metrics, 3-way routing, threshold rationale |
| [guides/quick-start.md](./guides/quick-start.md) | Install, configure, index, run UI and API |
| [guides/testing.md](./guides/testing.md) | How to verify documentation and run tests |
| [guides/prompt-management.md](./guides/prompt-management.md) | Jinja2 prompt registry, versioning, A/B testing |
| [guides/adding-questions.md](./guides/adding-questions.md) | How to add questions to the eval dataset |
| [evaluation/README.md](./evaluation/README.md) | Eval framework, metrics, LLM-as-judge, report format |
| [DATA_PIPELINE.md](./DATA_PIPELINE.md) | How documents are indexed: Docling, HybridChunker, embeddings, ChromaDB |
| [passport.md](./passport.md) | Project summary: stack, eval results, architecture bullets |
| [plans/backlog.md](./plans/backlog.md) | Active backlog and known issues |
