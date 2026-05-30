# Documentation

Regulatory RAG — a retrieval-augmented Q&A system for Russian regulatory documents (ГОСТ,
ТК РФ, СНиП, СП, federal laws) that answers with citations or explicitly abstains when
retrieval confidence is low. These docs follow a [Diátaxis](https://diataxis.fr/)-style
split: start here, learn *why* in Explanation, find recipes in How-to, look up facts in
Reference.

## Getting started

- [getting-started.md](./getting-started.md) — install, configure, index, run the UI and API

## Explanation — how and why it works

- [explanation/architecture.md](./explanation/architecture.md) — pipeline, the V7 graph, node map
- [explanation/triage.md](./explanation/triage.md) — how `evaluate_triage` decides sufficient vs escalate
- [explanation/design-decisions.md](./explanation/design-decisions.md) — **the 8 key design decisions, with evidence** (start here for the "why")

## How-to — task recipes

- [how-to/add-a-node.md](./how-to/add-a-node.md) — add a pipeline node
- [how-to/manage-prompts.md](./how-to/manage-prompts.md) — versioned Jinja2 prompts
- [how-to/add-eval-questions.md](./how-to/add-eval-questions.md) — extend the eval dataset
- [how-to/run-evaluation.md](./how-to/run-evaluation.md) — run a evaluation
- [how-to/run-tests.md](./how-to/run-tests.md) — tests, lint, docs freshness check

## Reference — facts and contracts

- [reference/FACTS.md](./reference/FACTS.md) — **canonical volatile facts** (models, thresholds, metrics, nodes) — single source of truth
- [reference/data-pipeline.md](./reference/data-pipeline.md) — indexing: Docling, HybridChunker, embeddings, ChromaDB
- [reference/evaluation.md](./reference/evaluation.md) — eval metrics and report format
- [reference/api.md](./reference/api.md) — REST endpoints

Historical plans and the V7 migration spec live in [archive/](./archive/).

---

## По-русски

Это документация RAG-системы по российским нормативным документам: отвечает с цитатами из
НПА либо честно отказывается, когда уверенности в найденном мало. Документы — на английском
(единый источник правды, без двойной поддержки). Начните с
[архитектуры](./explanation/architecture.md) и [ключевых решений](./explanation/design-decisions.md);
все актуальные цифры — в [FACTS](./reference/FACTS.md).
