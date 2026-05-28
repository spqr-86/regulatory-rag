# Prompt Management

## Overview

All V7 pipeline prompts (generation, verification, rewrite, expand) are stored as versioned
Jinja2 templates and loaded at runtime via `PromptManager`. This separates prompt text from
Python code and enables version switching without deploys.

- **Templates:** `prompts/` directory (Jinja2 `.j2` files)
- **Registry:** `prompts/registry.yaml` — maps logical `prompt_id` to versioned file paths
- **Related:** `config/term_glossary.yaml` — deterministic domain abbreviation expansion (not Jinja2, but part of prompt engineering)

---

## Directory Structure

```text
prompts/
├── registry.yaml              # Global version registry
├── common/
│   └── base_rules.j2          # BASE_RULES macro (10 edge cases)
├── agents/
│   ├── generate_answer_v1.j2  # Generate answer prompt (active)
│   ├── query_expand_v1.j2     # Query expansion prompt (active)
│   └── ...
└── chains/
    └── ...
```

---

## Registry (`registry.yaml`)

Maps a logical `prompt_id` to concrete versioned templates.

```yaml
applicability_retriever:
  active_version: "v2"
  versions:
    v1: "agents/query_expansion_v1.j2"
    v2: "agents/query_expansion_v2.j2"
```

---

## Usage in Code

Load prompts via `PromptManager`:

```python
from src.infra.prompt_manager import PromptManager

manager = PromptManager()

prompt_text = manager.render(
    prompt_id="research_agent",
    question="Что такое СИЗ?",
    documents=[doc1, doc2]
)

llm.invoke(prompt_text)
```

`PromptManager` uses `StrictUndefined` — if a template variable is missing, it raises an
exception immediately rather than sending a broken prompt to the LLM.

---

## Version Switching (Pinning)

Override the active version for a specific prompt via environment variable without touching code or registry:

```bash
# Format: PROMPT_{PROMPT_ID}_VERSION={VERSION}
export PROMPT_RESEARCH_AGENT_VERSION=v2
python app.py
```

Use cases:
- Local debugging of a new prompt version
- A/B testing in CI
- Quick rollback in production

---

## Security and Logging

1. **Path traversal protection:** The registry blocks absolute paths and `..` — cannot read arbitrary files from disk.
2. **Privacy-first logging:** By default only the SHA256 hash of the prompt and variable key names are logged. Prompt text (which may contain PII) is not logged. Set `DEBUG_PROMPTS=true` to log full prompt text for debugging.

---

## Advanced Techniques

### Chain-of-Thought (XML Tagging)

Templates use `<thought>` / `<answer>` XML tags to separate reasoning from the final answer.
Agent code parses these tags and the UI can display reasoning in a collapsible block.

### BASE_RULES Macro

`prompts/common/base_rules.j2` — shared macro `{% macro base_rules() %}`. Contains:
- Absolute prohibitions (no hallucinations, no extrapolation)
- Query preprocessing rules (substance vs. identifiers)
- 10 edge cases (outdated references, document overlaps, negative questions, glossary integration, fallback for unknown abbreviations, etc.)
- Answer format rules (direct answer, verbatim citations)

### Term Glossary

`config/term_glossary.yaml` — YAML dictionary of unofficial domain abbreviations.
Expansion logic is in `src/glossary.py` (`expand_query_with_glossary`), used by the V7 router.
Applied deterministically:
- Words > 4 letters matched by morphological stem; abbreviations ≤ 4 letters matched as whole word
- A `[Глоссарий: term → expansion]` block is appended to the query
- Fallback: LLM instruction for terms not in the glossary (BASE_RULES case 10)

---

## Validation

`scripts/validate_prompts.py` checks:
- YAML structure of the registry
- Existence of all files referenced in the registry
- No duplicate version keys

Run after any prompt changes:
```bash
python scripts/validate_prompts.py
```
