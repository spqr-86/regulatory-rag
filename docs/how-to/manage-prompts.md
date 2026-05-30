# How to manage prompts

Pipeline prompts are versioned Jinja2 templates loaded at runtime via `PromptManager`.
This keeps prompt text out of Python and lets you switch versions without a deploy.

- **Templates:** `prompts/` (`.j2` files)
- **Registry:** `prompts/registry.yaml` — maps a logical `prompt_id` to versioned file paths
- **Glossary:** `config/term_glossary.yaml` — deterministic domain-abbreviation expansion (not Jinja2, but part of prompt engineering)

## Live prompt families

The registry has **three** live families (current active versions:
[FACTS](../reference/FACTS.md#prompts)):

| `prompt_id` | role |
|---|---|
| `generate_answer` | answer synthesis from passages |
| `query_expand` | query expansion |
| `applicability_retriever` | applicability sub-retrieval |

`PromptManager.render()` is only ever called with these three ids. Older generate-answer
versions and the verifier / router / rag-simple / rag-complex templates were removed — only
the three families above remain.

## Registry shape

```yaml
generate_answer:
  active_version: "v8"
  versions:
    v7: "agents/generate_answer_v7.j2"
    v8: "agents/generate_answer_v8.j2"
```

## Rendering in code

```python
from src.infra.prompt_manager import PromptManager

manager = PromptManager()
prompt_text = manager.render(
    prompt_id="generate_answer",
    question="...",
    documents=[doc1, doc2],
)
llm.invoke(prompt_text)
```

`PromptManager` uses `StrictUndefined` — a missing template variable raises immediately
instead of sending a broken prompt to the LLM. This is also how you validate a template:
render it with sample context and confirm no error (there is no separate validation script).

## Switching versions without code changes

```bash
# Format: PROMPT_{PROMPT_ID}_VERSION={VERSION}
export PROMPT_GENERATE_ANSWER_VERSION=v7
python app.py
```

Use cases: local debugging of a new version, A/B in CI, quick rollback in production.

## Adding a new version

1. Create the file, e.g. `prompts/agents/generate_answer_v9.j2`.
2. Add it under the family's `versions:` in `registry.yaml`; bump `active_version` (or test via the env override above first).
3. Render-smoke-test with `PromptManager.render(...)` and run `pytest -m unit`.

## Security and logging

- **Path traversal protection:** the registry blocks absolute paths and `..` — it cannot read arbitrary files.
- **Privacy-first logging:** by default only the prompt's SHA256 hash and variable key names are logged, not the text (which may contain PII). Set `DEBUG_PROMPTS=true` to log full prompt text when debugging.

## Term glossary

`config/term_glossary.yaml` — unofficial domain abbreviations; expansion logic in
`src/glossary.py` (`expand_query_with_glossary`), applied by the V7 router:

- Words > 4 letters matched by morphological stem; abbreviations ≤ 4 letters matched as a whole word.
- A `[Глоссарий: term → expansion]` block is appended to the query.

To extend it, edit the YAML — no code change. See [adding eval questions](./add-eval-questions.md)
to test the effect on retrieval.
