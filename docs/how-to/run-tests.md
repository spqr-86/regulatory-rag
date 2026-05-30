# How to run tests and checks

Always work inside the project venv (`source venv/bin/activate`).

## Unit tests

```bash
pytest -m unit              # fast unit suite (the CI gate)
pytest -m integration       # integration tests (need real deps)
pytest -m "not slow"        # everything except slow tests
pytest tests/test_hard_gates.py -v   # a single file
```

Markers (`unit` / `integration` / `slow`) are configured in `pyproject.toml`. New features
and bugfixes must ship with unit tests in `tests/test_*.py` using `unittest.mock` for
injected dependencies.

## Lint

```bash
black . && ruff check . --fix
```

Run before every commit. Note: `ruff` strips unused imports — add an import and its first
use in the same edit, or it gets removed.

## Docs freshness check

```bash
python scripts/check_docs.py          # local: includes provider/chunk checks if available
python scripts/check_docs.py --ci     # committed-source checks only (what CI runs)
```

This verifies prompt versions against `prompts/registry.yaml` and greps live docs for
stale terms (removed nodes, old provider names, dead template versions). It is the
automated replacement for manually re-reading docs after a code change. If it flags a
real removed-thing mention that is intentional (e.g. "we removed the verifier"), exempt
that line with a `<!--freshness:ignore-->` marker rather than weakening the deny-list.

## End-to-end smoke

```bash
python scripts/trace_v7.py "для кого проводится повторный инструктаж?"
python scripts/trace_v7.py --no-chroma "привет как дела"   # stub mode
```

Prints the path taken and the final answer — the quickest confirmation the pipeline runs.
