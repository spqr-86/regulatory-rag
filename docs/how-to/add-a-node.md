# How to add a pipeline node

Nodes are thin orchestrators: read state → call a function → write state. Keep the logic
in `nlp_core.py` / `hard_gates.py` / a dedicated module, not in the node body.

1. **Create the node** — `src/v7/nodes/<name>.py`. Signature: `def <name>(state: RAGState) -> RAGState`. Read what you need from `state`, call the real logic, return the updated state (or a partial dict of changed keys).

2. **Register it in the graph** — `src/v7/graph.py`: add the node to the `nodes` dict, then wire its edges with `g.add_edge(...)` or `g.add_conditional_edges(...)`. For a branch, write a `route_*` function returning the next-node key and map keys → node names.

3. **Inject dependencies via the bridge** — if the node needs an LLM or vector search, do not import a provider directly. Add a setter in `src/v7/bridge.py` and inject the function in `init_v7_pipeline()`. This keeps the node testable and provider-agnostic.

4. **Write unit tests** — `tests/test_<name>.py`, marked `@pytest.mark.unit`. Use `unittest.mock` for injected dependencies. Cover: happy path, the routing decision (if any), and the no-dependency / no-op case.

5. **Smoke-test end to end** —

```bash
venv/bin/python scripts/trace_v7.py "your test question"
```

The trace prints the path taken (colored) and the final answer, so you can confirm your
node is on the path you expect.

See [architecture.md](../explanation/architecture.md) for the full node list and flow.
