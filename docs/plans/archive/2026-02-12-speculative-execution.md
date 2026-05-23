# Speculative Execution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce perceived latency by 1-2 seconds by running search in parallel with query routing.

**Architecture:**
- Use `concurrent.futures.ThreadPoolExecutor` in `stream_events` to run `RouterAgent.route` and `search_documents` (via retriever) concurrently.
- Inject the results (route decision and search results) into the `initial_state` of the LangGraph workflow.
- Update `_router_node` to respect pre-calculated routing decisions.
- Ensure the ReAct agent is aware of the pre-fetched documents.

**Tech Stack:** Python, LangGraph, Threading.

### Task 1: Verify Parallel Execution Capability

**Files:**
- Create: `tests/test_speculative_execution.py`

**Step 1: Write the test**
Create a test that simulates a slow router (1s) and a slow search (1s) running in parallel. Total time should be ~1s, not 2s.

```python
import time
import pytest
from concurrent.futures import ThreadPoolExecutor

def slow_router():
    time.sleep(1.0)
    return "rag_simple"

def slow_search():
    time.sleep(1.0)
    return ["doc1", "doc2"]

def test_parallel_execution():
    start = time.perf_counter()
    with ThreadPoolExecutor() as executor:
        f1 = executor.submit(slow_router)
        f2 = executor.submit(slow_search)
        route = f1.result()
        docs = f2.result()
    end = time.perf_counter()
    
    duration = end - start
    print(f"Duration: {duration:.2f}s")
    
    assert duration < 1.5
    assert route == "rag_simple"
    assert len(docs) == 2
```

**Step 2: Run test**
`pytest tests/test_speculative_execution.py -v -s`

**Step 3: Commit**
`git add tests/test_speculative_execution.py && git commit -m "test: verify thread pool parallelism"`

### Task 2: Implement Speculative Logic in `stream_events`

**Files:**
- Modify: `agents/multiagent_rag.py`

**Step 1: Write failing test (Conceptual)**
We can't easily unit test `stream_events` integration without mocking the whole graph. We will rely on manual verification or integration test later.
For this plan, we will proceed with implementation.

**Step 2: Modify `stream_events`**
Refactor `stream_events` to:
1. Define `run_router` and `run_search` helper functions.
2. Use `ThreadPoolExecutor` to run them.
3. Populate `initial_state` with results.
   - If search finishes, add to `chunks_found` and `searches_performed`.
   - If router finishes, add to `route_type` and set a flag (e.g., `router_done=True` inside state? Or just trust `route_type` if it's not default).

**Note on Router:**
We need to know if the route in `state` is the *final* decision or just the default `RAG`.
We can add a field `is_routed: bool` to `RAGState`.

**Step 3: Update `RAGState` definition**
Add `is_routed: bool` to `RAGState` in `agents/multiagent_rag.py`.

**Step 4: Update `_router_node`**
Modify `_router_node` to check `state.get("is_routed")`. If True, return existing state (pass-through).

**Step 5: Verify**
Run the app or a test script to see if it works.

### Task 3: Agent Awareness

**Files:**
- Modify: `prompts/agents/rag_agent_v1.j2` (Maybe? Or rely on `searches_performed`)

**Step 1: Check Prompt**
The prompt iterates over `searches_performed`.
```jinja2
{% for search in searches_performed %}
- "{{ search.query }}" → {{ search.results_count }} результатов
{% endfor %}
```
If we populate `searches_performed` in `initial_state`, the agent will see it.
We need to ensure `searches_performed` format matches: `{"query": str, "results_count": int}`.

**Step 2: Implementation Details in `stream_events`**
When populating `initial_state`:
```python
initial_state["searches_performed"] = [
    {"query": expanded_query, "results_count": len(chunks)}
]
initial_state["chunks_found"] = chunks
```

### Task 4: Cleanup

**Files:**
- Remove: `tests/test_speculative_execution.py`

**Step 1: Commit changes**
