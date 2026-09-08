# REST API reference

FastAPI backend (`api.py`), run alongside Streamlit:

```bash
uvicorn api:app --port 8503
```

Interactive docs: `http://localhost:8503/docs`. (The Streamlit UI runs separately on the
deploy port — see [FACTS](FACTS.md#deploy).)

Four endpoints: `POST /query` (full pipeline), `POST /retrieve` and `GET /corpus`
(retrieval only, for batch clients), `GET /health` (liveness).

## `POST /query` — main RAG pipeline

```bash
curl -X POST http://localhost:8503/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Как часто проводится повторный инструктаж?"}'
```

Request: `{"question": str}` (1–2000 chars).

```json
{
  "answer": "Повторный инструктаж проводится не реже одного раза в 6 месяцев...",
  "passages": [{"text": "...", "source": "2464.pdf", "score": 0.91}],
  "path": "rag_simple → evaluate_triage → generate_answer → END",
  "elapsed_sec": 4.2
}
```

`path` is a human-readable trace of the route taken, derived from graph state
(`_infer_path`) — e.g. `rag_simple → evaluate_triage → rag_complex → generate_answer → END`,
`... → abstain → END`, or `intent_gate → END (chitchat/oos)`.

## `POST /retrieve` — retrieval only

Hybrid search (vector + BM25 → RRF merge, optional rerank), no LLM. Built for
latency-sensitive batch clients.

```bash
curl -X POST http://localhost:8503/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "повторный инструктаж", "k": 5}'
```

Request: `{"question": str, "k": int = 5 (1–50), "source_filter": str | null}`.
When `source_filter` is set, retrieval is scoped to that one document via a native
metadata filter and ranked within it by BM25.

```json
{
  "passages": [{"text": "...", "source": "2464.pdf", "score": 12.4}],
  "elapsed_sec": 0.5
}
```

## `GET /corpus` — indexed documents

```bash
curl http://localhost:8503/corpus
# {"sources": ["1479 ппр.pdf", "2464.pdf", "426-ФЗ.pdf", ...]}
```

Sorted unique `metadata.source` values in the index; cached after the first call.

## `GET /health` — liveness

```bash
curl http://localhost:8503/health
# {"status": "ok"}          200 when the pipeline is loaded
# {"detail": "not ready"}   503 otherwise
```

## Hardening

- Rate limits (slowapi, per IP): 30/min default, **10/min** on `/query`,
  **600/min** on `/retrieve` and `/corpus`.
- Request body capped (`question` max 2000 chars).
- Errors return `internal error (request_id=…)` instead of internal detail;
  every response carries an `X-Request-ID` header.
