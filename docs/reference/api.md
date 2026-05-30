# REST API reference

FastAPI backend (`api.py`), run alongside Streamlit:

```bash
uvicorn api:app --port 8503
```

Interactive docs: `http://localhost:8503/docs`. (The Streamlit UI runs separately on the
deploy port — see [FACTS](FACTS.md#deploy).)

## `POST /query` — main RAG pipeline

```bash
curl -X POST http://localhost:8503/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Как часто проводится повторный инструктаж?"}'
```

```json
{
  "answer": "Повторный инструктаж проводится не реже одного раза в 6 месяцев...",
  "passages": [{"text": "...", "source": "2464.pdf", "score": 0.91}],
  "path": "rag_simple",
  "elapsed_sec": 4.2
}
```

`path` is the route taken (`rag_simple` / `rag_complex` / `abstain`).

## `POST /query/gosts` — GOST/SNiP corpus

Separate ChromaDB collection, separate generation model. Same request/response shape as
`/query`.

```bash
curl -X POST http://localhost:8503/query/gosts \
  -H "Content-Type: application/json" \
  -d '{"question": "Степень защиты IP55 — что означает?"}'
```

## `GET /health` — readiness

```bash
curl http://localhost:8503/health
# {"status": "ok", "pipeline_ready": true, "gosts_ready": true}
```

## `GET /` — service banner

Returns the API name/version.

## Hardening

- Rate limit: 10 requests/min per IP (slowapi).
- Request body capped (`question` max length).
- Errors return a `request_id` instead of internal detail; `X-Request-ID` header on responses.
