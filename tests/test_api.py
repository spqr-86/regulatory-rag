"""Tests for api.py — CARD-1.1 (request_id, exception hiding) and CARD-1.2 (rate limit, length cap)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# App setup: patch pipeline so we never touch ChromaDB in unit tests
# ---------------------------------------------------------------------------


def _make_client(pipeline_app=None):
    """Return a TestClient with app.state pre-populated (no lifespan startup)."""
    import api as api_module

    pipeline_stub = pipeline_app or MagicMock()
    api_module.app.state.pipeline = pipeline_stub

    client = TestClient(api_module.app, raise_server_exceptions=False)
    return client, api_module


# ---------------------------------------------------------------------------
# CARD-1.1: X-Request-ID header on every response
# ---------------------------------------------------------------------------


class TestRequestIdHeader:
    def test_health_has_request_id_header(self):
        client, api_module = _make_client()
        api_module.app.state.pipeline = MagicMock()  # mark as ready
        response = client.get("/health")
        assert "x-request-id" in response.headers, "X-Request-ID header missing"
        rid = response.headers["x-request-id"]
        assert len(rid) == 12, f"Expected 12-char hex, got {rid!r}"

    def test_query_has_request_id_header_on_success(self):
        mock_pipeline = MagicMock()
        mock_pipeline.invoke.return_value = {
            "answer": "test answer",
            "final_passages": [],
        }
        client, _ = _make_client(pipeline_app=mock_pipeline)
        response = client.post("/query", json={"question": "test question"})
        assert "x-request-id" in response.headers

    def test_query_has_request_id_header_on_error(self):
        mock_pipeline = MagicMock()
        mock_pipeline.invoke.side_effect = RuntimeError("boom")
        client, _ = _make_client(pipeline_app=mock_pipeline)
        response = client.post("/query", json={"question": "test question"})
        assert response.status_code == 500
        assert "x-request-id" in response.headers


# ---------------------------------------------------------------------------
# CARD-1.1: 500 error must NOT expose exception details
# ---------------------------------------------------------------------------


class TestExceptionHiding:
    def test_500_contains_request_id_not_exception_text(self):
        mock_pipeline = MagicMock()
        mock_pipeline.invoke.side_effect = RuntimeError(
            "secret internal path /home/user/venv"
        )
        client, _ = _make_client(pipeline_app=mock_pipeline)
        response = client.post("/query", json={"question": "test"})

        assert response.status_code == 500
        body = response.json()
        detail = body.get("detail", "")

        assert "secret internal path" not in detail
        assert "/home/user/venv" not in detail
        assert "RuntimeError" not in detail
        assert "request_id=" in detail

    def test_500_request_id_matches_header(self):
        mock_pipeline = MagicMock()
        mock_pipeline.invoke.side_effect = ValueError("internal details")
        client, _ = _make_client(pipeline_app=mock_pipeline)
        response = client.post("/query", json={"question": "test"})

        assert response.status_code == 500
        rid_header = response.headers.get("x-request-id", "")
        detail = response.json().get("detail", "")
        assert (
            rid_header in detail
        ), f"request_id in header ({rid_header!r}) not found in detail ({detail!r})"


# ---------------------------------------------------------------------------
# CARD-1.1: 503 responses must not expose internal details
# ---------------------------------------------------------------------------


class TestGenericErrorMessages:
    def test_health_503_is_generic(self):
        import api as api_module

        api_module.app.state.pipeline = None  # simulate not-ready
        client = TestClient(api_module.app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 503
        detail = response.json().get("detail", "")
        assert "chroma" not in detail.lower()
        assert "pipeline" not in detail.lower() or detail == "not ready"


# ---------------------------------------------------------------------------
# CARD-1.2: Input length cap (max_length=2000)
# ---------------------------------------------------------------------------


class TestInputLengthCap:
    def test_question_over_2000_chars_returns_422(self):
        mock_pipeline = MagicMock()
        client, _ = _make_client(pipeline_app=mock_pipeline)
        long_question = "A" * 2001
        response = client.post("/query", json={"question": long_question})
        assert response.status_code == 422

    def test_question_exactly_2000_chars_is_accepted(self):
        mock_pipeline = MagicMock()
        mock_pipeline.invoke.return_value = {"answer": "ok", "final_passages": []}
        client, _ = _make_client(pipeline_app=mock_pipeline)
        question = "A" * 2000
        response = client.post("/query", json={"question": question})
        assert response.status_code != 422

    def test_empty_question_returns_422(self):
        mock_pipeline = MagicMock()
        client, _ = _make_client(pipeline_app=mock_pipeline)
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# CARD-1.2: Rate limit — custom 429 handler
# ---------------------------------------------------------------------------


class TestRateLimitHandler:
    def test_rate_limit_exceeded_returns_429_without_details(self):
        """Custom handler must return 429 with generic message, no internals."""
        import api as api_module

        mock_pipeline = MagicMock()
        mock_pipeline.invoke.return_value = {"answer": "ok", "final_passages": []}
        api_module.app.state.pipeline = mock_pipeline

        from starlette.requests import Request as StarletteRequest
        import asyncio

        async def call_handler():
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/query",
                "query_string": b"",
                "headers": [],
            }
            request = StarletteRequest(scope)
            exc = RateLimitExceeded(limit=MagicMock())
            return await api_module.rate_limit_handler(request, exc)

        response = asyncio.get_event_loop().run_until_complete(call_handler())
        assert response.status_code == 429
        import json

        body = json.loads(response.body)
        assert "rate limit" in body["detail"].lower()
        assert "10/minute" not in body["detail"]
