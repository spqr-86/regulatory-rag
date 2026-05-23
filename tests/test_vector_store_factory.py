"""Tests for VectorStoreBackend factory (CARD-2.1)."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_factory_unknown_raises():
    """Unknown VECTOR_STORE must raise ValueError mentioning the bad value."""
    from src.backends.vector_store import get_vector_store_backend

    with patch("src.backends.vector_store.settings") as s:
        s.VECTOR_STORE = "qdrant"
        with pytest.raises(ValueError, match="qdrant"):
            get_vector_store_backend()


@pytest.mark.unit
def test_factory_case_insensitive():
    """VECTOR_STORE matching is case-insensitive."""
    from src.backends.vector_store import get_vector_store_backend

    with (
        patch("src.backends.vector_store.settings") as s,
        patch("src.vector_store.load_vector_store"),
    ):
        s.VECTOR_STORE = "CHROMA"
        # Should not raise
        backend = get_vector_store_backend(load_existing=True)
        assert backend is not None
