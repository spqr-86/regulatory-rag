"""One lazily loaded, serialized reranker per model configuration."""

# ANCHOR: model resource owns the load/predict lock, never a graph or request.
# Absolute monotonic deadlines bound lock waiting, not running blocking inference.

from __future__ import annotations

import math
import threading
import time
from typing import Callable


class RerankerError(RuntimeError):
    """Model loading or inference failed; callers must record degradation."""


class SharedReranker:
    def __init__(self, loader: Callable, wait_timeout_s: float = 120.0):
        if not math.isfinite(wait_timeout_s) or wait_timeout_s <= 0:
            raise ValueError("reranker wait timeout must be finite and positive")
        self._loader = loader
        self._wait_timeout_s = wait_timeout_s
        self._predict = None
        self._lock = threading.Lock()

    def __call__(self, query, passages, top_k, *, deadline: float | None = None):
        if not passages:
            return []
        now = time.monotonic()
        wait_deadline = now + self._wait_timeout_s
        if deadline is not None:
            if not math.isfinite(deadline):
                raise ValueError("reranker deadline must be finite")
            wait_deadline = min(wait_deadline, deadline)
        remaining = wait_deadline - now
        acquired = remaining > 0 and self._lock.acquire(timeout=remaining)
        if not acquired:
            raise TimeoutError("reranker wait deadline exceeded")
        try:
            if self._predict is None:
                self._predict = self._loader()
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("reranker deadline exceeded after load")
            return self._predict(query, passages, top_k)
        except TimeoutError:
            raise
        except Exception as exc:
            raise RerankerError(f"reranker load/predict failed: {exc}") from exc
        finally:
            self._lock.release()


_resources: dict[tuple, SharedReranker] = {}
_resources_lock = threading.Lock()


def shared_reranker(
    key: tuple, loader: Callable, *, wait_timeout_s: float = 120.0
) -> SharedReranker:
    with _resources_lock:
        resource = _resources.get(key)
        if resource is None:
            resource = SharedReranker(loader, wait_timeout_s)
            _resources[key] = resource
        return resource
