"""Resource lifetime, contention and bound-store tests; no real model loads."""

# ANCHOR: barriers/events expose races without relying on scheduling sleeps.

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock
import time

import pytest

pytestmark = pytest.mark.unit


def test_reranker_is_lazy_shared_and_serialized():
    from src.v7.reranker import shared_reranker

    loads = []
    entered = Event()
    release = Event()
    start = Barrier(2)
    active = 0
    max_active = 0
    guard = Lock()

    def load():
        loads.append(1)

        def predict(query, passages, top_k):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            entered.set()
            assert release.wait(5)
            with guard:
                active -= 1
            return passages[:top_k]

        return predict

    key = ("test", object())
    a = shared_reranker(key, load)
    b = shared_reranker(key, load)
    assert a is b
    assert loads == []
    assert a("q", [], 1) == []
    assert loads == []

    def run(fn):
        start.wait(5)
        return fn("q", [{"text": "p"}], 1)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, fn) for fn in (a, b)]
        assert entered.wait(5)
        release.set()
        assert all(f.result(timeout=5) == [{"text": "p"}] for f in futures)
    assert loads == [1]
    assert max_active == 1


def test_reranker_wait_honors_deadline_and_releases_after_failure():
    from src.v7.reranker import shared_reranker, RerankerError

    entered, release = Event(), Event()

    def predict(*args):
        entered.set()
        assert release.wait(5)
        raise ValueError("predict failed")

    resource = shared_reranker(("test", object()), lambda: predict)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(resource, "q", [{"text": "p"}], 1)
        assert entered.wait(5)
        try:
            with pytest.raises(TimeoutError):
                resource("q", [{"text": "p"}], 1, deadline=time.monotonic() + 0.02)
        finally:
            release.set()
        with pytest.raises(RerankerError, match="predict failed"):
            first.result(timeout=5)
    with pytest.raises(RerankerError):
        resource("q", [{"text": "p"}], 1, deadline=time.monotonic() + 1)


def test_runtime_builds_one_bm25_and_no_generation_clients(monkeypatch):
    from src.v7 import bridge
    from src.v7.retrieval import retrieve_context
    from unittest.mock import Mock

    class RawChromaStore:
        def get(self, *, include):
            assert include == ["metadatas", "documents"]
            return {
                "documents": ["медосмотр водителей", "пожарная безопасность"],
                "metadatas": [{"source": "A"}, {"source": "B"}],
            }

        def similarity_search_with_score(self, query, *, k, filter):
            return []

    store = RawChromaStore()
    index = Mock()
    index.search.return_value = []
    factory = Mock(return_value=index)
    monkeypatch.setattr(bridge, "BM25Index", factory)
    monkeypatch.setattr(
        bridge, "get_simple_llm", Mock(side_effect=AssertionError("LLM init"))
    )
    monkeypatch.setattr(
        bridge, "get_complex_llm", Mock(side_effect=AssertionError("LLM init"))
    )
    runtime = bridge.build_v7_runtime(store, rerank=None)
    retrieve_context("медосмотр водителей", runtime=runtime)
    retrieve_context("пожарная безопасность", runtime=runtime)
    factory.assert_called_once()
    assert index.search.call_count == 2


def test_bound_chroma_backends_do_not_use_default_singleton(monkeypatch):
    from src.backends.chroma_backend import ChromaBackend
    from src.indexing import vector_store
    from unittest.mock import Mock

    loader = Mock(side_effect=[object(), object()])
    monkeypatch.setattr(vector_store, "load_bound_vector_store", loader)
    a = ChromaBackend(path="/index/a", collection="A", embeddings="embedding-A")
    b = ChromaBackend(path="/index/b", collection="B", embeddings="embedding-B")
    assert a._vs is not b._vs
    assert loader.call_args_list[0].kwargs == {
        "path": "/index/a",
        "collection": "A",
        "embeddings": "embedding-A",
    }


def test_bm25_snapshot_and_results_do_not_share_mutable_metadata():
    from src.v7.nlp_core import BM25Index

    passages = [
        {"text": "медосмотр водителей", "metadata": {"source": "A"}},
        {"text": "пожарная безопасность", "metadata": {"source": "B"}},
    ]
    index = BM25Index(passages)
    passages[0]["metadata"]["source"] = "mutated input"
    first = index.search("медосмотр", filters={"source": "A"})
    assert len(first) == 1
    first[0]["metadata"]["source"] = "mutated result"
    assert len(index.search("медосмотр", filters={"source": "A"})) == 1


def test_reranker_load_failure_remains_visible_in_result(monkeypatch):
    import copy
    from src.v7.reranker import shared_reranker
    from src.v7.runtime import V7Runtime
    from src.v7.retrieval import retrieve_context
    from src.v7.config import v7_config
    from tests.v7.test_retrieval_context import CASES

    case = next(c for c in CASES if c["name"] == "escalation_accepted")
    for key, value in case["config"].items():
        monkeypatch.setattr(v7_config, key, value)

    def load():
        raise ValueError("model unavailable")

    resource = shared_reranker(("failure", object()), load)

    def search(query, top_k, **kwargs):
        return copy.deepcopy(case["simple" if top_k == 12 else "complex"])

    result = retrieve_context(
        case["query"], runtime=V7Runtime(vector_search=search, rerank=resource)
    )
    assert result.outcome == "failed"
    assert result.technical_failure
    assert result.final_context == []
    assert result.attempts[-1]["metrics"]["rerank_error"] == "RerankerError"


def test_runtime_crossref_uses_its_own_bm25_after_global_replacement(monkeypatch):
    from unittest.mock import Mock
    from src.backends.vector_store import VectorStoreBackend
    from src.v7 import bridge, cross_ref

    store = Mock(spec=VectorStoreBackend)
    store.get_by_filter.return_value = []
    index = Mock()
    index.search.return_value = []
    runtime = bridge.build_v7_runtime(store, bm25_index=index, rerank=None)
    monkeypatch.setattr(
        cross_ref, "bm25_search", Mock(side_effect=AssertionError("global BM25"))
    )
    runtime.crossref_expander(
        [{"text": "согласно пункту 99", "metadata": {"source": "A"}}], "медосмотр"
    )
    index.search.assert_called_once()
    store.get_by_filter.assert_called_once()


def test_default_reranker_wait_is_finite_without_caller_deadline():
    from src.v7.reranker import SharedReranker

    entered, release = Event(), Event()

    def predict(q, ps, k):
        entered.set()
        assert release.wait(5)
        return ps

    resource = SharedReranker(lambda: predict, wait_timeout_s=0.02)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(resource, "q", [{"text": "p"}], 1)
        assert entered.wait(5)
        try:
            with pytest.raises(TimeoutError):
                resource("q", [{"text": "p"}], 1)
        finally:
            release.set()
        first.result(timeout=5)


def test_legacy_capture_preserves_separate_simple_and_complex_callbacks(monkeypatch):
    from src.v7.runtime import capture_legacy_runtime
    from src.v7.nodes import rag_simple, rag_complex

    a, b = lambda **kw: [], lambda **kw: []
    monkeypatch.setattr(rag_simple, "_vector_search", a)
    monkeypatch.setattr(rag_complex, "_vector_search", b)
    runtime = capture_legacy_runtime()
    assert runtime.vector_search is a
    assert runtime.complex_vector_search is b


def test_capture_waits_for_atomic_legacy_initialization():
    from src.v7.runtime import capture_legacy_runtime, legacy_runtime_lock

    entered = Event()

    def capture():
        entered.set()
        return capture_legacy_runtime()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with legacy_runtime_lock:
            future = pool.submit(capture)
            assert entered.wait(5)
            assert not future.done()
        assert future.result(timeout=5) is not None
