"""Request-local embedding reuse for scoped retrieval."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.v7.retrieval import RequestEmbeddingMemo, bind_request_embedding
from src.v7.runtime import V7Runtime


@pytest.mark.unit
def test_concurrent_same_key_computes_embedding_once():
    memo = RequestEmbeddingMemo()
    calls = 0
    entered = threading.Barrier(2)
    release = threading.Event()

    def embed(text):
        nonlocal calls
        calls += 1
        entered.wait(timeout=2)
        release.wait(timeout=2)
        return [float(len(text))]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(memo.get_or_compute, space="model-a", text="q", embed=embed)
        entered.wait(timeout=2)
        second = pool.submit(
            memo.get_or_compute, space="model-a", text="q", embed=embed
        )
        release.set()
        assert first.result(timeout=2) == second.result(timeout=2) == [1.0]
    assert calls == 1
    assert memo.stats() == {"computations": 1, "api_attempts": 1, "cache_hits": 1}


@pytest.mark.unit
def test_space_and_exact_text_are_part_of_key():
    memo = RequestEmbeddingMemo()
    calls = []

    def embed(text):
        calls.append(text)
        return [float(len(calls))]

    assert memo.get_or_compute(space="a", text="q", embed=embed) == [1.0]
    assert memo.get_or_compute(space="a", text="q ", embed=embed) == [2.0]
    assert memo.get_or_compute(space="b", text="q", embed=embed) == [3.0]


@pytest.mark.unit
def test_failure_is_shared_without_retry_loop():
    memo = RequestEmbeddingMemo()
    calls = 0

    def embed(_text):
        nonlocal calls
        calls += 1
        raise RuntimeError("provider down")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="provider down"):
            memo.get_or_compute(space="a", text="q", embed=embed)
    assert calls == 1


@pytest.mark.unit
def test_bound_runtimes_share_vector_across_scopes_and_complex_search():
    memo = RequestEmbeddingMemo()
    embed_calls = []
    searches = []

    def embed(text):
        embed_calls.append(text)
        return [42.0]

    def runtime_for(corpus):
        def search(*, query, embedding, filters=None, top_k=12):
            searches.append((corpus, query, embedding, filters, top_k))
            return []

        return bind_request_embedding(
            V7Runtime(vector_search=search, complex_vector_search=search),
            memo=memo,
            space="embedding-model-v1",
            embed=embed,
        )

    external = runtime_for("external")
    internal = runtime_for("internal")

    external.vector_search(
        query="prepared question", filters={"corpus": "external"}, top_k=8
    )
    internal.vector_search(
        query="prepared question", filters={"corpus": "internal"}, top_k=8
    )
    internal.complex_vector_search(
        query="prepared question", filters={"corpus": "internal"}, top_k=24
    )

    assert embed_calls == ["prepared question"]
    assert searches == [
        ("external", "prepared question", [42.0], {"corpus": "external"}, 8),
        ("internal", "prepared question", [42.0], {"corpus": "internal"}, 8),
        ("internal", "prepared question", [42.0], {"corpus": "internal"}, 24),
    ]
