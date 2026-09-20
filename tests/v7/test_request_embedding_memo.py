"""Request-local embedding reuse for scoped retrieval."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from src.v7.retrieval import RequestEmbeddingMemo


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
