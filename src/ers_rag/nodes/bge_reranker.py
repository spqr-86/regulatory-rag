"""BGE-reranker-v2-m3 wrapper (ONNX int8) for ers_rag.

Drop-in replacement for the FlashRank reranker used in src/ers_rag/bridge.py.
Multilingual cross-encoder (XLM-RoBERTa base), strong on Russian technical text
(ГОСТ, СНиП, СП).

Public API:
    make_bge_rerank_fn(model_dir, max_length=512) -> Callable
        Returns fn(query, passages, top_k) -> List[dict] with the same
        contract as the previous FlashRank wrapper:
            - input passages: list of dicts with key 'text'
            - output: top_k reranked passages, each annotated with
              'vector_score' (preserved original score) and 'score'
              (BGE relevance, sigmoid of raw logit, range [0, 1]).

Model loading is singleton-cached per (model_dir, file_name) to avoid
re-loading the ~300MB int8 graph on every request.
"""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from typing import Callable, List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_LOAD_LOCK = threading.Lock()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@lru_cache(maxsize=4)
def _load_session(model_dir: str, file_name: str) -> tuple:
    """Load tokenizer + ORT session once per (dir, file). Cached singleton."""
    with _LOAD_LOCK:
        onnx_path = os.path.join(model_dir, file_name)
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"BGE reranker ONNX file not found: {onnx_path}. "
                "Run optimum-cli export + onnxruntime quantize."
            )
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        input_names = {i.name for i in session.get_inputs()}
        logger.info(
            "BGE reranker loaded: %s (inputs=%s)", onnx_path, sorted(input_names)
        )
        return tokenizer, session, input_names


def _pick_onnx_file(model_dir: str, prefer_quantized: bool) -> str:
    """Pick model_quantized.onnx when available, else model.onnx."""
    candidates = []
    if prefer_quantized:
        candidates.append("model_quantized.onnx")
    candidates.extend(["model.onnx", "model_optimized.onnx"])
    for name in candidates:
        if os.path.exists(os.path.join(model_dir, name)):
            return name
    raise FileNotFoundError(
        f"No ONNX file found in {model_dir}. Expected one of: {candidates}"
    )


def make_bge_rerank_fn(
    model_dir: str = "models/bge-reranker-v2-m3-onnx-int8",
    max_length: int = 512,
    prefer_quantized: bool = True,
) -> Callable[[str, List[dict], int], List[dict]]:
    """Create a BGE-reranker-v2-m3 rerank function for ers_rag.

    Signature matches the previous FlashRank wrapper:
        fn(query, passages, top_k) -> passages (reranked, top_k items).
    """
    file_name = _pick_onnx_file(model_dir, prefer_quantized=prefer_quantized)
    # Eager-load so init failures surface at bridge init, not first request.
    tokenizer, session, input_names = _load_session(model_dir, file_name)
    output_name = session.get_outputs()[0].name

    def _rerank(query: str, passages: List[dict], top_k: int) -> List[dict]:
        if not passages:
            return passages
        pairs = [[query, p.get("text", "")] for p in passages]
        enc = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        feed = {"input_ids": enc["input_ids"].astype(np.int64)}
        if "attention_mask" in input_names:
            feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
        if "token_type_ids" in input_names and "token_type_ids" in enc:
            feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)

        logits = session.run([output_name], feed)[0]  # shape (N, 1) or (N,)
        logits = np.asarray(logits).reshape(-1)
        scores = _sigmoid(logits)

        order = np.argsort(-scores)[:top_k]
        reranked: List[dict] = []
        for idx in order:
            orig = passages[int(idx)]
            reranked.append(
                {
                    **orig,
                    "vector_score": orig.get("score", 0.0),
                    "score": round(float(scores[int(idx)]), 4),
                }
            )
        return reranked

    return _rerank
