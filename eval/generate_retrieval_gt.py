"""Generate retrieval ground truth (A→Q*) from the real Chroma corpus.

For every non-junk chunk in the vector collection the generator asks an LLM for
``N`` natural questions whose answer is that chunk, then writes
``eval/data/retrieval_gt.jsonl`` with the schema::

    {"question": str, "chunk_id": str, "source": str, "chunk_preview": str}

``chunk_id`` uses the same ``"{source}#{chunk_id}"`` identity as retrieval fusion
(``src.v7.nlp_core.passage_identity``), so the file feeds
``eval.retrieval_metrics.evaluate_retrieval_batch`` directly.

The deterministic core (junk filter, normalisation, dedup, parsing, record
construction) imports nothing heavy and is unit-tested in
``tests/test_generate_retrieval_gt.py``. LangChain / Chroma are imported lazily
inside the functions that need them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Sequence

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

GT_PATH = REPO_ROOT / "eval" / "data" / "retrieval_gt.jsonl"

QUESTIONS_PER_CHUNK = 3
MIN_CHUNK_CHARS = 200
NEAR_DUP_THRESHOLD = 0.95
MAX_WORKERS = 6
COST_ABORT_USD = 2.0
PREVIEW_CHARS = 200

# gpt-4o-mini list price, USD per 1M tokens (2026-09).
PRICE_PER_1M = {"input": 0.15, "output": 0.60}


class Questions(BaseModel):
    """Structured-output schema for one LLM generation call."""

    questions: list[str] = Field(description="Natural questions answered by the chunk")


GEN_PROMPT = """Ты — специалист по охране труда, который ищет ответ в нормативной базе.
Ниже фрагмент нормативного документа. Сформулируй {n} естественных вопросов,
ответ на которые содержится ИМЕННО в этом фрагменте.

Требования к вопросам:
- используй как можно меньше слов из самого фрагмента (перефразируй);
- пиши так, как реально спрашивают коллеги — не канцелярит, не слишком коротко и не слишком длинно;
- каждый вопрос самодостаточен (без «здесь», «в этом пункте»);
- разные формулировки, не три перифраза одного и того же.

Фрагмент:
{chunk}
"""


# ─── Deterministic core (unit-tested, no heavy imports) ──────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_TOC_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")


def normalize_question(q: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — key for exact dedup."""
    q = _PUNCT_RE.sub(" ", q.lower())
    return _WS_RE.sub(" ", q).strip()


def is_junk_chunk(text: str, min_chars: int = MIN_CHUNK_CHARS) -> bool:
    """True for chunks not worth generating questions from: too short, or a
    table of contents (several lines ending in a dotted leader + page number,
    or an explicit 'содержание' / 'оглавление' header)."""
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        return True

    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines and lines[0].lower().rstrip(":.").strip() in {"содержание", "оглавление"}:
        return True

    leader_lines = sum(1 for ln in lines if _TOC_LEADER_RE.search(ln))
    if len(lines) >= 3 and leader_lines >= len(lines) - 1:
        return True

    return False


def parse_questions(raw: object) -> list[str]:
    """Extract a clean list[str] from a Questions model, a dict, or any object
    exposing ``.questions``. Trims each item, drops blanks."""
    if isinstance(raw, Questions):
        items = raw.questions
    elif isinstance(raw, dict):
        items = raw.get("questions", [])
    else:
        items = getattr(raw, "questions", []) or []
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def dedup_questions(
    records: list[dict],
    embed_fn: Callable[[list[str]], list[Sequence[float]]] | None = None,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
) -> tuple[list[dict], int]:
    """Remove exact duplicate questions (by :func:`normalize_question`) and, when
    ``embed_fn`` is given, near-duplicates (cosine > ``near_dup_threshold``).
    Returns ``(kept_records, removed_count)`` preserving input order."""
    kept: list[dict] = []
    seen_norm: set[str] = set()
    for rec in records:
        norm = normalize_question(rec["question"])
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            kept.append(rec)

    removed = len(records) - len(kept)
    if embed_fn is None or len(kept) < 2:
        return kept, removed

    vectors = embed_fn([r["question"] for r in kept])
    survivors: list[dict] = []
    survivor_vecs: list[Sequence[float]] = []
    for rec, vec in zip(kept, vectors):
        if any(_cosine(vec, sv) > near_dup_threshold for sv in survivor_vecs):
            removed += 1
            continue
        survivors.append(rec)
        survivor_vecs.append(vec)
    return survivors, removed


def _passage_identity(chunk: dict) -> str:
    """``"{source}#{chunk_id}"`` — mirrors src.v7.nlp_core.passage_identity for
    the chunk_id branch. GT generation controls the index and requires chunk_id,
    so the content-hash fallback is not needed here."""
    meta = chunk.get("metadata") or {}
    source = meta.get("source", chunk.get("source", ""))
    cid = chunk.get("chunk_id", meta.get("chunk_id"))
    return f"{source}#{cid}"


def build_gt_record(question: str, chunk: dict) -> dict:
    meta = chunk.get("metadata") or {}
    return {
        "question": question.strip(),
        "chunk_id": _passage_identity(chunk),
        "source": meta.get("source", chunk.get("source", "")),
        "chunk_preview": (chunk.get("text", "") or "")[:PREVIEW_CHARS],
    }


# ─── LLM + corpus (lazy heavy imports) ──────────────────────────────────────


def calc_total_price(usages: Iterable[dict]) -> float:
    """Sum USD cost from a list of ``{"input": n, "output": n}`` token counts."""
    total = 0.0
    for u in usages:
        total += u.get("input", 0) / 1_000_000 * PRICE_PER_1M["input"]
        total += u.get("output", 0) / 1_000_000 * PRICE_PER_1M["output"]
    return total


def estimate_cost(n_chunks: int, avg_chunk_tokens: int = 350) -> float:
    """Rough pre-flight estimate: prompt ≈ chunk + 150 tokens overhead,
    output ≈ 60 tokens per generation call."""
    per_call = calc_total_price([{"input": avg_chunk_tokens + 150, "output": 60}])
    return per_call * n_chunks


def iter_corpus_chunks(vs=None) -> list[dict]:
    """All chunks of the Chroma collection as ``{text, chunk_id, source, metadata}``."""
    if vs is None:
        from src.indexing.vector_store import get_vector_store  # noqa: PLC0415

        vs = get_vector_store()
    raw = vs.get(include=["documents", "metadatas"])
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []
    chunks: list[dict] = []
    for i, text in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        chunks.append(
            {
                "text": text,
                "chunk_id": meta.get("chunk_id"),
                "source": meta.get("source", ""),
                "metadata": meta,
            }
        )
    return chunks


def _make_llm():
    from src.infra.llm_factory import get_judge_llm  # noqa: PLC0415

    return get_judge_llm()


def generate_questions_for_chunk(chunk: dict, llm, n: int = QUESTIONS_PER_CHUNK):
    """One structured-output call. Returns ``(questions, usage)``. Retries on
    transient API errors via tenacity."""
    from openai import (  # noqa: PLC0415
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    from tenacity import (  # noqa: PLC0415
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )

    structured = llm.with_structured_output(Questions, include_raw=True)
    prompt = GEN_PROMPT.format(n=n, chunk=chunk["text"])

    # Retry only transient failures — 4xx (bad request / auth / schema) is a
    # caller bug and must surface immediately.
    @retry(
        retry=retry_if_exception_type(
            (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
        ),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call():
        return structured.invoke(prompt)

    result = _call()
    parsed = (
        parse_questions(result["parsed"])
        if isinstance(result, dict)
        else parse_questions(result)
    )
    usage = {}
    raw_msg = result.get("raw") if isinstance(result, dict) else None
    meta = getattr(raw_msg, "usage_metadata", None) or {}
    if meta:
        usage = {
            "input": meta.get("input_tokens", 0),
            "output": meta.get("output_tokens", 0),
        }
    return parsed[:n], usage


def run(
    limit: int | None = None,
    out_path: Path = GT_PATH,
    dry_run: bool = False,
) -> dict:
    chunks = [c for c in iter_corpus_chunks() if not is_junk_chunk(c["text"])]
    if limit:
        chunks = chunks[:limit]

    projected = estimate_cost(len(chunks))
    print(
        f"chunks (after junk filter): {len(chunks)}  projected cost: ${projected:.2f}"
    )
    if projected > COST_ABORT_USD:
        sys.exit(f"ABORT: projected ${projected:.2f} > ${COST_ABORT_USD:.2f} budget")
    if dry_run:
        return {"chunks": len(chunks), "projected_usd": projected}

    llm = _make_llm()
    records: list[dict] = []
    usages: list[dict] = []
    failures = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(generate_questions_for_chunk, c, llm): c for c in chunks}
        for fut in as_completed(futs):
            chunk = futs[fut]
            try:
                questions, usage = fut.result()
            except Exception as e:  # noqa: BLE001 — one bad chunk must not kill the run
                failures += 1
                print(f"  chunk {_passage_identity(chunk)} failed: {e}")
                continue
            if usage:
                usages.append(usage)
            for q in questions:
                records.append(build_gt_record(q, chunk))

    embed_fn = _embedding_fn()
    kept, removed = dedup_questions(records, embed_fn=embed_fn)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    spent = calc_total_price(usages)
    print(
        f"wrote {len(kept)} questions ({removed} dups removed, {failures} chunk failures) "
        f"to {out_path}  spent: ${spent:.4f}"
    )
    return {
        "questions": len(kept),
        "removed_dups": removed,
        "failures": failures,
        "spent_usd": spent,
    }


def _embedding_fn():
    try:
        from src.infra.llm_factory import get_embedding_model  # noqa: PLC0415

        model = get_embedding_model()
        return lambda texts: model.embed_documents(list(texts))
    except Exception as e:  # noqa: BLE001
        print(f"  near-dup dedup skipped (no embedding model): {e}")
        return None


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="cap chunks (smoke test)")
    p.add_argument("--out", type=Path, default=GT_PATH)
    p.add_argument(
        "--dry-run", action="store_true", help="estimate cost, write nothing"
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(limit=args.limit, out_path=args.out, dry_run=args.dry_run)
