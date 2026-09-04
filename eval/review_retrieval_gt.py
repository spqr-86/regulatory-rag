"""Review the synthetic retrieval GT sample with an adversarial LLM pass.

The generator writes questions from a chunk; this module tries to *reject* each
of them. Different job, opposite instruction, separate call — a pass that asks
"is this fine?" agrees with itself. Every question is judged against the chunk it
is tagged with and gets ``keep`` / ``drop`` plus a reason from a closed list, so
the rejects can be counted by cause instead of read one by one.

Two things stay with the human: every ``drop``, and any verdict below the
confidence threshold. The model's confident keeps are taken as they are.

Crash-safe like the generator: each verdict is appended to ``<out>.raw`` at once
and a rerun resumes from that log.

Usage::

    python eval/review_retrieval_gt.py                      # sample -> review TSV
    python eval/review_retrieval_gt.py --limit 20 --dry-run # cost estimate only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval.generate_retrieval_gt import (  # noqa: E402
    PREVIEW_CHARS,
    calc_total_price,
    estimate_cost,
)
from eval.sample_retrieval_gt import load_records  # noqa: E402

DEFAULT_SAMPLE = REPO_ROOT / "eval" / "data" / "retrieval_gt_sample.jsonl"
DEFAULT_OUT = REPO_ROOT / "eval" / "data" / "retrieval_gt_sample.review.tsv"

# Judged by the same cheap model the questions came from, but with the opposite
# instruction. A bigger judge is one flag away (--model) if agreement looks poor.
REVIEW_MODEL = "gpt-4o-mini"
# Арбитр по отказам — модель посильнее: спор идёт о том, есть ли ответ в тексте,
# и здесь ошибка стоит выброшенного годного вопроса.
ARBITER_MODEL = "gpt-4o"
# gpt-4o на этом аккаунте ограничен 30K токенов в минуту: шесть параллельных вызовов
# по ~1К токенов упираются в лимит и съедают все ретраи. Замер 04.09.2026.
ARBITER_WORKERS = 2
MAX_WORKERS = 6
COST_ABORT_USD = 2.0
# Below this the verdict is not trusted on its own and goes to the human queue.
CONFIDENCE_THRESHOLD = 0.75

DROP_REASONS = (
    # Ответ содержится в самой формулировке вопроса — поиск не проверяется.
    "answer_in_question",
    # Во фрагменте нет ответа: вопрос привязан не к тому чанку.
    "not_answerable",
    # Вопрос подходит к десяткам фрагментов: зацепки нет.
    "too_generic",
    # Вопрос опирается на контекст, которого у спрашивающего нет («в этом пункте»).
    "context_dependent",
    # Спрашивают реквизиты, а не содержание нормы.
    "about_requisites",
    # Обрывок, бессмыслица, не вопрос.
    "malformed",
    "other",
)

REVIEW_PROMPT = """Ты — придирчивый рецензент тестового набора для поисковой системы
по нормативной базе. Твоя задача — НАЙТИ ПРИЧИНУ ЗАБРАКОВАТЬ вопрос, а не одобрить его.
Одобряй только то, к чему претензий не нашлось.

Набор используется так: вопрос подают в поиск и проверяют, вернул ли поиск именно
этот фрагмент. Поэтому годен только вопрос, который (а) отвечается этим фрагментом,
(б) содержит зацепку, ведущую именно к нему, (в) понятен человеку, который фрагмента
не видел.

Проверяй ровно в этом порядке, первая сработавшая причина и есть ответ.

ШАГ 1. Есть ли ответ в фрагменте? Найди в тексте фрагмента фразу, которая прямо
отвечает на вопрос, и выпиши её ДОСЛОВНО в answer_span. Если выписать нечего —
verdict "drop", reason "not_answerable". Спрашивают срок, а во фрагменте сроков нет;
спрашивают объём, а есть только запреты — это not_answerable, и никакая другая
причина сюда не подставляется.

ШАГ 2. Ведёт ли вопрос именно к этому фрагменту? Ответ есть, но в вопросе нет ни
одной конкретной детали (срок, число, вид работ, условие, категория работников,
объект) — reason "too_generic". Эта причина ставится ТОЛЬКО когда ответ в фрагменте
найден, а зацепки нет. Вопрос с конкретной деталью — не too_generic, даже если
формулировка тебе не нравится.

ШАГ 3. Остальные пороки:
- answer_in_question — ответ виден из самой формулировки вопроса или угадывается
  из общих знаний, читать фрагмент не нужно;
- context_dependent — вопрос опирается на невидимый контекст: «в этом пункте»,
  «согласно указанной таблице», «в данном приложении». Формулировка «если сняты
  колпаки» — это условие из нормы, а не ссылка на контекст: она допустима;
- about_requisites — спрашивают не содержание нормы, а её реквизиты: каким актом
  введена, номер пункта, редакция, дата утраты силы;
- malformed — обрывок, бессмыслица, не вопрос;
- other — годной причины из списка нет, но вопрос всё равно негоден (объясни в note).

Если ни одна причина не сработала — verdict "keep", reason "none", и answer_span
обязателен: дословная фраза из фрагмента.

Заголовок раздела в начале фрагмента бывает от другого раздела — это дефект нарезки.
Суди по тексту нормы, а не по заголовку, и упомяни расхождение в note.

confidence — насколько ты уверен в вердикте, от 0 до 1. Ставь ниже 0.75, если
решение спорное: тогда вопрос уйдёт человеку.

ВОПРОС:
{question}

ФРАГМЕНТ (документ {source}):
{chunk}
"""


class Review(BaseModel):
    """Structured-output schema for one review call."""

    verdict: str = Field(description='"keep" or "drop"')
    reason: str = Field(description='drop reason from the closed list, or "none"')
    confidence: float = Field(description="0..1 confidence in the verdict")
    note: str = Field(default="", description="short free-text remark, optional")
    answer_span: str = Field(
        default="",
        description="verbatim quote from the chunk that answers the question (keep only)",
    )


# ─── Deterministic core (unit-tested, no heavy imports) ──────────────────────


def parse_review(review: Review | dict) -> dict:
    """Normalise one model verdict.

    A ``keep`` carries no reason, an unknown reason on a ``drop`` collapses to
    ``other`` (an invented label would split the counts), and confidence is
    clamped: a model that answers 7.0 must not outrank every honest verdict.
    """
    data = review if isinstance(review, dict) else review.model_dump()
    verdict = (
        "drop" if str(data.get("verdict", "")).strip().lower() == "drop" else "keep"
    )
    reason = str(data.get("reason", "")).strip().lower()
    if verdict == "keep":
        reason = ""
    elif reason not in DROP_REASONS:
        reason = "other"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return {
        "verdict": verdict,
        "reason": reason,
        "confidence": confidence,
        "note": str(data.get("note", "")).strip(),
        "answer_span": str(data.get("answer_span", "")).strip(),
    }


_SPAN_WS_RE = re.compile(r"\s+")
# Доверие к keep без подтверждающей цитаты. Ниже порога — уходит человеку.
UNVERIFIED_KEEP_CONFIDENCE = 0.5


def validate_answer_span(review: dict, chunk_text: str) -> dict:
    """Check a ``keep`` against its own evidence.

    The model must quote the phrase of the chunk that answers the question. A
    quote that is not actually in the chunk means the answer was reconstructed
    from general knowledge rather than found — exactly the failure the review is
    there to catch — so such a keep is demoted to the human queue instead of
    being flipped to a drop on the word of the same model.
    """
    if review["verdict"] != "keep":
        return review
    span = _SPAN_WS_RE.sub(" ", review.get("answer_span", "")).strip().lower()
    haystack = _SPAN_WS_RE.sub(" ", chunk_text or "").lower()
    if span and span in haystack:
        return review
    checked = dict(review)
    checked["confidence"] = min(review["confidence"], UNVERIFIED_KEEP_CONFIDENCE)
    flag = "span_not_found" if span else "span_missing"
    checked["note"] = f"{flag} {review.get('note', '')}".strip()
    return checked


def needs_human(review: dict, threshold: float = CONFIDENCE_THRESHOLD) -> bool:
    """Whether the verdict has to be looked at by a person.

    Every ``drop`` does — throwing a question out is the expensive mistake — and
    so does anything the model itself is unsure about.
    """
    return review["verdict"] == "drop" or review["confidence"] < threshold


def load_reviewed(raw_path: Path) -> dict[int, dict]:
    """Resume log ``n -> verdict``. A torn last line is dropped, not fatal."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return {}
    reviewed: dict[int, dict] = {}
    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "n" in rec:
                reviewed[int(rec["n"])] = rec
    return reviewed


def review_row(n: int, record: dict, review: dict) -> dict:
    """One row of the review TSV, same columns as the hand-review sheet."""
    reason = review.get("reason") or ""
    note = (
        f"{reason} ({review['confidence']:.2f})"
        if reason
        else f"({review['confidence']:.2f})"
    )
    if review.get("note"):
        note = f"{note} {review['note']}"
    return {
        "n": n,
        "verdict": review["verdict"],
        "status": review.get("status", ""),
        "note": note,
        "question": record["question"],
        "source": record.get("source", ""),
        "chunk_id": record.get("chunk_id", ""),
        "preview": (record.get("chunk_preview") or "")[:PREVIEW_CHARS],
    }


def attach_chunk_texts(records: list[dict], corpus: dict[str, str]) -> list[dict]:
    """Give every record the full chunk text it is tagged with.

    The GT file stores a 200-char preview only, and a verdict of "not_answerable"
    read off a truncated chunk is a false reject. A chunk that is no longer in
    the collection keeps the preview and is flagged, so the count of such cases
    is visible instead of silently degrading the review.
    """
    attached = []
    for rec in records:
        text = corpus.get(rec.get("chunk_id", ""))
        out = dict(rec)
        out["chunk_text"] = text if text else (rec.get("chunk_preview") or "")
        out["chunk_missing"] = not bool(text)
        attached.append(out)
    return attached


def merge_arbitration(review: dict, arbitration: dict | None) -> dict:
    """Combine the strict pass with the arbiter's second opinion.

    The strict reviewer rejects too eagerly — measured on 14 hand-checked drops,
    roughly a third were questions the chunk does answer. So a drop only stands
    when a second, stronger model looking for the answer also fails to find one;
    where the two disagree the question goes to a person rather than to either
    model's verdict.
    """
    merged = dict(review)
    if arbitration is None:
        merged["status"] = "single_pass"
        return merged
    if arbitration["verdict"] == "drop":
        merged["status"] = "drop_confirmed"
        return merged
    merged["status"] = "disputed"
    span = arbitration.get("answer_span", "")
    merged["note"] = (
        f"спор: арбитр оставляет — {arbitration.get('note') or span}".strip()
    )
    return merged


GT_FIELDS = ("question", "chunk_id", "source", "chunk_preview")


def build_reviewed_gt(
    records: list[dict],
    merged: dict[int, dict],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """The reviewed eval set: confident keeps only.

    A disputed question — one reviewer rejects it, the other saves it — is left
    out rather than argued over: the sample exists to measure retrieval, and a
    question two models cannot agree on measures the reviewers instead. Rejects
    stay in the sample file, so the exclusion is reversible.
    """
    kept = []
    for i, rec in enumerate(records, start=1):
        verdict = merged.get(i)
        if not verdict or verdict["verdict"] != "keep":
            continue
        if verdict["confidence"] < threshold:
            continue
        kept.append({k: rec[k] for k in GT_FIELDS if k in rec})
    return kept


def summarize(reviews: Iterable[dict], threshold: float = CONFIDENCE_THRESHOLD) -> dict:
    """Verdict counts, reasons by frequency and the size of the human queue."""
    reviews = list(reviews)
    reasons = Counter(
        r["reason"] for r in reviews if r["verdict"] == "drop" and r["reason"]
    )
    return {
        "total": len(reviews),
        "keep": sum(1 for r in reviews if r["verdict"] == "keep"),
        "drop": sum(1 for r in reviews if r["verdict"] == "drop"),
        "needs_human": sum(1 for r in reviews if needs_human(r, threshold)),
        "reasons": dict(reasons.most_common()),
    }


# ─── LLM pass ────────────────────────────────────────────────────────────────


def corpus_text_map() -> dict[str, str]:
    """``chunk_id -> full text`` for the whole collection, keyed the way the GT is."""
    from eval.generate_retrieval_gt import (  # noqa: PLC0415
        _passage_identity,
        iter_corpus_chunks,
    )

    return {_passage_identity(p): p.get("text", "") for p in iter_corpus_chunks()}


def _make_llm(model: str = REVIEW_MODEL):
    from src.infra import llm_factory  # noqa: PLC0415

    return llm_factory.get_judge_llm(model_name=model)


def review_one(record: dict, llm) -> tuple[dict, dict]:
    """One structured-output call. Returns ``(review, usage)``."""
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

    structured = llm.with_structured_output(Review, include_raw=True)
    prompt = REVIEW_PROMPT.format(
        question=record["question"],
        source=record.get("source", "н/д"),
        chunk=record.get("chunk_text") or record.get("chunk_preview", ""),
    )

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
    parsed = result["parsed"] if isinstance(result, dict) else result
    if parsed is None:
        review = {
            "verdict": "keep",
            "reason": "",
            "confidence": 0.0,
            "note": "unparsed model output",
            "answer_span": "",
        }
    else:
        review = validate_answer_span(
            parse_review(parsed),
            record.get("chunk_text") or record.get("chunk_preview", ""),
        )
    usage = {}
    raw_msg = result.get("raw") if isinstance(result, dict) else None
    meta = getattr(raw_msg, "usage_metadata", None) or {}
    if meta:
        usage = {
            "input": meta.get("input_tokens", 0),
            "output": meta.get("output_tokens", 0),
        }
    return review, usage


ARBITER_PROMPT = """Ты — арбитр в споре о тестовом вопросе для поисковой системы
по нормативной базе. Первый рецензент забраковал вопрос. Твоя задача — проверить его,
и склоняться в пользу вопроса: рецензент известен тем, что бракует лишнее.

Вопрос годен (verdict "keep"), если в тексте фрагмента есть фраза, отвечающая на него
хотя бы по существу — пусть неполно, пусть другими словами. Выпиши эту фразу дословно
в answer_span. Ответ, собранный из перечня, таблицы или кода классификатора, тоже
считается ответом.

Вопрос негоден (verdict "drop"), только если:
- ответа в фрагменте нет вовсе (reason "not_answerable") — спрашивают срок, а сроков нет;
  спрашивают число, а числа нет; фрагмент обрывается до ответа;
- ответ виден из самой формулировки вопроса (reason "answer_in_question");
- вопрос — обрывок или бессмыслица (reason "malformed");
- спрашивают реквизиты акта, а не норму (reason "about_requisites").

Формулировка «слишком общая» причиной забраковать НЕ является, если ответ в фрагменте
есть: такой вопрос проверяет поиск ровно так, как его проверяют живые пользователи.

ПРИЧИНА ПЕРВОГО РЕЦЕНЗЕНТА: {reason} — {note}

ВОПРОС:
{question}

ФРАГМЕНТ (документ {source}):
{chunk}
"""


def arbitrate_one(record: dict, review: dict, llm) -> tuple[dict, dict]:
    """Second opinion on one rejected question. Returns ``(review, usage)``."""
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

    structured = llm.with_structured_output(Review, include_raw=True)
    prompt = ARBITER_PROMPT.format(
        reason=review.get("reason", ""),
        note=review.get("note", ""),
        question=record["question"],
        source=record.get("source", "н/д"),
        chunk=record.get("chunk_text") or record.get("chunk_preview", ""),
    )

    @retry(
        retry=retry_if_exception_type(
            (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
        ),
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def _call():
        return structured.invoke(prompt)

    result = _call()
    parsed = result["parsed"] if isinstance(result, dict) else result
    arbitration = (
        parse_review(parsed)
        if parsed is not None
        else {
            "verdict": "drop",
            "reason": review.get("reason", "other"),
            "confidence": 0.0,
            "note": "unparsed arbiter output",
            "answer_span": "",
        }
    )
    usage = {}
    raw_msg = result.get("raw") if isinstance(result, dict) else None
    meta = getattr(raw_msg, "usage_metadata", None) or {}
    if meta:
        usage = {
            "input": meta.get("input_tokens", 0),
            "output": meta.get("output_tokens", 0),
        }
    return arbitration, usage


def write_tsv(rows: list[dict], out_path: Path) -> None:
    """Write the review sheet, hardest cases first: drops, then low confidence."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "n",
        "verdict",
        "status",
        "note",
        "question",
        "source",
        "chunk_id",
        "preview",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run(
    sample_path: Path = DEFAULT_SAMPLE,
    out_path: Path = DEFAULT_OUT,
    model: str = REVIEW_MODEL,
    limit: int | None = None,
    threshold: float = CONFIDENCE_THRESHOLD,
    dry_run: bool = False,
) -> dict:
    """Review the sample and write the TSV. Returns the summary."""
    records = load_records(Path(sample_path))
    if limit:
        records = records[:limit]

    if not dry_run:
        records = attach_chunk_texts(records, corpus_text_map())
        missing = sum(1 for r in records if r.get("chunk_missing"))
        if missing:
            print(f"⚠ {missing} чанков не найдено в коллекции — судятся по превью.")

    raw_path = Path(out_path).with_suffix(".raw.jsonl")
    reviewed = load_reviewed(raw_path)
    pending = [(i, r) for i, r in enumerate(records, start=1) if i not in reviewed]

    avg_chars = sum(len(r.get("chunk_preview") or "") for r in records) / max(
        len(records), 1
    )
    # Промпт рецензента тяжелее генераторского: инструкция ~450 токенов сверх чанка.
    projected = estimate_cost(len(pending), model=model, avg_chunk_tokens=800)
    print(
        f"К разметке: {len(pending)} из {len(records)} "
        f"(уже размечено {len(reviewed)}). Оценка ≈ ${projected:.2f}, модель {model}."
    )
    if dry_run:
        return {
            "pending": len(pending),
            "estimate_usd": projected,
            "avg_chars": avg_chars,
        }
    if projected > COST_ABORT_USD:
        raise SystemExit(
            f"Оценка ${projected:.2f} выше потолка ${COST_ABORT_USD:.2f} — прогон отменён."
        )

    usages: list[dict] = []
    if pending:
        llm = _make_llm(model)
        with raw_path.open("a", encoding="utf-8") as raw_fh:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
                futures = {pool.submit(review_one, rec, llm): n for n, rec in pending}
                done = 0
                for future in as_completed(futures):
                    n = futures[future]
                    review, usage = future.result()
                    reviewed[n] = {"n": n, **review}
                    usages.append(usage)
                    raw_fh.write(
                        json.dumps({"n": n, **review}, ensure_ascii=False) + "\n"
                    )
                    raw_fh.flush()
                    done += 1
                    if done % 50 == 0:
                        print(f"  {done}/{len(pending)}")

    rows = [
        review_row(i, rec, reviewed[i])
        for i, rec in enumerate(records, start=1)
        if i in reviewed
    ]
    rows.sort(key=lambda r: (r["verdict"] != "drop", r["note"]))
    write_tsv(rows, Path(out_path))

    stats = summarize(
        [reviewed[i] for i in sorted(reviewed) if i <= len(records)], threshold
    )
    stats["spent_usd"] = calc_total_price(usages, model=model) if usages else 0.0
    stats["out"] = str(out_path)
    return stats


def run_arbitration(
    sample_path: Path = DEFAULT_SAMPLE,
    out_path: Path = DEFAULT_OUT,
    model: str = ARBITER_MODEL,
    limit: int | None = None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict:
    """Second pass over the rejects of a finished review, then rewrite the TSV."""
    records = load_records(Path(sample_path))
    if limit:
        records = records[:limit]
    records = attach_chunk_texts(records, corpus_text_map())

    out_path = Path(out_path)
    reviewed = load_reviewed(out_path.with_suffix(".raw.jsonl"))
    if not reviewed:
        raise SystemExit("Нечего арбитрировать: сначала прогон разметки.")
    arb_path = out_path.with_suffix(".arb.jsonl")
    arbitrated = load_reviewed(arb_path)

    pending = [
        (n, records[n - 1])
        for n in sorted(reviewed)
        if n <= len(records)
        and reviewed[n]["verdict"] == "drop"
        and n not in arbitrated
    ]
    projected = estimate_cost(len(pending), model=model, avg_chunk_tokens=800)
    print(
        f"Арбитраж: {len(pending)} отказов, оценка ≈ ${projected:.2f}, модель {model}."
    )
    if projected > COST_ABORT_USD:
        raise SystemExit(f"Оценка ${projected:.2f} выше потолка ${COST_ABORT_USD:.2f}.")

    usages: list[dict] = []
    if pending:
        llm = _make_llm(model)
        with arb_path.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=ARBITER_WORKERS) as pool:
                futures = {
                    pool.submit(arbitrate_one, rec, reviewed[n], llm): n
                    for n, rec in pending
                }
                done = 0
                for future in as_completed(futures):
                    n = futures[future]
                    arbitration, usage = future.result()
                    arbitrated[n] = {"n": n, **arbitration}
                    usages.append(usage)
                    fh.write(
                        json.dumps({"n": n, **arbitration}, ensure_ascii=False) + "\n"
                    )
                    fh.flush()
                    done += 1
                    if done % 50 == 0:
                        print(f"  {done}/{len(pending)}")

    merged = {
        n: merge_arbitration(reviewed[n], arbitrated.get(n))
        for n in sorted(reviewed)
        if n <= len(records)
    }
    rows = [review_row(n, records[n - 1], merged[n]) for n in sorted(merged)]
    order = {"disputed": 0, "drop_confirmed": 1, "single_pass": 2}
    rows.sort(key=lambda r: (order.get(r["status"], 3), r["note"]))
    write_tsv(rows, out_path)

    gt_path = out_path.parent / "retrieval_gt_reviewed.jsonl"
    kept = build_reviewed_gt(records, merged, threshold)
    with gt_path.open("w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    statuses = Counter(m["status"] for m in merged.values())
    stats = summarize(list(merged.values()), threshold)
    stats["statuses"] = dict(statuses)
    stats["reviewed_gt"] = {"path": str(gt_path), "questions": len(kept)}
    stats["spent_usd"] = calc_total_price(usages, model=model) if usages else 0.0
    stats["out"] = str(out_path)
    return stats


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Adversarial review of the synthetic retrieval GT sample"
    )
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=REVIEW_MODEL)
    parser.add_argument(
        "--limit", type=int, default=None, help="review only the first N records"
    )
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the cost estimate and exit"
    )
    parser.add_argument(
        "--arbitrate",
        action="store_true",
        help="second pass over the rejects of a finished review",
    )
    parser.add_argument("--arbiter-model", default=ARBITER_MODEL)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.arbitrate:
        stats = run_arbitration(
            sample_path=args.sample,
            out_path=args.out,
            model=args.arbiter_model,
            limit=args.limit,
            threshold=args.threshold,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return
    stats = run(
        sample_path=args.sample,
        out_path=args.out,
        model=args.model,
        limit=args.limit,
        threshold=args.threshold,
        dry_run=args.dry_run,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
