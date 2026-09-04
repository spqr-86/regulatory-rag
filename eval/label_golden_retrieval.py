"""Held-out retrieval labelling: which chunk answers each hand-written question.

The synthetic GT is written *from* a chunk, so a question there is close to the
text it is tagged with by construction — Hit Rate on it drifts up for reasons
that have nothing to do with retrieval quality. The 57 questions in
``tests/dataset.csv`` were written by a person against the domain, not against a
chunk, so labelling them gives a second, independent number. Step 3 without this
one is a figure nobody can trust (docs/roadmap.md, шаг 4).

Labelling is what a person would spend an hour on: for every question the tool
retrieves a candidate pool and asks which candidates actually answer it. Two
passes with opposite instructions judge each pool — a strict one that has to be
convinced, a lenient one that gives the question the benefit of the doubt — and
only their agreement becomes ground truth. Every disagreement, every question
where nothing was found and a control sample of agreements go to a TSV for the
human: the point of the exercise is a number that is *not* the judge's opinion
of itself, and the control rows are what measures the judge against a person.

Candidates come from both retrieval paths merged round-robin, so the labels are
not tied to the path that produced them: a GT labelled off ``complex`` alone
would flatter ``complex`` when the metric is run.

Usage::

    .venv/bin/python eval/label_golden_retrieval.py --limit 5 --dry-run  # cost only
    .venv/bin/python eval/label_golden_retrieval.py --limit 5            # smoke
    .venv/bin/python eval/label_golden_retrieval.py                      # full run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval.generate_retrieval_gt import calc_total_price, price_for  # noqa: E402

DEFAULT_DATASET = REPO_ROOT / "tests" / "dataset.csv"
DEFAULT_OUT = REPO_ROOT / "eval" / "data" / "golden_retrieval_labeled.jsonl"

# Судья сильный намеренно: разметка релевантности — дорогая ошибка, а объём
# смешной (десятки вызовов), поэтому экономить тут нечего. Потолок нашего ключа
# (аккаунт не верифицирован, astra недоступен) — см. reference/llm_apis.md.
JUDGE_MODEL = "gpt-5.6-terra"
# Арбитр по всему, что дешёвая пара не закрыла сама. Замер 04.09 на 5 вопросах:
# terra дважды сослалась на цитату, которой в чанке нет, sol — ни разу; сильная
# модель зовётся только там, где решается эталон, и стоит поэтому копейки.
ARBITER_MODEL = "gpt-5.6-sol"
# Глубина пула на путь. Больше 20 человек не вычитает, а метрика меряется по
# top-12 — кандидат с 30-го места ничего в ней не меняет.
TOP_K = 20
# Сколько кандидатов уходит судье после объединения путей.
CANDIDATE_LIMIT = 30
PATHS = ("simple", "complex")
CONTROL_SAMPLE = 10
CONTROL_SEED = 20260904
MAX_WORKERS = 4
COST_ABORT_USD = 15.0

# Вопросы вне корпуса и с ложной посылкой размечать нечем: релевантного чанка
# для них не существует по построению, и в retrieval-метрику они не входят.
NOT_LABELABLE_OOS = ("out_of_scope", "false_premise")

STATUS_AGREED = "agreed"
STATUS_DISPUTED = "disputed"
STATUS_NONE_FOUND = "none_found"
STATUS_UNVERIFIED = "unverified_quote"
STATUS_ARBITRATED = "arbitrated"


class RelevantChunk(BaseModel):
    """One candidate the judge says answers the question."""

    index: int = Field(description="номер кандидата из списка, начиная с 1")
    quote: str = Field(
        description="дословная фраза из этого кандидата, отвечающая на вопрос"
    )


class LabelSet(BaseModel):
    """Structured-output schema for one labelling call."""

    relevant: list[RelevantChunk] = Field(
        default_factory=list, description="все кандидаты, отвечающие на вопрос"
    )
    note: str = Field(default="", description="короткое замечание, необязательно")


STRICT_PROMPT = """Ты размечаешь эталон для поисковой системы по нормативной базе
(охрана труда, пожарная безопасность). Тебе дан вопрос и пронумерованные фрагменты
документов, которые вернул поиск. Нужно указать те фрагменты, которые ДЕЙСТВИТЕЛЬНО
отвечают на вопрос.

Ты — строгий разметчик. По умолчанию фрагмент не подходит; включай его, только если
убедился. Критерий один: во фрагменте есть текст, из которого человек получит ответ
на заданный вопрос. Не «фрагмент про ту же тему», не «фрагмент из нужного документа»,
а именно ответ.

Для каждого включённого фрагмента выпиши в quote ДОСЛОВНУЮ фразу из него — ту самую,
которая отвечает. Если выписать нечего, фрагмент включать нельзя.

Фрагментов-ответов может быть несколько (норма продублирована, ответ разбит по пунктам) —
укажи все. Если ни один не отвечает, верни пустой список: это нормальный исход, поиск
мог не найти ничего.

Заголовок в начале фрагмента бывает от чужого раздела — это дефект нарезки корпуса.
Суди по тексту нормы, а не по заголовку.

ВОПРОС: {question}

ФРАГМЕНТЫ:
{candidates}
"""

LENIENT_PROMPT = """Ты размечаешь эталон для поисковой системы по нормативной базе
(охрана труда, пожарная безопасность). Тебе дан вопрос и пронумерованные фрагменты
документов, которые вернул поиск. Нужно указать те фрагменты, которые отвечают на вопрос.

Ты — снисходительный разметчик: первый рецензент известен тем, что бракует лишнее,
и твоя задача — не потерять годный фрагмент. Засчитывай фрагмент, если он отвечает
на вопрос хотя бы по существу — пусть неполно, пусть другими словами, пусть ответ
собирается из перечня, таблицы или строки классификатора. Фрагмент, обрывающийся
до самого ответа, засчитывать нельзя: обрыв — это отсутствие ответа.

Для каждого включённого фрагмента выпиши в quote ДОСЛОВНУЮ фразу из него — ту самую,
которая отвечает. Если выписать нечего, фрагмент включать нельзя.

Если ни один фрагмент не отвечает, верни пустой список.

Заголовок в начале фрагмента бывает от чужого раздела — это дефект нарезки корпуса.
Суди по тексту нормы, а не по заголовку.

ВОПРОС: {question}

ФРАГМЕНТЫ:
{candidates}
"""


ARBITER_PROMPT = """Ты — арбитр в разметке эталона для поисковой системы по нормативной
базе (охрана труда, пожарная безопасность). Два предыдущих разметчика — строгий и
снисходительный — разошлись во мнении или не нашли ответа вовсе. Решаешь ты.

Тебе дан вопрос и пронумерованные фрагменты. Укажи те, которые ДЕЙСТВИТЕЛЬНО отвечают
на вопрос: во фрагменте есть текст, из которого человек получит ответ. Не «фрагмент про
ту же тему», не «фрагмент из нужного документа», а ответ.

На каждый указанный фрагмент выпиши в quote ДОСЛОВНУЮ фразу из него. Цитата проверяется
автоматически: если такой фразы в тексте фрагмента нет, твой вердикт не будет засчитан.
Не восстанавливай норму по памяти — только то, что видишь в тексте.

Пустой список — нормальный ответ: поиск мог не найти ничего.

Заголовок в начале фрагмента бывает от чужого раздела — это дефект нарезки корпуса.
Суди по тексту нормы, а не по заголовку.

ПРЕДМЕТ СПОРА: фрагменты {disputed}

ВОПРОС: {question}

ФРАГМЕНТЫ:
{candidates}
"""


# ─── Deterministic core (unit-tested, no heavy imports) ──────────────────────


def load_golden_questions(path: Path = DEFAULT_DATASET) -> list[dict]:
    """The hand-written golden set, numbered by position in the file.

    Numbering follows the source file rather than the filtered result, so a run
    with ``--limit`` and a full run talk about the same question 17.
    """
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            oos = (row.get("oos_type") or "").strip()
            records.append(
                {
                    "n": i,
                    "question": question,
                    "ground_truth": (row.get("ground_truth") or "").strip(),
                    "oos_type": oos,
                    "in_scope": oos not in NOT_LABELABLE_OOS,
                }
            )
    return records


def merge_candidates(
    pools: Sequence[Sequence[dict]],
    limit: int = CANDIDATE_LIMIT,
    pool_names: Sequence[str] | None = None,
) -> list[dict]:
    """Round-robin union of candidate pools, deduplicated by chunk_id.

    Interleaving keeps both paths' top results near the top of the list: taking
    one path first and appending the other would push the second path's best
    candidates past the cut whenever the pools barely overlap.
    """
    names = list(pool_names) if pool_names else [f"pool{i}" for i in range(len(pools))]
    merged: list[dict] = []
    seen: dict[str, dict] = {}
    depth = max((len(p) for p in pools), default=0)
    for rank in range(depth):
        for pool_idx, pool in enumerate(pools):
            if rank >= len(pool):
                continue
            cand = pool[rank]
            cid = cand.get("chunk_id") or ""
            if not cid:
                continue
            name = names[pool_idx] if pool_idx < len(names) else f"pool{pool_idx}"
            if cid in seen:
                if name not in seen[cid]["pools"]:
                    seen[cid]["pools"].append(name)
                continue
            entry = dict(cand)
            entry["pools"] = [name]
            seen[cid] = entry
            merged.append(entry)
    # Кандидат, найденный обоими путями, мог прийти из хвоста — метка pools
    # ставится по факту нахождения, а обрезка идёт уже по общему порядку.
    return merged[:limit]


def format_candidates(candidates: Sequence[dict]) -> str:
    """Numbered candidate block for the prompt. Numbers start at 1."""
    blocks = []
    for i, cand in enumerate(candidates, start=1):
        source = cand.get("source") or (cand.get("chunk_id") or "").split("#")[0]
        blocks.append(f"[{i}] (документ {source})\n{cand.get('text', '')}")
    return "\n\n".join(blocks)


def parse_labels(labels: LabelSet | dict, n_candidates: int) -> dict:
    """Normalise one judge verdict.

    An index outside the list is dropped rather than clamped: a model that
    answers "31" out of 30 candidates is guessing, and clamping would silently
    turn a guess into a label on the last chunk.
    """
    data = labels if isinstance(labels, dict) else labels.model_dump()
    relevant: list[dict] = []
    seen: set[int] = set()
    for item in data.get("relevant") or []:
        raw = item if isinstance(item, dict) else item.model_dump()
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        if not 1 <= index <= n_candidates or index in seen:
            continue
        seen.add(index)
        relevant.append({"index": index, "quote": str(raw.get("quote", "")).strip()})
    return {"relevant": relevant, "note": str(data.get("note", "")).strip()}


_SPAN_WS_RE = re.compile(r"\s+")


def validate_label_spans(labels: dict, candidates: Sequence[dict]) -> dict:
    """Check every label against its own evidence.

    A quote that is not in the chunk means the answer was reconstructed from
    general knowledge instead of read off the text — the exact failure this
    labelling exists to avoid. The label is flagged rather than dropped: which
    of the two passes was right is a question for the human queue, not for the
    same model that produced the quote.
    """
    checked = dict(labels)
    out = []
    for item in labels.get("relevant") or []:
        text = ""
        idx = item["index"]
        if 1 <= idx <= len(candidates):
            text = candidates[idx - 1].get("text", "") or ""
        quote = _SPAN_WS_RE.sub(" ", item.get("quote", "")).strip().lower()
        haystack = _SPAN_WS_RE.sub(" ", text).lower()
        verified = bool(quote) and quote in haystack
        out.append({**item, "verified": verified})
    checked["relevant"] = out
    return checked


def _chunk_ids(indices: Iterable[int], candidates: Sequence[dict]) -> list[str]:
    ids = []
    for idx in indices:
        if 1 <= idx <= len(candidates):
            cid = candidates[idx - 1].get("chunk_id")
            if cid:
                ids.append(cid)
    return ids


def merge_passes(strict: dict, lenient: dict, candidates: Sequence[dict]) -> dict:
    """Combine the two opposed passes into one verdict.

    Only what both passes marked becomes ground truth. Everything else — one
    pass alone, or nothing at all — is a question for the person: measured on
    the GT sample review (04.09), the strict pass rejects roughly a third too
    much and the lenient one accepts fragments that merely look like an answer,
    so neither is trustworthy alone.
    """
    strict_idx = {item["index"]: item for item in strict.get("relevant") or []}
    lenient_idx = {item["index"]: item for item in lenient.get("relevant") or []}
    agreed = sorted(set(strict_idx) & set(lenient_idx))
    disputed = sorted(set(strict_idx) ^ set(lenient_idx))

    gold = _chunk_ids(agreed, candidates)
    if not agreed and not disputed:
        status = STATUS_NONE_FOUND
    elif disputed:
        status = STATUS_DISPUTED
    elif any(
        not (strict_idx[i].get("verified") and lenient_idx[i].get("verified"))
        for i in agreed
    ):
        status = STATUS_UNVERIFIED
    else:
        status = STATUS_AGREED

    unverified = [
        i
        for i in agreed
        if not (strict_idx[i].get("verified") and lenient_idx[i].get("verified"))
    ]
    return {
        "status": status,
        "gold_chunk_ids": gold,
        "disputed_chunk_ids": _chunk_ids(disputed, candidates),
        # Что именно предъявляется арбитру: спорные кандидаты и согласия,
        # подпёртые цитатой, которой в чанке нет.
        "disputed_indices": sorted(set(disputed) | set(unverified)),
        "quotes": [
            strict_idx.get(i, lenient_idx.get(i, {})).get("quote", "") for i in agreed
        ],
        "note": " ".join(
            x for x in (strict.get("note"), lenient.get("note")) if x
        ).strip(),
    }


def needs_human(result: dict) -> bool:
    """Whether the verdict has to be looked at by a person.

    Settled cases are a clean agreement of the two passes and a case the arbiter
    closed with a chunk to show for it. Everything else goes to the person: an
    open dispute, an unverified quote, and any question left without ground
    truth — it drops out of the metric, and a set that quietly loses its hard
    questions reads better than it is.
    """
    status = result.get("status")
    if status == STATUS_AGREED:
        return False
    if status == STATUS_ARBITRATED:
        return not result.get("gold_chunk_ids")
    return True


def needs_arbitration(result: dict) -> bool:
    """Whether the stronger judge is called for this question.

    Everything the cheap pair could not settle by itself: a disagreement, an
    agreement resting on a quote that is not in the chunk, and a question they
    found no answer for at all — the last is where a stronger model most often
    earns its price, since a weak judge missing the answer looks exactly like
    retrieval missing it.
    """
    return result.get("status") != STATUS_AGREED


def apply_arbitration(
    merged: dict, arbitration: dict | None, candidates: Sequence[dict]
) -> dict:
    """Settle the open part of a verdict with the stronger judge's answer.

    The arbiter rules only on what was open — the disputed candidates and the
    unverified agreements. It cannot relabel the rest of the pool: a chunk both
    cheap passes ignored was never in question, and letting a third opinion add
    it would make the эталон the arbiter's alone. Its own verdict is held to the
    same evidence rule: a quote absent from the chunk buys nothing.
    """
    if arbitration is None:
        return merged
    open_indices = set(merged.get("disputed_indices") or [])
    if not open_indices:
        # none_found: спорить не о чем, арбитр смотрит весь пул сам.
        open_indices = set(range(1, len(candidates) + 1))

    confirmed: list[int] = []
    unverified = False
    for item in arbitration.get("relevant") or []:
        idx = item.get("index")
        if idx not in open_indices:
            continue
        if item.get("verified"):
            confirmed.append(idx)
        else:
            unverified = True

    out = dict(merged)
    gold = list(merged.get("gold_chunk_ids") or [])
    for cid in _chunk_ids(sorted(confirmed), candidates):
        if cid not in gold:
            gold.append(cid)
    # Согласие, снятое арбитром, из эталона уходит: подтверждения цитатой нет.
    rejected = set(_chunk_ids(sorted(open_indices - set(confirmed)), candidates))
    gold = [
        cid
        for cid in gold
        if cid not in rejected or cid in _chunk_ids(confirmed, candidates)
    ]

    out["gold_chunk_ids"] = gold
    out["disputed_chunk_ids"] = []
    out["disputed_indices"] = []
    out["status"] = STATUS_UNVERIFIED if unverified else STATUS_ARBITRATED
    if arbitration.get("note"):
        out["note"] = f"{merged.get('note', '')} арбитр: {arbitration['note']}".strip()
    return out


def human_priority(result: dict) -> str:
    """How urgently a verdict needs a person.

    ``blocking`` — the question has no ground truth to measure with (nothing
    agreed, or the only agreement rests on a quote that is not in the chunk):
    left unchecked it silently leaves the metric, and a set that quietly drops
    its hard questions reads better than it is. ``optional`` — the passes agreed
    on a chunk and differ over an extra one: Hit Rate is already decided, only
    recall moves. Everything else is not queued.
    """
    if not needs_human(result):
        return ""
    if result.get("status") == STATUS_UNVERIFIED or not result.get("gold_chunk_ids"):
        return "blocking"
    return "optional"


def pick_controls(
    results: Sequence[dict], k: int = CONTROL_SAMPLE, seed: int = CONTROL_SEED
) -> list[dict]:
    """A deterministic sample of clean agreements for the human to check.

    Disputes reach the person anyway; these rows are what measures the judge
    where it was confident — without them "agreement" is a number the judge
    grades itself on.
    """
    pool = [r for r in results if not needs_human(r)]
    if len(pool) <= k:
        return list(pool)
    rng = random.Random(seed)
    return sorted(rng.sample(pool, k), key=lambda r: r.get("n", 0))


GT_FIELDS = ("question", "chunk_id", "relevant_chunk_ids", "source")


def build_labeled_gt(records: Sequence[dict], merged: dict[int, dict]) -> list[dict]:
    """The labelled eval set: questions that got at least one agreed chunk.

    ``chunk_id`` repeats the first gold chunk so the file loads in the existing
    runner unchanged; ``relevant_chunk_ids`` carries all of them for metrics
    that take a list. Disputed chunks are not written: a chunk one pass rejected
    would score a hit for retrieval nobody agreed was correct.
    """
    out = []
    for rec in records:
        verdict = merged.get(rec["n"])
        if not verdict:
            continue
        gold = verdict.get("gold_chunk_ids") or []
        if not gold:
            continue
        out.append(
            {
                "question": rec["question"],
                "chunk_id": gold[0],
                "relevant_chunk_ids": list(gold),
                "source": gold[0].split("#")[0],
            }
        )
    return out


def load_labeled(raw_path: Path) -> dict[int, dict]:
    """Resume log ``n -> verdict``. A torn last line is dropped, not fatal."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        return {}
    labeled: dict[int, dict] = {}
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
                labeled[int(rec["n"])] = rec
    return labeled


TSV_FIELDS = ("n", "role", "status", "question", "gold", "disputed", "quote", "note")


def label_row(record: dict, merged: dict, role: str) -> dict:
    """One row of the review sheet handed to the human."""
    quotes = [q for q in (merged.get("quotes") or []) if q]
    return {
        "n": record.get("n", ""),
        "role": role,
        "status": merged.get("status", ""),
        "question": record.get("question", ""),
        "gold": " | ".join(merged.get("gold_chunk_ids") or []),
        "disputed": " | ".join(merged.get("disputed_chunk_ids") or []),
        "quote": " | ".join(quotes),
        "note": merged.get("note", ""),
    }


def build_review_rows(
    records: dict[int, dict],
    results: Sequence[dict],
    controls: int = CONTROL_SAMPLE,
    seed: int = CONTROL_SEED,
) -> list[dict]:
    """The human's sheet: blocking cases, then optional ones, then controls.

    Order is the instruction: a person who stops after the first block has
    already unblocked the metric.
    """
    rows: list[dict] = []
    for role in ("blocking", "optional"):
        for res in results:
            if human_priority(res) != role:
                continue
            rec = records.get(res["n"])
            if rec:
                rows.append(label_row(rec, res, role=role))
    for res in pick_controls(results, k=controls, seed=seed):
        rec = records.get(res["n"])
        if rec:
            rows.append(label_row(rec, res, role="control"))
    return rows


def write_tsv(rows: Sequence[dict], out_path: Path) -> None:
    """Write the review sheet: disputes first, controls after."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(TSV_FIELDS), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TSV_FIELDS})


def summarize(results: Iterable[dict]) -> dict:
    """Status counts, the size of the human queue and how many got a gold chunk."""
    results = list(results)
    counts = Counter(r.get("status", "") for r in results)
    return {
        "total": len(results),
        "agreed": counts.get(STATUS_AGREED, 0),
        "arbitrated": counts.get(STATUS_ARBITRATED, 0),
        "disputed": counts.get(STATUS_DISPUTED, 0),
        "none_found": counts.get(STATUS_NONE_FOUND, 0),
        "unverified_quote": counts.get(STATUS_UNVERIFIED, 0),
        "needs_human": sum(1 for r in results if needs_human(r)),
        "with_gold": sum(1 for r in results if r.get("gold_chunk_ids")),
    }


def estimate_cost(
    n_questions: int,
    model: str = JUDGE_MODEL,
    avg_candidate_tokens: int = 180,
    n_candidates: int = CANDIDATE_LIMIT,
) -> float:
    """Pre-flight estimate: two passes over the same pool, ~120 tokens out each."""
    per_pass = calc_total_price(
        [{"input": avg_candidate_tokens * n_candidates + 300, "output": 120}],
        model=model,
    )
    return per_pass * 2 * n_questions


# ─── LLM pass ────────────────────────────────────────────────────────────────


def _make_llm(model: str = JUDGE_MODEL):
    from src.infra import llm_factory  # noqa: PLC0415

    return llm_factory.get_judge_llm(model_name=model)


def label_one(
    question: str, candidates: Sequence[dict], llm, prompt: str
) -> tuple[dict, dict]:
    """One structured-output labelling call. Returns ``(labels, usage)``."""
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

    structured = llm.with_structured_output(LabelSet, include_raw=True)
    rendered = prompt.format(
        question=question, candidates=format_candidates(candidates)
    )

    @retry(
        retry=retry_if_exception_type(
            (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def _call():
        return structured.invoke(rendered)

    result = _call()
    parsed = result["parsed"] if isinstance(result, dict) else result
    if parsed is None:
        # Пустая разметка отправила бы вопрос в none_found и тихо выкинула его
        # из метрики; помечаем явно, чтобы это увидел человек.
        labels = {"relevant": [], "note": "unparsed model output"}
    else:
        labels = validate_label_spans(parse_labels(parsed, len(candidates)), candidates)
    usage = {}
    raw_msg = result.get("raw") if isinstance(result, dict) else None
    meta = getattr(raw_msg, "usage_metadata", None) or {}
    if meta:
        usage = {
            "input": meta.get("input_tokens", 0),
            "output": meta.get("output_tokens", 0),
        }
    return labels, usage


def arbitrate_one(
    question: str, candidates: Sequence[dict], open_indices: Sequence[int], llm
) -> tuple[dict, dict]:
    """Stronger judge on one unsettled question. Returns ``(labels, usage)``."""
    disputed = ", ".join(f"[{i}]" for i in open_indices) or "весь список"
    prompt = ARBITER_PROMPT.replace("{disputed}", disputed)
    return label_one(question, candidates, llm, prompt)


def build_candidate_pools(question: str, top_k: int = TOP_K) -> list[list[dict]]:
    """Candidate pools for one question, one per retrieval path."""
    from eval.run_retrieval_eval import make_retrieval_fn  # noqa: PLC0415

    pools = []
    for path in PATHS:
        retrieve = _RETRIEVERS.setdefault(
            path, make_retrieval_fn(path, return_passages=True)
        )
        pools.append(list(retrieve(question))[:top_k])
    return pools


_RETRIEVERS: dict = {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=JUDGE_MODEL, help="модель обоих проходов")
    parser.add_argument(
        "--arbiter-model", default=ARBITER_MODEL, help="модель арбитра по спорным"
    )
    parser.add_argument(
        "--no-arbitrate", action="store_true", help="без третьего прохода"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="label only the first N questions"
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument(
        "--candidates",
        type=int,
        default=CANDIDATE_LIMIT,
        help="сколько кандидатов уходит судье после объединения путей",
    )
    parser.add_argument("--controls", type=int, default=CONTROL_SAMPLE)
    parser.add_argument("--seed", type=int, default=CONTROL_SEED)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument(
        "--dry-run", action="store_true", help="cost estimate, no calls"
    )
    args = parser.parse_args()

    records = [r for r in load_golden_questions(args.dataset) if r["in_scope"]]
    if args.limit:
        records = records[: args.limit]

    price = price_for(args.model)
    estimate = estimate_cost(
        len(records), model=args.model, n_candidates=args.candidates
    )
    line = (
        f"вопросов: {len(records)} · судья: {args.model} "
        f"(${price['input']}/${price['output']} за 1M)"
    )
    if not args.no_arbitrate:
        # По смоуку 04.09 арбитр зовётся примерно на половине вопросов.
        arb = (
            estimate_cost(
                len(records), model=args.arbiter_model, n_candidates=args.candidates
            )
            / 2
            * 0.5
        )
        estimate += arb
        price_a = price_for(args.arbiter_model)
        line += (
            f" · арбитр: {args.arbiter_model} "
            f"(${price_a['input']}/${price_a['output']} за 1M, ~половина вопросов)"
        )
    print(f"{line} · оценка: ${estimate:.2f}")
    if args.dry_run:
        return 0
    if estimate > COST_ABORT_USD:
        print(f"оценка выше предохранителя ${COST_ABORT_USD}: прогон не запущен")
        return 1

    from eval.run_retrieval_eval import init_engine  # noqa: PLC0415

    init_engine()
    llm = _make_llm(args.model)
    arbiter_llm = None if args.no_arbitrate else _make_llm(args.arbiter_model)

    raw_path = Path(str(args.out) + ".raw")
    done = load_labeled(raw_path)
    usages: list[dict] = []
    arbiter_usages: list[dict] = []
    results: dict[int, dict] = {}
    candidates_by_n: dict[int, list[dict]] = {}

    def _process(rec: dict) -> tuple[dict, list[dict], list[dict]]:
        pools = build_candidate_pools(rec["question"], top_k=args.top_k)
        candidates = merge_candidates(pools, limit=args.candidates, pool_names=PATHS)
        if not candidates:
            return (
                {
                    "status": STATUS_NONE_FOUND,
                    "gold_chunk_ids": [],
                    "disputed_chunk_ids": [],
                    "quotes": [],
                    "note": "поиск не вернул кандидатов",
                },
                [],
                [],
            )
        strict, u1 = label_one(rec["question"], candidates, llm, STRICT_PROMPT)
        lenient, u2 = label_one(rec["question"], candidates, llm, LENIENT_PROMPT)
        merged = merge_passes(strict, lenient, candidates)
        arb_usage: list[dict] = []
        if arbiter_llm is not None and needs_arbitration(merged):
            open_idx = merged.get("disputed_indices") or list(
                range(1, len(candidates) + 1)
            )
            arbitration, u3 = arbitrate_one(
                rec["question"], candidates, open_idx, arbiter_llm
            )
            merged = apply_arbitration(merged, arbitration, candidates)
            arb_usage = [u3]
        return merged, candidates, ([u1, u2], arb_usage)

    pending = [r for r in records if r["n"] not in done]
    for n, verdict in done.items():
        results[n] = verdict
    if pending:
        with (
            raw_path.open("a", encoding="utf-8") as raw_fh,
            ThreadPoolExecutor(max_workers=args.workers) as pool,
        ):
            futures = {pool.submit(_process, rec): rec for rec in pending}
            for fut in as_completed(futures):
                rec = futures[fut]
                verdict, candidates, (pass_usages, arb_usages) = fut.result()
                usages.extend(pass_usages)
                arbiter_usages.extend(arb_usages)
                results[rec["n"]] = verdict
                candidates_by_n[rec["n"]] = candidates
                raw_fh.write(
                    json.dumps({"n": rec["n"], **verdict}, ensure_ascii=False) + "\n"
                )
                raw_fh.flush()
                print(f"[{rec['n']:>3}] {verdict['status']:<16} {rec['question'][:60]}")

    ordered = [{**results[r["n"]], "n": r["n"]} for r in records if r["n"] in results]
    summary = summarize(ordered)
    spent = calc_total_price(usages, model=args.model) + calc_total_price(
        arbiter_usages, model=args.arbiter_model
    )

    gt = build_labeled_gt(records, results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in gt:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_n = {r["n"]: r for r in records}
    rows = build_review_rows(by_n, ordered, controls=args.controls, seed=args.seed)
    tsv_path = Path(str(args.out).replace(".jsonl", "")).with_suffix(".review.tsv")
    write_tsv(rows, tsv_path)

    print(
        f"\nитог: {summary['agreed']} согласий · {summary['arbitrated']} решено арбитром · "
        f"{summary['disputed']} споров · {summary['none_found']} без ответа · "
        f"{summary['unverified_quote']} без цитаты"
    )
    print(f"эталон: {len(gt)} вопросов → {args.out}")
    blocking = sum(1 for r in rows if r["role"] == "blocking")
    print(f"человеку: {len(rows)} строк ({blocking} обязательных) → {tsv_path}")
    print(f"потрачено: ${spent:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
