"""Generate retrieval ground truth (A→Q*) from the real Chroma corpus.

For every non-junk chunk in the vector collection the generator asks an LLM for
``N`` natural questions whose answer is that chunk, then writes
``eval/data/retrieval_gt.jsonl`` with the schema::

    {"question": str, "chunk_id": str, "source": str, "chunk_preview": str}

Generation is crash-safe: every finished chunk is appended to
``<out>.raw`` immediately, a restart resumes from that log (already generated
chunks are not re-paid for), and ``--finalize`` derives the deduped final file
from the log without generating anything again. Dedup itself embeds the surviving
questions, so those embeddings are cached next to the output (``<out>.embcache.npz``)
and a repeated ``--finalize`` costs nothing.

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
import os
import random
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
# Между разными чанками похожий вопрос ядовит: отвечают оба, размечен один.
CROSS_CHUNK_THRESHOLD = 0.88
MAX_WORKERS = 6
COST_ABORT_USD = 2.0
PREVIEW_CHARS = 200

# Model the generator runs on. Pinned here on purpose: writing questions from a
# given chunk is a cheap job, and inheriting the eval judge's model (get_judge_llm)
# would silently retag it at 17x the price if JUDGE_MODEL_NAME changes.
GEN_MODEL = "gpt-4o-mini"

# OpenAI list prices, USD per 1M tokens (checked 2026-09-02).
PRICE_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Промо-цена, действует минимум до конца ноября 2026 (базовая — $5/$30).
    # Длинный контекст (>272K) идёт вдвое дороже; наши промпты в него не входят.
    "gpt-5.6-sol": {"input": 4.00, "output": 20.00},
}


def price_for(model: str) -> dict:
    """Rate card for ``model``. Unknown model is an error, not a default: pricing
    it at some other model's rate is exactly how the cost guard gets fooled."""
    try:
        return PRICE_PER_1M[model]
    except KeyError:
        known = ", ".join(sorted(PRICE_PER_1M))
        raise ValueError(
            f"No price for model {model!r}. Known: {known}. "
            "Add its rate to PRICE_PER_1M before running."
        ) from None


class Questions(BaseModel):
    """Structured-output schema for one LLM generation call."""

    questions: list[str] = Field(description="Natural questions answered by the chunk")


GEN_PROMPT = """Ты — специалист по охране труда, который ищет ответ в нормативной базе.
Ниже фрагмент нормативного документа. Сформулируй {n} естественных вопросов,
ответ на которые содержится ИМЕННО в этом фрагменте и больше нигде.

Главное требование — зацепка. В каждом вопросе должна быть конкретная деталь
из фрагмента: срок, периодичность, числовое значение или порог, вид работ,
условие применения, категория работников или адресат обязанности. Без этой детали
вопрос отвечается половиной нормативной базы, и разметка становится ложью.

Не задавай вопросов, на которые можно ответить, не читая фрагмент:
«Какие документы регулируют охрану труда?», «Что говорит закон о медосмотрах?»,
«Каковы основные требования к обучению?» — такие формулировки запрещены.

Проверяй себя так: закрой фрагмент и попробуй ответить. Если ответ угадывается
из общих знаний или из самого вопроса — вопрос негодный. «Какова величина МРОТ?»,
«Какой возрастной порог установлен для лиц, трудоустройство которых имеет
особенности?», «Какие гарантии по оплате труда предусмотрены для работников?» —
всё это угадывается и запрещено. Ответом должна быть конкретика, которую без
этого фрагмента не назовёшь.

Не спрашивай, каким актом норма установлена: «Какой закон регулирует…»,
«Какое законодательство определяет…», «Какой документ устанавливает порядок…».
Ответ на такой вопрос — название документа, и он подходит к любому его фрагменту.

Не ссылайся на структуру источника: «согласно статье 213», «согласно графе 3»,
«в соответствии с настоящим Кодексом», «какое приложение содержит…». Спрашивающий
номера не знает — он их ищет.

Зацепка — из содержания нормы, а не из её реквизитов. Запрещены вопросы про
историю редакции и нумерацию: «Каким законом внесены изменения», «(в ред. …)»,
«Какой номер пункта указывает на …», «Когда статья утратила силу». Спрашивают
о том, что нужно сделать и в какие сроки, а не о том, каким актом это введено.

Остальные требования:
- перефразируй, не переписывай фразы фрагмента дословно;
- пиши так, как реально спрашивают коллеги — не канцелярит;
- каждый вопрос самодостаточен (без «здесь», «в этом пункте»);
- разные вопросы о разных деталях, не три перифраза одного и того же.

Если во фрагменте нет ни одной такой зацепки (это титульный лист, преамбула со
ссылками или оглавление) — верни пустой список вопросов.

Фрагмент:
{chunk}
"""


# ─── Deterministic core (unit-tested, no heavy imports) ──────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_TOC_LEADER_RE = re.compile(r"\.{3,}\s*\d+\s*$")

# Реквизиты: ссылка на источник опубликования и на статью/пункт другого акта.
_CITATION_RE = re.compile(
    r"собрание законодательства|ст\.\s*\d+|стать[ияею]{1,2}\s+\d+"
    r"|[nN№]\s*\d+-фз|пункт(?:ом|ами)?\s+\d+",
    flags=re.IGNORECASE | re.UNICODE,
)
# Долженствование и порядок действий — признак того, что во фрагменте есть норма,
# а не только перечень оснований, по которым акт издан.
_OBLIGATION_RE = re.compile(
    r"обязан|должен|должна|должны|вправе|запрещ|не допускается|подлежит"
    r"|устанавлива|проводится|проводятся|осуществля|включает|применяется"
    r"|оформляется|выда[её]тся|направляется|не реже|не позднее",
    flags=re.IGNORECASE | re.UNICODE,
)

# Строка, набранная в основном заглавными, — заголовок акта; чанкер повторяет его
# в каждом чанке документа, поэтому в объём содержания он не идёт.
HEADING_UPPERCASE_RATIO = 0.5
# Доля знаков, занятых реквизитами. Замер по корпусу 02.09.2026: преамбулы 0.19–0.29,
# обычные статьи со ссылками на редакции 0.06–0.16.
MAX_CITATION_SHARE = 0.17


def normalize_question(q: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — key for exact dedup."""
    q = _PUNCT_RE.sub(" ", q.lower())
    return _WS_RE.sub(" ", q).strip()


def _uppercase_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _body_without_headings(text: str) -> str:
    """Text minus the all-caps heading lines the chunker repeats in every chunk."""
    return "\n".join(
        ln
        for ln in text.splitlines()
        if _uppercase_ratio(ln) <= HEADING_UPPERCASE_RATIO
    ).strip()


def is_junk_chunk(text: str, min_chars: int = MIN_CHUNK_CHARS) -> bool:
    """True for chunks not worth generating questions from.

    Four cases, all seen in the corpus: too short; a table of contents; a title
    page (nothing but the act's caps header); a preamble of citations
    ("… (Собрание законодательства …, ст. 3), пунктом 6 статьи 34 …") that lists
    the grounds for issuing the act and states no rule of its own.

    The last two matter because they are long, so the length filter kept them,
    and every question generated from them ("какие документы регулируют охрану
    здоровья граждан?") is answered by half the corpus while the GT marks a
    single chunk — measuring the labelling, not the retrieval.

    Both new rules are deliberately about proportion, not count: the heading is
    excluded from the content budget rather than triggering a caps check, and
    citations are measured as a share of the text, because "(в ред. Федеральных
    законов от … N 90-ФЗ, …)" sits at the end of perfectly ordinary articles.
    """
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        return True

    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines and lines[0].lower().rstrip(":.").strip() in {"содержание", "оглавление"}:
        return True

    leader_lines = sum(1 for ln in lines if _TOC_LEADER_RE.search(ln))
    if len(lines) >= 3 and leader_lines >= len(lines) - 1:
        return True

    if len(_body_without_headings(stripped)) < min_chars:
        return True

    citation_chars = sum(len(m) for m in _CITATION_RE.findall(stripped))
    if citation_chars / len(
        stripped
    ) > MAX_CITATION_SHARE and not _OBLIGATION_RE.search(stripped):
        return True

    return False


# Вопрос о реквизитах акта, а не о его содержании: каким актом введена норма,
# когда утратила силу, какой у пункта номер. Поиску по нормативке их не задают.
_META_QUESTION_RE = re.compile(
    r"как(?:ой|им|ая|ие)\s+(?:федеральн\w+\s+)?закон\w*\s+"
    r"(?:ввёл|ввел|введ\w*|внес\w*|изменил|утратил|отменил|дополнил)"
    r"|каким\s+(?:актом|приказом|постановлением)\s+"
    r"|утратил\w*\s+силу"
    r"|номер\s+(?:пункта|статьи|части|подпункта)"
    r"|нов\w+\s+редакц|введена\s+редакц|в\s+ред\.",
    flags=re.IGNORECASE | re.UNICODE,
)


def is_meta_question(question: str) -> bool:
    """True для вопроса о реквизитах нормы (каким законом введена, когда утратила
    силу, какой номер у пункта) вместо её содержания."""
    return bool(_META_QUESTION_RE.search(question or ""))


# Вопрос о том, КАКОЙ акт что-то регулирует. Ответ — название документа, а не
# норма; для retrieval-GT бесполезен: подходит к любому чанку того же акта.
# Смоук 02.09.2026 (--sample 60 --seed 11): 8 таких вопросов из 173.
_SOURCE_REF_RE = re.compile(
    r"как(?:ой|ая|ое|ие|ого|их|им)\s+"
    # до двух слов между местоимением и существительным: «какого ТИПА закона»
    r"(?:\w+\s+){0,2}?"
    r"(?:закон\w*|законодательств\w*|кодекс\w*|документ\w*|акт\w*"
    r"|постановлени\w*|приказ\w*)\b"
    r"[^?]{0,60}?"
    r"\b(?:регулир\w+|определ\w+|устанавлив\w+|регламентир\w+"
    r"|действует|предусматрив\w+|осуществля\w+)",
    flags=re.IGNORECASE | re.UNICODE,
)

# Ссылка на структуру источника: «согласно статье 213», «согласно графе 3»,
# «какое приложение содержит». Вопрос либо несёт ответ-указатель, либо спрашивает
# о нумерации. Поиском по нормативке так не спрашивают. Тот же смоук: 12 из 173.
_STRUCTURAL_REF_RE = re.compile(
    r"(?:согласно|в\s+соответствии\s+со?)\s+"
    r"(?:настоящ\w+\s+|данн\w+\s+|указанн\w+\s+|трудов\w+\s+)*"
    r"(?:стать\w*|пункт\w*|част[ьияей]\w*|граф\w*|приложени\w*|раздел\w*"
    r"|глав\w*|списк\w*|кодекс\w*|таблиц\w*)"
    r"|в\s+стать[ье]\w*\s*\d"
    r"|как(?:ой|ое|ая|ие)\s+(?:пункт|стать|приложени|раздел|глав)",
    flags=re.IGNORECASE | re.UNICODE,
)


def is_source_reference_question(question: str) -> bool:
    """True для вопроса «какой закон/документ это регулирует» — ответом служит
    название акта, а не его норма."""
    return bool(_SOURCE_REF_RE.search(question or ""))


def has_structural_reference(question: str) -> bool:
    """True, если вопрос ссылается на структуру источника (статья, пункт, графа,
    приложение, «настоящий Кодекс») вместо того, чтобы спрашивать о содержании."""
    return bool(_STRUCTURAL_REF_RE.search(question or ""))


def is_weak_question(question: str) -> bool:
    """Объединяет три детерминированных признака непригодного для GT вопроса."""
    return (
        is_meta_question(question)
        or is_source_reference_question(question)
        or has_structural_reference(question)
    )


def filter_questions(questions: Iterable[str]) -> list[str]:
    """Отбрасывает мета-вопросы, вопросы о том, каким актом норма введена, и
    вопросы со ссылкой на структуру источника. Порядок сохраняется."""
    return [q for q in questions if not is_weak_question(q)]


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


def _greedy_near_dup_pass(
    records: list[dict],
    vectors: list[Sequence[float]],
    near_dup_threshold: float,
    cross_chunk_threshold: float,
) -> list[dict]:
    """Жадный проход: запись выживает, если ни с одним уже выжившим её косинус не
    выше порога (порог зависит от того, из одного ли они чанка).

    Проход векторизован намеренно. Наивный двойной цикл на чистом Python — это
    ``n**2 / 2`` косинусов; на реальном GT (15 тыс. вопросов, 1536 измерений) он
    считался больше часа и на smoke-прогоне в 160 вопросов это не проявлялось.
    Здесь каждая итерация — одно матрично-векторное умножение по выжившим."""
    import numpy as np  # noqa: PLC0415 — тяжёлый импорт только на реальном прогоне

    mat = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1)
    unit = np.zeros_like(mat)
    nonzero = norms > 0.0
    unit[nonzero] = mat[nonzero] / norms[nonzero, None]

    # chunk_id сравниваем как целочисленные коды: строковое сравнение внутри цикла
    # вернуло бы ту самую квадратичность, ради ухода от которой всё и переписано.
    code_of: dict[str, int] = {}
    codes = np.fromiter(
        (code_of.setdefault(r["chunk_id"], len(code_of)) for r in records),
        dtype=np.int64,
        count=len(records),
    )

    survivors: list[dict] = []
    survivor_unit = np.empty_like(unit)
    survivor_codes = np.empty(len(records), dtype=np.int64)
    count = 0
    for i, rec in enumerate(records):
        if count:
            sims = survivor_unit[:count] @ unit[i]
            thresholds = np.where(
                survivor_codes[:count] == codes[i],
                near_dup_threshold,
                cross_chunk_threshold,
            )
            if bool(np.any(sims > thresholds)):
                continue
        survivors.append(rec)
        survivor_unit[count] = unit[i]
        survivor_codes[count] = codes[i]
        count += 1
    return survivors


def dedup_questions(
    records: list[dict],
    embed_fn: Callable[[list[str]], list[Sequence[float]]] | None = None,
    near_dup_threshold: float = NEAR_DUP_THRESHOLD,
    cross_chunk_threshold: float = CROSS_CHUNK_THRESHOLD,
) -> tuple[list[dict], int]:
    """Remove exact duplicate questions (by :func:`normalize_question`) and, when
    ``embed_fn`` is given, near-duplicates by cosine similarity.

    Two thresholds. Inside one chunk the three questions deliberately paraphrase
    the same text, so only a near-identical pair is a duplicate
    (``near_dup_threshold``). Across chunks a merely similar pair is worse than a
    duplicate: both chunks answer it, but the GT marks one, so Hit Rate drops for
    a retrieval that was right — hence the stricter ``cross_chunk_threshold``.

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
    survivors = _greedy_near_dup_pass(
        kept, vectors, near_dup_threshold, cross_chunk_threshold
    )
    removed += len(kept) - len(survivors)
    return survivors, removed


def select_chunks(
    chunks: list[dict],
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> list[dict]:
    """Pick the chunks to generate from.

    ``sample`` draws a random subset with a fixed ``seed`` — a smoke run on
    ``chunks[:N]`` lands entirely inside the first document and says nothing about
    the corpus. ``limit`` keeps the old head-slice for reproducing a specific run;
    ``sample`` wins when both are given.
    """
    if sample:
        rng = random.Random(seed)
        if sample >= len(chunks):
            return list(chunks)
        return rng.sample(chunks, sample)
    if limit:
        return chunks[:limit]
    return list(chunks)


def _passage_identity(passage: dict) -> str:
    """Byte-for-byte mirror of ``src.v7.nlp_core.passage_identity``.

    Copied rather than imported because ``nlp_core`` instantiates a pymorphy3
    analyzer at module import, which the LLM-free unit tests must not pull in.
    Divergence here would silently zero out Hit Rate — the GT ids would stop
    matching the ids the retrieval runners emit — so
    ``test_identity_matches_nlp_core`` pins the two implementations together.
    """
    cid = passage.get("chunk_id")
    if cid is not None and cid != "":
        meta = passage.get("metadata") or {}
        source = meta.get("source", "")
        return f"{source}#{cid}"
    meta = passage.get("metadata") or {}
    source = meta.get("source", "")
    page_no = meta.get("page_no", "")
    text = (passage.get("text", "") or "")[:80]
    return f"{source}|{page_no}|{text}"


def build_gt_record(question: str, passage: dict) -> dict:
    meta = passage.get("metadata") or {}
    return {
        "question": question.strip(),
        "chunk_id": _passage_identity(passage),
        "source": meta.get("source", ""),
        "chunk_preview": (passage.get("text", "") or "")[:PREVIEW_CHARS],
    }


# ─── LLM + corpus (lazy heavy imports) ──────────────────────────────────────


def calc_total_price(usages: Iterable[dict], model: str = GEN_MODEL) -> float:
    """Sum USD cost from a list of ``{"input": n, "output": n}`` token counts."""
    rate = price_for(model)
    total = 0.0
    for u in usages:
        total += u.get("input", 0) / 1_000_000 * rate["input"]
        total += u.get("output", 0) / 1_000_000 * rate["output"]
    return total


def estimate_cost(
    n_chunks: int, model: str = GEN_MODEL, avg_chunk_tokens: int = 350
) -> float:
    """Rough pre-flight estimate: prompt ≈ chunk + 150 tokens overhead,
    output ≈ 60 tokens per generation call."""
    per_call = calc_total_price(
        [{"input": avg_chunk_tokens + 150, "output": 60}], model=model
    )
    return per_call * n_chunks


def to_passage(doc: dict) -> dict:
    """Shape a backend document as a v7 passage — same lift of ``chunk_id`` from
    metadata to top level that ``src.v7.bridge._doc_to_passage`` does, so
    :func:`_passage_identity` sees what the retrieval paths see."""
    meta = doc.get("metadata") or {}
    passage = {"text": doc.get("text", ""), "metadata": meta}
    if "chunk_id" in meta:
        passage["chunk_id"] = meta["chunk_id"]
    return passage


def iter_corpus_chunks(backend=None) -> list[dict]:
    """Every stored chunk as a v7 passage dict.

    Goes through the backend's own paginated ``iter_all_documents()`` — the same
    traversal the BM25 corpus build uses — rather than a raw collection query.
    """
    if backend is None:
        from src.backends.vector_store import (  # noqa: PLC0415
            get_vector_store_backend,
        )

        backend = get_vector_store_backend(load_existing=True)
    return [to_passage(doc) for doc in backend.iter_all_documents()]


def _make_llm(model: str = GEN_MODEL):
    from src.infra import llm_factory  # noqa: PLC0415

    # Explicit model_name wins over JUDGE_MODEL_NAME in the factory.
    return llm_factory.get_judge_llm(model_name=model)


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
    parsed = filter_questions(
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


def raw_path_for(out_path: Path) -> Path:
    """Append-only companion log of ``out_path``.

    The dedup pass needs every record at once, so the final file can only be
    written at the end — and a run that dies before it (killed shell, OOM,
    exhausted daily quota) used to lose hours of paid generation. Each finished
    chunk is therefore appended here immediately; the final file is derived from
    it by :func:`finalize`.
    """
    return Path(str(out_path) + ".raw")


def load_raw(raw_path: Path) -> tuple[set[str], list[dict]]:
    """``(chunk ids already generated, their records)`` from the raw log.

    A run killed mid-write leaves a truncated last line: it is skipped, not
    fatal — losing three questions beats losing the file.
    """
    done: set[str] = set()
    records: list[dict] = []
    if not raw_path.exists():
        return done, records
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"  raw log: skipping corrupt line in {raw_path}")
                continue
            chunk_id = entry.get("chunk_id")
            if not chunk_id:
                continue
            done.add(chunk_id)
            records.extend(entry.get("records") or [])
    return done, records


def _append_raw(handle, chunk_id: str, records: list[dict]) -> None:
    """One durable line per chunk. fsync on purpose: the whole point is
    surviving a process that never gets to flush."""
    handle.write(
        json.dumps({"chunk_id": chunk_id, "records": records}, ensure_ascii=False)
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())


def finalize(
    raw_path: Path,
    out_path: Path = GT_PATH,
    embed_fn: Callable[[list[str]], list[Sequence[float]]] | None = None,
) -> dict:
    """Dedup the raw log into the final GT file. No generation calls — a run that
    died after generation is finished with ``--finalize``, not re-paid for. The
    near-dup pass still embeds the questions; ``_embedding_fn`` caches those on disk."""
    _, records = load_raw(raw_path)
    kept, removed = dedup_questions(records, embed_fn=embed_fn)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {
        "questions": len(kept),
        "removed_dups": removed,
        "raw_records": len(records),
    }


def run(
    limit: int | None = None,
    out_path: Path = GT_PATH,
    dry_run: bool = False,
    model: str = GEN_MODEL,
    sample: int | None = None,
    seed: int = 0,
    resume: bool = True,
    max_workers: int = MAX_WORKERS,
) -> dict:
    chunks = [c for c in iter_corpus_chunks() if not is_junk_chunk(c["text"])]
    chunks = select_chunks(chunks, limit=limit, sample=sample, seed=seed)

    raw_path = raw_path_for(out_path)
    done: set[str] = set()
    if resume:
        done, _ = load_raw(raw_path)
        if done:
            print(f"resume: {len(done)} chunks already in {raw_path.name}")
    pending = [c for c in chunks if _passage_identity(c) not in done]

    projected = estimate_cost(len(pending), model=model)
    print(
        f"chunks (after junk filter): {len(chunks)}  to generate: {len(pending)}  "
        f"model: {model}  projected cost: ${projected:.2f}"
    )
    if projected > COST_ABORT_USD:
        sys.exit(f"ABORT: projected ${projected:.2f} > ${COST_ABORT_USD:.2f} budget")
    if dry_run:
        return {"chunks": len(pending), "projected_usd": projected, "model": model}

    llm = _make_llm(model)
    usages: list[dict] = []
    failures = 0

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8") as raw_f:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(generate_questions_for_chunk, c, llm): c for c in pending
            }
            for fut in as_completed(futs):
                chunk = futs[fut]
                try:
                    questions, usage = fut.result()
                except (
                    Exception
                ) as e:  # noqa: BLE001 — one bad chunk must not kill the run
                    failures += 1
                    print(f"  chunk {_passage_identity(chunk)} failed: {e}")
                    continue
                if usage:
                    usages.append(usage)
                _append_raw(
                    raw_f,
                    _passage_identity(chunk),
                    [build_gt_record(q, chunk) for q in questions],
                )

    stats = finalize(
        raw_path,
        out_path=out_path,
        embed_fn=_embedding_fn(emb_cache_path_for(out_path)),
    )

    spent = calc_total_price(usages, model=model)
    print(
        f"wrote {stats['questions']} questions ({stats['removed_dups']} dups removed, "
        f"{failures} chunk failures) to {out_path}  spent this run: ${spent:.4f}"
    )
    return {
        "questions": stats["questions"],
        "removed_dups": stats["removed_dups"],
        "failures": failures,
        "spent_usd": spent,
        "model": model,
    }


def emb_cache_path_for(out_path: Path) -> Path:
    """Companion embedding cache of ``out_path``.

    Dedup embeds every surviving question. Without a cache each ``--finalize``
    re-embeds the same 15 thousand questions from scratch — paid again and slow
    for a step whose whole point is that it costs nothing to repeat.
    """
    return Path(str(out_path) + ".embcache.npz")


def _disk_cached(
    embed_fn: Callable[[list[str]], list[Sequence[float]]], cache_path: Path
) -> Callable[[list[str]], list[Sequence[float]]]:
    """Wrap ``embed_fn`` in a disk cache keyed by the exact question text."""
    import numpy as np  # noqa: PLC0415

    def wrapped(texts):
        texts = list(texts)
        cache: dict[str, Sequence[float]] = {}
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    cache = dict(zip(data["keys"].tolist(), data["vecs"]))
            except Exception as e:  # noqa: BLE001 — битый кеш не должен ронять прогон
                print(f"  embedding cache ignored ({cache_path.name}): {e}")

        missing = [t for t in dict.fromkeys(texts) if t not in cache]
        if missing:
            print(f"  embedding {len(missing)} questions ({len(cache)} cached)")
            cache.update(zip(missing, embed_fn(missing)))
            keys = list(cache)
            np.savez(
                cache_path,
                keys=np.array(keys, dtype=np.str_),
                vecs=np.asarray([cache[k] for k in keys], dtype=np.float32),
            )
        return [cache[t] for t in texts]

    return wrapped


def _embedding_fn(cache_path: Path | None = None):
    try:
        from src.infra.llm_factory import get_embedding_model  # noqa: PLC0415

        model = get_embedding_model()
    except Exception as e:  # noqa: BLE001
        print(f"  near-dup dedup skipped (no embedding model): {e}")
        return None

    def embed(texts):
        return model.embed_documents(list(texts))

    return _disk_cached(embed, cache_path) if cache_path else embed


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--limit", type=int, default=None, help="first N chunks (reproduce a run)"
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="random N chunks (smoke test); wins over --limit",
    )
    p.add_argument("--seed", type=int, default=0, help="seed for --sample")
    p.add_argument("--out", type=Path, default=GT_PATH)
    p.add_argument(
        "--dry-run", action="store_true", help="estimate cost, write nothing"
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore the .raw log and regenerate every chunk (pays again)",
    )
    p.add_argument(
        "--finalize",
        action="store_true",
        help="dedup an existing .raw log into --out and exit; no generation calls",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"threads (default: {MAX_WORKERS})",
    )
    p.add_argument(
        "--model",
        default=GEN_MODEL,
        help=f"generation model (default: {GEN_MODEL}); must be priced in PRICE_PER_1M",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.finalize:
        raw = raw_path_for(args.out)
        stats = finalize(
            raw,
            out_path=args.out,
            embed_fn=_embedding_fn(emb_cache_path_for(args.out)),
        )
        print(
            f"finalize: {stats['raw_records']} raw records -> {stats['questions']} "
            f"questions ({stats['removed_dups']} dups removed) in {args.out}"
        )
        sys.exit(0)
    run(
        limit=args.limit,
        out_path=args.out,
        dry_run=args.dry_run,
        model=args.model,
        sample=args.sample,
        seed=args.seed,
        resume=not args.no_resume,
        max_workers=args.workers,
    )
