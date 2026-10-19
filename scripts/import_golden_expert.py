"""Append expert-answer candidates from eval/data/golden_expert/ to tests/dataset.csv.

ANCHOR: offline import, no LLM calls.
- In: batch1_ot / batch2_labor / batch3_fire (`;`-CSV), the corpus in --source-docs.
- Out: rows G057+ with reference_status=expert_candidate (run_v7_eval skips them,
  eval/ls/dataset.py uploads them only with --status expert_candidate), plus
  norm_check.md next to the batches: per-case act/point lookup and the risk list.
- corpus_support: supported (every cited act is in the corpus and every cited
  point/article is found in its text), partial (some of it), known_missing (none of
  the cited acts is in the corpus).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = REPO_ROOT / "eval" / "data" / "golden_expert"
BATCHES = ("batch1_ot", "batch2_labor", "batch3_fire")
EXCLUDED = {"OT-050"}  # 135н lost force; dropped by decision 2026-10-18
FIRST_CASE = 57

# Act pattern in norm_ref -> corpus file. КоАП is in the corpus as one article only.
ACT_FILES: list[tuple[str, str]] = [
    (r"№ 2464\b", "2464.md"),
    (r"№ 766н", "766н.md"),
    (r"№ 772н", "772н.md"),
    (r"№ 342н", "342н.md"),
    (r"№ 1206\b", "1206.md"),
    (r"№ 236\b", "236.md"),
    (r"№ 29н", "29н.pdf"),
    (r"426-ФЗ", "426-ФЗ.pdf"),
    (r"ТК РФ", "тк рф.pdf"),
    (r"№ 1479\b", "1479 ппр.pdf"),
    (r"69-ФЗ", "О пожарной безопасности.pdf"),
    (r"125-ФЗ", "125-ФЗ.md"),
    (r"№ 696н", "696н.md"),
    (r"КоАП РФ, ст\. 5\.27\.1\b", "Статья 5.27.1..pdf"),
]

POINT_RE = re.compile(r"\b(ст|пп?|подп)\.\s*([\d.]+(?:\s*[–-]\s*\d+)?)")


def act_text(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        if path.suffix == ".pdf":
            with pymupdf.open(path) as doc:
                cache[path] = "\n".join(page.get_text() for page in doc)
        else:
            cache[path] = path.read_text(encoding="utf-8")
    return cache[path]


def point_found(kind: str, number: str, text: str) -> bool:
    """Article as a heading, point as a numbered line — a lookup, not a parse."""
    number = re.split(r"\s*[–-]\s*", number)[0].rstrip(".")
    if kind == "ст":
        return re.search(rf"Стать[яи]\s+{re.escape(number)}(?!\.?\d)", text) is not None
    # Markdown acts escape the dot after a list number: "62\. ...".
    return re.search(rf"(?m)^\W*{re.escape(number)}\\?\.(?!\d)", text) is not None


def check_norms(
    norm_ref: str, source_docs: Path, cache: dict[Path, str]
) -> tuple[str, list[str]]:
    acts_total = acts_present = points_total = points_found = 0
    notes: list[str] = []
    for segment in norm_ref.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        acts_total += 1
        match = next(
            (name for pattern, name in ACT_FILES if re.search(pattern, segment)), None
        )
        if match is None or not (source_docs / match).exists():
            notes.append(f"нет в корпусе: {segment}")
            continue
        acts_present += 1
        text = act_text(source_docs / match, cache)
        # Points follow the act name; the act's own number is not a point.
        points = POINT_RE.findall(segment.split(",", 1)[1] if "," in segment else "")
        for kind, number in points:
            points_total += 1
            if point_found(kind, number, text):
                points_found += 1
            else:
                notes.append(f"не найдено в {match}: {kind}. {number}")
    if acts_present == 0:
        return "known_missing", notes
    if acts_present == acts_total and points_found == points_total:
        return "supported", notes
    return "partial", notes


def is_risky(risk: str) -> bool:
    return risk.strip().lower().startswith(("устарел", "расходятся"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "tests" / "dataset.csv"
    )
    parser.add_argument("--source-docs", type=Path, default=REPO_ROOT / "source_docs")
    args = parser.parse_args()

    with args.dataset.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        existing = list(reader)
    if any(r["reference_status"] == "expert_candidate" for r in existing):
        raise SystemExit("expert candidates already imported")
    if max(int(r["case_id"][1:]) for r in existing) >= FIRST_CASE:
        raise SystemExit(f"case ids from G{FIRST_CASE:03d} are taken")

    cache: dict[Path, str] = {}
    new_rows: list[dict[str, str]] = []
    report: list[str] = []
    risky: list[str] = []
    counts = {"supported": 0, "partial": 0, "known_missing": 0}
    number = FIRST_CASE
    for batch in BATCHES:
        with (BATCH_DIR / f"{batch}.csv").open(encoding="utf-8", newline="") as fh:
            for src in csv.DictReader(fh, delimiter=";"):
                if src["id"] in EXCLUDED:
                    continue
                case_id = f"G{number:03d}"
                number += 1
                support, notes = check_norms(src["norm_ref"], args.source_docs, cache)
                counts[support] += 1
                note = (
                    f"Кандидат {src['id']} ({batch}), тип {src['type']}; "
                    f"{src['answered_by']}: {src['url']}. Нормы: {src['norm_ref']}."
                )
                if src["risk"].strip():
                    note += f" Риск: {src['risk'].strip()}"
                new_rows.append(
                    {
                        **{name: "" for name in fieldnames},
                        "case_id": case_id,
                        "question": src["question"].strip(),
                        "ground_truth": src["expert_answer"].strip(),
                        "oos_type": (
                            "false_premise" if src["type"] == "false_premise" else ""
                        ),
                        "reference_status": "expert_candidate",
                        "review_note": note,
                        "corpus_support": support,
                        "legal_as_of": src["answer_date"].strip() or "unknown",
                    }
                )
                report.append(
                    f"| {case_id} | {src['id']} | {support} | "
                    + ("; ".join(notes) or "—").replace("|", "/")
                    + " |"
                )
                if is_risky(src["risk"]):
                    risk = src["risk"].strip().replace("|", "/")
                    risky.append(f"| {case_id} | {src['id']} | {risk} |")

    with args.dataset.open("a", encoding="utf-8", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fieldnames).writerows(new_rows)

    summary = ", ".join(f"{k} {v}" for k, v in counts.items())
    (BATCH_DIR / "norm_check.md").write_text(
        "# Локальная сверка норм golden_expert\n\n"
        f"{len(new_rows)} кандидатов → `tests/dataset.csv` "
        f"G{FIRST_CASE:03d}–G{number - 1:03d}, `reference_status=expert_candidate`. "
        f"corpus_support: {summary}. Поиск пункта эвристический (строка «N.» или "
        "«Статья N»): «найдено» значит, что номер есть в тексте акта, а не что "
        "пункт подтверждает ответ.\n\n"
        f"## Риск: устарел / расходятся ({len(risky)})\n\n"
        "| case_id | Кандидат | Риск |\n|---|---|---|\n" + "\n".join(risky) + "\n\n"
        "## Сверка по кейсам\n\n"
        "| case_id | Кандидат | corpus_support | Замечания |\n|---|---|---|---|\n"
        + "\n".join(report)
        + "\n",
        encoding="utf-8",
    )
    print(f"appended {len(new_rows)} rows; {summary}; risky {len(risky)}")


if __name__ == "__main__":
    main()
