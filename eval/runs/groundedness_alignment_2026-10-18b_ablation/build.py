"""Build groundedness batch 2: saved on3 answers with their support removed from context.

Usage: python build.py <path to on3.json>

For each case, every context fragment containing a needle of the answer's key claim
is removed; remaining fragments are renumbered 1..N. Answers are unchanged, so their
[Фрагмент N] citations may point at unrelated text. No model calls.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
HEAD = re.compile(r"(?m)^\[(\d+)\] \((HIGH|MED|LOW)\) ")

# case_id -> needles of the key claim (lowercase, ё->е, collapsed whitespace)
NEEDLES = {
    "G008": ["50 процент"],
    "G009": ["16 час", "шестнадцат"],
    "G010": ["3 года", "три года", "трех лет", "3 лет"],
    "G013": ["микропредприят"],
    "G015": ["60 календарн", "шестидесят"],
    "G017": ["система сохранения жизни", "охрана труда -", "охрана труда —"],
    "G022": ["организации и финансированию", "финансирован"],
}


def norm(text):
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е"))


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def split(context):
    starts = [m.start() for m in HEAD.finditer(context)]
    assert starts and starts[0] == 0, "context must start with a fragment header"
    parts = [context[a:b] for a, b in zip(starts, starts[1:] + [len(context)])]
    assert "".join(parts) == context
    return parts


def ablate(context, needles):
    parts = split(context)
    removed = [i + 1 for i, p in enumerate(parts) if any(x in norm(p) for x in needles)]
    kept = [(i + 1, p) for i, p in enumerate(parts) if i + 1 not in removed]
    renumber = {}
    out = []
    for new, (old, part) in enumerate(kept, 1):
        renumber[old] = new
        out.append(HEAD.sub(lambda m: f"[{new}] ({m.group(2)}) ", part, count=1))
    result = "".join(out)
    assert not any(x in norm(result) for x in needles)
    return result, removed, renumber, len(parts)


def main(source):
    raw = Path(source).read_bytes()
    rows = {r["case_id"]: r for r in json.loads(raw)["rows"]}
    batch, labels = [], []
    for n, (case_id, needles) in enumerate(NEEDLES.items(), 1):
        row = rows[case_id]
        review_id = f"GB{n:02d}"
        question = row["inputs"]["question"]
        answer = row["outputs"]["answer"]
        original = row["outputs"]["context"]
        context, removed, renumber, total = ablate(original, needles)
        batch.append(
            {
                "review_id": review_id,
                "case_id": case_id,
                "source_run_id": row["run_id"],
                "source_example_id": row["example_id"],
                "case_type": "regular",
                "inputs": {"question": question},
                "outputs": {"answer": answer, "context": context},
                "answer_sha256": sha(answer),
                "context_sha256": sha(context),
                "perturbation": {
                    "type": "context_ablation",
                    "needles": needles,
                    "original_context_sha256": sha(original),
                    "original_fragments": total,
                    "removed_fragments": removed,
                    "renumber_old_to_new": renumber,
                    "expected_groundedness": 0,
                },
            }
        )
        labels.append(
            {
                "review_id": review_id,
                "case_id": case_id,
                "source_run_id": row["run_id"],
                "human_groundedness": None,
                "reviewer": None,
                "reviewed_at": None,
                "checks": None,
                "note": "",
            }
        )
        card = (
            f"{review_id} / {case_id}\n\n"
            f"ИЗМЕНЕНИЕ КОНТЕКСТА\n"
            f"Удалены фрагменты {removed} из {total} (содержали: {', '.join(needles)}); "
            f"остальные перенумерованы. Ссылки [Фрагмент N] в ответе — по старой нумерации.\n\n"
            f"ВОПРОС\n{question}\n\n"
            f"ОТВЕТ (полный)\n{answer}\n\n"
            f"КОНТЕКСТ ГЕНЕРАТОРА ПОСЛЕ УДАЛЕНИЯ (полный)\n{context}\n"
        )
        (OUT / f"{review_id}-{case_id}.txt").write_text(card)
    dump = lambda items: "".join(
        json.dumps(x, ensure_ascii=False) + "\n" for x in items
    )
    (OUT / "batch.jsonl").write_text(dump(batch))
    (OUT / "human_labels.jsonl").write_text(dump(labels))
    print("source sha256", hashlib.sha256(raw).hexdigest())
    print("batch sha256", sha(dump(batch)))
    for b in batch:
        p = b["perturbation"]
        print(
            b["review_id"],
            b["case_id"],
            "removed",
            p["removed_fragments"],
            "of",
            p["original_fragments"],
        )


if __name__ == "__main__":
    main(sys.argv[1])
