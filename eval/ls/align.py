"""Judge alignment: how often does an LLM judge agree with Petr's manual labels?

CLI:
    python -m eval.ls.align <experiment> [--human-key human_correct] [--judge-key correctness]
    python -m eval.ls.align <experiment> --human-key human_citations --judge-key citations

Reads feedback of the experiment's root runs from LangSmith (free, no LLM calls).
Rows without a human label or with a judge abstention (None: oos, needs_review,
judge_format_error) are skipped and counted. Target: agreement ≥ 0.85 (plan, task 5).
"""

from __future__ import annotations

import argparse

TARGET_AGREEMENT = 0.85

# (case_id, human score, judge score); scores are 1/0/True/False or None.
Row = tuple[str, object, object]


def agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """Confusion counts for (human, judge) pairs.

    fp = judge True while human False: the judge missed an error (the worst case).
    """
    if not pairs:
        raise ValueError("no labelled pairs")
    tp = sum(1 for h, j in pairs if h and j)
    tn = sum(1 for h, j in pairs if not h and not j)
    fp = sum(1 for h, j in pairs if not h and j)
    fn = sum(1 for h, j in pairs if h and not j)
    return {
        "n": len(pairs),
        "agreement": (tp + tn) / len(pairs),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def split_rows(rows: list[Row]) -> tuple[list[tuple[bool, bool]], int, dict]:
    """Pairs for agreement(), the number of skipped rows, case_ids of fp/fn."""
    pairs: list[tuple[bool, bool]] = []
    skipped = 0
    disagreements: dict[str, list[str]] = {"fp": [], "fn": []}
    for case_id, human, judge in rows:
        if human is None or judge is None:
            skipped += 1
            continue
        h, j = bool(human), bool(judge)
        pairs.append((h, j))
        if j and not h:
            disagreements["fp"].append(case_id)
        elif h and not j:
            disagreements["fn"].append(case_id)
    return pairs, skipped, disagreements


def _created(fb) -> float:
    return fb.created_at.timestamp() if fb.created_at else 0.0


def fetch_rows(experiment: str, human_key: str, judge_key: str) -> list[Row]:
    """One row per root run of the experiment; after --rejudge the latest feedback wins."""
    from langsmith import Client

    client = Client()
    runs = list(client.list_runs(project_name=experiment, is_root=True))
    latest: dict = {}
    for fb in client.list_feedback(
        run_ids=[r.id for r in runs], feedback_key=[human_key, judge_key]
    ):
        slot = (fb.run_id, fb.key)
        if slot not in latest or _created(fb) > _created(latest[slot]):
            latest[slot] = fb

    example_ids = [r.reference_example_id for r in runs if r.reference_example_id]
    case_ids = {
        e.id: (e.metadata or {}).get("case_id")
        for e in client.list_examples(example_ids=example_ids)
    }

    def score(run_id, key):
        fb = latest.get((run_id, key))
        return None if fb is None else fb.score

    rows = [
        (
            case_ids.get(r.reference_example_id) or str(r.id)[:8],
            score(r.id, human_key),
            score(r.id, judge_key),
        )
        for r in runs
    ]
    return sorted(rows, key=lambda row: row[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge vs human agreement")
    parser.add_argument("experiment")
    parser.add_argument("--human-key", default="human_correct")
    parser.add_argument("--judge-key", default="correctness")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()
    rows = fetch_rows(args.experiment, args.human_key, args.judge_key)
    pairs, skipped, disagreements = split_rows(rows)
    print(f"experiment: {args.experiment}  {args.human_key} vs {args.judge_key}")
    print(f"runs: {len(rows)}  labelled pairs: {len(pairs)}  skipped: {skipped}")
    if not pairs:
        print("no rows with both labels")
        return
    a = agreement(pairs)
    verdict = "OK" if a["agreement"] >= TARGET_AGREEMENT else "BELOW TARGET"
    print(
        f"agreement: {a['agreement']:.2f} ({a['tp'] + a['tn']}/{a['n']}) {verdict}"
        f"  tp={a['tp']} tn={a['tn']} fp={a['fp']} fn={a['fn']}"
    )
    print(f"fp (judge missed an error): {', '.join(disagreements['fp']) or '-'}")
    print(
        f"fn (judge rejected a correct answer): {', '.join(disagreements['fn']) or '-'}"
    )


if __name__ == "__main__":
    main()
