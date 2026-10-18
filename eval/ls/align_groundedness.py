"""Groundedness judge vs Petr's labels on frozen local alignment batches.

CLI:
    python -m eval.ls.align_groundedness <batch_dir>...
    python -m eval.ls.align_groundedness <batch_dir>... --paid --model <id> --out v.jsonl
    python -m eval.ls.align_groundedness <batch_dir>... --only GR02 GB06 [--paid ...]

Without --paid only loads and checks the batches (free). With --paid calls the
groundedness judge once per row, writes verdicts to --out and prints agreement
with human_groundedness via eval.ls.align. Each verdict keeps the raw judge
reply ("raw") so a contract failure can be replayed offline. --model must equal
the configured JUDGE_MODEL_NAME: a worktree without .env would silently fall
back to defaults.
"""

# ANCHOR: batch.jsonl + human_labels.jsonl → groundedness judge → (human, judge) agreement.

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval.ls.align import TARGET_AGREEMENT, agreement, split_rows


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_rows(batch_dirs: list[Path]) -> list[dict]:
    """Batch rows joined with human labels; refuses edited answers/contexts."""
    rows = []
    for batch_dir in map(Path, batch_dirs):
        labels = {
            lab["review_id"]: lab for lab in _jsonl(batch_dir / "human_labels.jsonl")
        }
        for row in _jsonl(batch_dir / "batch.jsonl"):
            rid = row["review_id"]
            outputs = row["outputs"]
            if (
                _sha(outputs["answer"]) != row["answer_sha256"]
                or _sha(outputs["context"]) != row["context_sha256"]
            ):
                raise ValueError(f"{rid}: answer/context differ from frozen sha256")
            if rid not in labels:
                raise ValueError(f"{rid}: no human label")
            case_type = row["case_type"]
            rows.append(
                {
                    "review_id": rid,
                    "case_id": row["case_id"],
                    "inputs": row["inputs"],
                    "outputs": outputs,
                    "reference": {
                        "oos_type": "" if case_type == "regular" else case_type
                    },
                    "human": labels[rid]["human_groundedness"],
                }
            )
    return rows


def select_rows(rows: list[dict], review_ids: list[str] | None) -> list[dict]:
    """Rows with the given review_ids (all when None); unknown ids are an error."""
    if review_ids is None:
        return rows
    unknown = set(review_ids) - {r["review_id"] for r in rows}
    if unknown:
        raise ValueError(f"unknown review_id: {', '.join(sorted(unknown))}")
    return [r for r in rows if r["review_id"] in review_ids]


def judge_rows(rows: list[dict], evaluator, on_result=None) -> list[dict]:
    """on_result receives each verdict as soon as it is paid for."""
    results = []
    for row in rows:
        verdict = evaluator(row["inputs"], row["outputs"], row["reference"])
        result = {
            "review_id": row["review_id"],
            "case_id": row["case_id"],
            "human": row["human"],
            "judge": verdict["score"],
            "comment": verdict["comment"],
        } | ({"raw": verdict["raw"]} if "raw" in verdict else {})
        if on_result is not None:
            on_result(result)
        results.append(result)
    return results


def acceptance(results: list[dict]) -> dict:
    """Accepted only when every row is scored and agreement meets the target."""
    pairs, skipped, disagreements = split_rows(
        [(r["review_id"], r["human"], r["judge"]) for r in results]
    )
    a = agreement(pairs) if pairs else None
    return {
        "accepted": skipped == 0
        and a is not None
        and a["agreement"] >= TARGET_AGREEMENT,
        "skipped": skipped,
        "agreement": a,
        "disagreements": disagreements,
    }


def _evaluator(expected_model: str):
    from config.settings import settings
    from eval.ls.groundedness import MAX_TOKENS, make_groundedness_evaluator
    from src.infra.llm_factory import get_judge_llm

    if settings.JUDGE_MODEL_NAME != expected_model:
        raise SystemExit(
            f"JUDGE_MODEL_NAME is {settings.JUDGE_MODEL_NAME!r}, expected "
            f"{expected_model!r}: run where .env is loaded"
        )
    print(f"judge: {settings.JUDGE_LLM_PROVIDER} {settings.JUDGE_MODEL_NAME}")
    llm = get_judge_llm(max_tokens=MAX_TOKENS, temperature=0.0)
    # Bounded transport as in eval.ls.run: no paid retries.
    llm.root_client = llm.root_client.with_options(max_retries=0, timeout=300)
    llm.client = llm.root_client.chat.completions
    return make_groundedness_evaluator(llm, keep_raw=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Groundedness judge vs human labels")
    parser.add_argument("batch_dirs", nargs="+", type=Path)
    parser.add_argument("--paid", action="store_true", help="call the judge model")
    parser.add_argument("--model", help="expected JUDGE_MODEL_NAME (with --paid)")
    parser.add_argument("--out", type=Path, help="judge verdicts JSONL (with --paid)")
    parser.add_argument("--only", nargs="+", help="judge only these review_ids")
    args = parser.parse_args()

    rows = select_rows(load_rows(args.batch_dirs), args.only)
    chars = sum(
        len(r["inputs"]["question"])
        + len(r["outputs"]["answer"])
        + len(r["outputs"]["context"])
        for r in rows
    )
    print(f"rows: {len(rows)}  input chars (without prompt): {chars}")
    if not args.paid:
        print("dry run: pass --paid --model <id> --out <file> to call the judge")
        return
    if args.out is None or args.model is None:
        parser.error("--paid requires --model and --out")

    from dotenv import load_dotenv

    load_dotenv()
    # Open before the first paid call and flush per verdict: a failure mid-batch
    # keeps what was already paid for.
    with args.out.open("w", encoding="utf-8") as out:

        def save(result: dict) -> None:
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

        results = judge_rows(rows, _evaluator(args.model), on_result=save)
    verdict = acceptance(results)
    a = verdict["agreement"]
    print(f"rows: {len(results)}  unscored (judge abstained): {verdict['skipped']}")
    if a:
        print(
            f"agreement on scored: {a['agreement']:.2f} ({a['tp'] + a['tn']}/{a['n']})"
            f"  tp={a['tp']} tn={a['tn']} fp={a['fp']} fn={a['fn']}"
        )
    print("calibration: " + ("OK" if verdict["accepted"] else "NOT ACCEPTED"))
    disagreements = verdict["disagreements"]
    print(f"fp (judge missed unsupported): {', '.join(disagreements['fp']) or '-'}")
    print(f"fn (judge rejected grounded): {', '.join(disagreements['fn']) or '-'}")


if __name__ == "__main__":
    main()
