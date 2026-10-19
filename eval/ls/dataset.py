"""Build the LangSmith golden dataset from tests/dataset.csv + retrieval labels and upsert it.

CLI:
    python -m eval.ls.dataset --dry-run   # build and print counts, no network
    python -m eval.ls.dataset             # build and upsert to LangSmith
    python -m eval.ls.dataset --prune     # same, and delete examples no longer in the CSV
    python -m eval.ls.dataset --csv eval/runs/align_batch2_2026-10-16/align2.csv \
        --status assistant_checked --split align2   # a separate split, verified untouched
"""

# ANCHOR: golden set → LangSmith dataset. Input: tests/dataset.csv (verified rows) and
# retrieval labels of the active snapshot; output: examples with stable uuid5 ids, upserted
# into DATASET_NAME (create new / update changed / report or prune stale CSV examples).

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import uuid
from pathlib import Path

DATASET_NAME = "regrag-golden"
SPLIT_VERIFIED = "verified"
DEFAULT_CSV = Path("tests/dataset.csv")
# Labels remapped to chunk ids of the active snapshot; eval/data/ still holds old-db ids.
DEFAULT_LABELS = Path(
    "eval/runs/gt_label_j7ceu4r4_2026-10-18/golden_retrieval_labeled.jsonl"
)


def build_examples(
    csv_path: Path,
    labels_path: Path,
    status: str = "verified",
    split: str = SPLIT_VERIFIED,
) -> list[dict]:
    """CSV rows with reference_status == status → LangSmith examples in `split`.

    Example shape:
        {"id": str (uuid5 of case_id, stable between runs),
         "inputs": {"question"},
         "outputs": {"ground_truth", "ground_truth_extra", "relevant_chunk_ids",
                     "forbidden_claims", "must_not_contain", "oos_type", "legal_as_of"},
         "metadata": {"case_id", "oos_type", "corpus_support", "csv_sha256", "split"},
         "split": split}
    In-scope row without labels → ValueError naming the case_id. OOS → no chunk ids.
    """
    # Hash the exact bytes: same hash ⇒ experiments ran on the same golden set.
    raw = Path(csv_path).read_bytes()
    csv_sha256 = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    verified_rows = [row for row in rows if row["reference_status"] == status]

    # question → relevant chunk ids. Joined by question text: the labels file has no
    # case_id. Older label rows carry a single "chunk_id" instead of the list.
    labels = {}
    for line in Path(labels_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        labels[item["question"].strip()] = item.get("relevant_chunk_ids") or [
            item["chunk_id"]
        ]

    examples = []
    for row in verified_rows:
        question = row["question"].strip()
        oos_type = row["oos_type"]

        if oos_type:
            # Out of scope: the right answer is a refusal, nothing to retrieve.
            chunk_ids = []
        else:
            # Fail loudly: [] would look like OOS and drop out of retrieval_hit.
            if question not in labels:
                raise ValueError(
                    f"{row['case_id']}: no retrieval labels for question "
                    f"{question!r} in {labels_path}"
                )
            chunk_ids = labels[question]

        examples.append(
            {
                # uuid5 of case_id: a rerun updates the same example, no duplicates.
                "id": str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{DATASET_NAME}/{row['case_id']}")
                ),
                "inputs": {"question": question},
                "outputs": {
                    "ground_truth": row["ground_truth"],
                    # Supplementary facts: their absence is not an error (16.10).
                    "ground_truth_extra": row.get("ground_truth_extra") or "",
                    "relevant_chunk_ids": chunk_ids,
                    "forbidden_claims": row["forbidden_claims"],
                    "must_not_contain": row["must_not_contain"],
                    "oos_type": oos_type,
                    "legal_as_of": row["legal_as_of"],
                },
                "metadata": {
                    "case_id": row["case_id"],
                    "oos_type": oos_type,
                    "corpus_support": row["corpus_support"],
                    "csv_sha256": csv_sha256,
                    # Copy of the split: stale/prune only look within their own split.
                    "split": split,
                },
                "split": split,
            }
        )
    return examples


def upsert_examples(
    client,
    examples: list[dict],
    dataset_name: str = DATASET_NAME,
    prune: bool = False,
    split: str = SPLIT_VERIFIED,
) -> dict[str, int]:
    """Create the dataset if missing; create new ids, update changed ones, skip the rest.

    "stale" counts examples built from the CSV earlier (they carry csv_sha256) that are no
    longer in `examples`, e.g. a row lost its verified status. With prune=True they are
    deleted (soft delete, earlier dataset versions keep them). Only examples of `split` are
    considered; ones uploaded before metadata carried "split" count as verified. Examples
    added by other means (no csv_sha256, e.g. from traces) are never touched.
    """
    if prune and not examples:
        raise ValueError(
            "refusing to prune with no examples: it would empty the dataset"
        )
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(dataset_name)

    existing = {
        str(ex.id): ex for ex in client.list_examples(dataset_name=dataset_name)
    }

    to_create, to_update, unchanged = [], [], 0
    for e in examples:
        old = existing.get(e["id"])
        if old is None:
            to_create.append(e)
        elif _changed(old, e):
            to_update.append(e)
        else:
            unchanged += 1

    # Copies, so the client never holds references to the caller's dicts.
    if to_create:
        client.create_examples(
            dataset_name=dataset_name, examples=copy.deepcopy(to_create)
        )
    if to_update:
        client.update_examples(
            dataset_name=dataset_name, updates=copy.deepcopy(to_update)
        )
    # Stale = ours (has csv_sha256), same split, no longer built; foreign ones are kept.
    new_ids = {e["id"] for e in examples}
    stale = [
        ex_id
        for ex_id, ex in existing.items()
        if ex_id not in new_ids
        and "csv_sha256" in (ex.metadata or {})
        and ex.metadata.get("split", SPLIT_VERIFIED) == split
    ]

    result = {
        "created": len(to_create),
        "updated": len(to_update),
        "unchanged": unchanged,
        "stale": len(stale),
    }
    if prune:
        if stale:
            client.delete_examples(example_ids=stale)
        result["deleted"] = len(stale)
    return result


def _changed(old, new: dict) -> bool:
    # LangSmith adds its own metadata keys (e.g. dataset_split): compare only ours.
    old_meta = {k: (old.metadata or {}).get(k) for k in new["metadata"]}
    return (
        old.inputs != new["inputs"]
        or old.outputs != new["outputs"]
        or old_meta != new["metadata"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload the verified golden set to LangSmith"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--status", default="verified", help="reference_status of rows to upload"
    )
    parser.add_argument("--split", default=SPLIT_VERIFIED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="delete examples of --split that are no longer in the CSV",
    )
    args = parser.parse_args()

    examples = build_examples(args.csv, args.labels, args.status, args.split)
    oos = sum(1 for e in examples if e["outputs"]["oos_type"])
    print(f"{len(examples)} examples: {len(examples) - oos} in-scope, {oos} OOS")
    print(f"split: {args.split}, status: {args.status}, labels: {args.labels}")
    if args.dry_run:
        return

    from dotenv import load_dotenv
    from langsmith import Client

    load_dotenv()
    print(upsert_examples(Client(), examples, prune=args.prune, split=args.split))


if __name__ == "__main__":
    main()
