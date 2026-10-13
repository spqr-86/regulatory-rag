"""Build the LangSmith golden dataset from tests/dataset.csv + retrieval labels and upsert it.

CLI:
    python -m eval.ls.dataset --dry-run   # build and print counts, no network
    python -m eval.ls.dataset             # build and upsert to LangSmith
"""

from __future__ import annotations

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
    "eval/runs/gt_remap_build-tjid2k6g_2026-10-12/golden_retrieval_labeled.jsonl"
)


def build_examples(csv_path: Path, labels_path: Path) -> list[dict]:
    """Verified CSV rows → LangSmith examples.

    Example shape:
        {"id": str (uuid5 of case_id, stable between runs),
         "inputs": {"question"},
         "outputs": {"ground_truth", "relevant_chunk_ids", "forbidden_claims",
                     "must_not_contain", "oos_type"},
         "metadata": {"case_id", "oos_type", "corpus_support", "csv_sha256"},
         "split": "verified"}
    In-scope row without labels → ValueError naming the case_id. OOS → no chunk ids.
    """
    raw = Path(csv_path).read_bytes()
    csv_sha256 = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"), newline="")))
    verified_rows = [row for row in rows if row["reference_status"] == "verified"]

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
            chunk_ids = []
        else:
            if question not in labels:
                raise ValueError(
                    f"{row['case_id']}: нет разметки для вопроса {question!r}"
                )
            chunk_ids = labels[question]

        examples.append(
            {
                "id": str(
                    uuid.uuid5(uuid.NAMESPACE_URL, f"{DATASET_NAME}/{row['case_id']}")
                ),
                "inputs": {"question": question},
                "outputs": {
                    "ground_truth": row["ground_truth"],
                    "relevant_chunk_ids": chunk_ids,
                    "forbidden_claims": row["forbidden_claims"],
                    "must_not_contain": row["must_not_contain"],
                    "oos_type": oos_type,
                },
                "metadata": {
                    "case_id": row["case_id"],
                    "oos_type": oos_type,
                    "corpus_support": row["corpus_support"],
                    "csv_sha256": csv_sha256,
                },
                "split": SPLIT_VERIFIED,
            }
        )
    return examples


def upsert_examples(
    client, examples: list[dict], dataset_name: str = DATASET_NAME
) -> dict[str, int]:
    """Create the dataset if missing; create new ids, update changed ones, skip the rest."""
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
    return {
        "created": len(to_create),
        "updated": len(to_update),
        "unchanged": unchanged,
    }


def _changed(old, new: dict) -> bool:
    # LangSmith adds its own metadata keys (e.g. dataset_split): compare only ours.
    old_meta = {k: (old.metadata or {}).get(k) for k in new["metadata"]}
    return (
        old.inputs != new["inputs"]
        or old.outputs != new["outputs"]
        or old_meta != new["metadata"]
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload the verified golden set to LangSmith"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    examples = build_examples(args.csv, args.labels)
    oos = sum(1 for e in examples if e["outputs"]["oos_type"])
    print(f"{len(examples)} examples: {len(examples) - oos} in-scope, {oos} OOS")
    if args.dry_run:
        return

    from dotenv import load_dotenv
    from langsmith import Client

    load_dotenv()
    print(upsert_examples(Client(), examples))


if __name__ == "__main__":
    main()
