"""Allowlisted configuration and frozen references, without constructing providers."""

# ANCHOR: selected SDK examples + local files → secret-safe, versioned run manifest.

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from eval.ls.policy import case_type

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_KEYS = (
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL_NAME",
    "TEMPERATURE",
    "RERANKER_BACKEND",
    "RERANKING_MODEL",
    "CROSSENCODER_MODEL",
    "SIMPLE_LLM_PROVIDER",
    "SIMPLE_MODEL_NAME",
    "COMPLEX_LLM_PROVIDER",
    "COMPLEX_MODEL_NAME",
    "JUDGE_LLM_PROVIDER",
    "JUDGE_MODEL_NAME",
    "CITATION_JUDGE_MODEL_NAME",
    "CITATION_JUDGE_MAX_TOKENS",
    "OPENROUTER_REASONING_EFFORT",
    "SIMPLE_REASONING_EFFORT",
    "OPENROUTER_MAX_TOKENS",
    "OPENROUTER_PROVIDER_SORT",
    "CHROMA_DB_PATH",
    "CHROMA_COLLECTION_NAME",
    "REQUEST_TIMEOUT",
)
V7_KEYS = (
    "HARD_GATE_THRESHOLD",
    "TRIAGE_SOFT_THRESHOLD",
    "MIN_PASSAGES",
    "MIN_KEYWORD_OVERLAP_ACTIVE",
    "MAX_SINGLE_DOC_RATIO",
    "SIMPLE_TOP_K",
    "SIMPLE_TIMEOUT_MS",
    "ABSTAIN_ON_EMPTY_EVIDENCE",
    "COMPLEX_THRESHOLD",
    "COMPLEX_MIN_PASSAGES",
    "COMPLEX_MIN_KW_OVERLAP",
    "COMPLEX_MAX_SINGLE_DOC_RATIO",
    "COMPLEX_BORDERLINE_THRESHOLD",
    "COMPLEX_TOP_K",
    "FINAL_MERGE_TOP_K",
    "RERANK_CANDIDATE_CAP",
    "COMPLEX_TIMEOUT_MS",
    "RRF_K",
    "MMR_LAMBDA",
    "MIN_KEYWORD_OVERLAP_ORIGINAL",
    "MAX_REWRITE_ATTEMPTS",
    "MAX_CHUNKS_FOR_LLM",
    "PACK_TOKEN_BUDGET",
    "PARENT_CONTEXT_ENABLE",
    "PARENT_MAX_CHARS",
    "PROMPT_TOKEN_BUDGET",
    "VERIFIER_CONFIDENCE_ANCHOR",
    "COVERAGE_DROP_PCT",
    "MAX_INPUT_LENGTH",
    "BLOCKED_PATTERNS",
    "DOMAIN_GATE_THRESHOLD",
    "V8_ENABLE_MULTI_QUERY",
    "V8_EXPAND_N",
    "TELEMETRY_ENABLED",
    "TELEMETRY_WRITER",
)
METADATA_KEYS = (
    "case_id",
    "oos_type",
    "corpus_support",
    "csv_sha256",
    "labels_sha256",
    "case_type",
    "retrieval_labelled",
    "split",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_example(example) -> dict:
    return {
        "id": str(example.id),
        "inputs": example.inputs,
        "outputs": example.outputs,
        "metadata": {
            k: v for k, v in (example.metadata or {}).items() if k in METADATA_KEYS
        },
        "version": str(getattr(example, "modified_at", None)),
    }


def make_manifest(
    examples,
    *,
    run_id,
    split,
    dry_run,
    skip_judge,
    git_sha,
    labels_snapshot,
    labels_path,
):
    from config.settings import settings
    from src.v7.config import v7_config

    configuration = {key: getattr(settings, key) for key in SETTINGS_KEYS}
    configuration["SIMPLE_REASONING_EFFORT"] = (
        settings.SIMPLE_REASONING_EFFORT
        if settings.SIMPLE_REASONING_EFFORT is not None
        else settings.OPENROUTER_REASONING_EFFORT
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    snapshot = Path(settings.CHROMA_DB_PATH).resolve()
    prompts = list((ROOT / "eval/prompts").glob("*.md")) + list(
        (ROOT / "src/prompts").rglob("*.md")
    )
    prompts += [
        p for p in (ROOT / "prompts").rglob("*") if p.suffix in {".md", ".j2", ".yaml"}
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "skip_judge": skip_judge,
        "split": split,
        "git_sha": git_sha,
        "git_dirty": bool(dirty.stdout.strip()),
        "settings": configuration,
        "v7": {k: getattr(v7_config, k) for k in V7_KEYS},
        "index_snapshot": snapshot.name,
        "index_path": str(snapshot),
        "labels_snapshot": labels_snapshot,
        "labels_sha256": digest(labels_path),
        "csv_hashes": sorted(
            {(e.metadata or {}).get("csv_sha256", "unknown") for e in examples}
        ),
        "pricing_sha256": digest(ROOT / "src/pricing.py"),
        "prompt_version_overrides": {
            name: os.environ[name]
            for name in (
                "PROMPT_GENERATE_ANSWER_VERSION",
                "PROMPT_QUERY_EXPAND_VERSION",
                "PROMPT_QUERY_EXPANSION_VERSION",
            )
            if name in os.environ
        },
        "snapshot_report_sha256": (
            digest(snapshot / "snapshot-report.json")
            if (snapshot / "snapshot-report.json").is_file()
            else None
        ),
        "prompt_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in sorted(set(prompts))
        },
        "dataset_versions": sorted(
            {str(getattr(e, "modified_at", None)) for e in examples}
        ),
        "references": [frozen_example(e) for e in examples],
        "case_counts": {
            kind: sum(case_type(e.outputs or {}) == kind for e in examples)
            for kind in ("regular", "false_premise", "out_of_scope")
        },
        "unlabelled_case_ids": [
            (e.metadata or {}).get("case_id")
            for e in examples
            if case_type(e.outputs or {}) != "out_of_scope"
            and not (e.outputs or {}).get("relevant_chunk_ids")
        ],
        "judge_policy": {
            "max_retries": 0,
            "timeout_sec": 300,
            "correctness_max_tokens": 16000,
            "groundedness_max_tokens": 16000,
        },
    }
