"""Docs freshness check: forbidden stale terms + FACTS-vs-source value sync + link check.

CI scope (committed sources only): stale-term grep, broken relative markdown links,
prompt-version sync. Run via ``python scripts/check_docs.py --ci``.

Also in CI: every ``run_retrieval_eval.py --path X`` in docs, benchmarks and READMEs
names a path the CLI accepts.

Local-only (needs .env / chroma): provider/model names, chunk count. Those are NOT
checked in CI by design — .env is a gitignored secret and absent there. When .env is
missing the script says so instead of pretending it verified everything.

Exemptions for the stale-term grep:
- anything under ``docs/archive/`` or ``docs/superpowers/`` (history / specs / plans
  legitimately describe removed designs)
- the decision-record file ``design-decisions.md`` (its job is to discuss removed designs)
- any line carrying a ``<!--freshness:ignore-->`` marker
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

STALE_DENYLIST: list[str] = [
    "llm_verifier",
    "rewriter",
    "FlashRank rerank",
    "generate_answer_v1",
    "Gemini 2.5 Flash (simple",
    "Gemini 3 Flash (complex",
    "validate_prompts",
]

IGNORE_MARKER = "<!--freshness:ignore-->"
EXCLUDE_DIRS = {"archive", "superpowers"}
EXCLUDE_FILES = {"design-decisions.md"}


@dataclass
class StaleHit:
    path: Path
    line: int
    term: str


@dataclass
class VersionMismatch:
    name: str
    registry_version: str
    facts_version: str


@dataclass
class BrokenLink:
    path: Path
    line: int
    target: str


LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def find_broken_links(root: Path) -> list[BrokenLink]:
    """Flag relative markdown links (to other tracked files) that resolve to nothing.

    Scoped to ``git ls-files`` (committed sources), matching this script's CI scope.
    External links (http/mailto) and in-page anchors (#...) are not checked.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    broken: list[BrokenLink] = []
    for rel in tracked:
        md = root / rel
        for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            if IGNORE_MARKER in line:
                continue
            for m in LINK_RE.finditer(line):
                target = m.group(1).strip()
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target_path = target.split("#", 1)[0]
                if not target_path:
                    continue
                resolved = (md.parent / target_path).resolve()
                if not resolved.exists():
                    broken.append(BrokenLink(path=md, line=i, target=target))
    return broken


def find_stale_terms(docs_dir: Path) -> list[StaleHit]:
    hits: list[StaleHit] = []
    for md in sorted(docs_dir.rglob("*.md")):
        rel = md.relative_to(docs_dir)
        if EXCLUDE_DIRS & set(rel.parts) or md.name in EXCLUDE_FILES:
            continue
        for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            if IGNORE_MARKER in line:
                continue
            for term in STALE_DENYLIST:
                if term in line:
                    hits.append(StaleHit(path=md, line=i, term=term))
    return hits


RETRIEVAL_EVAL_PATH_RE = re.compile(r"run_retrieval_eval\.py\b.*?--path[ =]([\w|]+)")


def find_retrieval_eval_path_drift(
    md_files: list[Path], valid_paths: tuple[str, ...]
) -> list[StaleHit]:
    """Flag ``run_retrieval_eval.py --path X`` where X is not a CLI choice."""
    hits: list[StaleHit] = []
    for md in md_files:
        for i, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            if IGNORE_MARKER in line:
                continue
            for m in RETRIEVAL_EVAL_PATH_RE.finditer(line):
                for path in m.group(1).split("|"):
                    if path not in valid_paths:
                        hits.append(StaleHit(path=md, line=i, term=path))
    return hits


def _retrieval_eval_md_files(root: Path) -> list[Path]:
    docs = [
        md
        for md in sorted((root / "docs").rglob("*.md"))
        if not EXCLUDE_DIRS & set(md.relative_to(root / "docs").parts)
    ]
    return (
        docs
        + sorted((root / "benchmarks").glob("*.md"))
        + sorted(root.glob("README*.md"))
    )


def _facts_prompt_versions(facts: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    in_prompts = False
    for line in facts.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("## prompts"):
            in_prompts = True
            continue
        if in_prompts and line.startswith("## "):
            break
        m = re.match(r"\s*-\s*([\w_]+):\s*(v\d+)", line)
        if in_prompts and m:
            out[m.group(1)] = m.group(2)
    return out


def check_prompt_versions(registry: Path, facts: Path) -> list[VersionMismatch]:
    reg = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    facts_versions = _facts_prompt_versions(facts)
    mismatches: list[VersionMismatch] = []
    for name, fv in facts_versions.items():
        rv = (reg.get(name) or {}).get("active_version")
        if rv is not None and rv != fv:
            mismatches.append(
                VersionMismatch(name=name, registry_version=rv, facts_version=fv)
            )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ci", action="store_true", help="committed-source checks only"
    )
    parser.add_argument("--root", default=".", type=Path)
    args = parser.parse_args(argv)

    root: Path = args.root
    docs_dir = root / "docs"
    registry = root / "prompts" / "registry.yaml"
    facts = docs_dir / "reference" / "FACTS.md"

    failures = 0

    stale = find_stale_terms(docs_dir)
    if stale:
        failures += len(stale)
        print("STALE TERMS (use <!--freshness:ignore--> for intentional mentions):")
        for h in stale:
            print(f"  {h.path}:{h.line}  '{h.term}'")

    broken_links = find_broken_links(root)
    if broken_links:
        failures += len(broken_links)
        print("BROKEN MARKDOWN LINKS:")
        for b in broken_links:
            print(f"  {b.path}:{b.line}  {b.target}")

    mism = check_prompt_versions(registry, facts)
    if mism:
        failures += len(mism)
        print("PROMPT VERSION DRIFT:")
        for m in mism:
            print(f"  {m.name}: FACTS={m.facts_version} registry={m.registry_version}")

    sys.path.insert(0, str(root.resolve()))
    from eval.run_retrieval_eval import PATHS

    drift = find_retrieval_eval_path_drift(_retrieval_eval_md_files(root), PATHS)
    if drift:
        failures += len(drift)
        print(f"RETRIEVAL EVAL CLI DRIFT (--path must be one of {', '.join(PATHS)}):")
        for h in drift:
            print(f"  {h.path}:{h.line}  --path {h.term}")

    if not args.ci and not (root / ".env").exists():
        print("NOTE: .env absent — skipping provider/model/chunk checks (local-only).")

    if failures:
        print(f"\ncheck_docs: FAIL ({failures} issue(s))")
        return 1
    print("check_docs: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
