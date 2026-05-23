# V7 Stage 0: State Types & Config — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the foundational contracts (`state_types.py`) and configuration (`config.py`) for the v7 RAG pipeline, enabling all subsequent modules to import typed schemas from day one.

**Architecture:** Two files in `src/v7/` — `state_types.py` with all TypedDict/Literal types from the v7 spec, and `config.py` with a pydantic-settings `V7Config` class exposing every threshold via `V7_`-prefixed env vars. Both are leaf modules with zero dependencies on other project code.

**Tech Stack:** Python 3.11+, typing (TypedDict, Literal, Annotated), operator, pydantic-settings.

**Design doc:** `docs/plans/2026-02-16-v7-migration-design.md`
**Source spec:** `docs/feature/migration-v7` (lines 160-323 — types, lines 72-90 — state design)

---

### Task 1: Create `src/v7/` package

**Files:**
- Create: `src/v7/__init__.py`

**Step 1: Create the package directory and init file**

```python
# src/v7/__init__.py
"""RAG Pipeline v7 — modular implementation."""
```

**Step 2: Verify import works**

Run: `source venv/bin/activate && python -c "import src.v7; print('ok')"`
Expected: `ok`

**Step 3: Commit**

```bash
git add src/v7/__init__.py
git commit -m "chore: create src/v7 package for v7 migration"
```

---

### Task 2: Write failing tests for state_types

**Files:**
- Create: `tests/v7/__init__.py`
- Create: `tests/v7/test_state_types.py`

**Step 1: Create test directory**

```python
# tests/v7/__init__.py
```

**Step 2: Write the failing tests**

```python
# tests/v7/test_state_types.py
"""Tests for v7 state type contracts."""
from __future__ import annotations

import operator
import pytest


class TestLiteralTypes:
    """Verify all Literal type aliases exist and have correct values."""

    def test_intent_values(self):
        from src.v7.state_types import Intent
        # Intent should accept "noise" and "domain"
        val: Intent = "noise"
        assert val == "noise"
        val2: Intent = "domain"
        assert val2 == "domain"

    def test_triage_category_values(self):
        from src.v7.state_types import TriageCategory
        val: TriageCategory = "sufficient"
        assert val in ("sufficient", "borderline", "clearly_bad")

    def test_verifier_verdict_values(self):
        from src.v7.state_types import VerifierVerdict
        val: VerifierVerdict = "rewrite"
        assert val in ("sufficient", "rewrite", "escalate")

    def test_routing_literal_types_exist(self):
        from src.v7.state_types import (
            NextAfterIntent,
            NextAfterRouter,
            NextAfterTriage,
            NextAfterVerifier,
            NextAfterEvalComplex,
        )
        # Just verify they are importable
        assert NextAfterIntent is not None


class TestRetrievalPlan:
    """RetrievalPlan is total=False TypedDict — all fields optional."""

    def test_empty_plan_is_valid(self):
        from src.v7.state_types import RetrievalPlan
        plan: RetrievalPlan = {}
        assert isinstance(plan, dict)

    def test_plan_with_fields(self):
        from src.v7.state_types import RetrievalPlan
        plan: RetrievalPlan = {
            "top_k": 10,
            "rerank": True,
            "threshold": 0.65,
            "require_multi_doc": False,
            "mmr_lambda": 0.9,
        }
        assert plan["top_k"] == 10
        assert plan["rerank"] is True

    def test_plan_has_expected_keys(self):
        from src.v7.state_types import RetrievalPlan
        expected_keys = {
            "top_k", "rerank", "timeout_ms", "threshold",
            "min_passages", "min_keyword_overlap",
            "max_single_doc_ratio", "borderline_threshold",
            "min_verifier_confidence",
            "require_multi_doc", "mmr_lambda",
        }
        # TypedDict.__annotations__ contains declared keys
        assert set(RetrievalPlan.__annotations__) == expected_keys


class TestRetrievalAttempt:
    """RetrievalAttempt is total=False TypedDict."""

    def test_attempt_with_fields(self):
        from src.v7.state_types import RetrievalAttempt
        attempt: RetrievalAttempt = {
            "retrieval_id": "abc123",
            "stage": "simple",
            "passages": [{"text": "test"}],
            "top_score": 0.85,
            "attempt_plan": {"top_k": 10},
            "metrics": {"overlap": 0.5},
        }
        assert attempt["stage"] == "simple"
        assert attempt["top_score"] == 0.85

    def test_attempt_has_expected_keys(self):
        from src.v7.state_types import RetrievalAttempt
        expected_keys = {
            "retrieval_id", "stage", "passages",
            "top_score", "attempt_plan", "metrics",
        }
        assert set(RetrievalAttempt.__annotations__) == expected_keys


class TestHardGateResult:
    """HardGateResult is total=True TypedDict — all fields required."""

    def test_full_result(self):
        from src.v7.state_types import HardGateResult
        result: HardGateResult = {
            "sufficient": True,
            "above_threshold": True,
            "enough_evidence": True,
            "keyword_overlap_ok": True,
            "top_score": 0.78,
            "passage_count": 5,
            "keyword_overlap_active": 0.6,
            "keyword_overlap_original": 0.4,
        }
        assert result["sufficient"] is True
        assert result["top_score"] == 0.78

    def test_has_expected_keys(self):
        from src.v7.state_types import HardGateResult
        expected_keys = {
            "sufficient", "above_threshold", "enough_evidence",
            "keyword_overlap_ok", "top_score", "passage_count",
            "keyword_overlap_active", "keyword_overlap_original",
        }
        assert set(HardGateResult.__annotations__) == expected_keys


class TestSufficiencyResult:
    """SufficiencyResult is total=True — full triage result."""

    def test_full_result(self):
        from src.v7.state_types import SufficiencyResult
        result: SufficiencyResult = {
            "sufficient": True,
            "above_threshold": True,
            "enough_evidence": True,
            "keyword_overlap_ok": True,
            "diversity_ok": True,
            "escalation_hint": False,
            "triage": "sufficient",
            "top_score": 0.78,
            "keyword_overlap_active": 0.6,
            "keyword_overlap_original": 0.4,
            "passage_count": 5,
            "unique_docs": 3,
            "max_doc_ratio": 0.4,
        }
        assert result["triage"] == "sufficient"
        assert result["unique_docs"] == 3

    def test_has_expected_keys(self):
        from src.v7.state_types import SufficiencyResult
        expected_keys = {
            "sufficient", "above_threshold", "enough_evidence",
            "keyword_overlap_ok", "diversity_ok", "escalation_hint",
            "triage", "top_score", "keyword_overlap_active",
            "keyword_overlap_original", "passage_count",
            "unique_docs", "max_doc_ratio",
        }
        assert set(SufficiencyResult.__annotations__) == expected_keys


class TestVerificationResult:
    """VerificationResult is total=False TypedDict."""

    def test_partial_result(self):
        from src.v7.state_types import VerificationResult
        result: VerificationResult = {
            "verdict": "sufficient",
            "confidence": 0.9,
        }
        assert result["verdict"] == "sufficient"

    def test_has_expected_keys(self):
        from src.v7.state_types import VerificationResult
        expected_keys = {
            "verdict", "reason", "rewrite_hint",
            "missing_aspects", "confidence",
        }
        assert set(VerificationResult.__annotations__) == expected_keys


class TestRAGState:
    """RAGState is the main graph state TypedDict."""

    def test_minimal_state(self):
        from src.v7.state_types import RAGState
        state: RAGState = {"query": "тест", "filters": {}}
        assert state["query"] == "тест"

    def test_has_all_sections(self):
        """Verify INPUT, INTERNAL, OUTPUT, and UX fields are present."""
        from src.v7.state_types import RAGState
        annotations = set(RAGState.__annotations__)

        # INPUT
        assert "query" in annotations
        assert "filters" in annotations

        # INTERNAL
        assert "intent" in annotations
        assert "plan" in annotations
        assert "retrieval_id" in annotations
        assert "active_query" in annotations
        assert "retrieval_attempts" in annotations
        assert "sufficient" in annotations
        assert "verify_iteration" in annotations
        assert "verification" in annotations

        # OUTPUT
        assert "final_passages" in annotations
        assert "final_score" in annotations
        assert "fallback_passages" in annotations
        assert "fallback_score" in annotations
        assert "clarify_message" in annotations
        assert "abstain_reason" in annotations
        assert "sufficiency_details" in annotations

        # UX
        assert "status_message" in annotations

    def test_retrieval_attempts_uses_operator_add(self):
        """retrieval_attempts must use Annotated[..., operator.add] for LangGraph reducer."""
        from src.v7.state_types import RAGState
        import typing
        hints = typing.get_type_hints(RAGState, include_extras=True)
        attempts_hint = hints["retrieval_attempts"]
        # Check it's Annotated with operator.add
        assert hasattr(attempts_hint, "__metadata__")
        assert operator.add in attempts_hint.__metadata__


class TestConstants:
    """Verify exported constants."""

    def test_max_verify_iterations(self):
        from src.v7.state_types import MAX_VERIFY_ITERATIONS
        assert MAX_VERIFY_ITERATIONS == 2

    def test_allowed_filter_keys(self):
        from src.v7.state_types import ALLOWED_FILTER_KEYS
        assert "doc_type" in ALLOWED_FILTER_KEYS
        assert "doc_id" in ALLOWED_FILTER_KEYS
        assert isinstance(ALLOWED_FILTER_KEYS, (set, frozenset))
```

**Step 3: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/v7/test_state_types.py -v 2>&1 | head -30`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.v7.state_types'`

**Step 4: Commit**

```bash
git add tests/v7/__init__.py tests/v7/test_state_types.py
git commit -m "test: add failing tests for v7 state_types"
```

---

### Task 3: Implement state_types.py

**Files:**
- Create: `src/v7/state_types.py`

**Step 1: Implement all types**

Source: `docs/feature/migration-v7` lines 182-323.

```python
# src/v7/state_types.py
"""V7 RAG pipeline — data contracts and type definitions.

All TypedDicts used across the v7 pipeline are defined here.
This module has zero dependencies on other project code.

Source spec: docs/feature/migration-v7 (lines 182-323).
"""
from __future__ import annotations

import operator
from typing import Annotated, List, Literal, TypedDict

# ─── Literal type aliases ─────────────────────────────────────────────────────

Intent = Literal["noise", "domain"]
TriageCategory = Literal["sufficient", "borderline", "clearly_bad"]
VerifierVerdict = Literal["sufficient", "rewrite", "escalate"]

NextAfterIntent = Literal["end", "router"]
NextAfterRouter = Literal["rag_simple", "clarify_respond"]
NextAfterTriage = Literal["end", "llm_verifier", "rag_complex"]
NextAfterVerifier = Literal["end", "rewriter", "rag_complex"]
NextAfterEvalComplex = Literal["end", "abstain"]

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_VERIFY_ITERATIONS = 2

ALLOWED_FILTER_KEYS = frozenset(
    {"doc_type", "doc_id", "section", "category", "year"}
)

# ─── State schemas ────────────────────────────────────────────────────────────


class RetrievalPlan(TypedDict, total=False):
    """
    Search parameters. Created by router, updated on escalation.

    Retrieval: top_k, rerank, timeout_ms.
    Hard gates: threshold, min_passages, min_keyword_overlap.
    Soft signals: max_single_doc_ratio.
    Borderline zone: borderline_threshold.
    LLM verifier: min_verifier_confidence.
    Router-driven: require_multi_doc, mmr_lambda.
    """
    top_k: int
    rerank: bool
    timeout_ms: int
    threshold: float
    min_passages: int
    min_keyword_overlap: float
    max_single_doc_ratio: float
    borderline_threshold: float
    min_verifier_confidence: float
    require_multi_doc: bool
    mmr_lambda: float


class RetrievalAttempt(TypedDict, total=False):
    """
    Result of a single retrieval attempt. Append-only list in state.

    attempt_plan: snapshot of plan at retrieval time.
    metrics: per-attempt diagnostics for offline evaluation.
    """
    retrieval_id: str
    stage: Literal["simple", "complex"]
    passages: List[dict]
    top_score: float
    attempt_plan: dict
    metrics: dict


class HardGateResult(TypedDict):
    """
    Hard gates ONLY result. No triage, no soft signals.
    Used by evaluate_complex (final check).

    Dual overlap: keyword_overlap_active (working query),
    keyword_overlap_original (original — drift detection).
    """
    sufficient: bool
    above_threshold: bool
    enough_evidence: bool
    keyword_overlap_ok: bool
    top_score: float
    passage_count: int
    keyword_overlap_active: float
    keyword_overlap_original: float


class SufficiencyResult(TypedDict):
    """
    Full check result: hard gates + soft signals + triage.
    Used by evaluate_triage (3-way gate).
    """
    sufficient: bool
    above_threshold: bool
    enough_evidence: bool
    keyword_overlap_ok: bool
    diversity_ok: bool
    escalation_hint: bool
    triage: TriageCategory
    top_score: float
    keyword_overlap_active: float
    keyword_overlap_original: float
    passage_count: int
    unique_docs: int
    max_doc_ratio: float


class VerificationResult(TypedDict, total=False):
    """
    LLM verification result.

    verdict: sufficient / rewrite / escalate.
    rewrite_hint: instruction for rewriter.
    missing_aspects: uncovered aspects.
    """
    verdict: VerifierVerdict
    reason: str
    rewrite_hint: str
    missing_aspects: List[str]
    confidence: float


class RAGState(TypedDict, total=False):
    """
    V7 graph state.

    INPUT:     query (immutable), filters.
    INTERNAL:  intent, plan, retrieval_id, active_query,
               retrieval_attempts, sufficient, verify_iteration, verification.
    OUTPUT:    final_passages, final_score, fallback_passages, fallback_score,
               clarify_message, abstain_reason, sufficiency_details.
    UX:        status_message — progress for frontend streaming.
    """
    # INPUT
    query: str
    filters: dict

    # INTERNAL
    intent: Intent
    plan: RetrievalPlan
    retrieval_id: str
    active_query: str
    retrieval_attempts: Annotated[List[RetrievalAttempt], operator.add]
    sufficient: bool
    verify_iteration: int
    verification: VerificationResult

    # OUTPUT
    final_passages: List[dict]
    final_score: float
    fallback_passages: List[dict]
    fallback_score: float
    clarify_message: str
    abstain_reason: str
    sufficiency_details: SufficiencyResult

    # UX
    status_message: str
```

**Step 2: Run tests**

Run: `source venv/bin/activate && pytest tests/v7/test_state_types.py -v`
Expected: All PASS

**Step 3: Run linter**

Run: `source venv/bin/activate && black src/v7/state_types.py tests/v7/test_state_types.py && ruff check src/v7/state_types.py tests/v7/test_state_types.py --fix`
Expected: Clean or auto-fixed

**Step 4: Commit**

```bash
git add src/v7/state_types.py
git commit -m "feat(v7): implement state_types with all TypedDict contracts"
```

---

### Task 4: Write failing tests for config

**Files:**
- Create: `tests/v7/test_config.py`

**Step 1: Write the failing tests**

```python
# tests/v7/test_config.py
"""Tests for v7 configuration."""
from __future__ import annotations

import os
import pytest


class TestV7ConfigDefaults:
    """Verify all default values match the design doc."""

    def test_hard_gate_threshold(self):
        from src.v7.config import v7_config
        assert v7_config.HARD_GATE_THRESHOLD == 0.65

    def test_triage_soft_threshold(self):
        from src.v7.config import v7_config
        assert v7_config.TRIAGE_SOFT_THRESHOLD == 0.40

    def test_coverage_drop_pct(self):
        from src.v7.config import v7_config
        assert v7_config.COVERAGE_DROP_PCT == 0.30

    def test_rrf_k(self):
        from src.v7.config import v7_config
        assert v7_config.RRF_K == 60

    def test_mmr_lambda(self):
        from src.v7.config import v7_config
        assert v7_config.MMR_LAMBDA == 0.7

    def test_bm25_top_k(self):
        from src.v7.config import v7_config
        assert v7_config.BM25_TOP_K == 20

    def test_semantic_top_k(self):
        from src.v7.config import v7_config
        assert v7_config.SEMANTIC_TOP_K == 20

    def test_keyword_overlap_active(self):
        from src.v7.config import v7_config
        assert v7_config.MIN_KEYWORD_OVERLAP_ACTIVE == 0.3

    def test_keyword_overlap_original(self):
        from src.v7.config import v7_config
        assert v7_config.MIN_KEYWORD_OVERLAP_ORIGINAL == 0.2

    def test_max_rewrite_attempts(self):
        from src.v7.config import v7_config
        assert v7_config.MAX_REWRITE_ATTEMPTS == 2

    def test_max_chunks_for_llm(self):
        from src.v7.config import v7_config
        assert v7_config.MAX_CHUNKS_FOR_LLM == 10

    def test_verifier_confidence_anchor(self):
        from src.v7.config import v7_config
        assert v7_config.VERIFIER_CONFIDENCE_ANCHOR == 0.7

    def test_max_input_length(self):
        from src.v7.config import v7_config
        assert v7_config.MAX_INPUT_LENGTH == 2000

    def test_blocked_patterns(self):
        from src.v7.config import v7_config
        assert "ignore previous" in v7_config.BLOCKED_PATTERNS
        assert "system prompt" in v7_config.BLOCKED_PATTERNS
        assert "you are now" in v7_config.BLOCKED_PATTERNS


class TestV7ConfigEnvOverride:
    """Verify env vars with V7_ prefix override defaults."""

    def test_env_override_threshold(self, monkeypatch):
        monkeypatch.setenv("V7_HARD_GATE_THRESHOLD", "0.80")
        # Re-instantiate to pick up env
        from src.v7.config import V7Config
        cfg = V7Config()
        assert cfg.HARD_GATE_THRESHOLD == 0.80

    def test_env_override_rrf_k(self, monkeypatch):
        monkeypatch.setenv("V7_RRF_K", "100")
        from src.v7.config import V7Config
        cfg = V7Config()
        assert cfg.RRF_K == 100

    def test_env_override_blocked_patterns(self, monkeypatch):
        monkeypatch.setenv("V7_BLOCKED_PATTERNS", '["custom pattern"]')
        from src.v7.config import V7Config
        cfg = V7Config()
        assert cfg.BLOCKED_PATTERNS == ["custom pattern"]

    def test_extra_env_vars_ignored(self, monkeypatch):
        monkeypatch.setenv("V7_UNKNOWN_SETTING", "anything")
        from src.v7.config import V7Config
        cfg = V7Config()
        assert not hasattr(cfg, "UNKNOWN_SETTING")


class TestV7ConfigType:
    """Verify the config is a pydantic BaseSettings instance."""

    def test_is_base_settings(self):
        from pydantic_settings import BaseSettings
        from src.v7.config import V7Config
        assert issubclass(V7Config, BaseSettings)

    def test_singleton_exists(self):
        from src.v7.config import v7_config
        from src.v7.config import V7Config
        assert isinstance(v7_config, V7Config)
```

**Step 2: Run tests to verify they fail**

Run: `source venv/bin/activate && pytest tests/v7/test_config.py -v 2>&1 | head -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.v7.config'`

**Step 3: Commit**

```bash
git add tests/v7/test_config.py
git commit -m "test: add failing tests for v7 config"
```

---

### Task 5: Implement config.py

**Files:**
- Create: `src/v7/config.py`

**Step 1: Implement V7Config**

```python
# src/v7/config.py
"""V7 RAG pipeline — configuration.

All thresholds and limits externalized via pydantic-settings.
Env vars use V7_ prefix (e.g. V7_HARD_GATE_THRESHOLD=0.80).

Source: docs/plans/2026-02-16-v7-migration-design.md, section 2.1.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class V7Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="V7_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Hard gates
    HARD_GATE_THRESHOLD: float = 0.65
    TRIAGE_SOFT_THRESHOLD: float = 0.40
    COVERAGE_DROP_PCT: float = 0.30

    # Retrieval
    RRF_K: int = 60
    MMR_LAMBDA: float = 0.7
    BM25_TOP_K: int = 20
    SEMANTIC_TOP_K: int = 20

    # Keyword overlap (dual)
    MIN_KEYWORD_OVERLAP_ACTIVE: float = 0.3
    MIN_KEYWORD_OVERLAP_ORIGINAL: float = 0.2

    # LLM & Limits
    MAX_REWRITE_ATTEMPTS: int = 2
    MAX_CHUNKS_FOR_LLM: int = 10
    VERIFIER_CONFIDENCE_ANCHOR: float = 0.7

    # Anti-injection
    MAX_INPUT_LENGTH: int = 2000
    BLOCKED_PATTERNS: list[str] = [
        "ignore previous",
        "system prompt",
        "you are now",
    ]


v7_config = V7Config()
```

**Step 2: Run tests**

Run: `source venv/bin/activate && pytest tests/v7/test_config.py -v`
Expected: All PASS

**Step 3: Run linter**

Run: `source venv/bin/activate && black src/v7/config.py tests/v7/test_config.py && ruff check src/v7/config.py tests/v7/test_config.py --fix`
Expected: Clean

**Step 4: Commit**

```bash
git add src/v7/config.py
git commit -m "feat(v7): implement V7Config with all thresholds"
```

---

### Task 6: Update `src/v7/__init__.py` re-exports and run full suite

**Files:**
- Modify: `src/v7/__init__.py`

**Step 1: Add re-exports**

```python
# src/v7/__init__.py
"""RAG Pipeline v7 — modular implementation."""
from src.v7.config import V7Config, v7_config
from src.v7.state_types import (
    ALLOWED_FILTER_KEYS,
    MAX_VERIFY_ITERATIONS,
    HardGateResult,
    Intent,
    RAGState,
    RetrievalAttempt,
    RetrievalPlan,
    SufficiencyResult,
    TriageCategory,
    VerificationResult,
    VerifierVerdict,
)

__all__ = [
    "ALLOWED_FILTER_KEYS",
    "HardGateResult",
    "Intent",
    "MAX_VERIFY_ITERATIONS",
    "RAGState",
    "RetrievalAttempt",
    "RetrievalPlan",
    "SufficiencyResult",
    "TriageCategory",
    "V7Config",
    "VerificationResult",
    "VerifierVerdict",
    "v7_config",
]
```

**Step 2: Run all v7 tests**

Run: `source venv/bin/activate && pytest tests/v7/ -v`
Expected: All PASS

**Step 3: Run full project test suite (verify no regressions)**

Run: `source venv/bin/activate && pytest -v 2>&1 | tail -20`
Expected: All existing tests still PASS

**Step 4: Run linter on all v7 code**

Run: `source venv/bin/activate && black src/v7/ tests/v7/ && ruff check src/v7/ tests/v7/ --fix`
Expected: Clean

**Step 5: Commit**

```bash
git add src/v7/__init__.py
git commit -m "feat(v7): add re-exports to src/v7/__init__"
```

---

### Task 7: Update design doc status

**Files:**
- Modify: `docs/plans/2026-02-16-v7-migration-design.md`

**Step 1: Mark Этап 0 as done**

Change line:
```markdown
### Этап 0: Контракты и конфигурация
```
To:
```markdown
### Этап 0: Контракты и конфигурация ✅ Done (2026-02-16)
```

**Step 2: Commit**

```bash
git add docs/plans/2026-02-16-v7-migration-design.md
git commit -m "docs: mark v7 stage 0 as complete"
```
