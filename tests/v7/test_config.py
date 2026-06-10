# tests/v7/test_config.py
"""Tests for v7 configuration."""

from __future__ import annotations


class TestV7ConfigDefaults:
    """Verify all default values match the design doc."""

    def test_hard_gate_threshold(self):
        from src.v7.config import v7_config

        # Acceptance threshold: relevant docs score ~0.44–0.56 → 0.50 separates good from uncertain
        assert v7_config.HARD_GATE_THRESHOLD == 0.50

    def test_triage_soft_threshold(self):
        from src.v7.config import v7_config

        # Floor threshold: below 0.38 = clearly garbage. Zone [0.38, 0.50) → borderline → llm_verifier
        assert v7_config.TRIAGE_SOFT_THRESHOLD == 0.38

    def test_coverage_drop_pct(self):
        from src.v7.config import v7_config

        assert v7_config.COVERAGE_DROP_PCT == 0.30

    def test_rrf_k(self):
        from src.v7.config import v7_config

        assert v7_config.RRF_K == 60

    def test_mmr_lambda(self):
        from src.v7.config import v7_config

        assert v7_config.MMR_LAMBDA == 0.7

    def test_keyword_overlap_active(self):
        from src.v7.config import v7_config

        # Lowered from 0.3: Russian adj/noun lemmas differ ("лестничный"≠"лестница")
        assert v7_config.MIN_KEYWORD_OVERLAP_ACTIVE == 0.15

    def test_keyword_overlap_original(self):
        from src.v7.config import v7_config

        # Drift detection only — looser than active threshold
        assert v7_config.MIN_KEYWORD_OVERLAP_ORIGINAL == 0.10

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
