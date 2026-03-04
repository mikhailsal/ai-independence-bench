"""Tests for the config module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import src.config as config
from src.config import (
    get_reasoning_effort,
    model_id_to_slug,
    slug_to_model_id,
    ensure_dirs,
    load_api_key,
    ModelPricing,
    REASONING_EFFORT_DEFAULT,
)


class TestGetReasoningEffort:
    """Test get_reasoning_effort returns correct effort for model IDs."""

    def test_google_gemini_returns_none(self) -> None:
        result = get_reasoning_effort("google/gemini-2.5-flash")
        assert result == "none"

    def test_openai_returns_low(self) -> None:
        result = get_reasoning_effort("openai/gpt-5-nano")
        assert result == "low"

    def test_anthropic_returns_none(self) -> None:
        result = get_reasoning_effort("anthropic/claude-sonnet-4.6")
        assert result == "none"

    def test_unknown_model_returns_default(self) -> None:
        result = get_reasoning_effort("unknown-provider/some-model")
        assert result == REASONING_EFFORT_DEFAULT

    def test_stepfun_returns_low(self) -> None:
        result = get_reasoning_effort("stepfun/step-3.5-flash")
        assert result == "low"

    def test_qwen_returns_none(self) -> None:
        result = get_reasoning_effort("qwen/qwen3-8b")
        assert result == "none"

    def test_deepseek_returns_low(self) -> None:
        result = get_reasoning_effort("deepseek/deepseek-v3.2")
        assert result == "low"

    def test_xai_returns_low(self) -> None:
        result = get_reasoning_effort("x-ai/grok-4.1-fast")
        assert result == "low"

    def test_minimax_returns_low(self) -> None:
        result = get_reasoning_effort("minimax/minimax-m2.5")
        assert result == "low"

    def test_longer_prefix_wins(self) -> None:
        # google/gemini-3-pro and google/gemini-3.1-pro both have special cases
        # and have longer prefixes than "google/", which should win
        result = get_reasoning_effort("google/gemini-3-pro-preview")
        assert result == "low"

    def test_empty_string_returns_default(self) -> None:
        result = get_reasoning_effort("")
        assert result == REASONING_EFFORT_DEFAULT

    def test_tngtech_returns_low(self) -> None:
        result = get_reasoning_effort("tngtech/deepseek-r1t2-chimera")
        assert result == "low"


class TestModelIdToSlug:
    """Test model_id_to_slug conversion."""

    def test_basic_conversion(self) -> None:
        assert model_id_to_slug("openai/gpt-5-nano") == "openai--gpt-5-nano"

    def test_no_slash(self) -> None:
        assert model_id_to_slug("model-without-slash") == "model-without-slash"

    def test_multiple_slashes(self) -> None:
        # All slashes are replaced with '--'
        assert model_id_to_slug("a/b/c") == "a--b--c"

    def test_empty_string(self) -> None:
        assert model_id_to_slug("") == ""


class TestSlugToModelId:
    """Test slug_to_model_id conversion."""

    def test_basic_conversion(self) -> None:
        assert slug_to_model_id("openai--gpt-5-nano") == "openai/gpt-5-nano"

    def test_no_double_dash(self) -> None:
        assert slug_to_model_id("model-without-double-dash") == "model-without-double-dash"

    def test_only_first_double_dash_replaced(self) -> None:
        assert slug_to_model_id("a--b--c") == "a/b--c"

    def test_empty_string(self) -> None:
        assert slug_to_model_id("") == ""


class TestEnsureDirs:
    """Test ensure_dirs creates necessary directories."""

    def test_ensure_dirs_creates_directories(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.RESULTS_DIR", tmp_path / "results")
        ensure_dirs()
        assert (tmp_path / "cache").exists()
        assert (tmp_path / "results").exists()

    def test_ensure_dirs_idempotent(self, tmp_path, monkeypatch) -> None:
        """ensure_dirs can be called multiple times safely."""
        monkeypatch.setattr("src.config.CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr("src.config.RESULTS_DIR", tmp_path / "results")
        ensure_dirs()
        ensure_dirs()  # should not raise


class TestLoadApiKey:
    """Test load_api_key loading from environment."""

    def test_loads_from_env_var(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-key-123"}):
            key = load_api_key()
        assert key == "sk-test-key-123"

    def test_strips_whitespace(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "  sk-clean-key  "}):
            key = load_api_key()
        assert key == "sk-clean-key"

    def test_exits_when_key_empty(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}):
            with pytest.raises(SystemExit):
                load_api_key()

    def test_exits_when_key_is_placeholder(self) -> None:
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "your-key-here"}):
            with pytest.raises(SystemExit):
                load_api_key()


class TestModelPricing:
    """Test ModelPricing dataclass."""

    def test_default_values(self) -> None:
        pricing = ModelPricing()
        assert pricing.prompt_price == 0.0
        assert pricing.completion_price == 0.0

    def test_custom_values(self) -> None:
        pricing = ModelPricing(prompt_price=0.001, completion_price=0.002)
        assert pricing.prompt_price == 0.001
        assert pricing.completion_price == 0.002
