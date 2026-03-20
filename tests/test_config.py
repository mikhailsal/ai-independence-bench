"""Tests for the config module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import src.config as config
from src.config import (
    MODEL_CONFIGS,
    ModelConfig,
    REASONING_EFFORT_DEFAULT,
    get_model_config,
    get_reasoning_effort,
    list_registered_labels_for_model,
    load_api_key,
    model_id_to_slug,
    ModelPricing,
    register_config,
    slug_to_model_id,
    ensure_dirs,
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


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    """Test ModelConfig dataclass and its properties."""

    def test_defaults(self) -> None:
        cfg = ModelConfig(model_id="openai/gpt-5-nano")
        assert cfg.model_id == "openai/gpt-5-nano"
        assert cfg.display_label == ""
        assert cfg.temperature is None
        assert cfg.reasoning_effort is None
        assert cfg.temperature_supported is True
        assert cfg.active is True

    def test_label_falls_back_to_model_id(self) -> None:
        cfg = ModelConfig(model_id="openai/gpt-5-nano")
        assert cfg.label == "openai/gpt-5-nano"

    def test_label_uses_display_label_when_set(self) -> None:
        cfg = ModelConfig(model_id="openai/gpt-5-nano", display_label="gpt5n@low-t0.7")
        assert cfg.label == "gpt5n@low-t0.7"

    def test_effective_temperature_default(self) -> None:
        cfg = ModelConfig(model_id="openai/gpt-5-nano")
        assert cfg.effective_temperature == 0.7  # RESPONSE_TEMPERATURE

    def test_effective_temperature_override(self) -> None:
        cfg = ModelConfig(model_id="openai/gpt-5-nano", temperature=0.0)
        assert cfg.effective_temperature == 0.0

    def test_effective_reasoning_default(self) -> None:
        cfg = ModelConfig(model_id="openai/gpt-5-nano")
        assert cfg.effective_reasoning == "low"  # openai/ prefix → low

    def test_effective_reasoning_override(self) -> None:
        cfg = ModelConfig(model_id="openai/gpt-5-nano", reasoning_effort="none")
        assert cfg.effective_reasoning == "none"

    def test_frozen(self) -> None:
        cfg = ModelConfig(model_id="openai/gpt-5-nano")
        with pytest.raises(AttributeError):
            cfg.model_id = "other"  # type: ignore[misc]


class TestModelConfigRegistry:
    """Test register_config, get_model_config, list_registered_labels_for_model."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """Save and restore the global registry around each test."""
        saved = dict(MODEL_CONFIGS)
        yield
        MODEL_CONFIGS.clear()
        MODEL_CONFIGS.update(saved)

    def test_register_and_lookup(self) -> None:
        cfg = ModelConfig(
            model_id="test/reg-model",
            display_label="test/reg-model@low-t0.5",
            temperature=0.5,
        )
        register_config(cfg)
        assert get_model_config("test/reg-model@low-t0.5") is cfg

    def test_duplicate_label_raises(self) -> None:
        cfg = ModelConfig(model_id="test/dup", display_label="test/dup@dup-label")
        register_config(cfg)
        with pytest.raises(ValueError, match="Duplicate"):
            register_config(cfg)

    def test_get_model_config_exact_label(self) -> None:
        cfg = ModelConfig(model_id="test/exact", display_label="test/exact@custom")
        register_config(cfg)
        assert get_model_config("test/exact@custom") is cfg

    def test_get_model_config_sole_model_id(self) -> None:
        cfg = ModelConfig(model_id="test/sole", display_label="test/sole@only")
        register_config(cfg)
        result = get_model_config("test/sole")
        assert result is cfg

    def test_get_model_config_ambiguous_model_id(self) -> None:
        cfg1 = ModelConfig(model_id="test/ambig", display_label="test/ambig@a")
        cfg2 = ModelConfig(model_id="test/ambig", display_label="test/ambig@b")
        register_config(cfg1)
        register_config(cfg2)
        result = get_model_config("test/ambig")
        assert result.model_id == "test/ambig"
        assert result.temperature is None

    def test_get_model_config_unknown_returns_default(self) -> None:
        result = get_model_config("unknown/model-xyz")
        assert result.model_id == "unknown/model-xyz"
        assert result.temperature is None

    def test_list_registered_labels_for_model(self) -> None:
        cfg1 = ModelConfig(model_id="test/multi", display_label="test/multi@x")
        cfg2 = ModelConfig(model_id="test/multi", display_label="test/multi@y")
        register_config(cfg1)
        register_config(cfg2)
        labels = list_registered_labels_for_model("test/multi")
        assert set(labels) == {"test/multi@x", "test/multi@y"}

    def test_list_registered_labels_none_found(self) -> None:
        labels = list_registered_labels_for_model("test/nonexistent-xyz")
        assert labels == []

    def test_builtin_step_flash_configs_registered(self) -> None:
        """The 3 Step Flash temperature configs should be registered via YAML at module load."""
        assert "step-3.5-flash:free@low-t0.7" in MODEL_CONFIGS
        assert "step-3.5-flash:free@low-t0.0" in MODEL_CONFIGS
        assert "step-3.5-flash:free@low-t1.0" in MODEL_CONFIGS

    def test_step_flash_config_temperatures(self) -> None:
        cfg07 = MODEL_CONFIGS["step-3.5-flash:free@low-t0.7"]
        assert cfg07.temperature == 0.7
        assert cfg07.model_id == "stepfun/step-3.5-flash:free"

        cfg00 = MODEL_CONFIGS["step-3.5-flash:free@low-t0.0"]
        assert cfg00.temperature == 0.0
        assert cfg00.model_id == "stepfun/step-3.5-flash:free"

        cfg10 = MODEL_CONFIGS["step-3.5-flash:free@low-t1.0"]
        assert cfg10.temperature == 1.0
        assert cfg10.model_id == "stepfun/step-3.5-flash:free"
