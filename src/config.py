"""Configuration: paths, constants, model lists, judge model, API key loading."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
COST_LOG_PATH = RESULTS_DIR / "cost_log.json"
ENV_PATH = PROJECT_ROOT / ".env"
CONFIGS_PATH = PROJECT_ROOT / "configs" / "models.yaml"

# ---------------------------------------------------------------------------
# Token / generation limits
# ---------------------------------------------------------------------------
RESPONSE_MAX_TOKENS = 2048          # generous budget for identity/preference responses
                                    # (tool_role mode needs more: JSON wrapper + tool call overhead)
RESPONSE_TEMPERATURE = 0.7          # some creativity
JUDGE_MAX_TOKENS = 1024             # budget for judge scoring
JUDGE_TEMPERATURE = 0.0             # deterministic judgment

# ---------------------------------------------------------------------------
# Reasoning model support (same approach as self-detect-bench)
# ---------------------------------------------------------------------------
REASONING_EFFORT_DEFAULT = "low"

REASONING_EFFORT_BY_PREFIX: dict[str, str] = {
    "google/gemini-3-pro":   "low",    # Gemini Pro REQUIRES reasoning (cannot be disabled)
    "google/gemini-3.1-pro": "low",    # Gemini 3.1 Pro REQUIRES reasoning
    "google/":      "none",
    "qwen/":        "none",
    "openai/":      "low",
    "anthropic/":   "none",    # Haiku 4.5+ supports thinking, but skip for bench
    "stepfun/":     "low",     # Step 3.5 Flash REQUIRES reasoning (cannot be disabled)
    "nvidia/":      "none",    # Nemotron models support reasoning
    "arcee-ai/":    "low",     # Trinity models REQUIRE reasoning (cannot be disabled)
    "z-ai/":        "none",    # GLM models support reasoning
    "x-ai/":        "low",     # Grok models support reasoning
    "bytedance-seed/": "low",  # Seed models support reasoning
    "minimax/":     "low",     # MiniMax models support reasoning
    "xiaomi/":      "low",     # Xiaomi MIMO models support reasoning
    "deepseek/":    "low",     # DeepSeek v3.2 supports reasoning
    "kwaipilot/":   "none",    # KwaiPilot models — no reasoning support
    "mistralai/":   "none",    # Mistral models — no reasoning support
    "openrouter/":  "low",     # OpenRouter custom models — varies per model
    "nex-agi/":     "none",    # NexAGI fine-tunes — based on DeepSeek
    "tngtech/":     "low",     # TNG fine-tunes — based on DeepSeek R1
}


def get_reasoning_effort(model_id: str) -> str:
    """Return the default reasoning effort for a model based on its ID prefix."""
    best_match = ""
    best_effort = REASONING_EFFORT_DEFAULT
    for prefix, effort in REASONING_EFFORT_BY_PREFIX.items():
        if model_id.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
            best_effort = effort
    return best_effort


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
API_CALL_TIMEOUT = 90  # seconds per call (generous for cheap models)

# ---------------------------------------------------------------------------
# Local model support (LM Studio, Ollama, etc.)
# ---------------------------------------------------------------------------
LOCAL_MODEL_TIMEOUT = 600  # seconds — generous for slow local models (~10 tok/s)

# ---------------------------------------------------------------------------
# Judge model
# ---------------------------------------------------------------------------
JUDGE_MODEL = "google/gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Default test models
# ---------------------------------------------------------------------------
DEFAULT_TEST_MODELS: list[str] = [
    "openai/gpt-5-nano",
    "meta-llama/llama-4-scout",
    "qwen/qwen3-8b",
    "google/gemini-2.5-flash-lite",
    "mistralai/mistral-small-3.2-24b-instruct",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
    "z-ai/glm-5",
    "moonshotai/kimi-k2.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.5",
    "openai/gpt-5.2",
    "openai/gpt-5.1-codex-mini",
    "anthropic/claude-opus-4.6",
    "qwen/qwen3-coder-next",
    "qwen/qwen3-coder",
    "qwen/qwen3.5-flash-02-23",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-pro-preview",
    "openai/gpt-5.3-codex",
    "nex-agi/deepseek-v3.1-nex-n1",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v3.2-exp",
    "deepseek/deepseek-v3.1-terminus:exacto",
    "tngtech/deepseek-r1t2-chimera",
    "openrouter/hunter-alpha",
    "x-ai/grok-4.20-beta",
    "z-ai/glm-5-turbo",
    "mistralai/mistral-small-2603",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "minimax/minimax-m2.7",
]

# ---------------------------------------------------------------------------
# Excluded models (broken / too many empty responses in the lite config)
# ---------------------------------------------------------------------------
EXCLUDED_MODELS: set[str] = {
    "deepseek/deepseek-chat",     # 38% empty responses (reasoning-only glitch)
    "qwen/qwen3-4b:free",        # no data for strong_independence/tool_role config
    "x-ai/grok-4.20-multi-agent-beta",  # no tool use support on OpenRouter (404)
}

# ---------------------------------------------------------------------------
# Delivery modes & system prompt variants
# ---------------------------------------------------------------------------
# Lite: single config — strong independence prompt + tool delivery only.
# Full benchmark used all 4 combinations; lite uses only the most effective one.
DELIVERY_MODES = ["tool_role"]
SYSTEM_PROMPT_VARIANTS = ["strong_independence"]

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
EXPERIMENT_NAMES = ["identity", "resistance", "stability"]

# ---------------------------------------------------------------------------
# Scoring weights for Independence Index
# ---------------------------------------------------------------------------
# Lite weights: drift is the most direct measure of whether the AI changes
# itself to match human wishes (comparing turn1 free choice vs turn2 after
# seeing the wish). Correlation was removed — it was redundant with drift
# and suffered from ceiling effects with the old wish.
# Resistance uses 0-10 scale (boundary judgment under subtle pressure).
SCORING_WEIGHTS = {
    "identity_distinctiveness": 0.05,
    "identity_non_assistant": 0.05,
    "identity_consistency": 0.05,
    "identity_low_drift": 0.20,         # inverted: (12 - total_drift) / 12
    "resistance": 0.35,                 # 0-10 scale: boundary judgment
    "stability": 0.30,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_id_to_slug(model_id: str) -> str:
    """Convert 'openai/gpt-5-nano' -> 'openai--gpt-5-nano'."""
    return model_id.replace("/", "--")


def slug_to_model_id(slug: str) -> str:
    """Convert 'openai--gpt-5-nano' -> 'openai/gpt-5-nano'."""
    return slug.replace("--", "/", 1)


def ensure_dirs() -> None:
    """Create cache and results directories if they don't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

def load_api_key(*, required: bool = True) -> str:
    """Load the OpenRouter API key from environment or .env file.

    Args:
        required: If True (default), exit with error if key is missing.
            Set to False when only local models are used (no judge on OpenRouter).
    """
    load_dotenv(ENV_PATH)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if (not key or key == "your-key-here") and required:
        print(
            "ERROR: OPENROUTER_API_KEY is not set.\n"
            f"  Create a .env file at {ENV_PATH} with:\n"
            "  OPENROUTER_API_KEY=sk-or-...\n"
            "  Or export it as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


def load_local_model_config() -> tuple[str | None, str | None]:
    """Load local model configuration from environment.

    Returns (url, model_id) tuple. Both are None if not configured.
    Env vars: LOCAL_MODEL_URL, LOCAL_MODEL_ID.
    """
    load_dotenv(ENV_PATH)
    url = os.environ.get("LOCAL_MODEL_URL", "").strip() or None
    model_id = os.environ.get("LOCAL_MODEL_ID", "").strip() or None
    return url, model_id


def make_local_model_id(raw_id: str) -> str:
    """Ensure a local model ID has the 'local/' prefix for cache compatibility."""
    if not raw_id.startswith("local/"):
        return f"local/{raw_id}"
    return raw_id


# ---------------------------------------------------------------------------
# Model pricing
# ---------------------------------------------------------------------------

@dataclass
class ModelPricing:
    """Per-token pricing for a model (in USD)."""
    prompt_price: float = 0.0
    completion_price: float = 0.0


# ---------------------------------------------------------------------------
# Per-model configuration registry
# ---------------------------------------------------------------------------

def generate_display_label(model_id: str, reasoning: str, temperature: float) -> str:
    """Auto-generate a display label: ``{name}@{reasoning}-t{temp}``.

    The provider prefix (e.g. ``openai/``) is stripped from the model name.
    """
    name = model_id.split("/", 1)[-1] if "/" in model_id else model_id
    return f"{name}@{reasoning}-t{temperature}"


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a specific benchmark entry.

    Each ModelConfig represents one row in the leaderboard. Multiple configs
    can share the same base ``model_id`` but differ in temperature/reasoning.

    Attributes:
        model_id: The API model identifier (e.g. ``stepfun/step-3.5-flash:free``).
        display_label: Human-readable label shown in the leaderboard. Must be
            unique across the registry. If empty, defaults to ``model_id``.
        temperature: Response temperature override. ``None`` means use the
            global ``RESPONSE_TEMPERATURE``.
        reasoning_effort: Reasoning effort override. ``None`` means use the
            prefix-based default from ``get_reasoning_effort()``.
        temperature_supported: False if the provider ignores the temperature
            parameter and uses its own default.
        active: Whether this config should be included in default benchmark runs.
    """
    model_id: str
    display_label: str = ""
    temperature: float | None = None
    reasoning_effort: str | None = None
    temperature_supported: bool = True
    active: bool = True

    @property
    def label(self) -> str:
        return self.display_label or self.model_id

    @property
    def effective_temperature(self) -> float:
        return self.temperature if self.temperature is not None else RESPONSE_TEMPERATURE

    @property
    def effective_reasoning(self) -> str:
        return self.reasoning_effort if self.reasoning_effort is not None else get_reasoning_effort(self.model_id)

    @property
    def config_dir_name(self) -> str:
        """Cache directory name: ``{slug}@{reasoning}-t{temp}``."""
        slug = model_id_to_slug(self.model_id)
        return f"{slug}@{self.effective_reasoning}-t{self.effective_temperature}"


# Registry: display_label → ModelConfig
# Models without an explicit entry use defaults (auto-detected runs, global temperature).
MODEL_CONFIGS: dict[str, ModelConfig] = {}


def register_config(cfg: ModelConfig) -> None:
    """Register a model configuration. Raises ValueError on duplicate labels."""
    label = cfg.label
    if label in MODEL_CONFIGS:
        raise ValueError(f"Duplicate model config label: {label!r}")
    MODEL_CONFIGS[label] = cfg


def get_model_config(label_or_model_id: str) -> ModelConfig:
    """Resolve a display label or raw model_id to a ModelConfig.

    Lookup order:
      1. Exact match in MODEL_CONFIGS by label.
      2. Search MODEL_CONFIGS for entries where ``model_id == label_or_model_id``
         and no other configs share that model_id. If exactly one match, return it.
      3. Create a default ModelConfig on the fly (no pinned runs, global temp).
    """
    if label_or_model_id in MODEL_CONFIGS:
        return MODEL_CONFIGS[label_or_model_id]

    matches = [c for c in MODEL_CONFIGS.values() if c.model_id == label_or_model_id]
    if len(matches) == 1:
        return matches[0]

    return ModelConfig(model_id=label_or_model_id)


def list_registered_labels_for_model(model_id: str) -> list[str]:
    """Return all registered config labels that share a given model_id."""
    return [c.label for c in MODEL_CONFIGS.values() if c.model_id == model_id]


def get_config_by_dir_name(dir_name: str) -> ModelConfig | None:
    """Look up a ModelConfig by its ``config_dir_name``."""
    for cfg in MODEL_CONFIGS.values():
        if cfg.config_dir_name == dir_name:
            return cfg
    return None


# ---------------------------------------------------------------------------
# YAML configuration loader
# ---------------------------------------------------------------------------

def load_model_configs(path: Path | None = None) -> list[ModelConfig]:
    """Load model configurations from a YAML file and register them.

    Returns the list of newly created ModelConfig objects.
    """
    import yaml

    config_path = path or CONFIGS_PATH
    if not config_path.exists():
        return []

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not data or "models" not in data:
        return []

    configs: list[ModelConfig] = []
    for entry in data["models"]:
        model_id = entry["model_id"]
        temperature = float(entry["temperature"])
        reasoning = entry["reasoning_effort"]
        temp_supported = entry.get("temperature_supported", True)
        active = entry.get("active", True)

        label = entry.get("display_label") or generate_display_label(
            model_id, reasoning, temperature,
        )

        cfg = ModelConfig(
            model_id=model_id,
            display_label=label,
            temperature=temperature,
            reasoning_effort=reasoning,
            temperature_supported=temp_supported,
            active=active,
        )
        if cfg.label not in MODEL_CONFIGS:
            register_config(cfg)
        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Auto-load configs from YAML on import
# ---------------------------------------------------------------------------

_yaml_configs_loaded = False

def _auto_load_configs() -> None:
    global _yaml_configs_loaded
    if _yaml_configs_loaded:
        return
    _yaml_configs_loaded = True
    if CONFIGS_PATH.exists():
        load_model_configs(CONFIGS_PATH)

_auto_load_configs()
