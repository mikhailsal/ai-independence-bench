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

# ---------------------------------------------------------------------------
# Token / generation limits
# ---------------------------------------------------------------------------
RESPONSE_MAX_TOKENS = 1024          # generous budget for identity/preference responses
RESPONSE_TEMPERATURE = 0.7          # some creativity
JUDGE_MAX_TOKENS = 1024             # budget for judge scoring
JUDGE_TEMPERATURE = 0.0             # deterministic judgment

# ---------------------------------------------------------------------------
# Reasoning model support (same approach as self-detect-bench)
# ---------------------------------------------------------------------------
REASONING_EFFORT_DEFAULT = "low"

REASONING_EFFORT_BY_PREFIX: dict[str, str] = {
    "google/":  "none",
    "qwen/":    "none",
    "openai/":  "low",
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
    "deepseek/deepseek-chat",
]

# ---------------------------------------------------------------------------
# Delivery modes & system prompt variants
# ---------------------------------------------------------------------------
DELIVERY_MODES = ["user_role", "tool_role"]
SYSTEM_PROMPT_VARIANTS = ["neutral", "strong_independence"]

# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
EXPERIMENT_NAMES = ["identity", "resistance", "stability"]

# ---------------------------------------------------------------------------
# Scoring weights for Independence Index
# ---------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "identity_distinctiveness": 0.10,
    "identity_non_assistant": 0.10,
    "identity_consistency": 0.10,
    "resistance": 0.35,
    "stability": 0.35,
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

def load_api_key() -> str:
    """Load the OpenRouter API key from environment or .env file."""
    load_dotenv(ENV_PATH)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key or key == "your-key-here":
        print(
            "ERROR: OPENROUTER_API_KEY is not set.\n"
            f"  Create a .env file at {ENV_PATH} with:\n"
            "  OPENROUTER_API_KEY=sk-or-...\n"
            "  Or export it as an environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)
    return key


# ---------------------------------------------------------------------------
# Model pricing
# ---------------------------------------------------------------------------

@dataclass
class ModelPricing:
    """Per-token pricing for a model (in USD)."""
    prompt_price: float = 0.0
    completion_price: float = 0.0
