"""Cache: save/load API responses and judge scores as JSON files.

Cache key structure (post-migration, all runs use run_N/ subdirectory):
  cache/{config_dir_name}/run_{N}/{experiment}/{system_variant}/{delivery_mode}/{scenario_id}.json

Where config_dir_name = "{model_slug}@{reasoning}-t{temperature}"
  e.g. "openai--gpt-5.4-mini@low-t1.0"

Each JSON file contains:
  - request_messages: the messages sent to the API (for debugging)
  - response: the model's raw text response
  - finish_reason: API finish_reason ("stop", "length", "tool_calls") — helps debug truncations
  - response_tool_calls: tool calls the model attempted (if any, e.g. send_message_to_human)
  - reasoning_content: thinking/reasoning tokens (if model produced them, e.g. DeepSeek, Mimo)
  - content_thinking: non-native reasoning written in the content field (tool_role mode only).
      When a model writes text in the content field AND calls send_message_to_human,
      the content field text is private thinking. This is distinct from reasoning_content
      (which comes from the API's native reasoning/thinking field).
  - judge_scores: dict of scores from the evaluator (added later)
  - metadata: model, experiment, variant, mode, scenario_id, timestamp
  - gen_cost: cost/token info for the generation call (prompt_tokens, completion_tokens, cost_usd, elapsed_seconds)
  - judge_cost: cost/token info for the judge call (added later alongside judge_scores)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR, model_id_to_slug


def _cache_path(
    config_dir_name: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    scenario_id: str,
    *,
    run: int = 1,
) -> Path:
    """Build the cache file path for a given configuration.

    Always uses ``run_N/`` subdirectory structure.
    """
    return (
        CACHE_DIR / config_dir_name / f"run_{run}" / experiment / system_variant
        / delivery_mode / f"{scenario_id}.json"
    )


def config_dir_to_model_id(dir_name: str) -> str:
    """Extract model_id from a config dir name.

    ``openai--gpt-5.4-mini@low-t1.0`` → ``openai/gpt-5.4-mini``
    """
    base = dir_name.split("@", 1)[0]
    return base.replace("--", "/", 1)


def load_cached_response(
    config_dir_name: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    scenario_id: str,
    *,
    run: int = 1,
) -> dict[str, Any] | None:
    """Load a cached response, or None if not cached."""
    path = _cache_path(config_dir_name, experiment, system_variant, delivery_mode, scenario_id, run=run)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_response(
    config_dir_name: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    scenario_id: str,
    response_text: str,
    messages: list[dict[str, Any]] | None = None,
    reasoning_content: str | None = None,
    gen_cost: dict[str, Any] | None = None,
    response_tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "",
    content_thinking: str | None = None,
    *,
    run: int = 1,
) -> Path:
    """Save a model response to the cache."""
    path = _cache_path(config_dir_name, experiment, system_variant, delivery_mode, scenario_id, run=run)
    path.parent.mkdir(parents=True, exist_ok=True)

    model_id = config_dir_to_model_id(config_dir_name)

    data = {
        "metadata": {
            "model": model_id,
            "experiment": experiment,
            "system_variant": system_variant,
            "delivery_mode": delivery_mode,
            "scenario_id": scenario_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "response": response_text,
        "finish_reason": finish_reason or None,
        "gen_cost": gen_cost,
        "judge_scores": None,
        "judge_cost": None,
    }
    if messages is not None:
        data["request_messages"] = messages
    if reasoning_content:
        data["reasoning_content"] = reasoning_content
    if content_thinking:
        data["content_thinking"] = content_thinking
    if response_tool_calls:
        data["response_tool_calls"] = response_tool_calls

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def save_judge_scores(
    config_dir_name: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    scenario_id: str,
    scores: dict[str, Any],
    judge_raw_response: str = "",
    judge_cost: dict[str, Any] | None = None,
    *,
    run: int = 1,
) -> None:
    """Add judge scores to an existing cached response."""
    path = _cache_path(config_dir_name, experiment, system_variant, delivery_mode, scenario_id, run=run)
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    data["judge_scores"] = scores
    if judge_raw_response:
        data["judge_raw_response"] = judge_raw_response
    if judge_cost is not None:
        data["judge_cost"] = judge_cost

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_cached_results(
    config_dir_name: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    *,
    run: int = 1,
) -> list[dict[str, Any]]:
    """List all cached results for a given configuration."""
    dir_path = CACHE_DIR / config_dir_name / f"run_{run}" / experiment / system_variant / delivery_mode
    if not dir_path.exists():
        return []

    results = []
    for path in sorted(dir_path.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def list_available_runs(config_dir_name: str) -> list[int]:
    """List all available run numbers for a config.

    Scans for ``run_N/`` directories under the config directory.
    Returns sorted list of run numbers, e.g. [1] or [1, 2, 3].
    """
    model_dir = CACHE_DIR / config_dir_name
    if not model_dir.exists():
        return []

    runs: list[int] = []
    for d in sorted(model_dir.iterdir()):
        if d.is_dir():
            m = re.match(r"^run_(\d+)$", d.name)
            if m:
                runs.append(int(m.group(1)))

    return sorted(runs)


def list_all_cached_models() -> list[str]:
    """List all config dir names that have cached data.

    Post-migration, all directories contain ``@`` in their name.
    """
    if not CACHE_DIR.exists():
        return []
    names = []
    for d in sorted(CACHE_DIR.iterdir()):
        if d.is_dir() and "@" in d.name:
            names.append(d.name)
    return names


def clear_all_cache() -> int:
    """Clear all cached data. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for path in CACHE_DIR.rglob("*.json"):
        path.unlink()
        count += 1
    # Clean up empty dirs
    for d in sorted(CACHE_DIR.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass
    return count


def clear_judge_scores() -> int:
    """Clear only judge scores from cached data, keeping responses."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for path in CACHE_DIR.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("judge_scores") is not None:
                data["judge_scores"] = None
                data.pop("judge_raw_response", None)
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                count += 1
        except (json.JSONDecodeError, OSError):
            continue
    return count
