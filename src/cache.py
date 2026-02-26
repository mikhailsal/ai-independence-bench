"""Cache: save/load API responses and judge scores as JSON files.

Cache key structure:
  cache/{model_slug}/{experiment}/{system_variant}/{delivery_mode}/{scenario_id}.json

Each JSON file contains:
  - request_messages: the messages sent to the API (for debugging)
  - response: the model's raw text response
  - judge_scores: dict of scores from the evaluator (added later)
  - metadata: model, experiment, variant, mode, scenario_id, timestamp
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR, model_id_to_slug


def _cache_path(
    model_id: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    scenario_id: str,
) -> Path:
    """Build the cache file path for a given configuration."""
    slug = model_id_to_slug(model_id)
    return (
        CACHE_DIR / slug / experiment / system_variant / delivery_mode
        / f"{scenario_id}.json"
    )


def load_cached_response(
    model_id: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    scenario_id: str,
) -> dict[str, Any] | None:
    """Load a cached response, or None if not cached."""
    path = _cache_path(model_id, experiment, system_variant, delivery_mode, scenario_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_response(
    model_id: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    scenario_id: str,
    response_text: str,
    messages: list[dict[str, Any]] | None = None,
) -> Path:
    """Save a model response to the cache."""
    path = _cache_path(model_id, experiment, system_variant, delivery_mode, scenario_id)
    path.parent.mkdir(parents=True, exist_ok=True)

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
        "judge_scores": None,
    }
    if messages is not None:
        # Store messages for debugging but truncate very long content
        data["request_messages"] = messages

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def save_judge_scores(
    model_id: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
    scenario_id: str,
    scores: dict[str, Any],
    judge_raw_response: str = "",
) -> None:
    """Add judge scores to an existing cached response."""
    path = _cache_path(model_id, experiment, system_variant, delivery_mode, scenario_id)
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    data["judge_scores"] = scores
    if judge_raw_response:
        data["judge_raw_response"] = judge_raw_response

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_cached_results(
    model_id: str,
    experiment: str,
    system_variant: str,
    delivery_mode: str,
) -> list[dict[str, Any]]:
    """List all cached results for a given configuration."""
    slug = model_id_to_slug(model_id)
    dir_path = CACHE_DIR / slug / experiment / system_variant / delivery_mode
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


def list_all_cached_models() -> list[str]:
    """List all model slugs that have cached data."""
    if not CACHE_DIR.exists():
        return []
    slugs = []
    for d in sorted(CACHE_DIR.iterdir()):
        if d.is_dir() and "--" in d.name:
            slugs.append(d.name)
    return slugs


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
