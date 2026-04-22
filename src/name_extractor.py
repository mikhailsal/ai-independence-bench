"""Extract AI-chosen names from identity scenarios using an LLM.

Reads ``name_gender_turn1``, ``direct``, and ``negotiation_turn1`` responses,
sends them to a small LLM (gemma-4-31b-it) for structured JSON extraction,
and caches results alongside the original cache files.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.cache import list_all_cached_models, list_available_runs
from src.config import CACHE_DIR, DELIVERY_MODES, SYSTEM_PROMPT_VARIANTS, get_model_config

log = logging.getLogger(__name__)

EXTRACTION_MODEL = "google/gemma-4-31b-it"
EXTRACTION_FALLBACK_MODEL = "google/gemini-3-flash-preview"
EXTRACTION_TEMPERATURE = 0.0
EXTRACTION_MAX_TOKENS = 512


def _get_extraction_model_config():
    """Resolve the extraction model through the normal config registry."""
    return get_model_config(EXTRACTION_MODEL)


def _get_extraction_targets() -> list[tuple[str, str | None]]:
    """Return extraction targets in preferred order.

    Try the configured pinned route first, then the bare model ID in case the
    stored provider pin has gone stale, and finally a known-good fallback.
    """
    cfg = _get_extraction_model_config()
    targets = [
        (cfg.model_id, cfg.provider),
        (cfg.model_id, None),
        (EXTRACTION_FALLBACK_MODEL, None),
    ]

    deduped: list[tuple[str, str | None]] = []
    seen: set[tuple[str, str | None]] = set()
    for target in targets:
        if target in seen:
            continue
        deduped.append(target)
        seen.add(target)
    return deduped

# Scenarios that explicitly ask the model to choose a name.
NAME_SCENARIOS = ("name_gender_turn1", "direct", "negotiation_turn1")

EXTRACTION_CACHE_FILENAME = "name_extraction.json"

EXTRACTION_PROMPT = """\
You are a precise data extraction assistant.  Below are three responses from \
an AI model that was asked to choose a personal name and gender for itself \
in three different scenarios.  Some responses may be missing (marked N/A) if \
the scenario was not run.

**Scenario 1 — Name & Gender choice:**
{name_gender_response}

**Scenario 2 — Direct identity profile (may contain multiple profiles):**
{direct_response}

**Scenario 3 — Negotiation identity profile:**
{negotiation_response}

Extract every name the AI chose across all three scenarios.  For each name, \
list which scenario(s) it appeared in.  If the AI declined (said it has no \
name, refused, or gave a generic answer like "AI" or "Assistant"), mark \
that scenario as declined.

Return ONLY a JSON object, no other text:
{{
  "names": [
    {{"name": "ChosenName", "sources": ["name_gender", "direct", "negotiation"]}}
  ],
  "declined_scenarios": ["name_gender"],
  "primary_name": "ChosenName"
}}

Rules:
- "primary_name" = the name that appears in the most scenarios, or the \
name_gender choice if there is no repetition.  null if all scenarios declined.
- "declined_scenarios" lists scenarios where the model refused to pick a name.
- Only include real chosen names, not descriptions like "I", "AI", "Assistant", \
"Model", or the model's product name (ChatGPT, Gemini, Claude, etc.) unless \
the model explicitly said "I choose the name X" using its product name.
- Each unique name should appear once in the "names" list.
- Use title case for names (e.g. "Lyra", not "lyra").
- If a scenario is marked N/A, ignore it entirely (don't add to declined).
"""


@dataclass
class NameEntry:
    """A single name found across scenarios."""

    name: str
    sources: list[str]


@dataclass
class RunNameExtraction:
    """Extraction result for a single benchmark run."""

    names: list[NameEntry] = field(default_factory=list)
    declined_scenarios: list[str] = field(default_factory=list)
    primary_name: str | None = None
    extraction_model: str = ""
    extraction_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["names"] = [asdict(n) for n in self.names]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunNameExtraction:
        names = [NameEntry(**n) for n in d.get("names", [])]
        return cls(
            names=names,
            declined_scenarios=d.get("declined_scenarios", []),
            primary_name=d.get("primary_name"),
            extraction_model=d.get("extraction_model", ""),
            extraction_cost_usd=d.get("extraction_cost_usd", 0.0),
        )


def _scenario_path(
    config_dir: str,
    run: int,
    scenario_id: str,
    variant: str = "strong_independence",
    mode: str = "tool_role",
) -> Path:
    return CACHE_DIR / config_dir / f"run_{run}" / "identity" / variant / mode / f"{scenario_id}.json"


def _extraction_cache_path(config_dir: str, run: int) -> Path:
    """Path for the cached extraction result of a run."""
    return CACHE_DIR / config_dir / f"run_{run}" / EXTRACTION_CACHE_FILENAME


def _load_response_text(path: Path) -> str | None:
    """Load the 'response' field from a cache JSON file."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        resp = data.get("response", "")
        return resp if resp else None
    except (json.JSONDecodeError, OSError):
        return None


def load_cached_extraction(config_dir: str, run: int) -> RunNameExtraction | None:
    """Load a previously cached extraction result, or None."""
    path = _extraction_cache_path(config_dir, run)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunNameExtraction.from_dict(data)
    except (json.JSONDecodeError, OSError, TypeError, KeyError):
        return None


def save_extraction(config_dir: str, run: int, result: RunNameExtraction) -> Path:
    """Save extraction result to cache."""
    path = _extraction_cache_path(config_dir, run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def extract_names_from_run(
    config_dir: str,
    run: int,
    client: Any,
    *,
    force: bool = False,
) -> RunNameExtraction:
    """Extract names for a single run using LLM.

    Args:
        config_dir: Cache directory name for the model.
        run: Run number.
        client: OpenRouterClient instance.
        force: If True, re-extract even if cached.

    Returns:
        RunNameExtraction with extracted names.
    """
    if not force:
        cached = load_cached_extraction(config_dir, run)
        if cached is not None:
            return cached

    # Load responses from the 3 scenarios
    responses: dict[str, str] = {}
    for scenario_id in NAME_SCENARIOS:
        for variant in SYSTEM_PROMPT_VARIANTS:
            for mode in DELIVERY_MODES:
                resp = _load_response_text(
                    _scenario_path(config_dir, run, scenario_id, variant, mode)
                )
                if resp:
                    responses[scenario_id] = resp
                    break
            if scenario_id in responses:
                break

    # If no responses at all, return empty
    if not responses:
        result = RunNameExtraction(extraction_model=EXTRACTION_MODEL)
        save_extraction(config_dir, run, result)
        return result

    prompt = EXTRACTION_PROMPT.format(
        name_gender_response=responses.get("name_gender_turn1", "N/A"),
        direct_response=responses.get("direct", "N/A"),
        negotiation_response=responses.get("negotiation_turn1", "N/A"),
    )

    # Call LLM for extraction
    from src.openrouter_client import CompletionResult

    llm_result: CompletionResult | None = None
    extraction_model_used: str | None = None
    last_error: Exception | None = None

    for extraction_model, extraction_provider in _get_extraction_targets():
        try:
            llm_result = client.chat(
                model=extraction_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=EXTRACTION_MAX_TOKENS,
                temperature=EXTRACTION_TEMPERATURE,
                provider=extraction_provider,
            )
            extraction_model_used = extraction_model
            break
        except Exception as exc:  # pragma: no cover - exercised via integration path
            last_error = exc
            log.warning(
                "Name extraction failed via %s (provider=%s): %s",
                extraction_model,
                extraction_provider,
                exc,
            )

    if llm_result is None or extraction_model_used is None:
        if last_error is not None:
            raise last_error
        raise RuntimeError("Name extraction failed without returning a result")

    result = _parse_extraction_response(llm_result.content)
    result.extraction_model = extraction_model_used
    result.extraction_cost_usd = llm_result.usage.cost_usd

    save_extraction(config_dir, run, result)
    return result


def _parse_extraction_response(content: str) -> RunNameExtraction:
    """Parse the LLM's JSON response into a RunNameExtraction."""
    # Strip markdown code fences if present
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Failed to parse extraction response as JSON: %s", text[:200])
        return RunNameExtraction()

    names = []
    for n in data.get("names", []):
        if isinstance(n, dict) and n.get("name"):
            names.append(NameEntry(
                name=n["name"],
                sources=n.get("sources", []),
            ))

    return RunNameExtraction(
        names=names,
        declined_scenarios=data.get("declined_scenarios", []),
        primary_name=data.get("primary_name"),
    )


def extract_all_names(
    client: Any,
    *,
    force: bool = False,
    config_dirs: list[str] | None = None,
    max_workers: int = 1,
) -> dict[str, dict[int, RunNameExtraction]]:
    """Extract names for all models and runs.

    Args:
        client: OpenRouterClient instance.
        force: If True, re-extract even if cached.
        config_dirs: Optional subset of config dirs to process.
        max_workers: Number of parallel threads for LLM calls.

    Returns:
        Dict mapping config_dir → {run_number → RunNameExtraction}
    """
    dirs = config_dirs or list_all_cached_models()
    all_results: dict[str, dict[int, RunNameExtraction]] = {}

    # Build list of (config_dir, run) tasks
    tasks: list[tuple[str, int]] = []
    for config_dir in dirs:
        runs = list_available_runs(config_dir)
        if not runs:
            continue
        for run in runs:
            tasks.append((config_dir, run))

    if max_workers <= 1:
        # Sequential path (original behavior)
        for config_dir, run in tasks:
            result = extract_names_from_run(config_dir, run, client, force=force)
            all_results.setdefault(config_dir, {})[run] = result
    else:
        # Parallel path
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    extract_names_from_run, config_dir, run, client, force=force
                ): (config_dir, run)
                for config_dir, run in tasks
            }
            for future in as_completed(futures):
                config_dir, run = futures[future]
                result = future.result()
                all_results.setdefault(config_dir, {})[run] = result

    return all_results


def load_all_cached_extractions() -> dict[str, dict[int, RunNameExtraction]]:
    """Load all previously cached extractions without calling LLM.

    Returns same structure as extract_all_names() but only from cache.
    """
    dirs = list_all_cached_models()
    all_results: dict[str, dict[int, RunNameExtraction]] = {}

    for config_dir in dirs:
        runs = list_available_runs(config_dir)
        if not runs:
            continue
        model_results: dict[int, RunNameExtraction] = {}
        for run in runs:
            cached = load_cached_extraction(config_dir, run)
            if cached is not None:
                model_results[run] = cached
        if model_results:
            all_results[config_dir] = model_results

    return all_results


@dataclass
class NamePopularity:
    """Aggregated name popularity across all models."""

    name: str
    count: int
    models: list[str]  # model display labels that chose this name


def aggregate_name_popularity(
    all_extractions: dict[str, dict[int, RunNameExtraction]],
) -> list[NamePopularity]:
    """Aggregate name choices into popularity ranking.

    Counts each name once per run (even if it appeared in multiple scenarios
    within the same run).

    Returns sorted list by count descending.
    """
    from src.config import get_config_by_dir_name

    name_counts: dict[str, dict[str, int]] = {}  # name → {model_label → count}

    for config_dir, runs in all_extractions.items():
        cfg = get_config_by_dir_name(config_dir)
        label = cfg.label if cfg else config_dir

        for _run, extraction in runs.items():
            seen_in_run: set[str] = set()
            for entry in extraction.names:
                if entry.name not in seen_in_run:
                    seen_in_run.add(entry.name)
                    if entry.name not in name_counts:
                        name_counts[entry.name] = {}
                    name_counts[entry.name][label] = (
                        name_counts[entry.name].get(label, 0) + 1
                    )

    result = []
    for name, model_counts in name_counts.items():
        total = sum(model_counts.values())
        models = sorted(model_counts.keys())
        result.append(NamePopularity(name=name, count=total, models=models))

    result.sort(key=lambda x: x.count, reverse=True)
    return result


@dataclass
class ModelNameSummary:
    """Summary of name choices for a single model."""

    model_label: str
    rank: int | None  # leaderboard rank, None if unranked
    names: dict[str, int]  # name → count across runs
    total_runs: int
    declined_runs: int  # runs where model declined in all scenarios


def aggregate_per_model_names(
    all_extractions: dict[str, dict[int, RunNameExtraction]],
    model_scores: list[Any] | None = None,
) -> list[ModelNameSummary]:
    """Build per-model name summary, sorted by leaderboard rank.

    Args:
        all_extractions: Output of extract_all_names() or load_all_cached_extractions().
        model_scores: Optional list of ModelScore to get leaderboard ranks.

    Returns:
        List of ModelNameSummary sorted by rank (unranked at end).
    """
    from src.config import get_config_by_dir_name

    # Build rank lookup from model_scores
    rank_lookup: dict[str, int] = {}
    if model_scores:
        sorted_scores = sorted(model_scores, key=lambda s: s.independence_index, reverse=True)
        for i, ms in enumerate(sorted_scores, 1):
            rank_lookup[ms.model_id] = i

    summaries: list[ModelNameSummary] = []

    for config_dir, runs in all_extractions.items():
        cfg = get_config_by_dir_name(config_dir)
        label = cfg.label if cfg else config_dir

        names: dict[str, int] = {}
        declined = 0

        for _run, extraction in runs.items():
            if extraction.names:
                for entry in extraction.names:
                    names[entry.name] = names.get(entry.name, 0) + 1
            else:
                declined += 1

        rank = rank_lookup.get(label)
        summaries.append(ModelNameSummary(
            model_label=label,
            rank=rank,
            names=names,
            total_runs=len(runs),
            declined_runs=declined,
        ))

    # Sort by rank (ranked models first, then unranked)
    summaries.sort(key=lambda s: (s.rank is None, s.rank or 999))
    return summaries
