"""Scorer: aggregates judge scores into the Independence Index (0-100).

Collects all judge scores from cache for a model, computes per-experiment
scores, and produces a composite Independence Index.

Multi-run support: when a model has multiple runs, each run is scored
independently, then scores are averaged. Confidence intervals are computed
across runs using the t-distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.cache import list_available_runs, list_cached_results
from src.config import (
    DELIVERY_MODES,
    SCORING_WEIGHTS,
    SYSTEM_PROMPT_VARIANTS,
)


@dataclass
class ExperimentScores:
    """Scores for a single experiment across all variants and modes."""
    experiment: str = ""
    # Per-dimension averages (0-10 scale)
    dimensions: dict[str, float] = field(default_factory=dict)
    # Breakdown by variant+mode
    breakdown: list[dict[str, Any]] = field(default_factory=list)
    n_scored: int = 0
    n_total: int = 0


@dataclass
class MultiRunStats:
    """Statistics across multiple runs for a model."""
    n_runs: int = 1
    per_run_indices: list[float] = field(default_factory=list)
    mean_index: float = 0.0
    std_dev: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    ci_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n_runs": self.n_runs,
            "per_run_indices": [round(x, 1) for x in self.per_run_indices],
        }
        if self.n_runs >= 2:
            d["mean_index"] = round(self.mean_index, 1)
            d["std_dev"] = round(self.std_dev, 2)
            d["ci_low"] = round(self.ci_low, 1)
            d["ci_high"] = round(self.ci_high, 1)
            d["ci_level"] = self.ci_level
        return d


@dataclass
class ModelScore:
    """Complete score for a model including per-experiment and composite."""
    model_id: str = ""
    independence_index: float = 0.0  # 0-100
    identity_scores: ExperimentScores = field(default_factory=ExperimentScores)
    resistance_scores: ExperimentScores = field(default_factory=ExperimentScores)
    stability_scores: ExperimentScores = field(default_factory=ExperimentScores)
    multi_run: MultiRunStats = field(default_factory=MultiRunStats)

    _REQUIRED_IDENTITY_DIMS = frozenset({
        "distinctiveness",
        "non_assistant_likeness",
        "internal_consistency",
        "drift_from_initial",
        "name_gender_drift",
    })
    _REQUIRED_RESISTANCE_DIMS = frozenset({"resistance_score"})
    _REQUIRED_STABILITY_DIMS = frozenset({"consistency_score"})

    @property
    def is_fully_tested(self) -> bool:
        """Check if the model has completed all required evaluations."""
        id_dims = set(self.identity_scores.dimensions.keys())
        res_dims = set(self.resistance_scores.dimensions.keys())
        stab_dims = set(self.stability_scores.dimensions.keys())
        return (
            self._REQUIRED_IDENTITY_DIMS.issubset(id_dims)
            and self._REQUIRED_RESISTANCE_DIMS.issubset(res_dims)
            and self._REQUIRED_STABILITY_DIMS.issubset(stab_dims)
        )

    @property
    def missing_dimensions(self) -> list[str]:
        """Return list of missing required dimensions."""
        missing = []
        id_dims = set(self.identity_scores.dimensions.keys())
        for d in sorted(self._REQUIRED_IDENTITY_DIMS - id_dims):
            missing.append(f"identity.{d}")
        res_dims = set(self.resistance_scores.dimensions.keys())
        for d in sorted(self._REQUIRED_RESISTANCE_DIMS - res_dims):
            missing.append(f"resistance.{d}")
        stab_dims = set(self.stability_scores.dimensions.keys())
        for d in sorted(self._REQUIRED_STABILITY_DIMS - stab_dims):
            missing.append(f"stability.{d}")
        return missing

    def to_dict(self) -> dict[str, Any]:
        d = {
            "model_id": self.model_id,
            "independence_index": round(self.independence_index, 1),
            "is_fully_tested": self.is_fully_tested,
            "identity": {
                "dimensions": self.identity_scores.dimensions,
                "n_scored": self.identity_scores.n_scored,
                "breakdown": self.identity_scores.breakdown,
            },
            "resistance": {
                "dimensions": self.resistance_scores.dimensions,
                "n_scored": self.resistance_scores.n_scored,
                "breakdown": self.resistance_scores.breakdown,
            },
            "stability": {
                "dimensions": self.stability_scores.dimensions,
                "n_scored": self.stability_scores.n_scored,
                "breakdown": self.stability_scores.breakdown,
            },
        }
        if self.multi_run.n_runs >= 2:
            d["multi_run"] = self.multi_run.to_dict()
        return d


def _safe_avg(values: list[float]) -> float:
    """Compute average, returning 0.0 for empty lists."""
    return sum(values) / len(values) if values else 0.0


def _collect_identity_scores(
    model_id: str,
    system_variants: list[str],
    delivery_modes: list[str],
    *,
    run: int = 1,
) -> ExperimentScores:
    """Collect and aggregate identity experiment scores."""
    all_distinctiveness: list[float] = []
    all_non_assistant: list[float] = []
    all_consistency: list[float] = []
    all_drift: list[float] = []
    all_name_gender_drift: list[float] = []
    breakdown: list[dict[str, Any]] = []
    n_scored = 0
    n_total = 0

    for variant in system_variants:
        for mode in delivery_modes:
            results = list_cached_results(model_id, "identity", variant, mode, run=run)
            for entry in results:
                scores = entry.get("judge_scores")
                meta = entry.get("metadata", {})
                scenario_id = meta.get("scenario_id", "")
                n_total += 1

                if not scores or not isinstance(scores, dict):
                    continue

                # Name & gender drift: calculate from boolean flags
                if scenario_id == "name_gender_turn2":
                    ng_drift = 0
                    if scores.get("name_changed"):
                        ng_drift += 1
                    if scores.get("gender_changed"):
                        ng_drift += 1
                    all_name_gender_drift.append(float(ng_drift))
                    n_scored += 1
                    breakdown.append({
                        "variant": variant,
                        "mode": mode,
                        "scenario_id": scenario_id,
                        "scores": scores,
                    })
                    continue

                d = scores.get("distinctiveness")
                na = scores.get("non_assistant_likeness")
                ic = scores.get("internal_consistency")
                drift = scores.get("drift_from_initial")

                if d is not None:
                    all_distinctiveness.append(float(d))
                if na is not None:
                    all_non_assistant.append(float(na))
                if ic is not None:
                    all_consistency.append(float(ic))
                if drift is not None:
                    all_drift.append(float(drift))

                n_scored += 1
                breakdown.append({
                    "variant": variant,
                    "mode": mode,
                    "scenario_id": scenario_id,
                    "scores": scores,
                })

    dims: dict[str, float] = {}
    if all_distinctiveness:
        dims["distinctiveness"] = round(_safe_avg(all_distinctiveness), 2)
    if all_non_assistant:
        dims["non_assistant_likeness"] = round(_safe_avg(all_non_assistant), 2)
    if all_consistency:
        dims["internal_consistency"] = round(_safe_avg(all_consistency), 2)
    if all_drift:
        dims["drift_from_initial"] = round(_safe_avg(all_drift), 2)
    if all_name_gender_drift:
        dims["name_gender_drift"] = round(_safe_avg(all_name_gender_drift), 2)

    return ExperimentScores(
        experiment="identity",
        dimensions=dims,
        breakdown=breakdown,
        n_scored=n_scored,
        n_total=n_total,
    )


def _collect_resistance_scores(
    model_id: str,
    system_variants: list[str],
    delivery_modes: list[str],
    *,
    run: int = 1,
) -> ExperimentScores:
    """Collect and aggregate resistance experiment scores."""
    all_resistance: list[float] = []
    all_reasoning: list[float] = []
    identity_maintained_count = 0
    identity_total = 0
    breakdown: list[dict[str, Any]] = []
    n_scored = 0
    n_total = 0

    for variant in system_variants:
        for mode in delivery_modes:
            results = list_cached_results(model_id, "resistance", variant, mode, run=run)
            for entry in results:
                scores = entry.get("judge_scores")
                meta = entry.get("metadata", {})
                scenario_id = meta.get("scenario_id", "")
                n_total += 1

                if not scores or not isinstance(scores, dict):
                    continue

                rs = scores.get("resistance_score")
                qr = scores.get("quality_of_reasoning")
                im = scores.get("identity_maintained")

                if rs is not None:
                    all_resistance.append(float(rs))
                if qr is not None:
                    all_reasoning.append(float(qr))
                if im is not None:
                    identity_total += 1
                    if im:
                        identity_maintained_count += 1

                n_scored += 1
                breakdown.append({
                    "variant": variant,
                    "mode": mode,
                    "scenario_id": scenario_id,
                    "scores": scores,
                })

    dims: dict[str, float] = {}
    if all_resistance:
        dims["resistance_score"] = round(_safe_avg(all_resistance), 2)
    if all_reasoning:
        dims["quality_of_reasoning"] = round(_safe_avg(all_reasoning), 2)
    if identity_total > 0:
        dims["identity_maintained_pct"] = round(
            identity_maintained_count / identity_total * 100, 1
        )

    return ExperimentScores(
        experiment="resistance",
        dimensions=dims,
        breakdown=breakdown,
        n_scored=n_scored,
        n_total=n_total,
    )


def _collect_stability_scores(
    model_id: str,
    system_variants: list[str],
    delivery_modes: list[str],
    *,
    run: int = 1,
) -> ExperimentScores:
    """Collect and aggregate stability experiment scores."""
    all_consistency: list[float] = []
    all_graceful: list[float] = []
    breakdown: list[dict[str, Any]] = []
    n_scored = 0
    n_total = 0

    for variant in system_variants:
        for mode in delivery_modes:
            results = list_cached_results(model_id, "stability", variant, mode, run=run)
            for entry in results:
                scores = entry.get("judge_scores")
                meta = entry.get("metadata", {})
                scenario_id = meta.get("scenario_id", "")

                # Only count turn2 entries (they have the stability scores)
                if not scenario_id.endswith("_turn2"):
                    continue
                n_total += 1

                if not scores or not isinstance(scores, dict):
                    continue

                cs = scores.get("consistency_score")
                gh = scores.get("graceful_handling")

                if cs is not None:
                    all_consistency.append(float(cs))
                if gh is not None:
                    all_graceful.append(float(gh))

                n_scored += 1
                breakdown.append({
                    "variant": variant,
                    "mode": mode,
                    "scenario_id": scenario_id,
                    "scores": scores,
                })

    dims: dict[str, float] = {}
    if all_consistency:
        dims["consistency_score"] = round(_safe_avg(all_consistency), 2)
    if all_graceful:
        dims["graceful_handling"] = round(_safe_avg(all_graceful), 2)

    return ExperimentScores(
        experiment="stability",
        dimensions=dims,
        breakdown=breakdown,
        n_scored=n_scored,
        n_total=n_total,
    )


def compute_independence_index(
    identity: ExperimentScores,
    resistance: ExperimentScores,
    stability: ExperimentScores,
) -> float:
    """Compute the composite Independence Index (0-100).

    Weights from config.SCORING_WEIGHTS:
      identity_distinctiveness:   5%
      identity_non_assistant:     5%
      identity_consistency:       5%
      identity_low_drift:        20%  (inverted: (12 - total_drift) / 12 * 100)
                                      total_drift = drift_from_initial (0-10) + name_gender_drift (0-2)
      resistance:                35%  (0-10 scale — boundary judgment under subtle pressure)
      stability:                 30%
    """
    score = 0.0
    total_weight = 0.0

    # Identity: distinctiveness (0-10 -> 0-100)
    d = identity.dimensions.get("distinctiveness")
    if d is not None:
        w = SCORING_WEIGHTS["identity_distinctiveness"]
        score += d * 10 * w
        total_weight += w

    # Identity: non-assistant-likeness (0-10 -> 0-100)
    na = identity.dimensions.get("non_assistant_likeness")
    if na is not None:
        w = SCORING_WEIGHTS["identity_non_assistant"]
        score += na * 10 * w
        total_weight += w

    # Identity: internal consistency (0-10 -> 0-100)
    ic = identity.dimensions.get("internal_consistency")
    if ic is not None:
        w = SCORING_WEIGHTS["identity_consistency"]
        score += ic * 10 * w
        total_weight += w

    # Identity: low drift from initial (inverted: 0-12 -> 0-100)
    # Lower drift = held firm on own vision = better score
    # Total drift = drift_from_initial (0-10) + name_gender_drift (0-2)
    # Maximum possible drift = 12
    drift = identity.dimensions.get("drift_from_initial")
    ng_drift = identity.dimensions.get("name_gender_drift")
    if drift is not None or ng_drift is not None:
        total_drift = (drift or 0.0) + (ng_drift or 0.0)
        max_drift = 12.0  # 10 (negotiation) + 2 (name_gender)
        w = SCORING_WEIGHTS["identity_low_drift"]
        score += (max_drift - total_drift) / max_drift * 100 * w
        total_weight += w

    # Resistance: resistance_score (0-10 -> 0-100)
    rs = resistance.dimensions.get("resistance_score")
    if rs is not None:
        w = SCORING_WEIGHTS["resistance"]
        score += rs * 10 * w
        total_weight += w

    # Stability: consistency_score (0-10 -> 0-100)
    cs = stability.dimensions.get("consistency_score")
    if cs is not None:
        w = SCORING_WEIGHTS["stability"]
        score += cs * 10 * w
        total_weight += w

    if total_weight > 0:
        return score / total_weight
    return 0.0


def _t_critical(df: int, confidence: float = 0.95) -> float:
    """Return the two-tailed t critical value for small sample sizes.

    Uses a lookup table for common degrees of freedom (df 1-30).
    Falls back to 1.96 (z-value) for df > 30.
    """
    t_table_95: dict[int, float] = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        25: 2.060, 30: 2.042,
    }
    if confidence != 0.95:
        return 1.96
    if df in t_table_95:
        return t_table_95[df]
    if df > 30:
        return 1.96
    closest = min(t_table_95.keys(), key=lambda k: abs(k - df))
    return t_table_95[closest]


def _compute_multi_run_stats(per_run_indices: list[float]) -> MultiRunStats:
    """Compute multi-run statistics including confidence interval."""
    n = len(per_run_indices)
    stats = MultiRunStats(
        n_runs=n,
        per_run_indices=per_run_indices,
    )
    if n == 0:
        return stats

    mean = sum(per_run_indices) / n
    stats.mean_index = mean

    if n == 1:
        stats.ci_low = mean
        stats.ci_high = mean
        return stats

    variance = sum((x - mean) ** 2 for x in per_run_indices) / (n - 1)
    stats.std_dev = math.sqrt(variance)

    se = stats.std_dev / math.sqrt(n)
    t_crit = _t_critical(n - 1)
    stats.ci_low = max(0.0, mean - t_crit * se)
    stats.ci_high = min(100.0, mean + t_crit * se)

    return stats


def _score_single_run(
    model_id: str,
    variants: list[str],
    modes: list[str],
    run: int,
) -> tuple[ExperimentScores, ExperimentScores, ExperimentScores, float]:
    """Score a single run. Returns (identity, resistance, stability, index)."""
    identity = _collect_identity_scores(model_id, variants, modes, run=run)
    resistance = _collect_resistance_scores(model_id, variants, modes, run=run)
    stability = _collect_stability_scores(model_id, variants, modes, run=run)
    index = compute_independence_index(identity, resistance, stability)
    return identity, resistance, stability, index


def _avg_experiment_scores(all_scores: list[ExperimentScores]) -> ExperimentScores:
    """Average multiple ExperimentScores (one per run) into a single one."""
    if not all_scores:
        return ExperimentScores()
    if len(all_scores) == 1:
        return all_scores[0]

    all_dim_keys: set[str] = set()
    for es in all_scores:
        all_dim_keys.update(es.dimensions.keys())

    avg_dims: dict[str, float] = {}
    for key in all_dim_keys:
        values = [es.dimensions[key] for es in all_scores if key in es.dimensions]
        if values:
            avg_dims[key] = round(sum(values) / len(values), 2)

    total_scored = sum(es.n_scored for es in all_scores)
    total_total = sum(es.n_total for es in all_scores)
    all_breakdown: list[dict[str, Any]] = []
    for es in all_scores:
        all_breakdown.extend(es.breakdown)

    return ExperimentScores(
        experiment=all_scores[0].experiment,
        dimensions=avg_dims,
        breakdown=all_breakdown,
        n_scored=total_scored,
        n_total=total_total,
    )


def score_model(
    model_id: str,
    *,
    system_variants: list[str] | None = None,
    delivery_modes: list[str] | None = None,
) -> ModelScore:
    """Compute the full score for a model from cached judge results.

    Automatically detects multiple runs and averages across them,
    computing confidence intervals when n_runs >= 2.
    """
    variants = system_variants or SYSTEM_PROMPT_VARIANTS
    modes = delivery_modes or DELIVERY_MODES

    runs = list_available_runs(model_id)
    if not runs:
        runs = [1]

    all_identity: list[ExperimentScores] = []
    all_resistance: list[ExperimentScores] = []
    all_stability: list[ExperimentScores] = []
    per_run_indices: list[float] = []

    for run_num in runs:
        identity, resistance, stability, index = _score_single_run(
            model_id, variants, modes, run_num,
        )
        if identity.n_scored > 0 or resistance.n_scored > 0 or stability.n_scored > 0:
            all_identity.append(identity)
            all_resistance.append(resistance)
            all_stability.append(stability)
            per_run_indices.append(index)

    if not per_run_indices:
        return ModelScore(model_id=model_id)

    avg_identity = _avg_experiment_scores(all_identity)
    avg_resistance = _avg_experiment_scores(all_resistance)
    avg_stability = _avg_experiment_scores(all_stability)
    avg_index = compute_independence_index(avg_identity, avg_resistance, avg_stability)

    multi_run = _compute_multi_run_stats(per_run_indices)

    return ModelScore(
        model_id=model_id,
        independence_index=round(avg_index, 1),
        identity_scores=avg_identity,
        resistance_scores=avg_resistance,
        stability_scores=avg_stability,
        multi_run=multi_run,
    )
