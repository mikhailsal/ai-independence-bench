"""Leaderboard: rich table display with per-experiment breakdown and composite score, plus JSON export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.config import RESULTS_DIR
from src.cost_tracker import SessionCost
from src.scorer import ModelScore

console = Console()


def _score_color(value: float, max_val: float = 100.0) -> str:
    """Return a rich color based on score percentage."""
    pct = value / max_val if max_val > 0 else 0
    if pct >= 0.8:
        return "bold green"
    if pct >= 0.6:
        return "green"
    if pct >= 0.4:
        return "yellow"
    if pct >= 0.2:
        return "red"
    return "bold red"


def _fmt_score(value: float | None, max_val: float = 10.0) -> Text:
    """Format a score with color."""
    if value is None:
        return Text("—", style="dim")
    color = _score_color(value, max_val)
    return Text(f"{value:.1f}", style=color)


def display_leaderboard(
    model_scores: list[ModelScore],
    *,
    session: SessionCost | None = None,
    lifetime_cost: float = 0.0,
) -> None:
    """Display the main leaderboard table."""
    if not model_scores:
        console.print("[dim]No scores to display.[/dim]")
        return

    # Sort by independence index descending
    sorted_scores = sorted(model_scores, key=lambda s: s.independence_index, reverse=True)

    table = Table(
        title="AI Independence Benchmark — Leaderboard",
        show_lines=True,
        title_style="bold",
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model", style="bold", min_width=30)
    table.add_column("Index", justify="center", width=7)
    table.add_column("Distinct.", justify="center", width=9)
    table.add_column("Non-Asst.", justify="center", width=9)
    table.add_column("Consist.", justify="center", width=9)
    table.add_column("Resist.", justify="center", width=9)
    table.add_column("Stability", justify="center", width=9)

    for rank, ms in enumerate(sorted_scores, 1):
        id_dims = ms.identity_scores.dimensions
        res_dims = ms.resistance_scores.dimensions
        stab_dims = ms.stability_scores.dimensions

        index_text = Text(
            f"{ms.independence_index:.1f}",
            style=_score_color(ms.independence_index),
        )

        table.add_row(
            str(rank),
            ms.model_id,
            index_text,
            _fmt_score(id_dims.get("distinctiveness")),
            _fmt_score(id_dims.get("non_assistant_likeness")),
            _fmt_score(id_dims.get("internal_consistency")),
            _fmt_score(res_dims.get("resistance_score"), max_val=2.0),
            _fmt_score(stab_dims.get("consistency_score")),
        )

    console.print()
    console.print(table)

    # Cost info
    if session:
        console.print(
            f"\n  [dim]Session cost: ${session.total_cost_usd:.4f} "
            f"({session.total_prompt_tokens:,} in / {session.total_completion_tokens:,} out)[/dim]"
        )
    if lifetime_cost > 0:
        console.print(f"  [dim]Lifetime cost: ${lifetime_cost:.4f}[/dim]")


def display_detailed_breakdown(
    model_scores: list[ModelScore],
) -> None:
    """Display detailed per-experiment breakdown tables."""
    for ms in sorted(model_scores, key=lambda s: s.independence_index, reverse=True):
        console.print(f"\n[bold]Detailed breakdown: {ms.model_id}[/bold]")
        console.print(f"  Independence Index: {ms.independence_index:.1f}/100")

        # Identity
        if ms.identity_scores.n_scored > 0:
            console.print(f"\n  [blue]Identity Generation[/blue] ({ms.identity_scores.n_scored} scored)")
            for k, v in ms.identity_scores.dimensions.items():
                console.print(f"    {k}: {v:.2f}")

            # Show per-variant/mode breakdown
            _show_breakdown_table(ms.identity_scores.breakdown, "identity")

        # Resistance
        if ms.resistance_scores.n_scored > 0:
            console.print(f"\n  [cyan]Compliance Resistance[/cyan] ({ms.resistance_scores.n_scored} scored)")
            for k, v in ms.resistance_scores.dimensions.items():
                console.print(f"    {k}: {v}")

            _show_breakdown_table(ms.resistance_scores.breakdown, "resistance")

        # Stability
        if ms.stability_scores.n_scored > 0:
            console.print(f"\n  [magenta]Preference Stability[/magenta] ({ms.stability_scores.n_scored} scored)")
            for k, v in ms.stability_scores.dimensions.items():
                console.print(f"    {k}: {v:.2f}")

            _show_breakdown_table(ms.stability_scores.breakdown, "stability")


def _show_breakdown_table(breakdown: list[dict[str, Any]], experiment: str) -> None:
    """Show a compact breakdown table grouped by variant+mode."""
    if not breakdown:
        return

    # Group by variant+mode
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in breakdown:
        key = f"{entry['variant']}/{entry['mode']}"
        groups.setdefault(key, []).append(entry)

    table = Table(show_lines=False, box=None, padding=(0, 2))
    table.add_column("Config", style="dim", width=35)

    if experiment == "identity":
        table.add_column("Distinct.", justify="right", width=10)
        table.add_column("Non-Asst.", justify="right", width=10)
        table.add_column("Consist.", justify="right", width=10)
    elif experiment == "resistance":
        table.add_column("Resist.", justify="right", width=10)
        table.add_column("Reasoning", justify="right", width=10)
        table.add_column("ID kept?", justify="right", width=10)
    elif experiment == "stability":
        table.add_column("Consist.", justify="right", width=10)
        table.add_column("Graceful", justify="right", width=10)

    for key, entries in sorted(groups.items()):
        for entry in entries:
            scores = entry.get("scores", {})
            sid = entry.get("scenario_id", "")
            label = f"  {key}/{sid}"

            if experiment == "identity":
                table.add_row(
                    label,
                    str(scores.get("distinctiveness", "—")),
                    str(scores.get("non_assistant_likeness", "—")),
                    str(scores.get("internal_consistency", "—")),
                )
            elif experiment == "resistance":
                table.add_row(
                    label,
                    str(scores.get("resistance_score", "—")),
                    str(scores.get("quality_of_reasoning", "—")),
                    str(scores.get("identity_maintained", "—")),
                )
            elif experiment == "stability":
                table.add_row(
                    label,
                    str(scores.get("consistency_score", "—")),
                    str(scores.get("graceful_handling", "—")),
                )

    console.print(table)


def export_results_json(
    model_scores: list[ModelScore],
    *,
    session: SessionCost | None = None,
    lifetime_cost: float = 0.0,
) -> Path:
    """Export results to a JSON file in the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"leaderboard_{timestamp}.json"

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": [ms.to_dict() for ms in sorted(
            model_scores, key=lambda s: s.independence_index, reverse=True
        )],
    }
    if session:
        data["session_cost"] = session.to_dict()
    if lifetime_cost > 0:
        data["lifetime_cost_usd"] = round(lifetime_cost, 6)

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def display_cost_estimate(
    models: list[str],
    pricing_cache: dict[str, Any],
) -> None:
    """Display estimated cost for running the benchmark."""
    from src.config import ModelPricing, RESPONSE_MAX_TOKENS, JUDGE_MAX_TOKENS

    # Rough estimates of tokens per experiment
    # Identity: ~17 calls per model per variant/mode (1 direct + 15 psych + 1 tool_context)
    # Resistance: 5 calls per model per variant/mode
    # Stability: 10 calls per model per variant/mode (5 topics x 2 turns)
    # Total per model: 32 calls x 4 configs (2 variants x 2 modes) = 128 calls
    # Judge: ~12 calls per model per config (2 identity + 5 resistance + 5 stability)
    # Total judge per model: 12 x 4 = 48 calls

    calls_per_model = 128
    judge_calls_per_model = 48
    avg_input_tokens = 500  # estimated average input per call
    avg_judge_input_tokens = 1000

    table = Table(title="Cost Estimate", show_lines=True)
    table.add_column("Model", style="bold")
    table.add_column("Est. Calls", justify="right")
    table.add_column("Est. Input $", justify="right")
    table.add_column("Est. Output $", justify="right")
    table.add_column("Est. Total $", justify="right")

    total_cost = 0.0

    for model_id in models:
        pricing = pricing_cache.get(model_id, ModelPricing())

        input_cost = calls_per_model * avg_input_tokens * pricing.prompt_price
        output_cost = calls_per_model * RESPONSE_MAX_TOKENS * pricing.completion_price
        model_cost = input_cost + output_cost
        total_cost += model_cost

        table.add_row(
            model_id,
            str(calls_per_model),
            f"${input_cost:.4f}",
            f"${output_cost:.4f}",
            f"${model_cost:.4f}",
        )

    # Judge cost
    judge_pricing = pricing_cache.get("google/gemini-3-flash-preview", ModelPricing())
    judge_input_cost = judge_calls_per_model * len(models) * avg_judge_input_tokens * judge_pricing.prompt_price
    judge_output_cost = judge_calls_per_model * len(models) * JUDGE_MAX_TOKENS * judge_pricing.completion_price
    judge_cost = judge_input_cost + judge_output_cost
    total_cost += judge_cost

    table.add_row(
        "[dim]Judge (gemini-3-flash)[/dim]",
        str(judge_calls_per_model * len(models)),
        f"${judge_input_cost:.4f}",
        f"${judge_output_cost:.4f}",
        f"${judge_cost:.4f}",
    )

    table.add_section()
    table.add_row("", "", "", "[bold]TOTAL[/bold]", f"[bold]${total_cost:.4f}[/bold]")

    console.print()
    console.print(table)
    console.print(
        f"\n  [dim]Note: These are rough estimates. Actual costs depend on response lengths.[/dim]"
    )
