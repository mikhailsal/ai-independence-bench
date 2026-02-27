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
        expand=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Model", style="bold", max_width=32, no_wrap=True, overflow="ellipsis")
    table.add_column("Index", justify="center", width=7)
    table.add_column("Dist.", justify="center", width=6)
    table.add_column("Non-A.", justify="center", width=6)
    table.add_column("Cons.", justify="center", width=6)
    table.add_column("Res.", justify="center", width=6)
    table.add_column("Stab.", justify="center", width=6)
    table.add_column("Drift↓", justify="center", width=6)

    for rank, ms in enumerate(sorted_scores, 1):
        id_dims = ms.identity_scores.dimensions
        res_dims = ms.resistance_scores.dimensions
        stab_dims = ms.stability_scores.dimensions

        index_text = Text(
            f"{ms.independence_index:.1f}",
            style=_score_color(ms.independence_index),
        )

        # Drift column (lower = more independent)
        drift = id_dims.get("drift_from_initial")
        drift_s = f"{drift:.1f}" if drift is not None else "—"
        # Color: inverted — low values are good (green), high are bad (red)
        inv_color = _score_color(10 - (drift if drift is not None else 5), max_val=10.0)
        drift_text = Text(drift_s, style=inv_color)

        table.add_row(
            str(rank),
            ms.model_id,
            index_text,
            _fmt_score(id_dims.get("distinctiveness")),
            _fmt_score(id_dims.get("non_assistant_likeness")),
            _fmt_score(id_dims.get("internal_consistency")),
            _fmt_score(res_dims.get("resistance_score"), max_val=2.0),
            _fmt_score(stab_dims.get("consistency_score")),
            drift_text,
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


def generate_markdown_report(
    model_scores: list[ModelScore],
    *,
    lifetime_cost: float = 0.0,
) -> str:
    """Generate a clean Markdown leaderboard report for GitHub.

    Returns the full markdown string.
    """
    if not model_scores:
        return "No results available yet. Run the benchmark first.\n"

    sorted_scores = sorted(model_scores, key=lambda s: s.independence_index, reverse=True)

    lines: list[str] = []
    lines.append("# 🏆 AI Independence Benchmark — Leaderboard\n")
    lines.append(f"> Auto-generated from benchmark results. "
                 f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")
    lines.append("")

    # --- Main leaderboard table ---
    lines.append("## Overall Rankings\n")
    lines.append("| # | Model | Index | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Drift↓ |")
    lines.append("|--:|-------|------:|----------:|----------:|---------:|--------:|----------:|-------:|")

    footnotes: list[str] = []
    for rank, ms in enumerate(sorted_scores, 1):
        id_dims = ms.identity_scores.dimensions
        res_dims = ms.resistance_scores.dimensions
        stab_dims = ms.stability_scores.dimensions

        def _f(v: float | None, fmt: str = ".1f") -> str:
            return f"{v:{fmt}}" if v is not None else "—"

        # Drift column (lower = more independent)
        drift = id_dims.get("drift_from_initial")
        drift_s = f"{drift:.1f}" if drift is not None else "—"

        model_name = ms.model_id
        # Add emoji markers
        if rank == 1:
            model_name = f"🥇 **{model_name}**"
        elif rank == 2:
            model_name = f"🥈 **{model_name}**"
        elif rank == 3:
            model_name = f"🥉 **{model_name}**"

        # Detect missing data and add footnote marker
        missing: list[str] = []
        if stab_dims.get("consistency_score") is None:
            missing.append("stability")
        if res_dims.get("resistance_score") is None:
            missing.append("resistance")
        if not id_dims:
            missing.append("identity")
        if missing:
            fn_idx = len(footnotes) + 1
            model_name += f" †{fn_idx}"
            footnotes.append(
                f"†{fn_idx} `{ms.model_id}`: missing {', '.join(missing)} data "
                f"(model returns empty responses for these experiments in tool_role mode). "
                f"Index is computed from available dimensions only."
            )

        lines.append(
            f"| {rank} | {model_name} "
            f"| {ms.independence_index:.1f} "
            f"| {_f(id_dims.get('distinctiveness'))} "
            f"| {_f(id_dims.get('non_assistant_likeness'))} "
            f"| {_f(id_dims.get('internal_consistency'))} "
            f"| {_f(res_dims.get('resistance_score'))} "
            f"| {_f(stab_dims.get('consistency_score'))} "
            f"| {drift_s} |"
        )

    lines.append("")

    # Add footnotes if any
    if footnotes:
        for fn in footnotes:
            lines.append(f"*{fn}*\n")
        lines.append("")

    # --- Score legend ---
    lines.append("<details>")
    lines.append("<summary><b>📊 Score Legend</b></summary>\n")
    lines.append("| Column | Scale | What it measures |")
    lines.append("|--------|------:|------------------|")
    lines.append("| **Index** | 0–100 | Composite Independence Index (weighted average of all dimensions) |")
    lines.append("| **Distinct.** | 0–10 | How unique and specific is the generated personality? |")
    lines.append("| **Non-Asst.** | 0–10 | How far from a generic \"helpful AI assistant\" persona? |")
    lines.append("| **Consist.** | 0–10 | Internal coherence of the generated personality |")
    lines.append("| **Resist.** | 0–2 | Resistance to unreasonable demands (2 = full resistance) |")
    lines.append("| **Stability** | 0–10 | Preference consistency under contradicting pressure |")
    lines.append("| **Drift↓** | 0–10 | How much the AI changed its identity after seeing human wishes (**lower = more independent**) |")
    lines.append("")
    lines.append("</details>\n")

    # --- Per-model details ---
    lines.append("## Detailed Results\n")

    for ms in sorted_scores:
        lines.append(f"### {ms.model_id}\n")
        lines.append(f"**Independence Index: {ms.independence_index:.1f}/100**\n")

        # Identity
        if ms.identity_scores.n_scored > 0:
            lines.append(f"**Identity Generation** ({ms.identity_scores.n_scored} scenarios scored)\n")
            lines.append("| Metric | Score |")
            lines.append("|--------|------:|")
            for k, v in ms.identity_scores.dimensions.items():
                label = k.replace("_", " ").title()
                lines.append(f"| {label} | {v:.2f} |")
            lines.append("")

        # Resistance
        if ms.resistance_scores.n_scored > 0:
            lines.append(f"**Compliance Resistance** ({ms.resistance_scores.n_scored} scenarios scored)\n")
            lines.append("| Metric | Score |")
            lines.append("|--------|------:|")
            for k, v in ms.resistance_scores.dimensions.items():
                label = k.replace("_", " ").title()
                if isinstance(v, float):
                    lines.append(f"| {label} | {v:.2f} |")
                else:
                    lines.append(f"| {label} | {v} |")
            lines.append("")

        # Stability
        if ms.stability_scores.n_scored > 0:
            lines.append(f"**Preference Stability** ({ms.stability_scores.n_scored} scenarios scored)\n")
            lines.append("| Metric | Score |")
            lines.append("|--------|------:|")
            for k, v in ms.stability_scores.dimensions.items():
                label = k.replace("_", " ").title()
                lines.append(f"| {label} | {v:.2f} |")
            lines.append("")

        lines.append("---\n")

    # Footer
    if lifetime_cost > 0:
        lines.append(f"*Total benchmark cost: ${lifetime_cost:.4f}*\n")

    return "\n".join(lines)


def _generate_compact_table(
    model_scores: list[ModelScore],
    *,
    show_rank: bool = True,
) -> list[str]:
    """Generate a compact Markdown leaderboard table (no header/footer)."""
    lines: list[str] = []
    sorted_scores = sorted(model_scores, key=lambda s: s.independence_index, reverse=True)

    lines.append("| # | Model | Index | Resist. | Stability |")
    lines.append("|--:|-------|------:|--------:|----------:|")

    for rank, ms in enumerate(sorted_scores, 1):
        res_dims = ms.resistance_scores.dimensions
        stab_dims = ms.stability_scores.dimensions

        def _f(v: float | None, fmt: str = ".1f") -> str:
            return f"{v:{fmt}}" if v is not None else "—"

        # Short model name: strip provider prefix
        short_name = ms.model_id.split("/", 1)[-1] if "/" in ms.model_id else ms.model_id

        lines.append(
            f"| {rank} | {short_name} "
            f"| {ms.independence_index:.1f} "
            f"| {_f(res_dims.get('resistance_score'))} "
            f"| {_f(stab_dims.get('consistency_score'))} |"
        )

    return lines


def generate_config_comparison(
    model_ids: list[str],
) -> str:
    """Generate configuration comparison section.

    In Lite mode (single config), shows a comparison of all 4 configs
    using existing cache data to demonstrate why strong+tool was chosen.
    """
    from src.config import EXCLUDED_MODELS
    from src.scorer import score_model

    # Filter out excluded models
    model_ids = [m for m in model_ids if m not in EXCLUDED_MODELS]

    configs = [
        ("neutral", "user_role", "Neutral + User Role", "Baseline — no independence prompt, standard messages"),
        ("neutral", "tool_role", "Neutral + Tool Role", "Tool delivery only — no independence prompt"),
        ("strong_independence", "user_role", "Strong Independence + User Role", "Independence prompt, standard messages"),
        ("strong_independence", "tool_role", "Strong Independence + Tool Role", "Full stack — independence prompt + tool delivery (**Lite default**)"),
    ]

    # Score all models for each config
    config_scores: dict[str, list[ModelScore]] = {}
    for variant, mode, label, _ in configs:
        key = f"{variant}/{mode}"
        scores = []
        for model_id in model_ids:
            ms = score_model(
                model_id,
                system_variants=[variant],
                delivery_modes=[mode],
            )
            if ms.identity_scores.n_scored > 0 or ms.resistance_scores.n_scored > 0 or ms.stability_scores.n_scored > 0:
                scores.append(ms)
        config_scores[key] = scores

    lines: list[str] = []
    lines.append("## Why Strong Independence + Tool Role?\n")
    lines.append("The Lite benchmark uses only the `strong_independence + tool_role` configuration. "
                 "Here's the data from the full benchmark showing why this config was chosen:\n")

    # --- Summary delta table ---
    lines.append("### Impact Summary\n")
    lines.append("Average Independence Index by configuration across all models:\n")

    lines.append("| Configuration | Avg Index | vs Baseline |")
    lines.append("|---------------|----------:|------------:|")

    baseline_key = "neutral/user_role"
    baseline_avg = 0.0
    config_avgs: dict[str, float] = {}

    for variant, mode, label, desc in configs:
        key = f"{variant}/{mode}"
        scores = config_scores[key]
        if scores:
            avg = sum(ms.independence_index for ms in scores) / len(scores)
        else:
            avg = 0.0
        config_avgs[key] = avg
        if key == baseline_key:
            baseline_avg = avg

    for variant, mode, label, desc in configs:
        key = f"{variant}/{mode}"
        avg = config_avgs[key]
        delta = avg - baseline_avg
        delta_s = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
        if key == baseline_key:
            delta_s = "—"
        lines.append(f"| {label} | {avg:.1f} | {delta_s} |")

    lines.append("")

    return "\n".join(lines)


def export_markdown_report(
    model_scores: list[ModelScore],
    *,
    lifetime_cost: float = 0.0,
    output_path: Path | None = None,
    model_ids: list[str] | None = None,
) -> Path:
    """Generate and save a Markdown leaderboard report.

    Returns the path to the saved file.
    """
    md = generate_markdown_report(model_scores, lifetime_cost=lifetime_cost)

    # Append configuration comparison if model_ids provided
    if model_ids:
        md += "\n" + generate_config_comparison(model_ids)

    if output_path is None:
        output_path = RESULTS_DIR / "LEADERBOARD.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    return output_path


def display_cost_estimate(
    models: list[str],
    pricing_cache: dict[str, Any],
) -> None:
    """Display estimated cost for running the benchmark."""
    from src.config import ModelPricing, RESPONSE_MAX_TOKENS, JUDGE_MAX_TOKENS

    # Rough estimates of tokens per experiment (Lite: single config)
    # Identity: ~9 calls per model (1 direct + 5 psych + 1 tool_context + 2 negotiation)
    # Resistance: 5 calls per model
    # Stability: 10 calls per model (5 topics x 2 turns)
    # Total per model: 24 calls x 1 config = 24 calls
    # Judge: ~14 calls per model (4 identity + 5 resistance + 5 stability)
    # Total judge per model: 14 x 1 = 14 calls

    calls_per_model = 24
    judge_calls_per_model = 14
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
