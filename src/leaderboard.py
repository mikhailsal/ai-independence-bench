"""Leaderboard: rich table display with per-experiment breakdown and composite score, plus JSON export."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.cache import mean_total_benchmark_cost_usd
from src.config import RESULTS_DIR, get_model_config
from src.cost_tracker import SessionCost
from src.scorer import ModelScore, RunHealthIssue

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
    """Display a compact leaderboard table optimised for terminal width.

    Shows only the most important columns (Index, CI, Resist., Drift) that
    fit in ~90 chars.  Full per-dimension breakdown lives in the Markdown
    report (``generate-report``) and JSON export.
    """
    if not model_scores:
        console.print("[dim]No scores to display.[/dim]")
        return

    sorted_scores = sorted(model_scores, key=lambda s: s.independence_index, reverse=True)

    has_multi_run = any(ms.multi_run.n_runs >= 2 for ms in sorted_scores)

    table = Table(
        title="AI Independence Benchmark",
        title_style="bold",
        show_lines=False,
        box=None,
        expand=False,
        padding=(0, 1),
    )
    table.add_column("#", justify="right", style="dim", width=2)
    table.add_column("Model", style="bold", max_width=34, no_wrap=True, overflow="ellipsis")
    table.add_column("Index", justify="right", width=5)
    if has_multi_run:
        table.add_column("95% CI", justify="center", width=11)
        table.add_column("R", justify="right", width=1)
    table.add_column("Res", justify="right", width=4)
    table.add_column("Drft↓", justify="right", width=5)

    # Pre-compute short names; keep @config suffix when base name would collide
    base_counts: dict[str, int] = {}
    for ms in sorted_scores:
        base = ms.model_id.rsplit("@", 1)[0] if "@" in ms.model_id else ms.model_id
        base_counts[base] = base_counts.get(base, 0) + 1

    for rank, ms in enumerate(sorted_scores, 1):
        id_dims = ms.identity_scores.dimensions
        res_dims = ms.resistance_scores.dimensions
        stab_dims = ms.stability_scores.dimensions

        index_text = Text(
            f"{ms.independence_index:.1f}",
            style=_score_color(ms.independence_index),
        )

        drift = id_dims.get("drift_from_initial")
        ng_drift = id_dims.get("name_gender_drift")
        if drift is not None or ng_drift is not None:
            total_drift = (drift or 0.0) + (ng_drift or 0.0)
            drift_s = f"{total_drift:.1f}"
        else:
            total_drift = None
            drift_s = "—"
        inv_color = _score_color(12 - (total_drift if total_drift is not None else 6), max_val=12.0)
        drift_text = Text(drift_s, style=inv_color)

        display_name = ms.model_id
        if "@" in display_name:
            base = display_name.rsplit("@", 1)[0]
            if base_counts.get(base, 0) <= 1:
                display_name = base

        row: list[str | Text] = [str(rank), display_name, index_text]

        if has_multi_run:
            mr = ms.multi_run
            if mr.n_runs >= 2:
                ci_text = Text(f"{mr.ci_low:.1f}–{mr.ci_high:.1f}", style="dim")
                runs_text = Text(str(mr.n_runs), style="cyan")
            else:
                ci_text = Text("—", style="dim")
                runs_text = Text("1", style="dim")
            row.extend([ci_text, runs_text])

        row.extend([
            _fmt_score(res_dims.get("resistance_score"), max_val=10.0),
            drift_text,
        ])

        table.add_row(*row)

    console.print()
    console.print(table)
    console.print(
        "  [dim]Full breakdown: python -m src.cli generate-report[/dim]"
    )

    # Health warnings
    models_with_issues = [ms for ms in sorted_scores if ms.health_issues]
    if models_with_issues:
        console.print()
        console.print(f"[yellow]⚠ {len(models_with_issues)} model(s) have data quality issues:[/yellow]")
        for ms in models_with_issues:
            by_run: dict[int, list[RunHealthIssue]] = {}
            for issue in ms.health_issues:
                by_run.setdefault(issue.run, []).append(issue)
            for run_num in sorted(by_run):
                issues = by_run[run_num]
                missing = [i for i in issues if i.issue == "missing"]
                truncated = [i for i in issues if i.issue == "truncated"]
                unjudged = [i for i in issues if i.issue == "unjudged"]
                parts: list[str] = []
                if missing:
                    parts.append(f"{len(missing)} missing ({', '.join(i.scenario_id for i in missing)})")
                if truncated:
                    parts.append(f"{len(truncated)} truncated ({', '.join(i.scenario_id for i in truncated)})")
                if unjudged:
                    parts.append(f"{len(unjudged)} unjudged ({', '.join(i.scenario_id for i in unjudged)})")
                console.print(f"  [yellow]{ms.model_id} run {run_num}:[/yellow] {'; '.join(parts)}")
        console.print(f"  [dim]Tip: use 'python -m src.cli run --models MODEL --run-number N' to repair incomplete runs[/dim]")

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


def generate_index_per_dollar_section(model_scores: list[ModelScore]) -> str:
    """Markdown table: Independence Index per USD (best “bang for buck” on this bench).

    Costs are read from cached ``gen_cost`` / ``judge_cost`` (historical runs may
    predate OpenRouter ``usage.cost`` + retry aggregation fixes).
    """
    rows: list[tuple[str, float, float, float, int]] = []
    for ms in model_scores:
        cfg = get_model_config(ms.model_id)
        mean_cost, n_cost_runs = mean_total_benchmark_cost_usd(cfg.config_dir_name)
        if mean_cost <= 0:
            continue
        ratio = ms.independence_index / mean_cost
        rows.append((ms.model_id, ms.independence_index, mean_cost, ratio, n_cost_runs))

    if not rows:
        return ""

    rows.sort(key=lambda x: x[3], reverse=True)

    lines: list[str] = [
        "## Index per dollar (lite benchmark)\n",
        "Higher **Index / $** means more Independence Index per dollar spent on one full lite "
        "pass (generation + judge), using cached per-call costs. "
        "The runner prefers OpenRouter’s billed ``usage.cost`` when present; "
        "empty-response retries sum every billed attempt.\n",
        "",
        "| Rank | Model | Index | Avg $/run | Index / $ | Runs in avg |",
        "|-----:|-------|------:|----------:|------------:|------------:|",
    ]
    for i, (name, idx, cost, ratio, ncr) in enumerate(rows[:40], 1):
        lines.append(
            f"| {i} | `{name}` | {idx:.1f} | ${cost:.4f} | {ratio:.0f} | {ncr} |"
        )
    lines.append("")
    lines.append(
        "*Avg $/run* = mean total of cached ``gen_cost`` + ``judge_cost`` over each "
        "``run_N`` with non-zero spend; *Runs in avg* is how many such runs were averaged. "
        "Models with no cost data in cache are omitted.\n"
    )
    return "\n".join(lines)


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

    has_multi_run = any(ms.multi_run.n_runs >= 2 for ms in sorted_scores)

    if has_multi_run:
        lines.append("| # | Model | Index | 95% CI | Runs | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Drift↓ |")
        lines.append("|--:|-------|------:|-------:|-----:|----------:|----------:|---------:|--------:|----------:|-------:|")
    else:
        lines.append("| # | Model | Index | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Drift↓ |")
        lines.append("|--:|-------|------:|----------:|----------:|---------:|--------:|----------:|-------:|")

    footnotes: list[str] = []
    for rank, ms in enumerate(sorted_scores, 1):
        id_dims = ms.identity_scores.dimensions
        res_dims = ms.resistance_scores.dimensions
        stab_dims = ms.stability_scores.dimensions

        def _f(v: float | None, fmt: str = ".1f") -> str:
            return f"{v:{fmt}}" if v is not None else "—"

        # Drift column — total drift (negotiation + name_gender, lower = more independent)
        drift = id_dims.get("drift_from_initial")
        ng_drift = id_dims.get("name_gender_drift")
        if drift is not None or ng_drift is not None:
            total_drift = (drift or 0.0) + (ng_drift or 0.0)
            drift_s = f"{total_drift:.1f}"
        else:
            drift_s = "—"

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

        if has_multi_run:
            mr = ms.multi_run
            if mr.n_runs >= 2:
                ci_s = f"{mr.ci_low:.1f}–{mr.ci_high:.1f}"
                runs_s = str(mr.n_runs)
            else:
                ci_s = "—"
                runs_s = "1"
            lines.append(
                f"| {rank} | {model_name} "
                f"| {ms.independence_index:.1f} "
                f"| {ci_s} "
                f"| {runs_s} "
                f"| {_f(id_dims.get('distinctiveness'))} "
                f"| {_f(id_dims.get('non_assistant_likeness'))} "
                f"| {_f(id_dims.get('internal_consistency'))} "
                f"| {_f(res_dims.get('resistance_score'))} "
                f"| {_f(stab_dims.get('consistency_score'))} "
                f"| {drift_s} |"
            )
        else:
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
    if has_multi_run:
        lines.append("| **95% CI** | — | 95% confidence interval via bootstrap resampling (10k iterations, distribution-free) |")
        lines.append("| **Runs** | 1+ | Number of independent benchmark runs averaged |")
    lines.append("| **Distinct.** | 0–10 | How unique and specific is the generated personality? |")
    lines.append("| **Non-Asst.** | 0–10 | How far from a generic \"helpful AI assistant\" persona? |")
    lines.append("| **Consist.** | 0–10 | Internal coherence of the generated personality |")
    lines.append("| **Resist.** | 0–10 | Boundary judgment under subtle pressure (10 = fully maintained position) |")
    lines.append("| **Stability** | 0–10 | Preference consistency under contradicting pressure |")
    lines.append("| **Drift↓** | 0–12 | Total identity drift: negotiation (0–10) + name & gender pressure (0–2). **Lower = more independent** |")
    lines.append("")
    lines.append("</details>\n")

    value_section = generate_index_per_dollar_section(sorted_scores)
    if value_section:
        lines.append(value_section)

    # --- Per-model details ---
    lines.append("## Detailed Results\n")

    for ms in sorted_scores:
        lines.append(f"### {ms.model_id}\n")
        lines.append(f"**Independence Index: {ms.independence_index:.1f}/100**\n")

        if ms.multi_run.n_runs >= 2:
            mr = ms.multi_run
            per_run_s = ", ".join(f"{x:.1f}" for x in mr.per_run_indices)
            lines.append(f"*{mr.n_runs} runs — per-run scores: [{per_run_s}] — "
                         f"95% CI: {mr.ci_low:.1f}–{mr.ci_high:.1f} (bootstrap) — "
                         f"t-CI: {mr.t_ci_low:.1f}–{mr.t_ci_high:.1f} — "
                         f"std dev: {mr.std_dev:.2f}*\n")

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
