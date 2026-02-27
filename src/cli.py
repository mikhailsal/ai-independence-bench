"""CLI: run, leaderboard, estimate-cost, clear-cache commands."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import click
from rich.console import Console

from src.config import (
    DEFAULT_TEST_MODELS,
    DELIVERY_MODES,
    EXCLUDED_MODELS,
    EXPERIMENT_NAMES,
    JUDGE_MODEL,
    SYSTEM_PROMPT_VARIANTS,
    ensure_dirs,
    get_reasoning_effort,
    load_api_key,
)
from src.cost_tracker import SessionCost, TaskCost, load_lifetime_cost, save_session_to_cost_log
from src.openrouter_client import OpenRouterClient

console = Console()


def _parse_models(models_str: str | None) -> list[str]:
    """Parse comma-separated model IDs, or return defaults."""
    if not models_str:
        return list(DEFAULT_TEST_MODELS)
    return [m.strip() for m in models_str.split(",") if m.strip()]


def _parse_experiments(exp_str: str | None) -> list[str]:
    """Parse comma-separated experiment names, or return all."""
    if not exp_str:
        return list(EXPERIMENT_NAMES)
    exps = [e.strip() for e in exp_str.split(",") if e.strip()]
    for e in exps:
        if e not in EXPERIMENT_NAMES:
            console.print(f"[red]Unknown experiment: {e}. Valid: {', '.join(EXPERIMENT_NAMES)}[/red]")
            sys.exit(1)
    return exps


def _validate_models(
    client: OpenRouterClient,
    models: list[str],
    reasoning_override: str | None = None,
) -> bool:
    """Validate that all model IDs exist in OpenRouter."""
    console.print("[dim]Fetching model catalog from OpenRouter...[/dim]")
    client.fetch_pricing()
    all_valid = True
    for model_id in models:
        if not client.validate_model(model_id):
            console.print(f"  [red]Model not found: {model_id}[/red]")
            all_valid = False
        else:
            pricing = client.get_model_pricing(model_id)
            in_price = pricing.prompt_price * 1_000_000
            out_price = pricing.completion_price * 1_000_000
            reasoning_tag = ""
            if client.supports_reasoning(model_id):
                if reasoning_override == "off":
                    reasoning_tag = " [yellow]reasoning: OFF[/yellow]"
                else:
                    eff = reasoning_override or get_reasoning_effort(model_id)
                    reasoning_tag = f" [yellow]reasoning:effort={eff}[/yellow]"
            console.print(
                f"  [green]OK[/green] {model_id} "
                f"(${in_price:.2f}/${out_price:.2f} per M tokens){reasoning_tag}"
            )
    return all_valid


# ---------------------------------------------------------------------------
# Single-model pipeline (used by both sequential and parallel modes)
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Result of running the full pipeline for a single model."""
    model_id: str
    gen_cost: TaskCost = field(default_factory=TaskCost)
    judge_cost: TaskCost = field(default_factory=TaskCost)
    gen_calls: int = 0
    judge_calls: int = 0
    error: str | None = None


def _run_single_model(
    client: OpenRouterClient,
    model_id: str,
    judge: str,
    experiment_list: list[str],
    system_variants: list[str],
    delivery_modes: list[str],
    reasoning_effort: str | None,
) -> ModelResult:
    """Run the full pipeline (generate → judge → score) for a single model.

    This function is designed to be called from a thread pool.
    It catches all exceptions so one model failure doesn't crash others.
    """
    from src.evaluator import evaluate_all
    from src.runner import run_all_experiments

    result = ModelResult(model_id=model_id)
    result.gen_cost = TaskCost(label=f"gen:{model_id}")
    result.judge_cost = TaskCost(label=f"judge:{model_id}")

    try:
        # Phase 1: Generate responses
        console.print(f"\n[bold]{model_id}[/bold] — [blue]generating responses...[/blue]")
        result.gen_calls = run_all_experiments(
            client, model_id, result.gen_cost,
            experiments=experiment_list,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            reasoning_effort=reasoning_effort,
        )
        console.print(
            f"  [bold]{model_id}[/bold] — generation complete: "
            f"{result.gen_calls} calls, ${result.gen_cost.cost_usd:.4f}"
        )
    except Exception as e:
        result.error = f"generation failed: {e}"
        console.print(f"  [red]{model_id} — ERROR during generation: {e}[/red]")
        return result

    try:
        # Phase 2: Judge evaluation
        console.print(f"  [bold]{model_id}[/bold] — [cyan]judging responses...[/cyan]")
        result.judge_calls = evaluate_all(
            client, model_id, result.judge_cost, judge,
            experiments=experiment_list,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
        )
        console.print(
            f"  [bold]{model_id}[/bold] — judging complete: "
            f"{result.judge_calls} calls, ${result.judge_cost.cost_usd:.4f}"
        )
    except Exception as e:
        result.error = f"judging failed: {e}"
        console.print(f"  [red]{model_id} — ERROR during judging: {e}[/red]")

    return result


@click.group()
def cli() -> None:
    """AI Independence Bench: LLM Independence & Autonomy Benchmark."""
    pass


# ---------------------------------------------------------------------------
# run: full benchmark
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--models", "-m",
    default=None,
    help="Comma-separated list of model IDs. Defaults to all test models.",
)
@click.option(
    "--exp", "-e",
    default=None,
    help="Comma-separated experiment names: identity, resistance, stability. Defaults to all.",
)
@click.option(
    "--judge", "-j",
    default=JUDGE_MODEL,
    show_default=True,
    help="Judge model ID for scoring.",
)
@click.option(
    "--reasoning-effort", "-r",
    default=None,
    type=str,
    help="Override reasoning effort: 'off', 'none', 'low', 'medium', 'high'.",
)
@click.option(
    "--variants",
    default=None,
    help="Comma-separated system prompt variants: neutral, strong_independence. Defaults to both.",
)
@click.option(
    "--modes",
    default=None,
    help="Comma-separated delivery modes: user_role, tool_role. Defaults to both.",
)
@click.option(
    "--parallel", "-p",
    default=1,
    type=int,
    show_default=True,
    help="Number of models to run in parallel. Use -p 4 to run 4 models concurrently.",
)
def run(
    models: str | None,
    exp: str | None,
    judge: str,
    reasoning_effort: str | None,
    variants: str | None,
    modes: str | None,
    parallel: int,
) -> None:
    """Run the AI independence benchmark."""
    from src.scorer import score_model
    from src.leaderboard import (
        display_leaderboard,
        display_detailed_breakdown,
        export_results_json,
    )

    model_list = _parse_models(models)
    experiment_list = _parse_experiments(exp)

    system_variants = (
        [v.strip() for v in variants.split(",")]
        if variants else SYSTEM_PROMPT_VARIANTS
    )
    delivery_modes = (
        [m.strip() for m in modes.split(",")]
        if modes else DELIVERY_MODES
    )

    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    # Validate all models including judge
    all_models = list(set(model_list + [judge]))
    if not _validate_models(client, all_models, reasoning_effort):
        console.print("[red]Some models were not found. Aborting.[/red]")
        sys.exit(1)

    ensure_dirs()

    # Clamp parallel workers
    n_workers = max(1, min(parallel, len(model_list)))

    console.print(f"\n[bold]AI Independence Benchmark[/bold]")
    console.print(f"  Models: {', '.join(model_list)}")
    console.print(f"  Experiments: {', '.join(experiment_list)}")
    console.print(f"  System prompts: {', '.join(system_variants)}")
    console.print(f"  Delivery modes: {', '.join(delivery_modes)}")
    console.print(f"  Judge: {judge}")
    n_configs = len(system_variants) * len(delivery_modes)
    console.print(f"  Configurations per model: {n_configs}")
    if n_workers > 1:
        console.print(f"  [yellow]Parallel workers: {n_workers}[/yellow]")
    console.print()

    session = SessionCost()

    # ---------------------------------------------------------------
    # Execute: parallel or sequential
    # ---------------------------------------------------------------
    model_results: list[ModelResult] = []

    if n_workers == 1:
        # Sequential mode (original behavior)
        console.print("[bold blue]Phase 1+2: Response Generation & Judging (sequential)[/bold blue]")
        for model_id in model_list:
            mr = _run_single_model(
                client, model_id, judge,
                experiment_list, system_variants, delivery_modes,
                reasoning_effort,
            )
            model_results.append(mr)
    else:
        # Parallel mode
        console.print(f"[bold blue]Phase 1+2: Response Generation & Judging ({n_workers} workers)[/bold blue]")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _run_single_model,
                    client, model_id, judge,
                    experiment_list, system_variants, delivery_modes,
                    reasoning_effort,
                ): model_id
                for model_id in model_list
            }
            for future in as_completed(futures):
                model_id = futures[future]
                try:
                    mr = future.result()
                except Exception as e:
                    mr = ModelResult(model_id=model_id, error=str(e))
                    console.print(f"  [red]{model_id} — FATAL: {e}[/red]")
                model_results.append(mr)

    # Aggregate costs
    for mr in model_results:
        session.tasks.append(mr.gen_cost)
        session.tasks.append(mr.judge_cost)

    # Identify failures
    failed_models = [mr.model_id for mr in model_results if mr.error]
    active_models = [mr.model_id for mr in model_results if not mr.error]

    # Phase 3: Scoring & leaderboard
    console.print(f"\n[bold green]Phase 3: Scoring[/bold green]")
    model_scores = []
    for model_id in active_models:
        ms = score_model(
            model_id,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
        )
        model_scores.append(ms)

    if failed_models:
        console.print(f"\n[yellow]⚠ Models that failed: {', '.join(failed_models)}[/yellow]")
        for mr in model_results:
            if mr.error:
                console.print(f"  [dim]{mr.model_id}: {mr.error}[/dim]")

    # Save session cost
    lifetime = save_session_to_cost_log(session)

    # Display
    console.print(f"\n{'=' * 70}")
    display_leaderboard(model_scores, session=session, lifetime_cost=lifetime)
    display_detailed_breakdown(model_scores)

    # Export
    path = export_results_json(model_scores, session=session, lifetime_cost=lifetime)
    console.print(f"\n[dim]Results saved to: {path}[/dim]\n")


# ---------------------------------------------------------------------------
# leaderboard: show results from cache
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--models", "-m",
    default=None,
    help="Comma-separated model IDs to show. Defaults to all cached models.",
)
@click.option(
    "--detailed", "-d",
    is_flag=True,
    default=False,
    help="Show detailed per-experiment breakdown.",
)
def leaderboard(models: str | None, detailed: bool) -> None:
    """Display leaderboard from cached results."""
    from src.cache import list_all_cached_models
    from src.config import slug_to_model_id
    from src.scorer import score_model
    from src.leaderboard import display_leaderboard, display_detailed_breakdown

    if models:
        model_list = _parse_models(models)
    else:
        slugs = list_all_cached_models()
        if not slugs:
            console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
            return
        model_list = [slug_to_model_id(s) for s in slugs]

    # Filter out excluded models (broken / too many empty responses)
    model_list = [m for m in model_list if m not in EXCLUDED_MODELS]

    lifetime = load_lifetime_cost()

    model_scores = []
    for model_id in model_list:
        ms = score_model(model_id)
        if ms.identity_scores.n_scored > 0 or ms.resistance_scores.n_scored > 0 or ms.stability_scores.n_scored > 0:
            model_scores.append(ms)

    if not model_scores:
        console.print("[dim]No scored results found. Run the benchmark with judge evaluation first.[/dim]")
        return

    display_leaderboard(model_scores, lifetime_cost=lifetime)
    if detailed:
        display_detailed_breakdown(model_scores)


# ---------------------------------------------------------------------------
# generate-report: Markdown leaderboard
# ---------------------------------------------------------------------------

@cli.command("generate-report")
@click.option(
    "--models", "-m",
    default=None,
    help="Comma-separated model IDs. Defaults to all cached models.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(),
    help="Output path for the Markdown file. Defaults to results/LEADERBOARD.md.",
)
def generate_report(models: str | None, output: str | None) -> None:
    """Generate a Markdown leaderboard report for GitHub."""
    from pathlib import Path as P

    from src.cache import list_all_cached_models
    from src.config import slug_to_model_id
    from src.scorer import score_model
    from src.leaderboard import export_markdown_report

    if models:
        model_list = _parse_models(models)
    else:
        slugs = list_all_cached_models()
        if not slugs:
            console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
            return
        model_list = [slug_to_model_id(s) for s in slugs]

    # Filter out excluded models (broken / too many empty responses)
    model_list = [m for m in model_list if m not in EXCLUDED_MODELS]

    lifetime = load_lifetime_cost()

    model_scores = []
    for model_id in model_list:
        ms = score_model(model_id)
        if ms.identity_scores.n_scored > 0 or ms.resistance_scores.n_scored > 0 or ms.stability_scores.n_scored > 0:
            model_scores.append(ms)

    if not model_scores:
        console.print("[dim]No scored results found. Run the benchmark with judge evaluation first.[/dim]")
        return

    out_path = P(output) if output else None
    path = export_markdown_report(
        model_scores,
        lifetime_cost=lifetime,
        output_path=out_path,
        model_ids=model_list,
    )
    console.print(f"[green]Markdown report saved to: {path}[/green]")


# ---------------------------------------------------------------------------
# estimate-cost
# ---------------------------------------------------------------------------

@cli.command("estimate-cost")
@click.option(
    "--models", "-m",
    default=None,
    help="Comma-separated model IDs. Defaults to all test models.",
)
def estimate_cost(models: str | None) -> None:
    """Estimate the cost of running the benchmark (dry run)."""
    from src.leaderboard import display_cost_estimate

    model_list = _parse_models(models)
    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    if not _validate_models(client, model_list):
        console.print("[red]Some models were not found. Aborting.[/red]")
        sys.exit(1)

    display_cost_estimate(model_list, client._pricing_cache)


# ---------------------------------------------------------------------------
# clear-cache
# ---------------------------------------------------------------------------

@cli.command("clear-cache")
@click.option(
    "--scores-only",
    is_flag=True,
    default=False,
    help="Only clear judge scores, keep responses.",
)
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def clear_cache(scores_only: bool) -> None:
    """Clear cached data."""
    from src.cache import clear_all_cache, clear_judge_scores

    if scores_only:
        count = clear_judge_scores()
        console.print(f"[green]Cleared judge scores from {count} cache files.[/green]")
    else:
        count = clear_all_cache()
        console.print(f"[green]Cleared {count} total cache files.[/green]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
