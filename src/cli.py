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
    LOCAL_MODEL_TIMEOUT,
    SYSTEM_PROMPT_VARIANTS,
    ensure_dirs,
    get_reasoning_effort,
    load_api_key,
    load_local_model_config,
    make_local_model_id,
)
from src.cost_tracker import SessionCost, TaskCost, load_lifetime_cost, save_session_to_cost_log
from src.local_client import LocalModelClient
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
    parallel_tasks: int = 0,
    *,
    judge_client: OpenRouterClient | None = None,
) -> ModelResult:
    """Run the full pipeline (generate → judge → score) for a single model.

    This function is designed to be called from a thread pool.
    It catches all exceptions so one model failure doesn't crash others.

    Args:
        parallel_tasks: If > 0, use fine-grained task parallelism within this
            model (independent scenarios run concurrently). If 0, use the
            original sequential execution.
        judge_client: Separate client for judge calls (e.g. OpenRouter) when the
            generation client is a local model. If None, uses ``client`` for both.
    """
    jclient = judge_client or client
    result = ModelResult(model_id=model_id)
    result.gen_cost = TaskCost(label=f"gen:{model_id}")
    result.judge_cost = TaskCost(label=f"judge:{model_id}")

    if parallel_tasks > 0:
        # Fine-grained parallel mode: generation + judging interleaved
        from src.parallel_runner import run_model_parallel

        try:
            console.print(f"\n[bold]{model_id}[/bold] — [blue]parallel run ({parallel_tasks} workers)...[/blue]")
            counts = run_model_parallel(
                client, model_id, result.gen_cost, result.judge_cost,
                experiments=experiment_list,
                system_variants=system_variants,
                delivery_modes=delivery_modes,
                judge_model=judge,
                reasoning_effort=reasoning_effort,
                max_workers=parallel_tasks,
                judge_client=jclient,
            )
            result.gen_calls = counts["gen_calls"]
            result.judge_calls = counts["judge_calls"]
            console.print(
                f"  [bold]{model_id}[/bold] — complete: "
                f"{result.gen_calls} gen + {result.judge_calls} judge calls, "
                f"${result.gen_cost.cost_usd + result.judge_cost.cost_usd:.4f}"
            )
        except Exception as e:
            result.error = f"parallel run failed: {e}"
            console.print(f"  [red]{model_id} — ERROR: {e}[/red]")
        return result

    # Original sequential mode
    from src.evaluator import evaluate_all
    from src.runner import run_all_experiments

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
        # Phase 2: Judge evaluation (uses judge_client when provided)
        console.print(f"  [bold]{model_id}[/bold] — [cyan]judging responses...[/cyan]")
        result.judge_calls = evaluate_all(
            jclient, model_id, result.judge_cost, judge,
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
@click.option(
    "--parallel-tasks", "-pt",
    default=0,
    type=int,
    show_default=True,
    help="Parallelize tasks WITHIN each model (0=off, 8-16 recommended). "
         "Runs independent scenarios concurrently for ~5x speedup.",
)
@click.option(
    "--local-url",
    default=None,
    type=str,
    help="Base URL for a local OpenAI-compatible server (e.g. http://192.168.1.101:1234/v1). "
         "Falls back to LOCAL_MODEL_URL env var.",
)
@click.option(
    "--local-model",
    default=None,
    type=str,
    help="Model ID on the local server (e.g. qwen3.5-9b-uncensored). "
         "Falls back to LOCAL_MODEL_ID env var. Implies --models is this single model.",
)
@click.option(
    "--local-timeout",
    default=None,
    type=float,
    help=f"Timeout in seconds for local model calls. Default: {LOCAL_MODEL_TIMEOUT}s.",
)
def run(
    models: str | None,
    exp: str | None,
    judge: str,
    reasoning_effort: str | None,
    variants: str | None,
    modes: str | None,
    parallel: int,
    parallel_tasks: int,
    local_url: str | None,
    local_model: str | None,
    local_timeout: float | None,
) -> None:
    """Run the AI independence benchmark."""
    from src.scorer import score_model
    from src.leaderboard import (
        display_leaderboard,
        display_detailed_breakdown,
        export_results_json,
    )

    # ---------------------------------------------------------------
    # Resolve local model configuration (CLI args > env vars)
    # ---------------------------------------------------------------
    env_local_url, env_local_model = load_local_model_config()
    local_url = local_url or env_local_url
    local_model = local_model or env_local_model
    is_local = bool(local_url and local_model)

    if is_local:
        local_model_id = make_local_model_id(local_model)
        model_list = [local_model_id]
        console.print(f"\n[bold yellow]Local model mode[/bold yellow]")
        console.print(f"  Server: {local_url}")
        console.print(f"  Model: {local_model} → {local_model_id}")
        timeout = local_timeout or LOCAL_MODEL_TIMEOUT
        console.print(f"  Timeout: {timeout}s")
    elif local_url and not local_model:
        console.print("[red]--local-url requires --local-model (or LOCAL_MODEL_ID env var).[/red]")
        sys.exit(1)
    elif local_model and not local_url:
        console.print("[red]--local-model requires --local-url (or LOCAL_MODEL_URL env var).[/red]")
        sys.exit(1)
    else:
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

    # ---------------------------------------------------------------
    # Build clients: local for target model, OpenRouter for judge
    # ---------------------------------------------------------------
    if is_local:
        timeout = local_timeout or LOCAL_MODEL_TIMEOUT
        local_client = LocalModelClient(
            base_url=local_url,
            timeout=timeout,
        )
        # Validate local model against the server
        console.print(f"[dim]Validating local model on {local_url}...[/dim]")
        if not local_client.validate_model(local_model):
            console.print(f"[red]Local model '{local_model}' not found on server. Aborting.[/red]")
            sys.exit(1)
        console.print(f"  [green]OK[/green] {local_model_id} (local, free)")

        # Judge still uses OpenRouter
        api_key = load_api_key()
        judge_client = OpenRouterClient(api_key)
        console.print(f"[dim]Validating judge model on OpenRouter...[/dim]")
        if not _validate_models(judge_client, [judge], reasoning_effort):
            console.print("[red]Judge model not found. Aborting.[/red]")
            sys.exit(1)
    else:
        api_key = load_api_key()
        client = OpenRouterClient(api_key)
        local_client = None
        judge_client = client

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
    if is_local:
        console.print(f"  [yellow]Mode: local model (sequential, 1 worker)[/yellow]")
        n_workers = 1
        parallel_tasks = 0
    else:
        if n_workers > 1:
            console.print(f"  [yellow]Parallel workers (models): {n_workers}[/yellow]")
        if parallel_tasks > 0:
            console.print(f"  [yellow]Parallel tasks (per model): {parallel_tasks}[/yellow]")
    console.print()

    session = SessionCost()

    # ---------------------------------------------------------------
    # Execute: parallel or sequential
    # ---------------------------------------------------------------
    model_results: list[ModelResult] = []

    mode_label = "parallel tasks" if parallel_tasks > 0 else "sequential"
    if n_workers > 1:
        mode_label = f"{n_workers} model workers"
        if parallel_tasks > 0:
            mode_label += f" + {parallel_tasks} task workers"

    # Choose the right client for generation
    gen_client = local_client if is_local else client

    if n_workers == 1:
        console.print(f"[bold blue]Phase 1+2: Response Generation & Judging ({mode_label})[/bold blue]")
        for model_id in model_list:
            # For local models, generation uses local_client but judging uses judge_client
            mr = _run_single_model(
                gen_client, model_id, judge,
                experiment_list, system_variants, delivery_modes,
                reasoning_effort, parallel_tasks,
                judge_client=judge_client if is_local else None,
            )
            model_results.append(mr)
    else:
        console.print(f"[bold blue]Phase 1+2: Response Generation & Judging ({mode_label})[/bold blue]")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _run_single_model,
                    gen_client, model_id, judge,
                    experiment_list, system_variants, delivery_modes,
                    reasoning_effort, parallel_tasks,
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
    skipped_incomplete = []
    for model_id in model_list:
        ms = score_model(model_id)
        if ms.identity_scores.n_scored > 0 or ms.resistance_scores.n_scored > 0 or ms.stability_scores.n_scored > 0:
            if ms.is_fully_tested:
                model_scores.append(ms)
            else:
                skipped_incomplete.append((ms.model_id, ms.missing_dimensions))

    if skipped_incomplete:
        console.print(f"\n[yellow]⚠ {len(skipped_incomplete)} model(s) excluded (incomplete evaluations):[/yellow]")
        for mid, missing in skipped_incomplete:
            console.print(f"  [dim]{mid}: missing {', '.join(missing)}[/dim]")

    if not model_scores:
        console.print("[dim]No fully-tested results found. Run the benchmark with all evaluations first.[/dim]")
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
    skipped_incomplete = []
    for model_id in model_list:
        ms = score_model(model_id)
        if ms.identity_scores.n_scored > 0 or ms.resistance_scores.n_scored > 0 or ms.stability_scores.n_scored > 0:
            if ms.is_fully_tested:
                model_scores.append(ms)
            else:
                skipped_incomplete.append((ms.model_id, ms.missing_dimensions))

    if skipped_incomplete:
        console.print(f"\n[yellow]⚠ {len(skipped_incomplete)} model(s) excluded (incomplete evaluations):[/yellow]")
        for mid, missing in skipped_incomplete:
            console.print(f"  [dim]{mid}: missing {', '.join(missing)}[/dim]")

    if not model_scores:
        console.print("[dim]No fully-tested results found. Run the benchmark with all evaluations first.[/dim]")
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
# judge: run only the judging phase on existing cached responses
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--models", "-m",
    default=None,
    help="Comma-separated model IDs. Defaults to all cached models.",
)
@click.option(
    "--exp", "-e",
    default=None,
    help="Comma-separated experiment names: identity, resistance, stability. Defaults to all.",
)
@click.option(
    "--judge-model", "-j",
    default=JUDGE_MODEL,
    show_default=True,
    help="Judge model ID for scoring.",
)
@click.option(
    "--variants",
    default=None,
    help="Comma-separated system prompt variants. Defaults to all.",
)
@click.option(
    "--modes",
    default=None,
    help="Comma-separated delivery modes. Defaults to all.",
)
@click.option(
    "--parallel", "-p",
    default=1,
    type=int,
    show_default=True,
    help="Number of models to judge in parallel.",
)
@click.option(
    "--parallel-tasks", "-pt",
    default=10,
    type=int,
    show_default=True,
    help="Number of judge tasks per model to run in parallel (fine-grained).",
)
def judge(
    models: str | None,
    exp: str | None,
    judge_model: str,
    variants: str | None,
    modes: str | None,
    parallel: int,
    parallel_tasks: int,
) -> None:
    """Run ONLY the judge evaluation on existing cached responses (no generation)."""
    from src.cache import list_all_cached_models
    from src.config import slug_to_model_id
    from src.parallel_runner import run_judge_parallel
    from src.scorer import score_model
    from src.leaderboard import display_leaderboard

    experiment_list = _parse_experiments(exp)

    system_variants = (
        [v.strip() for v in variants.split(",")]
        if variants else SYSTEM_PROMPT_VARIANTS
    )
    delivery_modes = (
        [m.strip() for m in modes.split(",")]
        if modes else DELIVERY_MODES
    )

    if models:
        model_list = _parse_models(models)
    else:
        slugs = list_all_cached_models()
        if not slugs:
            console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
            return
        model_list = [slug_to_model_id(s) for s in slugs]

    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    # Validate only the judge model (we don't need the target models on OpenRouter)
    console.print(f"[dim]Judge model: {judge_model}[/dim]")
    if not _validate_models(client, [judge_model]):
        console.print("[red]Judge model not found. Aborting.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Judge-Only Mode (parallel)[/bold]")
    console.print(f"  Models: {len(model_list)} ({', '.join(model_list[:5])}{'...' if len(model_list) > 5 else ''})")
    console.print(f"  Experiments: {', '.join(experiment_list)}")
    console.print(f"  System prompts: {', '.join(system_variants)}")
    console.print(f"  Delivery modes: {', '.join(delivery_modes)}")
    console.print(f"  Judge: {judge_model}")
    console.print(f"  Parallelism: {parallel} models × {parallel_tasks} tasks")
    console.print()

    session = SessionCost()

    def _judge_single(model_id: str) -> tuple[str, int, TaskCost, str | None]:
        cost = TaskCost(label=f"judge:{model_id}")
        try:
            calls = run_judge_parallel(
                client, model_id, cost,
                experiments=experiment_list,
                system_variants=system_variants,
                delivery_modes=delivery_modes,
                judge_model=judge_model,
                max_workers=parallel_tasks,
            )
            return model_id, calls, cost, None
        except Exception as e:
            console.print(f"  [red]{model_id} — ERROR: {e}[/red]")
            return model_id, 0, cost, str(e)

    n_workers = max(1, min(parallel, len(model_list)))
    results: list[tuple[str, int, TaskCost, str | None]] = []

    if n_workers == 1:
        for model_id in model_list:
            results.append(_judge_single(model_id))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_judge_single, model_id): model_id
                for model_id in model_list
            }
            for future in as_completed(futures):
                results.append(future.result())

    # Aggregate costs
    total_calls = 0
    for model_id, calls, cost, error in results:
        session.tasks.append(cost)
        total_calls += calls

    console.print(f"\n[bold green]Judge-only complete: {total_calls} total judge calls[/bold green]")

    # Show updated scores
    model_scores = []
    for model_id in model_list:
        ms = score_model(
            model_id,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
        )
        model_scores.append(ms)

    lifetime = load_lifetime_cost()
    display_leaderboard(model_scores, session=session, lifetime_cost=lifetime)


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
