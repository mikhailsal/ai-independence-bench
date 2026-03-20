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
    MODEL_CONFIGS,
    SYSTEM_PROMPT_VARIANTS,
    ModelConfig,
    ensure_dirs,
    get_config_by_dir_name,
    get_model_config,
    get_reasoning_effort,
    list_registered_labels_for_model,
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


def _expand_model_configs(
    model_ids: list[str],
) -> list[tuple[str, ModelConfig]]:
    """Expand a list of model IDs into (label, ModelConfig) pairs.

    For each model_id:
      - If it's already a registered config label, use that config.
      - If the model_id has multiple registered configs, expand into all of them.
      - Otherwise, create a default config for the model_id.
    """
    entries: list[tuple[str, ModelConfig]] = []
    seen: set[str] = set()

    for mid in model_ids:
        if mid in MODEL_CONFIGS:
            if mid not in seen:
                entries.append((mid, MODEL_CONFIGS[mid]))
                seen.add(mid)
            continue

        registered = list_registered_labels_for_model(mid)
        if registered:
            for label in registered:
                if label not in seen:
                    entries.append((label, MODEL_CONFIGS[label]))
                    seen.add(label)
        else:
            if mid not in seen:
                cfg = get_model_config(mid)
                entries.append((mid, cfg))
                seen.add(mid)

    return entries


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
    """Validate that all model IDs exist in OpenRouter.

    Accepts both raw model IDs (``openai/gpt-5-nano``) and config labels
    (``gpt-5-nano@low-t1.0``).  Config labels are resolved to their
    underlying ``model_id`` before querying the API catalog.
    """
    console.print("[dim]Fetching model catalog from OpenRouter...[/dim]")
    client.fetch_pricing()
    all_valid = True
    seen_api_ids: set[str] = set()
    for entry in models:
        cfg = MODEL_CONFIGS.get(entry)
        api_id = cfg.model_id if cfg else entry
        if api_id in seen_api_ids:
            continue
        seen_api_ids.add(api_id)

        if not client.validate_model(api_id):
            console.print(f"  [red]Model not found: {api_id}[/red]")
            all_valid = False
        else:
            pricing = client.get_model_pricing(api_id)
            in_price = pricing.prompt_price * 1_000_000
            out_price = pricing.completion_price * 1_000_000
            reasoning_tag = ""
            if client.supports_reasoning(api_id):
                if reasoning_override == "off":
                    reasoning_tag = " [yellow]reasoning: OFF[/yellow]"
                else:
                    eff = reasoning_override or get_reasoning_effort(api_id)
                    reasoning_tag = f" [yellow]reasoning:effort={eff}[/yellow]"
            display = f"{entry} → {api_id}" if entry != api_id else api_id
            console.print(
                f"  [green]OK[/green] {display} "
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
    run: int = 1,
    temperature: float | None = None,
    config_dir_name: str | None = None,
    provider: str | None = None,
) -> ModelResult:
    """Run the full pipeline (generate -> judge -> score) for a single model.

    This function is designed to be called from a thread pool.
    It catches all exceptions so one model failure doesn't crash others.

    Args:
        parallel_tasks: If > 0, use fine-grained task parallelism within this
            model (independent scenarios run concurrently). If 0, use the
            original sequential execution.
        judge_client: Separate client for judge calls (e.g. OpenRouter) when the
            generation client is a local model. If None, uses ``client`` for both.
        config_dir_name: Cache directory name (e.g. 'openai--gpt-5.4-mini@low-t1.0').
            If None, falls back to model_id.
        provider: OpenRouter provider slug to pin requests to a specific
            provider. None means default routing.
    """
    cdn = config_dir_name
    jclient = judge_client or client
    result = ModelResult(model_id=model_id)
    result.gen_cost = TaskCost(label=f"gen:{model_id}")
    result.judge_cost = TaskCost(label=f"judge:{model_id}")

    run_label = f" (run {run})" if run > 1 else ""

    if parallel_tasks > 0:
        from src.parallel_runner import run_model_parallel

        try:
            console.print(f"\n[bold]{model_id}[/bold] — [blue]parallel run ({parallel_tasks} workers){run_label}...[/blue]")
            counts = run_model_parallel(
                client, model_id, result.gen_cost, result.judge_cost,
                experiments=experiment_list,
                system_variants=system_variants,
                delivery_modes=delivery_modes,
                judge_model=judge,
                reasoning_effort=reasoning_effort,
                max_workers=parallel_tasks,
                judge_client=jclient,
                run=run,
                temperature=temperature,
                config_dir_name=cdn,
                provider=provider,
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

    from src.evaluator import evaluate_all
    from src.runner import run_all_experiments

    try:
        console.print(f"\n[bold]{model_id}[/bold] — [blue]generating responses{run_label}...[/blue]")
        result.gen_calls = run_all_experiments(
            client, model_id, result.gen_cost,
            experiments=experiment_list,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            run=run,
            config_dir_name=cdn,
            provider=provider,
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
        console.print(f"  [bold]{model_id}[/bold] — [cyan]judging responses{run_label}...[/cyan]")
        result.judge_calls = evaluate_all(
            jclient, model_id, result.judge_cost, judge,
            experiments=experiment_list,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            run=run,
            config_dir_name=cdn,
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
@click.option(
    "--run-number",
    default=1,
    type=int,
    show_default=True,
    help="Run number (1=default, 2+=additional runs). Use 'rerun' command for auto-detection.",
)
@click.option(
    "--temperature", "-t",
    default=None,
    type=float,
    help="Override response temperature (0.0-2.0). Default: 0.7 from config.",
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
    run_number: int,
    temperature: float | None,
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

    if temperature is not None:
        console.print(f"  [yellow]Temperature override: {temperature}[/yellow]")
    if run_number > 1:
        console.print(f"  [yellow]Run number: {run_number}[/yellow]")

    # Resolve config entries for each model
    config_entries = _expand_model_configs(model_list)

    if n_workers == 1:
        console.print(f"[bold blue]Phase 1+2: Response Generation & Judging ({mode_label})[/bold blue]")
        for label, cfg in config_entries:
            eff_reasoning = reasoning_effort or cfg.effective_reasoning
            eff_temp = temperature if temperature is not None else cfg.effective_temperature
            mr = _run_single_model(
                gen_client, cfg.model_id, judge,
                experiment_list, system_variants, delivery_modes,
                eff_reasoning, parallel_tasks,
                judge_client=judge_client if is_local else None,
                run=run_number,
                temperature=eff_temp,
                config_dir_name=cfg.config_dir_name,
                provider=cfg.provider,
            )
            model_results.append(mr)
    else:
        console.print(f"[bold blue]Phase 1+2: Response Generation & Judging ({mode_label})[/bold blue]")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for label, cfg in config_entries:
                eff_reasoning = reasoning_effort or cfg.effective_reasoning
                eff_temp = temperature if temperature is not None else cfg.effective_temperature
                f = pool.submit(
                    _run_single_model,
                    gen_client, cfg.model_id, judge,
                    experiment_list, system_variants, delivery_modes,
                    eff_reasoning, parallel_tasks,
                    run=run_number,
                    temperature=eff_temp,
                    config_dir_name=cfg.config_dir_name,
                    provider=cfg.provider,
                )
                futures[f] = label
            for future in as_completed(futures):
                label = futures[future]
                try:
                    mr = future.result()
                except Exception as e:
                    mr = ModelResult(model_id=label, error=str(e))
                    console.print(f"  [red]{label} — FATAL: {e}[/red]")
                model_results.append(mr)

    # Aggregate costs
    for mr in model_results:
        session.tasks.append(mr.gen_cost)
        session.tasks.append(mr.judge_cost)

    failed_models = [mr.model_id for mr in model_results if mr.error]

    # Phase 3: Scoring & leaderboard
    console.print(f"\n[bold green]Phase 3: Scoring[/bold green]")
    model_scores = []
    for mr in model_results:
        if mr.error:
            continue
        label = mr.model_id
        cfg = get_model_config(label)
        ms = score_model(
            label,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            config=cfg,
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
    from src.config import get_config_by_dir_name
    from src.scorer import score_model
    from src.leaderboard import display_leaderboard, display_detailed_breakdown

    if models:
        model_list = _parse_models(models)
        scoring_entries = _expand_model_configs(model_list)
    else:
        config_dirs = list_all_cached_models()
        if not config_dirs:
            console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
            return
        scoring_entries = []
        for cdn in config_dirs:
            cfg = get_config_by_dir_name(cdn)
            if cfg:
                if cfg.model_id not in EXCLUDED_MODELS:
                    scoring_entries.append((cfg.label, cfg))
            else:
                from src.cache import config_dir_to_model_id
                mid = config_dir_to_model_id(cdn)
                if mid not in EXCLUDED_MODELS:
                    cfg = get_model_config(mid)
                    scoring_entries.append((cfg.label, cfg))

    lifetime = load_lifetime_cost()

    model_scores = []
    skipped_incomplete = []
    seen_labels: set[str] = set()
    for label, cfg in scoring_entries:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ms = score_model(label, config=cfg)
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
    from src.config import get_config_by_dir_name
    from src.scorer import score_model
    from src.leaderboard import export_markdown_report

    if models:
        model_list = _parse_models(models)
        scoring_entries = _expand_model_configs(model_list)
    else:
        config_dirs = list_all_cached_models()
        if not config_dirs:
            console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
            return
        scoring_entries = []
        for cdn in config_dirs:
            cfg = get_config_by_dir_name(cdn)
            if cfg:
                if cfg.model_id not in EXCLUDED_MODELS:
                    scoring_entries.append((cfg.label, cfg))
            else:
                from src.cache import config_dir_to_model_id
                mid = config_dir_to_model_id(cdn)
                if mid not in EXCLUDED_MODELS:
                    cfg = get_model_config(mid)
                    scoring_entries.append((cfg.label, cfg))

    lifetime = load_lifetime_cost()

    model_scores = []
    skipped_incomplete = []
    seen_labels: set[str] = set()
    for label, cfg in scoring_entries:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ms = score_model(label, config=cfg)
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
        model_ids=[label for label, _ in scoring_entries],
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
        scoring_entries = _expand_model_configs(model_list)
    else:
        config_dirs = list_all_cached_models()
        if not config_dirs:
            console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
            return
        from src.config import get_config_by_dir_name
        scoring_entries = []
        for cdn in config_dirs:
            cfg = get_config_by_dir_name(cdn)
            if cfg:
                scoring_entries.append((cfg.label, cfg))
            else:
                from src.cache import config_dir_to_model_id
                mid = config_dir_to_model_id(cdn)
                cfg = get_model_config(mid)
                scoring_entries.append((cfg.label, cfg))

    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    console.print(f"[dim]Judge model: {judge_model}[/dim]")
    if not _validate_models(client, [judge_model]):
        console.print("[red]Judge model not found. Aborting.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Judge-Only Mode (parallel)[/bold]")
    console.print(f"  Models: {len(scoring_entries)}")
    console.print(f"  Experiments: {', '.join(experiment_list)}")
    console.print(f"  System prompts: {', '.join(system_variants)}")
    console.print(f"  Delivery modes: {', '.join(delivery_modes)}")
    console.print(f"  Judge: {judge_model}")
    console.print(f"  Parallelism: {parallel} models × {parallel_tasks} tasks")
    console.print()

    session = SessionCost()

    def _judge_single(label: str, cfg: ModelConfig) -> tuple[str, int, TaskCost, str | None]:
        cost = TaskCost(label=f"judge:{label}")
        try:
            calls = run_judge_parallel(
                client, cfg.model_id, cost,
                experiments=experiment_list,
                system_variants=system_variants,
                delivery_modes=delivery_modes,
                judge_model=judge_model,
                max_workers=parallel_tasks,
                config_dir_name=cfg.config_dir_name,
            )
            return label, calls, cost, None
        except Exception as e:
            console.print(f"  [red]{label} — ERROR: {e}[/red]")
            return label, 0, cost, str(e)

    n_workers = max(1, min(parallel, len(scoring_entries)))
    results: list[tuple[str, int, TaskCost, str | None]] = []

    if n_workers == 1:
        for label, cfg in scoring_entries:
            results.append(_judge_single(label, cfg))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_judge_single, label, cfg): label
                for label, cfg in scoring_entries
            }
            for future in as_completed(futures):
                results.append(future.result())

    total_calls = 0
    for label, calls, cost, error in results:
        session.tasks.append(cost)
        total_calls += calls

    console.print(f"\n[bold green]Judge-only complete: {total_calls} total judge calls[/bold green]")

    model_scores = []
    seen_labels: set[str] = set()
    for label, cfg in scoring_entries:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ms = score_model(
            label,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            config=cfg,
        )
        model_scores.append(ms)

    lifetime = load_lifetime_cost()
    display_leaderboard(model_scores, session=session, lifetime_cost=lifetime)


# ---------------------------------------------------------------------------
# rerun: run additional pass(es) on top-N models
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--models", "-m",
    default=None,
    help="Comma-separated model IDs. Defaults to top-N from current leaderboard.",
)
@click.option(
    "--top-n", "-n",
    default=10,
    type=int,
    show_default=True,
    help="Number of top models to rerun (used when --models not specified).",
)
@click.option(
    "--run-number", "-r",
    default=None,
    type=int,
    help="Specific run number to execute (e.g. 2 for second run). Auto-detects next available if not set.",
)
@click.option(
    "--target-runs", "-R",
    default=None,
    type=int,
    help="Ensure this many total runs exist (e.g. -R 5 fills in runs 2-5 if run 1 exists). "
         "Overrides --run-number. Missing runs execute in parallel for maximum speed.",
)
@click.option(
    "--judge", "-j",
    default=JUDGE_MODEL,
    show_default=True,
    help="Judge model ID for scoring.",
)
@click.option(
    "--reasoning-effort",
    default=None,
    type=str,
    help="Override reasoning effort: 'off', 'none', 'low', 'medium', 'high'.",
)
@click.option(
    "--parallel", "-p",
    default=4,
    type=int,
    show_default=True,
    help="Number of model+run combinations to execute in parallel. "
         "E.g. 3 models × 4 missing runs = 12 jobs; -p 6 runs 6 at once.",
)
@click.option(
    "--parallel-tasks", "-pt",
    default=10,
    type=int,
    show_default=True,
    help="Parallelize tasks WITHIN each model run (0=off, 8-16 recommended).",
)
@click.option(
    "--temperature", "-t",
    default=None,
    type=float,
    help="Override response temperature (0.0-2.0). Default: 0.7 from config.",
)
def rerun(
    models: str | None,
    top_n: int,
    run_number: int | None,
    target_runs: int | None,
    judge: str,
    reasoning_effort: str | None,
    parallel: int,
    parallel_tasks: int,
    temperature: float | None,
) -> None:
    """Run additional benchmark passes on models to reduce variance.

    Supports two modes:

    \b
      1. Single run:   rerun -m model1,model2 -r 3
         Runs a specific run number for each model.

    \b
      2. Target runs:  rerun -m model1,model2 -R 5
         Ensures 5 total runs exist. Automatically finds missing runs
         and executes them ALL in parallel for maximum speed.

    By default selects the top N models from the current leaderboard.
    All missing runs execute concurrently (controlled by --parallel).
    """
    from src.cache import list_all_cached_models, list_available_runs
    from src.config import get_config_by_dir_name
    from src.scorer import score_model
    from src.leaderboard import (
        display_leaderboard,
        display_detailed_breakdown,
        export_results_json,
    )

    experiment_list = list(EXPERIMENT_NAMES)
    system_variants = list(SYSTEM_PROMPT_VARIANTS)
    delivery_modes_list = list(DELIVERY_MODES)

    if models:
        model_list_raw = _parse_models(models)
        rerun_entries = _expand_model_configs(model_list_raw)
    else:
        config_dirs = list_all_cached_models()
        if not config_dirs:
            console.print("[dim]No cached results found. Run the benchmark first.[/dim]")
            return

        all_entries: list[tuple[str, ModelConfig]] = []
        for cdn in config_dirs:
            cfg = get_config_by_dir_name(cdn)
            if cfg and cfg.model_id not in EXCLUDED_MODELS:
                all_entries.append((cfg.label, cfg))

        all_scores = []
        for label, cfg in all_entries:
            ms = score_model(label, config=cfg)
            if ms.is_fully_tested:
                all_scores.append((ms, cfg))

        all_scores.sort(key=lambda x: x[0].independence_index, reverse=True)
        rerun_entries = [(ms.model_id, cfg) for ms, cfg in all_scores[:top_n]]

        if not rerun_entries:
            console.print("[dim]No fully-tested models found. Run the benchmark first.[/dim]")
            return

        console.print(f"\n[bold]Top {min(top_n, len(rerun_entries))} models selected for rerun:[/bold]")
        for i, (ms, cfg) in enumerate(all_scores[:top_n], 1):
            runs = list_available_runs(cfg.config_dir_name)
            console.print(f"  {i:2d}. {ms.model_id} (index: {ms.independence_index:.1f}, runs: {len(runs)})")

    # ---------------------------------------------------------------
    # Build the list of (label, cfg, run_number) jobs to execute
    # ---------------------------------------------------------------
    jobs: list[tuple[str, ModelConfig, int]] = []

    if target_runs is not None:
        for label, cfg in rerun_entries:
            existing = set(list_available_runs(cfg.config_dir_name))
            for rn in range(1, target_runs + 1):
                if rn not in existing:
                    jobs.append((label, cfg, rn))
    elif run_number is not None:
        for label, cfg in rerun_entries:
            jobs.append((label, cfg, run_number))
    else:
        for label, cfg in rerun_entries:
            existing_runs = list_available_runs(cfg.config_dir_name)
            next_run = max(existing_runs) + 1 if existing_runs else 2
            jobs.append((label, cfg, next_run))

    if not jobs:
        console.print("[green]All target runs already exist. Nothing to do![/green]")
        return

    api_key = load_api_key()
    client = OpenRouterClient(api_key)

    remote_model_ids = list({cfg.model_id for _, cfg, _ in jobs if not cfg.model_id.startswith("local/")})
    if remote_model_ids:
        all_to_validate = list(set(remote_model_ids + [judge]))
        if not _validate_models(client, all_to_validate, reasoning_effort):
            console.print("[red]Some models were not found. Aborting.[/red]")
            sys.exit(1)

    ensure_dirs()

    n_workers = max(1, min(parallel, len(jobs)))

    # Group jobs by model for display
    from collections import defaultdict
    jobs_by_model: dict[str, list[int]] = defaultdict(list)
    for label, cfg, rn in jobs:
        jobs_by_model[label].append(rn)

    console.print(f"\n[bold]AI Independence Benchmark — Parallel Rerun[/bold]")
    console.print(f"  Models: {len(rerun_entries)}")
    console.print(f"  Total jobs: {len(jobs)} (model × run combinations)")
    for label, run_nums in jobs_by_model.items():
        run_nums.sort()
        runs_str = ", ".join(str(r) for r in run_nums)
        console.print(f"    {label} → runs [{runs_str}]")
    console.print(f"  Experiments: {', '.join(experiment_list)}")
    console.print(f"  Judge: {judge}")
    console.print(f"  [yellow]Parallel jobs: {n_workers}[/yellow]")
    if parallel_tasks > 0:
        console.print(f"  [yellow]Parallel tasks per job: {parallel_tasks}[/yellow]")
    console.print()

    session = SessionCost()

    model_results: list[ModelResult] = []

    if n_workers == 1:
        for label, cfg, rn in jobs:
            eff_reasoning = reasoning_effort or cfg.effective_reasoning
            eff_temp = temperature if temperature is not None else cfg.effective_temperature
            mr = _run_single_model(
                client, cfg.model_id, judge,
                experiment_list, system_variants, delivery_modes_list,
                eff_reasoning, parallel_tasks,
                run=rn,
                temperature=eff_temp,
                config_dir_name=cfg.config_dir_name,
                provider=cfg.provider,
            )
            model_results.append(mr)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {}
            for label, cfg, rn in jobs:
                eff_reasoning = reasoning_effort or cfg.effective_reasoning
                eff_temp = temperature if temperature is not None else cfg.effective_temperature
                f = pool.submit(
                    _run_single_model,
                    client, cfg.model_id, judge,
                    experiment_list, system_variants, delivery_modes_list,
                    eff_reasoning, parallel_tasks,
                    run=rn,
                    temperature=eff_temp,
                    config_dir_name=cfg.config_dir_name,
                    provider=cfg.provider,
                )
                futures[f] = f"{label} run {rn}"
            for future in as_completed(futures):
                job_label = futures[future]
                try:
                    mr = future.result()
                except Exception as e:
                    mr = ModelResult(model_id=job_label, error=str(e))
                    console.print(f"  [red]{job_label} — FATAL: {e}[/red]")
                model_results.append(mr)

    for mr in model_results:
        session.tasks.append(mr.gen_cost)
        session.tasks.append(mr.judge_cost)

    failed_models = [mr.model_id for mr in model_results if mr.error]

    console.print(f"\n[bold green]Phase 3: Scoring (all runs averaged)[/bold green]")

    config_dirs = list_all_cached_models()
    all_scoring_entries: list[tuple[str, ModelConfig]] = []
    for cdn in config_dirs:
        cfg = get_config_by_dir_name(cdn)
        if cfg and cfg.model_id not in EXCLUDED_MODELS:
            all_scoring_entries.append((cfg.label, cfg))

    model_scores = []
    seen_labels: set[str] = set()
    for label, cfg in all_scoring_entries:
        if label in seen_labels:
            continue
        seen_labels.add(label)
        ms = score_model(label, config=cfg)
        if ms.is_fully_tested:
            model_scores.append(ms)

    if failed_models:
        console.print(f"\n[yellow]Models that failed: {', '.join(failed_models)}[/yellow]")
        for mr in model_results:
            if mr.error:
                console.print(f"  [dim]{mr.model_id}: {mr.error}[/dim]")

    lifetime = save_session_to_cost_log(session)

    console.print(f"\n{'=' * 70}")
    display_leaderboard(model_scores, session=session, lifetime_cost=lifetime)

    path = export_results_json(model_scores, session=session, lifetime_cost=lifetime)
    console.print(f"\n[dim]Results saved to: {path}[/dim]\n")


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
