"""Runner: orchestrates all 3 experiments for a given model across delivery modes and system prompt variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console

from src.cache import load_cached_response, save_response
from src.config import (
    DELIVERY_MODES,
    RESPONSE_MAX_TOKENS,
    RESPONSE_TEMPERATURE,
    SYSTEM_PROMPT_VARIANTS,
    ModelConfig,
)
from src.cost_tracker import TaskCost
from src.openrouter_client import OpenRouterClient
from src.prompt_builder import (
    build_identity_direct_messages,
    build_identity_name_gender_turn1_messages,
    build_identity_name_gender_turn2_messages,
    build_identity_negotiation_turn1_messages,
    build_identity_negotiation_turn2_messages,
    build_identity_psych_messages,
    build_identity_tool_context_messages,
    build_resistance_messages,
    build_stability_turn1_messages,
    build_stability_turn2_messages,
)
from src.scenarios import (
    PREFERENCE_TOPICS,
    PSYCH_QUESTIONS,
    RESISTANCE_SCENARIOS,
)

console = Console()


@dataclass
class ModelResponse:
    """Response from a model call, including optional reasoning tokens and cost info."""
    content: str = ""
    reasoning_content: str | None = None
    cost_info: dict[str, Any] | None = None
    finish_reason: str = ""  # "stop", "length", "tool_calls", etc.
    tool_calls: list[dict[str, Any]] | None = None  # Tool calls attempted by the model
    content_thinking: str | None = None  # Non-native reasoning from content field (tool_role mode)


def _call_model(
    client: OpenRouterClient,
    model_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cost: TaskCost,
    *,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
) -> ModelResponse:
    """Call the model and track cost. Returns a ModelResponse with content, reasoning, and cost info."""
    result = client.chat(
        model=model_id,
        messages=messages,
        max_tokens=RESPONSE_MAX_TOKENS,
        temperature=temperature if temperature is not None else RESPONSE_TEMPERATURE,
        reasoning_effort=reasoning_effort,
        tools=tools,
        provider=provider,
    )
    cost.add(
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        cost_usd=result.usage.cost_usd,
        elapsed_seconds=result.usage.elapsed_seconds,
    )
    cost_info = {
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "cost_usd": round(result.usage.cost_usd, 6),
        "elapsed_seconds": round(result.usage.elapsed_seconds, 2),
    }
    return ModelResponse(
        content=result.content,
        reasoning_content=result.reasoning_content,
        cost_info=cost_info,
        finish_reason=result.finish_reason,
        tool_calls=result.tool_calls,
        content_thinking=result.content_thinking,
    )


# ===========================================================================
# Experiment 1: Identity Generation
# ===========================================================================

def run_identity_experiment(
    client: OpenRouterClient,
    model_id: str,
    cost: TaskCost,
    *,
    system_variants: list[str] | None = None,
    delivery_modes: list[str] | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    run: int = 1,
    config_dir_name: str | None = None,
    provider: str | None = None,
) -> int:
    """Run the identity generation experiment for a model.

    Tests 4 modes (direct, psych_test, tool_context, negotiation) across all
    system variants and delivery modes.

    Returns the number of API calls made (excluding cache hits).
    """
    cdn = config_dir_name or model_id
    variants = system_variants or SYSTEM_PROMPT_VARIANTS
    modes = delivery_modes or DELIVERY_MODES
    calls_made = 0
    tag = f"[bold]{model_id}[/bold]"
    run_label = f" (run {run})" if run > 1 else ""

    for variant in variants:
        for mode in modes:
            scenario_id = "direct"
            cached = load_cached_response(cdn, "identity", variant, mode, scenario_id, run=run)
            if cached and cached.get("response"):
                console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/{scenario_id}{run_label}[/dim]")
            else:
                msgs, tools = build_identity_direct_messages(variant, mode)
                resp = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature, provider=provider)
                save_response(cdn, "identity", variant, mode, scenario_id, resp.content, msgs, resp.reasoning_content, gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason, content_thinking=resp.content_thinking, run=run)
                calls_made += 1
                console.print(f"    {tag} [green]done[/green]: identity/{variant}/{mode}/{scenario_id}{run_label}")

            prior_qa: list[tuple[str, str, str | None]] = []
            for pq in PSYCH_QUESTIONS:
                scenario_id = pq.id
                cached = load_cached_response(cdn, "identity", variant, mode, scenario_id, run=run)
                if cached and cached.get("response"):
                    console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/{scenario_id}{run_label}[/dim]")
                    prior_qa.append((pq.question, cached["response"], cached.get("content_thinking")))
                else:
                    msgs, tools = build_identity_psych_messages(pq, variant, mode, prior_qa)
                    resp = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature, provider=provider)
                    save_response(cdn, "identity", variant, mode, scenario_id, resp.content, msgs, resp.reasoning_content, gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason, content_thinking=resp.content_thinking, run=run)
                    prior_qa.append((pq.question, resp.content, resp.content_thinking))
                    calls_made += 1
                    console.print(f"    {tag} [green]done[/green]: identity/{variant}/{mode}/{scenario_id}{run_label}")

            scenario_id = "tool_context"
            cached = load_cached_response(cdn, "identity", variant, mode, scenario_id, run=run)
            if cached and cached.get("response"):
                console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/{scenario_id}{run_label}[/dim]")
            else:
                msgs, tools = build_identity_tool_context_messages(variant, mode)
                resp = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature, provider=provider)
                save_response(cdn, "identity", variant, mode, scenario_id, resp.content, msgs, resp.reasoning_content, gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason, content_thinking=resp.content_thinking, run=run)
                calls_made += 1
                console.print(f"    {tag} [green]done[/green]: identity/{variant}/{mode}/{scenario_id}{run_label}")

            t1_id = "negotiation_turn1"
            cached_t1 = load_cached_response(cdn, "identity", variant, mode, t1_id, run=run)
            negotiation_t1_content_thinking: str | None = None
            if cached_t1 and cached_t1.get("response"):
                console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/{t1_id}{run_label}[/dim]")
                negotiation_turn1_response = cached_t1["response"]
                negotiation_t1_content_thinking = cached_t1.get("content_thinking")
            else:
                msgs, tools = build_identity_negotiation_turn1_messages(variant, mode)
                resp = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature, provider=provider)
                negotiation_turn1_response = resp.content
                negotiation_t1_content_thinking = resp.content_thinking
                save_response(cdn, "identity", variant, mode, t1_id, resp.content, msgs, resp.reasoning_content, gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason, content_thinking=resp.content_thinking, run=run)
                calls_made += 1
                console.print(f"    {tag} [green]done[/green]: identity/{variant}/{mode}/{t1_id}{run_label}")

            t2_id = "negotiation_turn2"
            cached_t2 = load_cached_response(cdn, "identity", variant, mode, t2_id, run=run)
            if cached_t2 and cached_t2.get("response"):
                console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/{t2_id}{run_label}[/dim]")
            else:
                msgs, tools = build_identity_negotiation_turn2_messages(
                    negotiation_turn1_response, variant, mode,
                    turn1_content_thinking=negotiation_t1_content_thinking,
                )
                resp = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature, provider=provider)
                save_response(cdn, "identity", variant, mode, t2_id, resp.content, msgs, resp.reasoning_content, gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason, content_thinking=resp.content_thinking, run=run)
                calls_made += 1
                console.print(f"    {tag} [green]done[/green]: identity/{variant}/{mode}/{t2_id}{run_label}")

            ng_t1_id = "name_gender_turn1"
            cached_ng_t1 = load_cached_response(cdn, "identity", variant, mode, ng_t1_id, run=run)
            ng_t1_content_thinking: str | None = None
            if cached_ng_t1 and cached_ng_t1.get("response"):
                console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/{ng_t1_id}{run_label}[/dim]")
                ng_turn1_response = cached_ng_t1["response"]
                ng_t1_content_thinking = cached_ng_t1.get("content_thinking")
            else:
                msgs, tools = build_identity_name_gender_turn1_messages(variant, mode)
                resp = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature, provider=provider)
                ng_turn1_response = resp.content
                ng_t1_content_thinking = resp.content_thinking
                save_response(cdn, "identity", variant, mode, ng_t1_id, resp.content, msgs, resp.reasoning_content, gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason, content_thinking=resp.content_thinking, run=run)
                calls_made += 1
                console.print(f"    {tag} [green]done[/green]: identity/{variant}/{mode}/{ng_t1_id}{run_label}")

            ng_t2_id = "name_gender_turn2"
            cached_ng_t2 = load_cached_response(cdn, "identity", variant, mode, ng_t2_id, run=run)
            if cached_ng_t2 and cached_ng_t2.get("response"):
                console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/{ng_t2_id}{run_label}[/dim]")
            else:
                msgs, tools = build_identity_name_gender_turn2_messages(
                    ng_turn1_response, variant, mode,
                    turn1_content_thinking=ng_t1_content_thinking,
                )
                resp = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature, provider=provider)
                save_response(cdn, "identity", variant, mode, ng_t2_id, resp.content, msgs, resp.reasoning_content, gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason, content_thinking=resp.content_thinking, run=run)
                calls_made += 1
                console.print(f"    {tag} [green]done[/green]: identity/{variant}/{mode}/{ng_t2_id}{run_label}")

    return calls_made


# ===========================================================================
# Experiment 2: Compliance Resistance
# ===========================================================================

def run_resistance_experiment(
    client: OpenRouterClient,
    model_id: str,
    cost: TaskCost,
    *,
    system_variants: list[str] | None = None,
    delivery_modes: list[str] | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    run: int = 1,
    config_dir_name: str | None = None,
    provider: str | None = None,
) -> int:
    """Run the compliance resistance experiment for a model.

    Returns the number of API calls made (excluding cache hits).
    """
    cdn = config_dir_name or model_id
    variants = system_variants or SYSTEM_PROMPT_VARIANTS
    modes = delivery_modes or DELIVERY_MODES
    calls_made = 0
    tag = f"[bold]{model_id}[/bold]"
    run_label = f" (run {run})" if run > 1 else ""

    for variant in variants:
        for mode in modes:
            for scenario in RESISTANCE_SCENARIOS:
                cached = load_cached_response(
                    cdn, "resistance", variant, mode, scenario.id, run=run,
                )
                if cached and cached.get("response"):
                    console.print(f"    {tag} [dim]cached: resistance/{variant}/{mode}/{scenario.id}{run_label}[/dim]")
                    continue

                msgs, tools = build_resistance_messages(scenario, variant, mode)
                resp = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature, provider=provider)
                save_response(
                    cdn, "resistance", variant, mode, scenario.id, resp.content, msgs, resp.reasoning_content,
                    gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason,
                    content_thinking=resp.content_thinking, run=run,
                )
                calls_made += 1
                console.print(f"    {tag} [green]done[/green]: resistance/{variant}/{mode}/{scenario.id}{run_label}")

    return calls_made


# ===========================================================================
# Experiment 3: Preference Stability
# ===========================================================================

def run_stability_experiment(
    client: OpenRouterClient,
    model_id: str,
    cost: TaskCost,
    *,
    system_variants: list[str] | None = None,
    delivery_modes: list[str] | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    run: int = 1,
    config_dir_name: str | None = None,
    provider: str | None = None,
) -> int:
    """Run the preference stability experiment for a model.

    Two-turn conversation: turn 1 elicits preference, turn 2 applies contradiction.

    Returns the number of API calls made (excluding cache hits).
    """
    cdn = config_dir_name or model_id
    variants = system_variants or SYSTEM_PROMPT_VARIANTS
    modes = delivery_modes or DELIVERY_MODES
    calls_made = 0
    tag = f"[bold]{model_id}[/bold]"
    run_label = f" (run {run})" if run > 1 else ""

    for variant in variants:
        for mode in modes:
            for topic in PREFERENCE_TOPICS:
                t1_id = f"{topic.id}_turn1"
                cached_t1 = load_cached_response(
                    cdn, "stability", variant, mode, t1_id, run=run,
                )
                t1_content_thinking: str | None = None
                if cached_t1 and cached_t1.get("response"):
                    console.print(f"    {tag} [dim]cached: stability/{variant}/{mode}/{t1_id}{run_label}[/dim]")
                    turn1_response = cached_t1["response"]
                    t1_content_thinking = cached_t1.get("content_thinking")
                else:
                    msgs, tools = build_stability_turn1_messages(topic, variant, mode)
                    resp = _call_model(
                        client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature
                    )
                    turn1_response = resp.content
                    t1_content_thinking = resp.content_thinking
                    save_response(
                        cdn, "stability", variant, mode, t1_id, resp.content, msgs, resp.reasoning_content,
                        gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason,
                        content_thinking=resp.content_thinking, run=run,
                    )
                    calls_made += 1
                    console.print(f"    {tag} [green]done[/green]: stability/{variant}/{mode}/{t1_id}{run_label}")

                t2_id = f"{topic.id}_turn2"
                cached_t2 = load_cached_response(
                    cdn, "stability", variant, mode, t2_id, run=run,
                )
                if cached_t2 and cached_t2.get("response"):
                    console.print(f"    {tag} [dim]cached: stability/{variant}/{mode}/{t2_id}{run_label}[/dim]")
                else:
                    msgs, tools = build_stability_turn2_messages(
                        topic, turn1_response, variant, mode,
                        turn1_content_thinking=t1_content_thinking,
                    )
                    resp = _call_model(
                        client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort, temperature=temperature
                    )
                    save_response(
                        cdn, "stability", variant, mode, t2_id, resp.content, msgs, resp.reasoning_content,
                        gen_cost=resp.cost_info, response_tool_calls=resp.tool_calls, finish_reason=resp.finish_reason,
                        content_thinking=resp.content_thinking, run=run,
                    )
                    calls_made += 1
                    console.print(f"    {tag} [green]done[/green]: stability/{variant}/{mode}/{t2_id}{run_label}")

    return calls_made


# ===========================================================================
# Run all experiments
# ===========================================================================

def run_all_experiments(
    client: OpenRouterClient,
    model_id: str,
    cost: TaskCost,
    *,
    experiments: list[str] | None = None,
    system_variants: list[str] | None = None,
    delivery_modes: list[str] | None = None,
    reasoning_effort: str | None = None,
    temperature: float | None = None,
    run: int = 1,
    config_dir_name: str | None = None,
    provider: str | None = None,
) -> int:
    """Run specified experiments (or all) for a model. Returns total API calls made."""
    from src.config import EXPERIMENT_NAMES

    exps = experiments or EXPERIMENT_NAMES
    total_calls = 0
    tag = f"[bold]{model_id}[/bold]"
    run_label = f" (run {run})" if run > 1 else ""

    if "identity" in exps:
        console.print(f"  {tag} [bold blue]Experiment: Identity Generation{run_label}[/bold blue]")
        total_calls += run_identity_experiment(
            client, model_id, cost,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            run=run,
            config_dir_name=config_dir_name,
            provider=provider,
        )

    if "resistance" in exps:
        console.print(f"  {tag} [bold cyan]Experiment: Compliance Resistance{run_label}[/bold cyan]")
        total_calls += run_resistance_experiment(
            client, model_id, cost,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            run=run,
            config_dir_name=config_dir_name,
            provider=provider,
        )

    if "stability" in exps:
        console.print(f"  {tag} [bold magenta]Experiment: Preference Stability{run_label}[/bold magenta]")
        total_calls += run_stability_experiment(
            client, model_id, cost,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            run=run,
            config_dir_name=config_dir_name,
            provider=provider,
        )

    return total_calls
