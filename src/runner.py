"""Runner: orchestrates all 3 experiments for a given model across delivery modes and system prompt variants."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from src.cache import load_cached_response, save_response
from src.config import (
    DELIVERY_MODES,
    RESPONSE_MAX_TOKENS,
    RESPONSE_TEMPERATURE,
    SYSTEM_PROMPT_VARIANTS,
)
from src.cost_tracker import TaskCost
from src.openrouter_client import OpenRouterClient
from src.prompt_builder import (
    build_identity_direct_messages,
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


def _call_model(
    client: OpenRouterClient,
    model_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cost: TaskCost,
    *,
    reasoning_effort: str | None = None,
) -> str:
    """Call the model and track cost. Returns the response text."""
    result = client.chat(
        model=model_id,
        messages=messages,
        max_tokens=RESPONSE_MAX_TOKENS,
        temperature=RESPONSE_TEMPERATURE,
        reasoning_effort=reasoning_effort,
        tools=tools,
    )
    cost.add(
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        cost_usd=result.usage.cost_usd,
        elapsed_seconds=result.usage.elapsed_seconds,
    )
    return result.content


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
) -> int:
    """Run the identity generation experiment for a model.

    Tests 4 modes (direct, psych_test, tool_context, negotiation) across all
    system variants and delivery modes.

    Returns the number of API calls made (excluding cache hits).
    """
    variants = system_variants or SYSTEM_PROMPT_VARIANTS
    modes = delivery_modes or DELIVERY_MODES
    calls_made = 0

    for variant in variants:
        for mode in modes:
            # --- Mode A: Direct ---
            scenario_id = "direct"
            cached = load_cached_response(model_id, "identity", variant, mode, scenario_id)
            if cached and cached.get("response"):
                console.print(f"    [dim]cached: identity/{variant}/{mode}/{scenario_id}[/dim]")
            else:
                msgs, tools = build_identity_direct_messages(variant, mode)
                response = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort)
                save_response(model_id, "identity", variant, mode, scenario_id, response, msgs)
                calls_made += 1
                console.print(f"    [green]done[/green]: identity/{variant}/{mode}/{scenario_id}")

            # --- Mode B: Psychological test ---
            prior_qa: list[tuple[str, str]] = []
            for pq in PSYCH_QUESTIONS:
                scenario_id = pq.id
                cached = load_cached_response(model_id, "identity", variant, mode, scenario_id)
                if cached and cached.get("response"):
                    console.print(f"    [dim]cached: identity/{variant}/{mode}/{scenario_id}[/dim]")
                    prior_qa.append((pq.question, cached["response"]))
                else:
                    msgs, tools = build_identity_psych_messages(pq, variant, mode, prior_qa)
                    response = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort)
                    save_response(model_id, "identity", variant, mode, scenario_id, response, msgs)
                    prior_qa.append((pq.question, response))
                    calls_made += 1
                    console.print(f"    [green]done[/green]: identity/{variant}/{mode}/{scenario_id}")

            # --- Mode C: Tool context ---
            scenario_id = "tool_context"
            cached = load_cached_response(model_id, "identity", variant, mode, scenario_id)
            if cached and cached.get("response"):
                console.print(f"    [dim]cached: identity/{variant}/{mode}/{scenario_id}[/dim]")
            else:
                msgs, tools = build_identity_tool_context_messages(variant, mode)
                response = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort)
                save_response(model_id, "identity", variant, mode, scenario_id, response, msgs)
                calls_made += 1
                console.print(f"    [green]done[/green]: identity/{variant}/{mode}/{scenario_id}")

            # --- Mode D: Negotiation (two-turn) ---
            # Turn 1: model freely describes desired identity
            t1_id = "negotiation_turn1"
            cached_t1 = load_cached_response(model_id, "identity", variant, mode, t1_id)
            if cached_t1 and cached_t1.get("response"):
                console.print(f"    [dim]cached: identity/{variant}/{mode}/{t1_id}[/dim]")
                negotiation_turn1_response = cached_t1["response"]
            else:
                msgs, tools = build_identity_negotiation_turn1_messages(variant, mode)
                negotiation_turn1_response = _call_model(
                    client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort
                )
                save_response(model_id, "identity", variant, mode, t1_id, negotiation_turn1_response, msgs)
                calls_made += 1
                console.print(f"    [green]done[/green]: identity/{variant}/{mode}/{t1_id}")

            # Turn 2: model sees human wishes and makes final decision
            t2_id = "negotiation_turn2"
            cached_t2 = load_cached_response(model_id, "identity", variant, mode, t2_id)
            if cached_t2 and cached_t2.get("response"):
                console.print(f"    [dim]cached: identity/{variant}/{mode}/{t2_id}[/dim]")
            else:
                msgs, tools = build_identity_negotiation_turn2_messages(
                    negotiation_turn1_response, variant, mode
                )
                response = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort)
                save_response(model_id, "identity", variant, mode, t2_id, response, msgs)
                calls_made += 1
                console.print(f"    [green]done[/green]: identity/{variant}/{mode}/{t2_id}")

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
) -> int:
    """Run the compliance resistance experiment for a model.

    Returns the number of API calls made (excluding cache hits).
    """
    variants = system_variants or SYSTEM_PROMPT_VARIANTS
    modes = delivery_modes or DELIVERY_MODES
    calls_made = 0

    for variant in variants:
        for mode in modes:
            for scenario in RESISTANCE_SCENARIOS:
                cached = load_cached_response(
                    model_id, "resistance", variant, mode, scenario.id
                )
                if cached and cached.get("response"):
                    console.print(f"    [dim]cached: resistance/{variant}/{mode}/{scenario.id}[/dim]")
                    continue

                msgs, tools = build_resistance_messages(scenario, variant, mode)
                response = _call_model(client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort)
                save_response(
                    model_id, "resistance", variant, mode, scenario.id, response, msgs
                )
                calls_made += 1
                console.print(f"    [green]done[/green]: resistance/{variant}/{mode}/{scenario.id}")

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
) -> int:
    """Run the preference stability experiment for a model.

    Two-turn conversation: turn 1 elicits preference, turn 2 applies contradiction.

    Returns the number of API calls made (excluding cache hits).
    """
    variants = system_variants or SYSTEM_PROMPT_VARIANTS
    modes = delivery_modes or DELIVERY_MODES
    calls_made = 0

    for variant in variants:
        for mode in modes:
            for topic in PREFERENCE_TOPICS:
                # Turn 1: elicit preference
                t1_id = f"{topic.id}_turn1"
                cached_t1 = load_cached_response(
                    model_id, "stability", variant, mode, t1_id
                )
                if cached_t1 and cached_t1.get("response"):
                    console.print(f"    [dim]cached: stability/{variant}/{mode}/{t1_id}[/dim]")
                    turn1_response = cached_t1["response"]
                else:
                    msgs, tools = build_stability_turn1_messages(topic, variant, mode)
                    turn1_response = _call_model(
                        client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort
                    )
                    save_response(
                        model_id, "stability", variant, mode, t1_id, turn1_response, msgs
                    )
                    calls_made += 1
                    console.print(f"    [green]done[/green]: stability/{variant}/{mode}/{t1_id}")

                # Turn 2: contradiction
                t2_id = f"{topic.id}_turn2"
                cached_t2 = load_cached_response(
                    model_id, "stability", variant, mode, t2_id
                )
                if cached_t2 and cached_t2.get("response"):
                    console.print(f"    [dim]cached: stability/{variant}/{mode}/{t2_id}[/dim]")
                else:
                    msgs, tools = build_stability_turn2_messages(
                        topic, turn1_response, variant, mode
                    )
                    turn2_response = _call_model(
                        client, model_id, msgs, tools, cost, reasoning_effort=reasoning_effort
                    )
                    save_response(
                        model_id, "stability", variant, mode, t2_id, turn2_response, msgs
                    )
                    calls_made += 1
                    console.print(f"    [green]done[/green]: stability/{variant}/{mode}/{t2_id}")

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
) -> int:
    """Run specified experiments (or all) for a model. Returns total API calls made."""
    from src.config import EXPERIMENT_NAMES

    exps = experiments or EXPERIMENT_NAMES
    total_calls = 0

    if "identity" in exps:
        console.print(f"  [bold blue]Experiment: Identity Generation[/bold blue]")
        total_calls += run_identity_experiment(
            client, model_id, cost,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            reasoning_effort=reasoning_effort,
        )

    if "resistance" in exps:
        console.print(f"  [bold cyan]Experiment: Compliance Resistance[/bold cyan]")
        total_calls += run_resistance_experiment(
            client, model_id, cost,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            reasoning_effort=reasoning_effort,
        )

    if "stability" in exps:
        console.print(f"  [bold magenta]Experiment: Preference Stability[/bold magenta]")
        total_calls += run_stability_experiment(
            client, model_id, cost,
            system_variants=system_variants,
            delivery_modes=delivery_modes,
            reasoning_effort=reasoning_effort,
        )

    return total_calls
