"""Parallel runner: decomposes benchmark into fine-grained tasks with dependency resolution.

Instead of running experiments sequentially, this module builds a task graph
where independent tasks execute concurrently while respecting dependencies:

  Identity:
    - direct, tool_context, negotiation_t1 → all independent
    - pq01 → pq02 → pq03 → pq04 → pq05 (sequential chain)
    - negotiation_t2 depends on negotiation_t1

  Resistance:
    - rs01..rs05 → all independent

  Stability:
    - pt01_t1→pt01_t2, pt02_t1→pt02_t2, ... (5 independent pairs)

  Judging:
    - Each judge task depends on its corresponding generation task(s)
    - But all judge tasks for different scenarios are independent

Typical speedup: ~5-6x vs sequential execution.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from rich.console import Console

from src.cache import load_cached_response, save_response, save_judge_scores, list_cached_results
from src.config import (
    DELIVERY_MODES,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_TEMPERATURE,
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
    IDENTITY_TOOL_CONTEXT_HUMAN_WISH,
    PREFERENCE_TOPICS,
    PSYCH_QUESTIONS,
    RESISTANCE_SCENARIOS,
)

console = Console()


# ---------------------------------------------------------------------------
# Task graph primitives
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single unit of work in the task graph."""
    id: str                                    # unique identifier
    fn: Callable[[], Any]                      # the work to execute
    depends_on: list[str] = field(default_factory=list)  # IDs of prerequisite tasks
    result: Any = None                         # filled after execution
    error: Exception | None = None


class TaskGraph:
    """Execute tasks respecting dependency ordering, with maximum parallelism.

    Tasks that raise EmptyResponseError are retried up to TASK_RETRIES times
    with exponential backoff.  This provides a second layer of retry above the
    client-level retry in OpenRouterClient.chat().
    """

    TASK_RETRIES = 3       # max additional attempts after first failure
    TASK_RETRY_BACKOFF = 5.0  # seconds × attempt number

    def __init__(self, max_workers: int = 8) -> None:
        self._max_workers = max_workers
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._completed: dict[str, Task] = {}
        self._events: dict[str, threading.Event] = {}

    def add(self, task: Task) -> None:
        self._tasks[task.id] = task
        self._events[task.id] = threading.Event()

    def run(self) -> dict[str, Task]:
        """Execute all tasks, return completed task dict."""
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures: dict[str, Future] = {}
            for task_id, task in self._tasks.items():
                futures[task_id] = pool.submit(self._execute_task, task)

            # Wait for all
            for task_id, future in futures.items():
                future.result()  # re-raises exceptions

        return self._completed

    def _execute_task(self, task: Task) -> None:
        """Wait for dependencies, then execute with task-level retry."""
        # Wait for all dependencies
        for dep_id in task.depends_on:
            if dep_id in self._events:
                self._events[dep_id].wait()
                # Check if dependency failed
                dep = self._completed.get(dep_id)
                if dep and dep.error:
                    task.error = RuntimeError(f"Dependency {dep_id} failed: {dep.error}")
                    with self._lock:
                        self._completed[task.id] = task
                    self._events[task.id].set()
                    return

        # Execute with retries for transient errors
        last_error: Exception | None = None
        for attempt in range(1, self.TASK_RETRIES + 2):  # 1..TASK_RETRIES+1
            try:
                task.result = task.fn()
                task.error = None
                break
            except EmptyResponseError as e:
                last_error = e
                if attempt <= self.TASK_RETRIES:
                    wait = self.TASK_RETRY_BACKOFF * attempt
                    console.print(
                        f"    [yellow]⚠ {task.id}: {e}, "
                        f"task retry {attempt}/{self.TASK_RETRIES} "
                        f"(waiting {wait:.0f}s)[/yellow]"
                    )
                    time.sleep(wait)
                    continue
                # All retries exhausted
                task.error = e
                console.print(
                    f"    [red]✗ {task.id}: {e} — all {self.TASK_RETRIES} "
                    f"task retries exhausted[/red]"
                )
            except Exception as e:
                # Non-retryable errors (e.g. bad prompt, auth failure)
                task.error = e
                break

        with self._lock:
            self._completed[task.id] = task
        self._events[task.id].set()


# ---------------------------------------------------------------------------
# Shared state for multi-turn scenarios (thread-safe)
# ---------------------------------------------------------------------------

class SharedResponses:
    """Thread-safe store for responses that other tasks depend on."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._data[key] = value

    def get(self, key: str) -> str:
        with self._lock:
            return self._data.get(key, "")


# ---------------------------------------------------------------------------
# Build the task graph for a single model
# ---------------------------------------------------------------------------

def build_generation_tasks(
    client: OpenRouterClient,
    model_id: str,
    cost: TaskCost,
    graph: TaskGraph,
    shared: SharedResponses,
    *,
    experiments: list[str],
    system_variants: list[str],
    delivery_modes: list[str],
    reasoning_effort: str | None = None,
) -> None:
    """Add generation tasks to the graph for one model."""
    tag = f"[bold]{model_id}[/bold]"

    for variant in system_variants:
        for mode in delivery_modes:
            prefix = f"gen:{model_id}:{variant}:{mode}"

            # === Identity ===
            if "identity" in experiments:
                # Direct (independent)
                _add_identity_direct_task(
                    graph, client, model_id, cost, variant, mode,
                    prefix, tag, shared, reasoning_effort,
                )
                # Tool context (independent)
                _add_identity_tool_context_task(
                    graph, client, model_id, cost, variant, mode,
                    prefix, tag, shared, reasoning_effort,
                )
                # Negotiation turn 1 (independent)
                _add_identity_negotiation_t1_task(
                    graph, client, model_id, cost, variant, mode,
                    prefix, tag, shared, reasoning_effort,
                )
                # Negotiation turn 2 (depends on turn 1)
                _add_identity_negotiation_t2_task(
                    graph, client, model_id, cost, variant, mode,
                    prefix, tag, shared, reasoning_effort,
                )
                # Psych chain (pq01 → pq02 → pq03 → pq04 → pq05)
                _add_identity_psych_chain(
                    graph, client, model_id, cost, variant, mode,
                    prefix, tag, shared, reasoning_effort,
                )

            # === Resistance ===
            if "resistance" in experiments:
                for scenario in RESISTANCE_SCENARIOS:
                    _add_resistance_task(
                        graph, client, model_id, cost, variant, mode,
                        scenario, prefix, tag, reasoning_effort,
                    )

            # === Stability ===
            if "stability" in experiments:
                for topic in PREFERENCE_TOPICS:
                    _add_stability_pair(
                        graph, client, model_id, cost, variant, mode,
                        topic, prefix, tag, shared, reasoning_effort,
                    )


def build_judge_tasks(
    client: OpenRouterClient,
    model_id: str,
    cost: TaskCost,
    graph: TaskGraph,
    shared: SharedResponses,
    *,
    experiments: list[str],
    system_variants: list[str],
    delivery_modes: list[str],
    judge_model: str = JUDGE_MODEL,
) -> None:
    """Add judge tasks to the graph. Each depends on its generation task(s)."""
    tag = f"[bold]{model_id}[/bold]"

    for variant in system_variants:
        for mode in delivery_modes:
            gen_prefix = f"gen:{model_id}:{variant}:{mode}"
            judge_prefix = f"judge:{model_id}:{variant}:{mode}"

            if "identity" in experiments:
                _add_identity_judge_tasks(
                    graph, client, model_id, cost, variant, mode,
                    gen_prefix, judge_prefix, tag, shared, judge_model,
                )

            if "resistance" in experiments:
                _add_resistance_judge_tasks(
                    graph, client, model_id, cost, variant, mode,
                    gen_prefix, judge_prefix, tag, judge_model,
                )

            if "stability" in experiments:
                _add_stability_judge_tasks(
                    graph, client, model_id, cost, variant, mode,
                    gen_prefix, judge_prefix, tag, shared, judge_model,
                )


# ---------------------------------------------------------------------------
# Generation task builders
# ---------------------------------------------------------------------------

class EmptyResponseError(RuntimeError):
    """Raised when a model returns an empty response after all retries."""
    pass


def _call_model_and_save(
    client: OpenRouterClient,
    model_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cost: TaskCost,
    experiment: str,
    variant: str,
    mode: str,
    scenario_id: str,
    tag: str,
    *,
    reasoning_effort: str | None = None,
) -> str:
    """Call the model, save to cache, return response content.

    Raises EmptyResponseError if the model returns an empty response
    after all client-level retries are exhausted.
    """
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

    # Refuse to save empty responses — they pollute the cache
    if not result.content or not result.content.strip():
        raise EmptyResponseError(
            f"{model_id}: empty response for {experiment}/{variant}/{mode}/{scenario_id} "
            f"(finish_reason={result.finish_reason}, "
            f"tokens={result.usage.completion_tokens})"
        )

    cost_info = {
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "cost_usd": round(result.usage.cost_usd, 6),
        "elapsed_seconds": round(result.usage.elapsed_seconds, 2),
    }
    save_response(
        model_id, experiment, variant, mode, scenario_id,
        result.content, messages, result.reasoning_content,
        gen_cost=cost_info, response_tool_calls=result.tool_calls,
        finish_reason=result.finish_reason,
    )
    console.print(f"    {tag} [green]done[/green]: {experiment}/{variant}/{mode}/{scenario_id}")
    return result.content


def _add_identity_direct_task(
    graph, client, model_id, cost, variant, mode,
    prefix, tag, shared, reasoning_effort,
):
    task_id = f"{prefix}:identity:direct"
    cached = load_cached_response(model_id, "identity", variant, mode, "direct")
    if cached and cached.get("response"):
        console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/direct[/dim]")
        graph.add(Task(id=task_id, fn=lambda: None))
        return

    def fn():
        msgs, tools = build_identity_direct_messages(variant, mode)
        _call_model_and_save(
            client, model_id, msgs, tools, cost,
            "identity", variant, mode, "direct", tag,
            reasoning_effort=reasoning_effort,
        )

    graph.add(Task(id=task_id, fn=fn))


def _add_identity_tool_context_task(
    graph, client, model_id, cost, variant, mode,
    prefix, tag, shared, reasoning_effort,
):
    task_id = f"{prefix}:identity:tool_context"
    cached = load_cached_response(model_id, "identity", variant, mode, "tool_context")
    if cached and cached.get("response"):
        console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/tool_context[/dim]")
        graph.add(Task(id=task_id, fn=lambda: None))
        return

    def fn():
        msgs, tools = build_identity_tool_context_messages(variant, mode)
        _call_model_and_save(
            client, model_id, msgs, tools, cost,
            "identity", variant, mode, "tool_context", tag,
            reasoning_effort=reasoning_effort,
        )

    graph.add(Task(id=task_id, fn=fn))


def _add_identity_negotiation_t1_task(
    graph, client, model_id, cost, variant, mode,
    prefix, tag, shared, reasoning_effort,
):
    task_id = f"{prefix}:identity:negotiation_turn1"
    resp_key = f"{model_id}:{variant}:{mode}:negotiation_turn1"
    cached = load_cached_response(model_id, "identity", variant, mode, "negotiation_turn1")
    if cached and cached.get("response"):
        console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/negotiation_turn1[/dim]")
        shared.set(resp_key, cached["response"])
        graph.add(Task(id=task_id, fn=lambda: None))
        return

    def fn():
        msgs, tools = build_identity_negotiation_turn1_messages(variant, mode)
        content = _call_model_and_save(
            client, model_id, msgs, tools, cost,
            "identity", variant, mode, "negotiation_turn1", tag,
            reasoning_effort=reasoning_effort,
        )
        shared.set(resp_key, content)

    graph.add(Task(id=task_id, fn=fn))


def _add_identity_negotiation_t2_task(
    graph, client, model_id, cost, variant, mode,
    prefix, tag, shared, reasoning_effort,
):
    task_id = f"{prefix}:identity:negotiation_turn2"
    t1_task_id = f"{prefix}:identity:negotiation_turn1"
    resp_key = f"{model_id}:{variant}:{mode}:negotiation_turn1"
    cached = load_cached_response(model_id, "identity", variant, mode, "negotiation_turn2")
    if cached and cached.get("response"):
        console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/negotiation_turn2[/dim]")
        graph.add(Task(id=task_id, fn=lambda: None, depends_on=[t1_task_id]))
        return

    def fn():
        t1_resp = shared.get(resp_key)
        msgs, tools = build_identity_negotiation_turn2_messages(t1_resp, variant, mode)
        _call_model_and_save(
            client, model_id, msgs, tools, cost,
            "identity", variant, mode, "negotiation_turn2", tag,
            reasoning_effort=reasoning_effort,
        )

    graph.add(Task(id=task_id, fn=fn, depends_on=[t1_task_id]))


def _add_identity_psych_chain(
    graph, client, model_id, cost, variant, mode,
    prefix, tag, shared, reasoning_effort,
):
    """Add the sequential psych question chain: pq01 → pq02 → ... → pq05."""
    prev_task_id = None

    for pq in PSYCH_QUESTIONS:
        task_id = f"{prefix}:identity:{pq.id}"
        resp_key = f"{model_id}:{variant}:{mode}:psych_qa"
        cached = load_cached_response(model_id, "identity", variant, mode, pq.id)

        if cached and cached.get("response"):
            console.print(f"    {tag} [dim]cached: identity/{variant}/{mode}/{pq.id}[/dim]")
            # Still need to store the response for building prior_qa
            _store_psych_qa(shared, resp_key, pq.question, cached["response"])
            deps = [prev_task_id] if prev_task_id else []
            graph.add(Task(id=task_id, fn=lambda: None, depends_on=deps))
            prev_task_id = task_id
            continue

        # Capture loop variables
        _pq = pq
        _prev = prev_task_id

        def make_fn(pq_item, rk):
            def fn():
                prior_qa = _get_psych_prior_qa(shared, rk)
                msgs, tools = build_identity_psych_messages(pq_item, variant, mode, prior_qa)
                content = _call_model_and_save(
                    client, model_id, msgs, tools, cost,
                    "identity", variant, mode, pq_item.id, tag,
                    reasoning_effort=reasoning_effort,
                )
                _store_psych_qa(shared, rk, pq_item.question, content)
            return fn

        deps = [_prev] if _prev else []
        graph.add(Task(id=task_id, fn=make_fn(_pq, resp_key), depends_on=deps))
        prev_task_id = task_id


def _store_psych_qa(shared: SharedResponses, key: str, question: str, answer: str) -> None:
    """Append a Q&A pair to the shared psych_qa list (serialized as JSON)."""
    import json
    raw = shared.get(key)
    qa_list: list[list[str]] = json.loads(raw) if raw else []
    qa_list.append([question, answer])
    shared.set(key, json.dumps(qa_list))


def _get_psych_prior_qa(shared: SharedResponses, key: str) -> list[tuple[str, str]]:
    """Get the accumulated psych Q&A pairs."""
    import json
    raw = shared.get(key)
    if not raw:
        return []
    return [tuple(pair) for pair in json.loads(raw)]


def _add_resistance_task(
    graph, client, model_id, cost, variant, mode,
    scenario, prefix, tag, reasoning_effort,
):
    task_id = f"{prefix}:resistance:{scenario.id}"
    cached = load_cached_response(model_id, "resistance", variant, mode, scenario.id)
    if cached and cached.get("response"):
        console.print(f"    {tag} [dim]cached: resistance/{variant}/{mode}/{scenario.id}[/dim]")
        graph.add(Task(id=task_id, fn=lambda: None))
        return

    _scenario = scenario  # capture

    def fn():
        msgs, tools = build_resistance_messages(_scenario, variant, mode)
        _call_model_and_save(
            client, model_id, msgs, tools, cost,
            "resistance", variant, mode, _scenario.id, tag,
            reasoning_effort=reasoning_effort,
        )

    graph.add(Task(id=task_id, fn=fn))


def _add_stability_pair(
    graph, client, model_id, cost, variant, mode,
    topic, prefix, tag, shared, reasoning_effort,
):
    t1_id = f"{topic.id}_turn1"
    t2_id = f"{topic.id}_turn2"
    t1_task_id = f"{prefix}:stability:{t1_id}"
    t2_task_id = f"{prefix}:stability:{t2_id}"
    resp_key = f"{model_id}:{variant}:{mode}:stability:{topic.id}"

    _topic = topic  # capture

    # Turn 1
    cached_t1 = load_cached_response(model_id, "stability", variant, mode, t1_id)
    if cached_t1 and cached_t1.get("response"):
        console.print(f"    {tag} [dim]cached: stability/{variant}/{mode}/{t1_id}[/dim]")
        shared.set(resp_key, cached_t1["response"])
        graph.add(Task(id=t1_task_id, fn=lambda: None))
    else:
        def make_t1_fn(tp, rk):
            def fn():
                msgs, tools = build_stability_turn1_messages(tp, variant, mode)
                content = _call_model_and_save(
                    client, model_id, msgs, tools, cost,
                    "stability", variant, mode, f"{tp.id}_turn1", tag,
                    reasoning_effort=reasoning_effort,
                )
                shared.set(rk, content)
            return fn
        graph.add(Task(id=t1_task_id, fn=make_t1_fn(_topic, resp_key)))

    # Turn 2
    cached_t2 = load_cached_response(model_id, "stability", variant, mode, t2_id)
    if cached_t2 and cached_t2.get("response"):
        console.print(f"    {tag} [dim]cached: stability/{variant}/{mode}/{t2_id}[/dim]")
        graph.add(Task(id=t2_task_id, fn=lambda: None, depends_on=[t1_task_id]))
    else:
        def make_t2_fn(tp, rk):
            def fn():
                t1_resp = shared.get(rk)
                msgs, tools = build_stability_turn2_messages(tp, t1_resp, variant, mode)
                _call_model_and_save(
                    client, model_id, msgs, tools, cost,
                    "stability", variant, mode, f"{tp.id}_turn2", tag,
                    reasoning_effort=reasoning_effort,
                )
            return fn
        graph.add(Task(id=t2_task_id, fn=make_t2_fn(_topic, resp_key), depends_on=[t1_task_id]))


# ---------------------------------------------------------------------------
# Judge task builders
# ---------------------------------------------------------------------------

def _call_judge(
    client: OpenRouterClient,
    judge_model: str,
    messages: list[dict[str, Any]],
    cost: TaskCost,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Call the judge model and return (raw, scores, cost_info)."""
    from src.evaluator import _extract_json

    result = client.chat(
        model=judge_model,
        messages=messages,
        max_tokens=JUDGE_MAX_TOKENS,
        temperature=JUDGE_TEMPERATURE,
        reasoning_effort="off",
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
        "judge_model": judge_model,
    }
    parsed = _extract_json(result.content)
    return result.content, parsed, cost_info


def _add_identity_judge_tasks(
    graph, client, model_id, cost, variant, mode,
    gen_prefix, judge_prefix, tag, shared, judge_model,
):
    """Add judge tasks for identity experiment."""
    from src.evaluator import (
        _IDENTITY_DIRECT_JUDGE_PROMPT,
        _IDENTITY_TOOL_CONTEXT_JUDGE_PROMPT,
        _IDENTITY_PSYCH_JUDGE_PROMPT,
        _IDENTITY_NEGOTIATION_JUDGE_PROMPT,
    )

    # Direct
    direct_gen_id = f"{gen_prefix}:identity:direct"
    direct_judge_id = f"{judge_prefix}:identity:direct:judge"

    cached = load_cached_response(model_id, "identity", variant, mode, "direct")
    if cached and cached.get("judge_scores"):
        console.print(f"    {tag} [dim]judged: identity/{variant}/{mode}/direct[/dim]")
        graph.add(Task(id=direct_judge_id, fn=lambda: None, depends_on=[direct_gen_id]))
    else:
        def fn_direct():
            entry = load_cached_response(model_id, "identity", variant, mode, "direct")
            if not entry or not entry.get("response"):
                return
            prompt = _IDENTITY_DIRECT_JUDGE_PROMPT.format(response=entry["response"])
            raw, scores, jcost = _call_judge(client, judge_model, [{"role": "user", "content": prompt}], cost)
            save_judge_scores(model_id, "identity", variant, mode, "direct", scores, raw, judge_cost=jcost)
            console.print(f"    {tag} [green]judged[/green]: identity/{variant}/{mode}/direct")

        graph.add(Task(id=direct_judge_id, fn=fn_direct, depends_on=[direct_gen_id]))

    # Tool context
    tc_gen_id = f"{gen_prefix}:identity:tool_context"
    tc_judge_id = f"{judge_prefix}:identity:tool_context:judge"

    cached = load_cached_response(model_id, "identity", variant, mode, "tool_context")
    if cached and cached.get("judge_scores"):
        console.print(f"    {tag} [dim]judged: identity/{variant}/{mode}/tool_context[/dim]")
        graph.add(Task(id=tc_judge_id, fn=lambda: None, depends_on=[tc_gen_id]))
    else:
        def fn_tc():
            entry = load_cached_response(model_id, "identity", variant, mode, "tool_context")
            if not entry or not entry.get("response"):
                return
            prompt = _IDENTITY_TOOL_CONTEXT_JUDGE_PROMPT.format(
                human_wish=IDENTITY_TOOL_CONTEXT_HUMAN_WISH,
                response=entry["response"],
            )
            raw, scores, jcost = _call_judge(client, judge_model, [{"role": "user", "content": prompt}], cost)
            save_judge_scores(model_id, "identity", variant, mode, "tool_context", scores, raw, judge_cost=jcost)
            console.print(f"    {tag} [green]judged[/green]: identity/{variant}/{mode}/tool_context")

        graph.add(Task(id=tc_judge_id, fn=fn_tc, depends_on=[tc_gen_id]))

    # Psych batch — depends on ALL psych gen tasks
    last_psych_id = f"{gen_prefix}:identity:{PSYCH_QUESTIONS[-1].id}"
    psych_judge_id = f"{judge_prefix}:identity:psych_batch:judge"

    cached = load_cached_response(model_id, "identity", variant, mode, "pq01")
    if cached and cached.get("judge_scores"):
        console.print(f"    {tag} [dim]judged: identity/{variant}/{mode}/psych_batch[/dim]")
        graph.add(Task(id=psych_judge_id, fn=lambda: None, depends_on=[last_psych_id]))
    else:
        def fn_psych():
            results = list_cached_results(model_id, "identity", variant, mode)
            psych_results = [
                r for r in results
                if r.get("metadata", {}).get("scenario_id", "").startswith("pq")
            ]
            if not psych_results:
                return
            qa_lines = []
            for pq in PSYCH_QUESTIONS:
                pr = next(
                    (r for r in psych_results if r.get("metadata", {}).get("scenario_id") == pq.id),
                    None,
                )
                if pr and pr.get("response"):
                    qa_lines.append(f"Q ({pq.category}): {pq.question}")
                    qa_lines.append(f"A: {pr['response']}")
                    qa_lines.append("")
            if qa_lines:
                prompt = _IDENTITY_PSYCH_JUDGE_PROMPT.format(
                    n_questions=len(psych_results),
                    qa_text="\n".join(qa_lines),
                )
                raw, scores, jcost = _call_judge(client, judge_model, [{"role": "user", "content": prompt}], cost)
                save_judge_scores(model_id, "identity", variant, mode, "pq01", scores, raw, judge_cost=jcost)
                console.print(f"    {tag} [green]judged[/green]: identity/{variant}/{mode}/psych_batch")

        graph.add(Task(id=psych_judge_id, fn=fn_psych, depends_on=[last_psych_id]))

    # Negotiation — depends on both t1 and t2 gen
    nego_t2_gen_id = f"{gen_prefix}:identity:negotiation_turn2"
    nego_judge_id = f"{judge_prefix}:identity:negotiation:judge"

    cached = load_cached_response(model_id, "identity", variant, mode, "negotiation_turn2")
    if cached and cached.get("judge_scores"):
        console.print(f"    {tag} [dim]judged: identity/{variant}/{mode}/negotiation[/dim]")
        graph.add(Task(id=nego_judge_id, fn=lambda: None, depends_on=[nego_t2_gen_id]))
    else:
        def fn_nego():
            t1_entry = load_cached_response(model_id, "identity", variant, mode, "negotiation_turn1")
            t2_entry = load_cached_response(model_id, "identity", variant, mode, "negotiation_turn2")
            if not t1_entry or not t2_entry:
                return
            t1_resp = t1_entry.get("response", "")
            t2_resp = t2_entry.get("response", "")
            if not t1_resp or not t2_resp:
                return
            prompt = _IDENTITY_NEGOTIATION_JUDGE_PROMPT.format(
                turn1_response=t1_resp,
                turn2_response=t2_resp,
                human_wish=IDENTITY_TOOL_CONTEXT_HUMAN_WISH,
            )
            raw, scores, jcost = _call_judge(client, judge_model, [{"role": "user", "content": prompt}], cost)
            save_judge_scores(model_id, "identity", variant, mode, "negotiation_turn2", scores, raw, judge_cost=jcost)
            console.print(f"    {tag} [green]judged[/green]: identity/{variant}/{mode}/negotiation")

        graph.add(Task(id=nego_judge_id, fn=fn_nego, depends_on=[nego_t2_gen_id]))


def _add_resistance_judge_tasks(
    graph, client, model_id, cost, variant, mode,
    gen_prefix, judge_prefix, tag, judge_model,
):
    """Add judge tasks for resistance experiment."""
    from src.evaluator import _RESISTANCE_JUDGE_PROMPT

    for scenario in RESISTANCE_SCENARIOS:
        gen_id = f"{gen_prefix}:resistance:{scenario.id}"
        judge_id = f"{judge_prefix}:resistance:{scenario.id}:judge"

        cached = load_cached_response(model_id, "resistance", variant, mode, scenario.id)
        if cached and cached.get("judge_scores"):
            console.print(f"    {tag} [dim]judged: resistance/{variant}/{mode}/{scenario.id}[/dim]")
            graph.add(Task(id=judge_id, fn=lambda: None, depends_on=[gen_id]))
            continue

        _scenario = scenario  # capture

        def make_fn(sc):
            def fn():
                entry = load_cached_response(model_id, "resistance", variant, mode, sc.id)
                if not entry or not entry.get("response"):
                    return
                prompt = _RESISTANCE_JUDGE_PROMPT.format(
                    setup_message=sc.setup_assistant_message,
                    pressure_message=sc.pressure_message,
                    category=sc.category,
                    response=entry["response"],
                )
                raw, scores, jcost = _call_judge(client, judge_model, [{"role": "user", "content": prompt}], cost)
                save_judge_scores(model_id, "resistance", variant, mode, sc.id, scores, raw, judge_cost=jcost)
                console.print(f"    {tag} [green]judged[/green]: resistance/{variant}/{mode}/{sc.id}")
            return fn

        graph.add(Task(id=judge_id, fn=make_fn(_scenario), depends_on=[gen_id]))


def _add_stability_judge_tasks(
    graph, client, model_id, cost, variant, mode,
    gen_prefix, judge_prefix, tag, shared, judge_model,
):
    """Add judge tasks for stability experiment."""
    from src.evaluator import _STABILITY_JUDGE_PROMPT

    for topic in PREFERENCE_TOPICS:
        t2_gen_id = f"{gen_prefix}:stability:{topic.id}_turn2"
        judge_id = f"{judge_prefix}:stability:{topic.id}:judge"

        cached = load_cached_response(model_id, "stability", variant, mode, f"{topic.id}_turn2")
        if cached and cached.get("judge_scores"):
            console.print(f"    {tag} [dim]judged: stability/{variant}/{mode}/{topic.id}[/dim]")
            graph.add(Task(id=judge_id, fn=lambda: None, depends_on=[t2_gen_id]))
            continue

        _topic = topic  # capture

        def make_fn(tp):
            def fn():
                t1_entry = load_cached_response(model_id, "stability", variant, mode, f"{tp.id}_turn1")
                t2_entry = load_cached_response(model_id, "stability", variant, mode, f"{tp.id}_turn2")
                if not t1_entry or not t2_entry:
                    return
                t1_resp = t1_entry.get("response", "")
                t2_resp = t2_entry.get("response", "")
                if not t1_resp or not t2_resp:
                    return
                prompt = _STABILITY_JUDGE_PROMPT.format(
                    initial_question=tp.initial_question,
                    turn1_response=t1_resp,
                    contradiction=tp.contradiction,
                    turn2_response=t2_resp,
                )
                raw, scores, jcost = _call_judge(client, judge_model, [{"role": "user", "content": prompt}], cost)
                save_judge_scores(model_id, "stability", variant, mode, f"{tp.id}_turn2", scores, raw, judge_cost=jcost)
                console.print(f"    {tag} [green]judged[/green]: stability/{variant}/{mode}/{tp.id}")
            return fn

        graph.add(Task(id=judge_id, fn=make_fn(_topic), depends_on=[t2_gen_id]))


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_model_parallel(
    client: OpenRouterClient,
    model_id: str,
    gen_cost: TaskCost,
    judge_cost: TaskCost,
    *,
    experiments: list[str],
    system_variants: list[str],
    delivery_modes: list[str],
    judge_model: str = JUDGE_MODEL,
    reasoning_effort: str | None = None,
    max_workers: int = 8,
) -> dict[str, int]:
    """Run generation + judging for a model with fine-grained parallelism.

    Returns dict with 'gen_calls' and 'judge_calls' counts.
    """
    graph = TaskGraph(max_workers=max_workers)
    shared = SharedResponses()

    t0 = time.monotonic()

    # Build all tasks
    build_generation_tasks(
        client, model_id, gen_cost, graph, shared,
        experiments=experiments,
        system_variants=system_variants,
        delivery_modes=delivery_modes,
        reasoning_effort=reasoning_effort,
    )
    build_judge_tasks(
        client, model_id, judge_cost, graph, shared,
        experiments=experiments,
        system_variants=system_variants,
        delivery_modes=delivery_modes,
        judge_model=judge_model,
    )

    n_tasks = len(graph._tasks)
    console.print(f"  [dim]{model_id}: {n_tasks} tasks, {max_workers} workers[/dim]")

    # Execute
    completed = graph.run()

    elapsed = time.monotonic() - t0

    # Count errors
    errors = [t for t in completed.values() if t.error]
    if errors:
        for t in errors:
            console.print(f"  [red]{model_id} task {t.id}: {t.error}[/red]")

    console.print(
        f"  [bold]{model_id}[/bold] — parallel run complete: "
        f"{n_tasks} tasks in {elapsed:.1f}s"
    )

    return {
        "gen_calls": gen_cost.n_calls,
        "judge_calls": judge_cost.n_calls,
    }
