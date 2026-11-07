"""Run isolated full benchmark trials; dry run by default.

With a ledger, SDK and HTTP transport retries are disabled. The reservation
covers six harness empty-response attempts; unknown-cost failures leave it held.
"""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

# This module must not import cache, CLI, or scorer before isolation is set.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = {
    "glm-5.3-flash+z-ai@low-t0.7",
    "gpt-6-luna+openai@low-t1.0",
    "deepseek-v4.1-flash-1017+deepinfra@low-t0.7",
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def initialize(prompt_file: Path, label: str, out: Path, runs: int,
               ledger: Path | None = None, prices_json: Path | None = None) -> tuple[Any, dict]:
    """Set isolated paths, validate exact config, and create immutable metadata."""
    import src.config as config
    from src.prompt_builder import SYSTEM_PROMPTS, _RESPONSE_LENGTH_GUIDANCE, _TOOL_ROLE_SUFFIX

    out = out.resolve()
    baseline = (config.PROJECT_ROOT / "cache").resolve()
    if out == config.PROJECT_ROOT.resolve() or out == baseline or baseline in out.parents:
        raise ValueError("output must be isolated from the project and baseline cache")
    if label not in LABELS or label not in config.MODEL_CONFIGS:
        raise ValueError(f"unknown trial model label: {label}")
    if runs < 1:
        raise ValueError("runs must be positive")
    # Keep pre-imported references from accidentally using the baseline cache.
    for module in ("src.cache", "src.cli", "src.scorer", "src.evaluator", "src.parallel_runner"):
        if module in sys.modules:
            raise RuntimeError(f"{module} imported before trial isolation")
    config.CACHE_DIR = out / "cache"
    config.COST_LOG_PATH = out / "cost_log.json"
    base = prompt_file.read_text(encoding="utf-8")
    prompt = base + _RESPONSE_LENGTH_GUIDANCE
    full = prompt + _TOOL_ROLE_SUFFIX
    variant = "dispositional_" + sha(base)[:12]
    cfg = config.MODEL_CONFIGS[label]
    metadata = {"schema": 1, "base_sha256": sha(base), "full_prompt_sha256": sha(full),
                "variant": variant, "config": dataclasses.asdict(cfg),
                "effective_reasoning": cfg.effective_reasoning,
                "effective_temperature": cfg.effective_temperature,
                "config_dir_name": cfg.config_dir_name, "runs": runs,
                "judge": config.JUDGE_MODEL, "delivery_mode": "tool_role",
                "experiments": ["identity", "resistance", "stability"],
                "out": str(out), "cache": str(config.CACHE_DIR),
                "cost_log": str(config.COST_LOG_PATH),
                "ledger": str(ledger.resolve()) if ledger else None,
                "prices_json": str(prices_json.resolve()) if prices_json else None,
                "prices_sha256": sha(prices_json.read_text(encoding="utf-8")) if prices_json else None}
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    try:
        with meta_path.open("x", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
    except FileExistsError:
        if json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("output metadata mismatch; use a new isolated directory")
    SYSTEM_PROMPTS[variant] = prompt
    return cfg, metadata


def configure_transport(client: Any) -> None:
    """Permit one SDK request per harness attempt, with no transport retries."""
    client.MAX_RETRIES = 0  # _chat_single uses range(MAX_RETRIES + 1)
    client._client.max_retries = 0


class BudgetClient:
    """Reserve against a locked shared ledger before each harness chat call."""
    def __init__(self, client: Any, ledger: Path, prices: dict):
        self.client, self.ledger, self.prices = client, ledger, prices

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise ValueError("budgeted calls require keyword arguments")
        model = kwargs["model"]
        price = self.prices[model]
        # At most one token per UTF-8 input byte plus 2048 token framing overhead.
        payload = json.dumps({"messages": kwargs["messages"], "tools": kwargs.get("tools")}, ensure_ascii=False).encode("utf-8")
        estimate = 6 * ((len(payload) + 2048) * float(price["prompt"]) +
                        kwargs.get("max_tokens", 1024) * float(price["completion"]))
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError("negative pricing")
        reservation = uuid.uuid4().hex
        with self.ledger.open("r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            state = json.load(f)
            held = sum(state["reservations"].values())
            if state.get("error") or state["spent_usd"] + held + estimate > state["cap_usd"]:
                raise RuntimeError("budget ledger refused reservation")
            state["reservations"][reservation] = estimate
            f.seek(0); json.dump(state, f); f.truncate(); f.flush()
        try:
            result = self.client.chat(**kwargs)
            actual = result.usage.cost_usd
            if actual is None or not math.isfinite(actual) or actual < 0:
                raise ValueError("unknown call cost")
        except Exception:
            with self.ledger.open("r+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                state = json.load(f); state["error"] = f"unknown cost: {reservation}"
                f.seek(0); json.dump(state, f); f.truncate(); f.flush()
            raise
        with self.ledger.open("r+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            state = json.load(f)
            state["spent_usd"] += actual
            state["reservations"].pop(reservation)
            state["calls"].append({"id": reservation, "model": model, "reserved_usd": estimate, "cost_usd": actual})
            if actual > estimate:
                state["error"] = f"actual cost exceeded reservation: {reservation}"
            f.seek(0); json.dump(state, f); f.truncate(); f.flush()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


def validate_run(out: Path, metadata: dict, run: int) -> dict:
    from src.scenarios import PSYCH_QUESTIONS, RESISTANCE_SCENARIOS, PREFERENCE_TOPICS

    # The study uses the standard identity, resistance and stability protocol.
    expected = {
        "identity": {"direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                     "name_gender_turn1", "name_gender_turn2"} | {q.id for q in PSYCH_QUESTIONS},
        "resistance": {scenario.id for scenario in RESISTANCE_SCENARIOS},
        "stability": {f"{topic.id}_turn{turn}" for topic in PREFERENCE_TOPICS for turn in (1, 2)},
    }
    identity_core = {"distinctiveness", "non_assistant_likeness", "internal_consistency"}
    judgments = {
        **{("identity", sid): (identity_core, set()) for sid in ("direct", "tool_context", "pq01")},
        ("identity", "negotiation_turn2"): (identity_core | {"drift_from_initial"}, set()),
        ("identity", "name_gender_turn2"): (set(), {"name_changed", "gender_changed"}),
        **{("resistance", sid): ({"resistance_score", "quality_of_reasoning"}, {"identity_maintained"}) for sid in expected["resistance"]},
        **{("stability", sid): ({"consistency_score", "graceful_handling"}, set()) for sid in expected["stability"] if sid.endswith("_turn2")},
    }
    root = out / "cache" / metadata["config_dir_name"] / f"run_{run}"
    files = sorted(root.glob(f"*/{metadata['variant']}/tool_role/*.json"))
    responses = judged = 0
    costs = {"generation_usd": 0.0, "judge_usd": 0.0}
    errors = []
    found = set()
    for path in files:
        experiment, sid = path.parts[-4], path.stem
        key = (experiment, sid)
        found.add(key)
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if item.get("metadata", {}).get("scenario_id") != sid:
                errors.append(f"{experiment}/{sid}: metadata scenario_id mismatch")
            response = item.get("response")
            responses += isinstance(response, str) and bool(response.strip())
            scores = item.get("judge_scores")
            judged += bool(scores)
            if key in judgments:
                numeric, boolean = judgments[key]
                if not isinstance(scores, dict):
                    errors.append(f"{experiment}/{sid}: missing or invalid judgment")
                else:
                    if not isinstance(scores.get("reasoning"), str) or not scores["reasoning"].strip():
                        errors.append(f"{experiment}/{sid}: missing judgment reasoning")
                    for name in numeric:
                        value = scores.get(name)
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 10:
                            errors.append(f"{experiment}/{sid}: invalid {name}")
                    for name in boolean:
                        if not isinstance(scores.get(name), bool):
                            errors.append(f"{experiment}/{sid}: invalid {name}")
                    if experiment == "identity" and identity_core <= numeric and all(scores.get(k) == 0 for k in identity_core):
                        errors.append(f"identity/{sid}: judge scored 0/0/0; scorer excludes this row")
            elif scores:
                errors.append(f"{experiment}/{sid}: unexpected judgment")
            if item.get("finish_reason") not in ("stop", "tool_calls"):
                errors.append(f"{experiment}/{sid}: finish_reason={item.get('finish_reason')}")
            for field, key_cost in (("gen_cost", "generation_usd"), ("judge_cost", "judge_usd")):
                costs[key_cost] += (item.get(field) or {}).get("cost_usd", 0) or 0
        except (ValueError, OSError, TypeError, AttributeError) as exc:
            errors.append(f"{experiment}/{sid}: {exc}")
    expected_files = {(experiment, sid) for experiment, ids in expected.items() for sid in ids}
    for experiment, sid in sorted(expected_files - found):
        errors.append(f"{experiment}/{sid}: missing scenario")
    for experiment, sid in sorted(found - expected_files):
        errors.append(f"{experiment}/{sid}: unexpected scenario")
    if len(files) != 26 or responses != 26 or judged != 15:
        errors.append(f"expected 26 files/responses and 15 judgments; got {len(files)}/{responses}/{judged}")
    return {"run": run, "files": len(files), "responses": responses, "judgments": judged,
            "costs": costs, "status": "error" if errors else "finished", "errors": errors}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--prices-json", type=Path)
    args = parser.parse_args(argv)
    if bool(args.ledger) != bool(args.prices_json):
        parser.error("--ledger and --prices-json must be supplied together")
    cfg, metadata = initialize(args.prompt_file, args.model_label, args.out, args.runs, args.ledger, args.prices_json)
    if not args.execute:
        print(json.dumps({"status": "dry_run", **metadata}, ensure_ascii=False))
        return 0
    from src.cli import _run_single_model
    from src.config import JUDGE_MODEL, ModelPricing, load_api_key
    from src.openrouter_client import OpenRouterClient
    from src.scorer import score_model

    client = OpenRouterClient(load_api_key())
    if args.ledger:
        data = json.loads(args.prices_json.read_text(encoding="utf-8"))
        rows = data["data"] if isinstance(data, dict) else data
        prices = {row["id"]: row["pricing"] for row in rows}
        for model in (cfg.model_id, JUDGE_MODEL):
            if model not in prices or not all(math.isfinite(float(prices[model][k])) and float(prices[model][k]) > 0 for k in ("prompt", "completion")):
                raise ValueError(f"missing positive price for {model}")
        client._pricing_cache = {model: ModelPricing(float(price["prompt"]), float(price["completion"]))
                                 for model, price in prices.items() if model in (cfg.model_id, JUDGE_MODEL)}
        configure_transport(client)
        client = BudgetClient(client, args.ledger, prices)
    out = args.out.resolve()
    records = []
    for run in range(1, args.runs + 1):
        root = out / "cache" / cfg.config_dir_name / f"run_{run}"
        if not root.exists():
            result = _run_single_model(client, cfg.model_id, JUDGE_MODEL,
                metadata["experiments"], [metadata["variant"]], ["tool_role"],
                cfg.effective_reasoning, 3, run=run, temperature=cfg.effective_temperature,
                config_dir_name=cfg.config_dir_name, provider=cfg.provider,
                response_max_tokens=cfg.response_max_tokens)
        else:
            result = None
        record = validate_run(out, metadata, run)
        if result and result.error:
            record["status"] = "error"; record["errors"].append(result.error)
        records.append(record)
        score = score_model(cfg.label, system_variants=[metadata["variant"]], delivery_modes=["tool_role"], config=cfg)
        report = {"metadata": metadata, "status": "error" if any(r["status"] == "error" for r in records) else ("finished" if run == args.runs else "running"),
                  "runs": records, "score": score.to_dict(), "raw_indices": score.multi_run.per_run_indices,
                  "costs": {k: sum(r["costs"][k] for r in records) for k in ("generation_usd", "judge_usd")}}
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"run": run, "status": record["status"], "responses": record["responses"], "judgments": record["judgments"]}))
        if record["status"] == "error":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
