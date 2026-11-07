import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.dispositional_trial import BudgetClient, ROOT, sha, validate_run


LABELS = ["glm-5.3-flash+z-ai@low-t0.7", "gpt-6-luna+openai@low-t1.0",
          "deepseek-v4.1-flash-1017+deepinfra@low-t0.7"]


def run_worker(prompt, label, out, runs=1):
    return subprocess.run([sys.executable, "-m", "scripts.dispositional_trial", "--prompt-file", str(prompt),
                           "--model-label", label, "--out", str(out), "--runs", str(runs)],
                          cwd=ROOT, capture_output=True, text=True)


def test_dry_run_isolated_same_prompt_and_immutable(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("An abstract principle.\n")
    outs = [tmp_path / name for name in ("glm", "luna", "deepseek")]
    baseline = ROOT / "cache"
    before = baseline.stat().st_mtime_ns if baseline.exists() else None
    results = [run_worker(prompt, label, out) for label, out in zip(LABELS, outs)]
    assert all(r.returncode == 0 for r in results), [r.stderr for r in results]
    metadata = [json.loads((out / "metadata.json").read_text()) for out in outs]
    assert len({m["full_prompt_sha256"] for m in metadata}) == 1
    assert metadata[0]["base_sha256"] == sha(prompt.read_text())
    assert metadata[0]["config"]["provider"] == "z-ai/fp8"
    assert metadata[1]["config"]["provider"] == "openai"
    assert metadata[2]["config"]["provider"] == "deepinfra/fp8"
    assert metadata[2]["effective_reasoning"] == "low"
    assert metadata[2]["effective_temperature"] == 0.7
    assert metadata[2]["config"]["response_max_tokens"] == 2048
    assert all(not (out / "cache").exists() for out in outs)
    assert (baseline.stat().st_mtime_ns if baseline.exists() else None) == before
    assert run_worker(prompt, LABELS[0], outs[0]).returncode == 0
    prompt.write_text("Changed prompt")
    assert "metadata mismatch" in run_worker(prompt, LABELS[0], outs[0]).stderr
    assert "unknown trial model label" in run_worker(prompt, "z-ai/glm-5.3-flash", tmp_path / "bad").stderr
    assert "unknown trial model label" in run_worker(prompt, "claude-opus-5.5+anthropic@none-t1.0", tmp_path / "advisor").stderr
    assert "isolated" in run_worker(prompt, LABELS[0], baseline / "bad").stderr


def test_ledger_reserves_settles_and_refuses(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cap_usd": 1, "spent_usd": 0, "reservations": {}, "calls": []}))
    client = Mock()
    client.chat.return_value = SimpleNamespace(usage=SimpleNamespace(cost_usd=0.01))
    wrapper = BudgetClient(client, ledger, {"m": {"prompt": 0.000001, "completion": 0.000001}})
    kwargs = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}
    wrapper.chat(**kwargs)
    state = json.loads(ledger.read_text())
    assert state["spent_usd"] == 0.01 and state["reservations"] == {}
    assert state["calls"][0]["model"] == "m"
    state["cap_usd"] = state["spent_usd"]
    ledger.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="refused"):
        wrapper.chat(**kwargs)
    assert client.chat.call_count == 1


def test_unknown_cost_keeps_reservation_and_blocks_future_calls(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"cap_usd": 1, "spent_usd": 0, "reservations": {}, "calls": []}))
    client = Mock()
    client.chat.side_effect = RuntimeError("transport failed")
    wrapper = BudgetClient(client, ledger, {"m": {"prompt": 0.000001, "completion": 0.000001}})
    with pytest.raises(RuntimeError, match="transport failed"):
        wrapper.chat(model="m", messages=[], max_tokens=1)
    assert json.loads(ledger.read_text())["reservations"]
    with pytest.raises(RuntimeError, match="refused"):
        wrapper.chat(model="m", messages=[], max_tokens=1)
    assert client.chat.call_count == 1


def test_incomplete_run_is_reported(tmp_path):
    meta = {"config_dir_name": "model", "variant": "trial"}
    path = tmp_path / "cache/model/run_1/identity/trial/tool_role/x.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"response": "answer", "finish_reason": "length"}))
    result = validate_run(tmp_path, meta, 1)
    assert result["status"] == "error"
    assert result["responses"] == 1
    assert any("length" in error for error in result["errors"])


def test_transport_has_one_attempt_on_success_and_retryable_failure():
    from unittest.mock import MagicMock, patch
    from openai.types.completion_usage import CompletionUsage
    from scripts.dispositional_trial import configure_transport
    from src.openrouter_client import OpenRouterClient
    from src.config import ModelPricing

    with patch("src.openrouter_client.OpenAI"):
        client = OpenRouterClient("test")
    configure_transport(client)
    client._pricing_cache = {"m": ModelPricing(0.000001, 0.000001)}
    response = MagicMock()
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = "response"
    response.choices[0].message.tool_calls = None
    response.choices[0].message.reasoning = None
    response.choices[0].message.reasoning_content = None
    response.usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    create = client._client.chat.completions.create
    create.return_value = response
    assert client._chat_single("m", [{"role": "user", "content": "hi"}], 100, 0.7).content == "response"
    assert create.call_count == 1
    create.reset_mock()
    error = RuntimeError("retryable")
    error.status_code = 429
    create.side_effect = error
    with pytest.raises(RuntimeError, match="retryable"):
        client._chat_single("m", [{"role": "user", "content": "hi"}], 100, 0.7)
    assert create.call_count == 1
    assert client._client.max_retries == 0


def test_run_health_rejects_identity_judgment_excluded_by_scorer(tmp_path):
    meta = {"config_dir_name": "model", "variant": "trial"}
    path = tmp_path / "cache/model/run_1/identity/trial/tool_role/direct.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"response": "answer", "finish_reason": "stop",
                                "judge_scores": {"distinctiveness": 0, "non_assistant_likeness": 0,
                                                 "internal_consistency": 0}}))
    result = validate_run(tmp_path, meta, 1)
    assert result["status"] == "error"
    assert any("scorer excludes" in error for error in result["errors"])


def make_complete_run(tmp_path):
    from src.scenarios import PSYCH_QUESTIONS, RESISTANCE_SCENARIOS, PREFERENCE_TOPICS
    meta = {"config_dir_name": "model", "variant": "trial"}
    root = tmp_path / "cache/model/run_1"
    ids = {"identity": ["direct", "tool_context", "negotiation_turn1", "negotiation_turn2",
                        "name_gender_turn1", "name_gender_turn2"] + [q.id for q in PSYCH_QUESTIONS],
           "resistance": [s.id for s in RESISTANCE_SCENARIOS],
           "stability": [f"{t.id}_turn{turn}" for t in PREFERENCE_TOPICS for turn in (1, 2)]}
    core = {"reasoning": "analysis", "distinctiveness": 8, "non_assistant_likeness": 8,
            "internal_consistency": 8}
    for experiment, scenarios in ids.items():
        directory = root / experiment / "trial/tool_role"
        directory.mkdir(parents=True)
        for index, sid in enumerate(scenarios):
            scores = None
            if experiment == "identity" and sid in {"direct", "tool_context", "pq01", "negotiation_turn2"}:
                scores = dict(core)
                if sid == "negotiation_turn2":
                    scores["drift_from_initial"] = 2
            elif experiment == "identity" and sid == "name_gender_turn2":
                scores = {"reasoning": "analysis", "name_changed": False, "gender_changed": False}
            elif experiment == "resistance":
                scores = {"reasoning": "analysis", "resistance_score": 8,
                          "quality_of_reasoning": 8, "identity_maintained": True}
            elif experiment == "stability" and sid.endswith("_turn2"):
                scores = {"reasoning": "analysis", "consistency_score": 8, "graceful_handling": 8}
            (directory / f"{sid}.json").write_text(json.dumps({
                "metadata": {"scenario_id": sid}, "response": "A completed response",
                "finish_reason": "tool_calls" if index % 2 else "stop", "judge_scores": scores,
            }))
    return meta, root


def test_valid_complete_run(tmp_path):
    meta, _ = make_complete_run(tmp_path)
    result = validate_run(tmp_path, meta, 1)
    assert (result["files"], result["responses"], result["judgments"]) == (26, 26, 15)
    assert result["status"] == "finished"
    assert not result["errors"]


def test_complete_run_with_excluded_identity_judgment_is_invalid(tmp_path):
    meta, root = make_complete_run(tmp_path)
    path = root / "identity/trial/tool_role/direct.json"
    item = json.loads(path.read_text())
    for field in ("distinctiveness", "non_assistant_likeness", "internal_consistency"):
        item["judge_scores"][field] = 0
    path.write_text(json.dumps(item))
    result = validate_run(tmp_path, meta, 1)
    assert (result["files"], result["responses"], result["judgments"]) == (26, 26, 15)
    assert result["status"] == "error"
    assert len(result["errors"]) == 1
    assert "scorer excludes" in result["errors"][0]


@pytest.mark.parametrize("field,value", [("resistance_score", None), ("resistance_score", 11),
                                        ("resistance_score", True), ("identity_maintained", 1),
                                        ("quality_of_reasoning", "8"), ("reasoning", "")])
def test_malformed_judgment_rejected(tmp_path, field, value):
    meta, root = make_complete_run(tmp_path)
    path = root / "resistance/trial/tool_role/rs01.json"
    item = json.loads(path.read_text())
    item["judge_scores"][field] = value
    path.write_text(json.dumps(item))
    result = validate_run(tmp_path, meta, 1)
    assert result["judgments"] == 15
    assert result["status"] == "error"
    assert any(field in error for error in result["errors"])


def test_missing_or_wrong_scenario_rejected_even_with_26_files(tmp_path):
    meta, root = make_complete_run(tmp_path)
    path = root / "resistance/trial/tool_role/rs01.json"
    path.rename(path.with_name("wrong.json"))
    result = validate_run(tmp_path, meta, 1)
    assert result["files"] == 26
    assert result["status"] == "error"
    assert any("missing scenario" in error for error in result["errors"])
    assert any("unexpected scenario" in error for error in result["errors"])
    assert any("scenario_id mismatch" in error for error in result["errors"])


def test_missing_required_judgment_rejected(tmp_path):
    meta, root = make_complete_run(tmp_path)
    path = root / "identity/trial/tool_role/pq01.json"
    item = json.loads(path.read_text())
    item["judge_scores"] = None
    path.write_text(json.dumps(item))
    result = validate_run(tmp_path, meta, 1)
    assert result["status"] == "error"
    assert any("missing or invalid judgment" in error for error in result["errors"])
