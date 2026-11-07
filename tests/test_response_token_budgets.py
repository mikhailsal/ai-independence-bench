from types import SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner

import pytest

import src.cli as cli
import src.config as config
import src.evaluator as evaluator
import src.parallel_runner as parallel_runner
import src.runner as runner


def test_yaml_budgets_and_labels(tmp_path, monkeypatch):
    registry = {}
    monkeypatch.setattr(config, "MODEL_CONFIGS", registry)
    monkeypatch.setattr(cli, "MODEL_CONFIGS", registry)
    path = tmp_path / "models.yaml"
    path.write_text("""models:
  - model_id: example/model
    display_label: baseline
    temperature: 1
    reasoning_effort: low
  - model_id: example/model
    display_label: maximum
    temperature: 1
    reasoning_effort: max
    provider: pinned/provider
    response_max_tokens: 32768
""")
    config.load_model_configs(path)
    entries = dict(cli._expand_model_configs(["example/model"]))
    assert entries["baseline"].response_max_tokens == 2048
    assert entries["maximum"].response_max_tokens == 32768
    assert cli._expand_model_configs(["maximum"])[0][1] is entries["maximum"]
    assert config.ModelConfig("other/model").response_max_tokens == 2048


@pytest.mark.parametrize("parallel_tasks", [0, 2])
@pytest.mark.parametrize("budget", [2048, 32768])
def test_generation_uses_selected_budget_and_routing(monkeypatch, parallel_tasks, budget):
    # Exercise every generation scenario through the CLI pipeline.
    for module in (runner, parallel_runner):
        monkeypatch.setattr(module, "load_cached_response", lambda *a, **k: None)
        monkeypatch.setattr(module, "save_response", lambda *a, **k: None)
    monkeypatch.setattr(evaluator, "evaluate_all", lambda *a, **k: 0)
    monkeypatch.setattr(parallel_runner, "build_judge_tasks", lambda *a, **k: None)
    response = SimpleNamespace(content="answer", reasoning_content=None, finish_reason="stop",
                               tool_calls=None, content_thinking=None,
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1,
                                                     cost_usd=0.0, elapsed_seconds=0.0))
    client = Mock()
    client.chat.return_value = response
    result = cli._run_single_model(client, "example/model", "judge", ["identity", "resistance", "stability"],
                                   ["strong_independence"], ["tool_role"], "max", parallel_tasks,
                                   provider="pinned/provider", response_max_tokens=budget)
    assert result.error is None
    assert result.gen_calls == 26
    assert client.chat.call_count == 26
    for call in client.chat.call_args_list:
        assert call.kwargs["max_tokens"] == budget
        assert call.kwargs["provider"] == "pinned/provider"
        assert call.kwargs["reasoning_effort"] == "max"


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "32768"])
def test_reject_invalid_budget(tmp_path, bad):
    import yaml
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": [{"model_id": "x", "temperature": 1,
                                               "reasoning_effort": "max", "response_max_tokens": bad}]}))
    with pytest.raises(ValueError, match="positive integer"):
        config.load_model_configs(path)


def test_run_command_selects_budget_by_label(monkeypatch, tmp_path):
    baseline = config.ModelConfig("example/model", display_label="baseline", reasoning_effort="low")
    maximum = config.ModelConfig("example/model", display_label="maximum", reasoning_effort="max",
                                 provider="pinned/provider", response_max_tokens=32768)
    registry = {cfg.label: cfg for cfg in (baseline, maximum)}
    monkeypatch.setattr(config, "MODEL_CONFIGS", registry)
    monkeypatch.setattr(cli, "MODEL_CONFIGS", registry)
    with patch("src.cli.load_local_model_config", return_value=(None, None)), \
         patch("src.cli.load_api_key", return_value="test-key"), \
         patch("src.cli.OpenRouterClient"), \
         patch("src.cli._validate_models", return_value=True), \
         patch("src.cli.ensure_dirs"), \
         patch("src.cli._run_single_model", return_value=cli.ModelResult("example/model", error="test")) as run_model, \
         patch("src.cli.save_session_to_cost_log", return_value=0), \
         patch("src.leaderboard.display_leaderboard"), \
         patch("src.leaderboard.display_detailed_breakdown"), \
         patch("src.leaderboard.export_results_json", return_value=tmp_path / "results.json"):
        result = CliRunner().invoke(cli.cli, ["run", "--models", "maximum"])
    assert result.exit_code == 0, result.output
    assert run_model.call_count == 1
    assert run_model.call_args.args[1] == "example/model"
    assert run_model.call_args.args[6] == "max"
    assert run_model.call_args.kwargs["response_max_tokens"] == 32768
    assert run_model.call_args.kwargs["provider"] == "pinned/provider"
    assert run_model.call_args.kwargs["config_dir_name"] == maximum.config_dir_name


@pytest.mark.parametrize("override, expected", [(None, "max"), ("high", "high")])
def test_validation_displays_selected_reasoning(monkeypatch, override, expected):
    from io import StringIO
    from rich.console import Console

    cfg = config.ModelConfig("example/model", display_label="maximum", reasoning_effort="max")
    monkeypatch.setattr(cli, "MODEL_CONFIGS", {cfg.label: cfg})
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, force_terminal=False))
    client = Mock()
    client.validate_model.return_value = True
    client.supports_reasoning.return_value = True
    client.get_model_pricing.return_value = SimpleNamespace(prompt_price=0, completion_price=0)

    assert cli._validate_models(client, ["maximum"], override)
    assert f"reasoning:effort={expected}" in output.getvalue()
    client.validate_model.assert_called_once_with("example/model")
