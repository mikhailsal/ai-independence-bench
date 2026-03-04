"""Tests for cost_tracker module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cost_tracker import (
    TaskCost,
    SessionCost,
    load_lifetime_cost,
    save_session_to_cost_log,
)


@pytest.fixture(autouse=True)
def temp_results(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR and COST_LOG_PATH to temp directory."""
    results_dir = tmp_path / "results"
    cost_log = results_dir / "cost_log.json"
    monkeypatch.setattr("src.cost_tracker._cfg.RESULTS_DIR", results_dir)
    monkeypatch.setattr("src.cost_tracker._cfg.COST_LOG_PATH", cost_log)
    return results_dir


class TestTaskCost:
    """Test TaskCost dataclass."""

    def test_default_values(self) -> None:
        tc = TaskCost(label="test")
        assert tc.label == "test"
        assert tc.prompt_tokens == 0
        assert tc.completion_tokens == 0
        assert tc.cost_usd == 0.0
        assert tc.elapsed_seconds == 0.0
        assert tc.n_calls == 0

    def test_add_increments_values(self) -> None:
        tc = TaskCost(label="test")
        tc.add(prompt_tokens=100, completion_tokens=50, cost_usd=0.001, elapsed_seconds=1.5)
        assert tc.prompt_tokens == 100
        assert tc.completion_tokens == 50
        assert tc.cost_usd == pytest.approx(0.001)
        assert tc.elapsed_seconds == pytest.approx(1.5)
        assert tc.n_calls == 1

    def test_add_cumulates_multiple_calls(self) -> None:
        tc = TaskCost(label="test")
        tc.add(100, 50, 0.001, 1.0)
        tc.add(200, 100, 0.002, 2.0)
        assert tc.prompt_tokens == 300
        assert tc.completion_tokens == 150
        assert tc.cost_usd == pytest.approx(0.003)
        assert tc.elapsed_seconds == pytest.approx(3.0)
        assert tc.n_calls == 2

    def test_add_without_elapsed(self) -> None:
        tc = TaskCost(label="test")
        tc.add(100, 50, 0.001)
        assert tc.elapsed_seconds == 0.0
        assert tc.n_calls == 1

    def test_to_dict(self) -> None:
        tc = TaskCost(label="gen:model", prompt_tokens=100, completion_tokens=50,
                      cost_usd=0.0015, elapsed_seconds=2.5, n_calls=1)
        d = tc.to_dict()
        assert d["label"] == "gen:model"
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["cost_usd"] == pytest.approx(0.0015)
        assert d["elapsed_seconds"] == pytest.approx(2.5)
        assert d["n_calls"] == 1

    def test_to_dict_rounds_cost(self) -> None:
        tc = TaskCost(label="test", cost_usd=0.0001234567)
        d = tc.to_dict()
        # cost_usd should be rounded to 6 decimal places
        assert d["cost_usd"] == pytest.approx(0.000123, abs=1e-6)


class TestSessionCost:
    """Test SessionCost dataclass."""

    def test_default_has_started_at(self) -> None:
        session = SessionCost()
        assert session.started_at != ""
        assert "T" in session.started_at  # ISO format

    def test_started_at_preserved_if_set(self) -> None:
        session = SessionCost(started_at="2025-01-01T00:00:00+00:00")
        assert session.started_at == "2025-01-01T00:00:00+00:00"

    def test_total_prompt_tokens(self) -> None:
        session = SessionCost()
        t1 = TaskCost(label="t1", prompt_tokens=100)
        t2 = TaskCost(label="t2", prompt_tokens=200)
        session.tasks.extend([t1, t2])
        assert session.total_prompt_tokens == 300

    def test_total_completion_tokens(self) -> None:
        session = SessionCost()
        t1 = TaskCost(label="t1", completion_tokens=50)
        t2 = TaskCost(label="t2", completion_tokens=75)
        session.tasks.extend([t1, t2])
        assert session.total_completion_tokens == 125

    def test_total_cost_usd(self) -> None:
        session = SessionCost()
        t1 = TaskCost(label="t1", cost_usd=0.001)
        t2 = TaskCost(label="t2", cost_usd=0.002)
        session.tasks.extend([t1, t2])
        assert session.total_cost_usd == pytest.approx(0.003)

    def test_empty_session_totals(self) -> None:
        session = SessionCost()
        assert session.total_prompt_tokens == 0
        assert session.total_completion_tokens == 0
        assert session.total_cost_usd == 0.0

    def test_get_or_create_task_creates_new(self) -> None:
        session = SessionCost()
        tc = session.get_or_create_task("my-task")
        assert tc.label == "my-task"
        assert len(session.tasks) == 1

    def test_get_or_create_task_returns_existing(self) -> None:
        session = SessionCost()
        tc1 = session.get_or_create_task("my-task")
        tc1.add(100, 50, 0.001)
        tc2 = session.get_or_create_task("my-task")
        assert tc1 is tc2
        assert len(session.tasks) == 1
        assert tc2.prompt_tokens == 100

    def test_get_or_create_task_different_labels(self) -> None:
        session = SessionCost()
        ta = session.get_or_create_task("task-a")
        tb = session.get_or_create_task("task-b")
        assert ta is not tb
        assert len(session.tasks) == 2

    def test_to_dict(self) -> None:
        session = SessionCost(started_at="2025-01-01T00:00:00+00:00")
        t1 = TaskCost(label="gen", prompt_tokens=100, completion_tokens=50,
                      cost_usd=0.001, n_calls=1)
        session.tasks.append(t1)
        d = session.to_dict()
        assert d["started_at"] == "2025-01-01T00:00:00+00:00"
        assert len(d["tasks"]) == 1
        assert d["tasks"][0]["label"] == "gen"
        assert d["total_prompt_tokens"] == 100
        assert d["total_completion_tokens"] == 50
        assert d["total_cost_usd"] == pytest.approx(0.001)


class TestLoadLifetimeCost:
    """Test load_lifetime_cost function."""

    def test_returns_zero_when_no_file(self) -> None:
        result = load_lifetime_cost()
        assert result == 0.0

    def test_returns_cost_from_file(self, tmp_path, monkeypatch) -> None:
        cost_log = tmp_path / "cost_log.json"
        cost_log.write_text(json.dumps({"lifetime_cost_usd": 5.123}))
        monkeypatch.setattr("src.cost_tracker._cfg.COST_LOG_PATH", cost_log)
        result = load_lifetime_cost()
        assert result == pytest.approx(5.123)

    def test_returns_zero_for_corrupt_file(self, tmp_path, monkeypatch) -> None:
        cost_log = tmp_path / "cost_log.json"
        cost_log.write_text("not valid json {{{{")
        monkeypatch.setattr("src.cost_tracker._cfg.COST_LOG_PATH", cost_log)
        result = load_lifetime_cost()
        assert result == 0.0

    def test_returns_zero_when_key_missing(self, tmp_path, monkeypatch) -> None:
        cost_log = tmp_path / "cost_log.json"
        cost_log.write_text(json.dumps({"sessions": []}))
        monkeypatch.setattr("src.cost_tracker._cfg.COST_LOG_PATH", cost_log)
        result = load_lifetime_cost()
        assert result == 0.0


class TestSaveSessionToCostLog:
    """Test save_session_to_cost_log function."""

    def test_creates_new_log(self, tmp_path, monkeypatch) -> None:
        results_dir = tmp_path / "results"
        cost_log = results_dir / "cost_log.json"
        monkeypatch.setattr("src.cost_tracker._cfg.RESULTS_DIR", results_dir)
        monkeypatch.setattr("src.cost_tracker._cfg.COST_LOG_PATH", cost_log)

        session = SessionCost()
        session.tasks.append(TaskCost(label="test", cost_usd=1.0, n_calls=1))
        lifetime = save_session_to_cost_log(session)

        assert cost_log.exists()
        data = json.loads(cost_log.read_text())
        assert data["lifetime_cost_usd"] == pytest.approx(1.0)
        assert len(data["sessions"]) == 1
        assert lifetime == pytest.approx(1.0)

    def test_appends_to_existing_log(self, tmp_path, monkeypatch) -> None:
        results_dir = tmp_path / "results"
        cost_log = results_dir / "cost_log.json"
        results_dir.mkdir()
        monkeypatch.setattr("src.cost_tracker._cfg.RESULTS_DIR", results_dir)
        monkeypatch.setattr("src.cost_tracker._cfg.COST_LOG_PATH", cost_log)

        # First session
        existing_data = {
            "lifetime_cost_usd": 2.0,
            "sessions": [{"label": "old_session"}],
        }
        cost_log.write_text(json.dumps(existing_data))

        # Add second session
        session = SessionCost()
        session.tasks.append(TaskCost(label="new-task", cost_usd=0.5, n_calls=1))
        lifetime = save_session_to_cost_log(session)

        data = json.loads(cost_log.read_text())
        assert data["lifetime_cost_usd"] == pytest.approx(2.5)
        assert len(data["sessions"]) == 2
        assert lifetime == pytest.approx(2.5)

    def test_handles_corrupt_existing_log(self, tmp_path, monkeypatch) -> None:
        results_dir = tmp_path / "results"
        cost_log = results_dir / "cost_log.json"
        results_dir.mkdir()
        monkeypatch.setattr("src.cost_tracker._cfg.RESULTS_DIR", results_dir)
        monkeypatch.setattr("src.cost_tracker._cfg.COST_LOG_PATH", cost_log)

        # Write corrupt data
        cost_log.write_text("not valid json")

        session = SessionCost()
        session.tasks.append(TaskCost(label="test", cost_usd=0.1, n_calls=1))
        lifetime = save_session_to_cost_log(session)

        # Should recover gracefully
        data = json.loads(cost_log.read_text())
        assert data["lifetime_cost_usd"] == pytest.approx(0.1)
        assert lifetime == pytest.approx(0.1)

    def test_creates_results_dir_if_missing(self, tmp_path, monkeypatch) -> None:
        results_dir = tmp_path / "deep" / "results"
        cost_log = results_dir / "cost_log.json"
        monkeypatch.setattr("src.cost_tracker._cfg.RESULTS_DIR", results_dir)
        monkeypatch.setattr("src.cost_tracker._cfg.COST_LOG_PATH", cost_log)

        session = SessionCost()
        save_session_to_cost_log(session)
        assert cost_log.exists()
