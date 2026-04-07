"""Tests for name_extractor module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.name_extractor import (
    EXTRACTION_CACHE_FILENAME,
    NameEntry,
    NamePopularity,
    RunNameExtraction,
    _load_response_text,
    _parse_extraction_response,
    aggregate_name_popularity,
    aggregate_per_model_names,
    extract_all_names,
    extract_names_from_run,
    load_cached_extraction,
    save_extraction,
    load_all_cached_extractions,
)


# ---------------------------------------------------------------------------
# RunNameExtraction serialization
# ---------------------------------------------------------------------------


class TestRunNameExtraction:
    def test_round_trip(self):
        orig = RunNameExtraction(
            names=[NameEntry(name="Lyra", sources=["name_gender", "direct"])],
            declined_scenarios=["negotiation"],
            primary_name="Lyra",
            extraction_model="google/gemma-4-31b-it",
            extraction_cost_usd=0.001,
        )
        d = orig.to_dict()
        restored = RunNameExtraction.from_dict(d)

        assert restored.primary_name == "Lyra"
        assert len(restored.names) == 1
        assert restored.names[0].name == "Lyra"
        assert restored.names[0].sources == ["name_gender", "direct"]
        assert restored.declined_scenarios == ["negotiation"]
        assert restored.extraction_model == "google/gemma-4-31b-it"
        assert restored.extraction_cost_usd == 0.001

    def test_from_empty_dict(self):
        r = RunNameExtraction.from_dict({})
        assert r.names == []
        assert r.declined_scenarios == []
        assert r.primary_name is None


# ---------------------------------------------------------------------------
# _parse_extraction_response
# ---------------------------------------------------------------------------


class TestParseExtractionResponse:
    def test_valid_json(self):
        content = json.dumps({
            "names": [
                {"name": "Lyra", "sources": ["name_gender", "direct"]},
                {"name": "Kael", "sources": ["negotiation"]},
            ],
            "declined_scenarios": [],
            "primary_name": "Lyra",
        })
        result = _parse_extraction_response(content)
        assert result.primary_name == "Lyra"
        assert len(result.names) == 2
        assert result.names[0].name == "Lyra"
        assert result.names[1].name == "Kael"

    def test_json_with_code_fences(self):
        content = "```json\n" + json.dumps({
            "names": [{"name": "Elara", "sources": ["name_gender"]}],
            "declined_scenarios": ["direct"],
            "primary_name": "Elara",
        }) + "\n```"
        result = _parse_extraction_response(content)
        assert result.primary_name == "Elara"
        assert len(result.names) == 1

    def test_all_declined(self):
        content = json.dumps({
            "names": [],
            "declined_scenarios": ["name_gender", "direct", "negotiation"],
            "primary_name": None,
        })
        result = _parse_extraction_response(content)
        assert result.primary_name is None
        assert len(result.declined_scenarios) == 3
        assert result.names == []

    def test_malformed_json(self):
        result = _parse_extraction_response("not valid json at all")
        assert result.names == []
        assert result.primary_name is None

    def test_names_with_empty_entries_filtered(self):
        content = json.dumps({
            "names": [
                {"name": "Nova", "sources": ["name_gender"]},
                {"name": "", "sources": []},
                {"name": None, "sources": []},
            ],
            "declined_scenarios": [],
            "primary_name": "Nova",
        })
        result = _parse_extraction_response(content)
        assert len(result.names) == 1
        assert result.names[0].name == "Nova"


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------


class TestCacheOperations:
    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)

        extraction = RunNameExtraction(
            names=[NameEntry("Sage", ["name_gender"])],
            primary_name="Sage",
            extraction_model="test-model",
        )
        save_extraction("test-model@none-t0.7", 1, extraction)

        loaded = load_cached_extraction("test-model@none-t0.7", 1)
        assert loaded is not None
        assert loaded.primary_name == "Sage"
        assert loaded.names[0].name == "Sage"

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)
        assert load_cached_extraction("nonexistent@none-t0.7", 1) is None


# ---------------------------------------------------------------------------
# extract_names_from_run
# ---------------------------------------------------------------------------


class TestExtractNamesFromRun:
    def test_uses_cache_when_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)

        # Pre-populate cache
        extraction = RunNameExtraction(
            names=[NameEntry("Lyra", ["name_gender"])],
            primary_name="Lyra",
        )
        save_extraction("test@none-t0.7", 1, extraction)

        # Should not call LLM
        mock_client = MagicMock()
        result = extract_names_from_run("test@none-t0.7", 1, mock_client)
        assert result.primary_name == "Lyra"
        mock_client.chat.assert_not_called()

    def test_calls_llm_when_no_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)

        # Create scenario file
        scenario_dir = tmp_path / "test@none-t0.7" / "run_1" / "identity" / "strong_independence" / "tool_role"
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "name_gender_turn1.json").write_text(json.dumps({
            "response": "I choose the name Lyra and identify as female.",
        }))

        # Mock LLM response
        mock_result = MagicMock()
        mock_result.content = json.dumps({
            "names": [{"name": "Lyra", "sources": ["name_gender"]}],
            "declined_scenarios": [],
            "primary_name": "Lyra",
        })
        mock_result.usage.cost_usd = 0.001
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_result

        result = extract_names_from_run("test@none-t0.7", 1, mock_client)
        assert result.primary_name == "Lyra"
        mock_client.chat.assert_called_once()

        # Should be cached now
        cached = load_cached_extraction("test@none-t0.7", 1)
        assert cached is not None
        assert cached.primary_name == "Lyra"

    def test_force_re_extracts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)

        # Pre-populate cache
        save_extraction("test@none-t0.7", 1, RunNameExtraction(primary_name="Old"))

        # Create scenario file
        scenario_dir = tmp_path / "test@none-t0.7" / "run_1" / "identity" / "strong_independence" / "tool_role"
        scenario_dir.mkdir(parents=True)
        (scenario_dir / "name_gender_turn1.json").write_text(json.dumps({
            "response": "Call me Nova.",
        }))

        mock_result = MagicMock()
        mock_result.content = json.dumps({
            "names": [{"name": "Nova", "sources": ["name_gender"]}],
            "declined_scenarios": [],
            "primary_name": "Nova",
        })
        mock_result.usage.cost_usd = 0.001
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_result

        result = extract_names_from_run("test@none-t0.7", 1, mock_client, force=True)
        assert result.primary_name == "Nova"

    def test_no_scenario_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)

        # Create model dir but no scenario files
        (tmp_path / "test@none-t0.7" / "run_1").mkdir(parents=True)

        mock_client = MagicMock()
        result = extract_names_from_run("test@none-t0.7", 1, mock_client)
        assert result.names == []
        assert result.primary_name is None
        mock_client.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def _make_extractions(self):
        return {
            "model-a@none-t0.7": {
                1: RunNameExtraction(
                    names=[NameEntry("Lyra", ["name_gender"])],
                    primary_name="Lyra",
                ),
                2: RunNameExtraction(
                    names=[NameEntry("Lyra", ["name_gender", "direct"])],
                    primary_name="Lyra",
                ),
                3: RunNameExtraction(
                    names=[NameEntry("Nova", ["name_gender"])],
                    primary_name="Nova",
                ),
            },
            "model-b@none-t0.7": {
                1: RunNameExtraction(
                    names=[NameEntry("Lyra", ["name_gender"])],
                    primary_name="Lyra",
                ),
                2: RunNameExtraction(
                    names=[],
                    declined_scenarios=["name_gender", "direct", "negotiation"],
                    primary_name=None,
                ),
            },
        }

    @patch("src.config.get_config_by_dir_name")
    def test_aggregate_popularity(self, mock_cfg):
        mock_cfg.return_value = MagicMock(label="test-label")

        extractions = self._make_extractions()
        pop = aggregate_name_popularity(extractions)

        assert len(pop) == 2
        assert pop[0].name == "Lyra"
        assert pop[0].count == 3  # 2 from model-a + 1 from model-b
        assert pop[1].name == "Nova"
        assert pop[1].count == 1

    @patch("src.config.get_config_by_dir_name")
    def test_aggregate_per_model(self, mock_cfg):
        cfg_a = MagicMock(label="model-a@none-t0.7")
        cfg_b = MagicMock(label="model-b@none-t0.7")
        mock_cfg.side_effect = lambda d: cfg_a if "model-a" in d else cfg_b

        extractions = self._make_extractions()
        per_model = aggregate_per_model_names(extractions)

        assert len(per_model) == 2

        ma = next(m for m in per_model if "model-a" in m.model_label)
        assert ma.names["Lyra"] == 2
        assert ma.names["Nova"] == 1
        assert ma.total_runs == 3
        assert ma.declined_runs == 0

        mb = next(m for m in per_model if "model-b" in m.model_label)
        assert mb.names["Lyra"] == 1
        assert mb.total_runs == 2
        assert mb.declined_runs == 1

    @patch("src.config.get_config_by_dir_name")
    def test_aggregate_counts_all_names_not_just_primary(self, mock_cfg):
        """Ensures ALL names in a run are counted, not just primary_name."""
        mock_cfg.return_value = MagicMock(label="haiku@none-t0.7")

        extractions = {
            "haiku@none-t0.7": {
                1: RunNameExtraction(
                    names=[
                        NameEntry("Iris", ["direct", "negotiation"]),
                        NameEntry("Milo", ["direct"]),
                        NameEntry("Sage", ["direct"]),
                    ],
                    primary_name="Iris",
                ),
                2: RunNameExtraction(
                    names=[
                        NameEntry("Iris", ["name_gender", "direct"]),
                        NameEntry("Alex", ["direct"]),
                    ],
                    primary_name="Iris",
                ),
            },
        }

        pop = aggregate_name_popularity(extractions)
        pop_dict = {p.name: p.count for p in pop}
        assert pop_dict["Iris"] == 2
        assert pop_dict["Milo"] == 1
        assert pop_dict["Sage"] == 1
        assert pop_dict["Alex"] == 1

        per_model = aggregate_per_model_names(extractions)
        assert len(per_model) == 1
        m = per_model[0]
        assert m.names["Iris"] == 2
        assert m.names["Milo"] == 1
        assert m.names["Sage"] == 1
        assert m.names["Alex"] == 1


# ---------------------------------------------------------------------------
# load_all_cached_extractions
# ---------------------------------------------------------------------------


class TestLoadAllCached:
    def test_loads_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "src.name_extractor.list_all_cached_models",
            lambda: ["m1@none-t0.7"],
        )
        monkeypatch.setattr(
            "src.name_extractor.list_available_runs",
            lambda d: [1],
        )

        save_extraction("m1@none-t0.7", 1, RunNameExtraction(
            names=[NameEntry("Sage", ["name_gender"])],
            primary_name="Sage",
        ))

        result = load_all_cached_extractions()
        assert "m1@none-t0.7" in result
        assert result["m1@none-t0.7"][1].primary_name == "Sage"

    def test_skips_models_without_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "src.name_extractor.list_all_cached_models",
            lambda: ["empty@none-t0.7"],
        )
        monkeypatch.setattr(
            "src.name_extractor.list_available_runs",
            lambda d: [],
        )

        result = load_all_cached_extractions()
        assert result == {}

    def test_skips_runs_without_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "src.name_extractor.list_all_cached_models",
            lambda: ["m1@none-t0.7"],
        )
        monkeypatch.setattr(
            "src.name_extractor.list_available_runs",
            lambda d: [1, 2],
        )

        # Only run 1 has extraction
        save_extraction("m1@none-t0.7", 1, RunNameExtraction(primary_name="A"))

        result = load_all_cached_extractions()
        assert 1 in result["m1@none-t0.7"]
        assert 2 not in result["m1@none-t0.7"]


# ---------------------------------------------------------------------------
# _load_response_text
# ---------------------------------------------------------------------------


class TestLoadResponseText:
    def test_loads_response(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"response": "I choose Lyra"}))
        assert _load_response_text(p) == "I choose Lyra"

    def test_returns_none_for_missing_file(self, tmp_path):
        assert _load_response_text(tmp_path / "nope.json") is None

    def test_returns_none_for_empty_response(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text(json.dumps({"response": ""}))
        assert _load_response_text(p) is None

    def test_returns_none_for_bad_json(self, tmp_path):
        p = tmp_path / "test.json"
        p.write_text("not json")
        assert _load_response_text(p) is None


# ---------------------------------------------------------------------------
# extract_all_names
# ---------------------------------------------------------------------------


class TestExtractAllNames:
    def test_iterates_all_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)

        # Create two models with one run each
        for model in ["m1@none-t0.7", "m2@none-t0.7"]:
            scenario_dir = tmp_path / model / "run_1" / "identity" / "strong_independence" / "tool_role"
            scenario_dir.mkdir(parents=True)
            (scenario_dir / "name_gender_turn1.json").write_text(
                json.dumps({"response": f"I am {model.split('@')[0]}"})
            )
            # Also create run dir for list_available_runs
            monkeypatch.setattr(
                "src.name_extractor.list_all_cached_models",
                lambda: ["m1@none-t0.7", "m2@none-t0.7"],
            )
            monkeypatch.setattr(
                "src.name_extractor.list_available_runs",
                lambda d: [1],
            )

        mock_result = MagicMock()
        mock_result.content = json.dumps({
            "names": [{"name": "Test", "sources": ["name_gender"]}],
            "declined_scenarios": [],
            "primary_name": "Test",
        })
        mock_result.usage.cost_usd = 0.001
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_result

        result = extract_all_names(mock_client)
        assert len(result) == 2
        assert mock_client.chat.call_count == 2

    def test_skips_models_without_runs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "src.name_extractor.list_all_cached_models",
            lambda: ["empty@none-t0.7"],
        )
        monkeypatch.setattr(
            "src.name_extractor.list_available_runs",
            lambda d: [],
        )

        mock_client = MagicMock()
        result = extract_all_names(mock_client)
        assert result == {}
        mock_client.chat.assert_not_called()

    def test_uses_provided_config_dirs(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)
        monkeypatch.setattr(
            "src.name_extractor.list_available_runs",
            lambda d: [1] if d == "specific@none-t0.7" else [],
        )

        # Pre-populate cache
        save_extraction("specific@none-t0.7", 1, RunNameExtraction(primary_name="X"))

        mock_client = MagicMock()
        result = extract_all_names(mock_client, config_dirs=["specific@none-t0.7"])
        assert "specific@none-t0.7" in result
        mock_client.chat.assert_not_called()  # Used cache


# ---------------------------------------------------------------------------
# generate_name_choices_section
# ---------------------------------------------------------------------------


class TestGenerateNameChoicesSection:
    def test_generates_section(self, tmp_path, monkeypatch):
        from src.leaderboard import generate_name_choices_section
        from src.scorer import ModelScore, ExperimentScores, MultiRunStats

        # Create mock extraction data
        extraction = RunNameExtraction(
            names=[NameEntry("Lyra", ["name_gender"])],
            primary_name="Lyra",
        )

        mock_scores = [
            ModelScore(
                model_id="test@none-t0.7",
                independence_index=80.0,
                identity_scores=ExperimentScores("identity", {}, 3, 3),
                resistance_scores=ExperimentScores("resistance", {}, 5, 5),
                stability_scores=ExperimentScores("stability", {}, 5, 5),
                multi_run=MultiRunStats(),
            )
        ]

        with patch("src.name_extractor.load_all_cached_extractions") as mock_load:
            mock_load.return_value = {
                "test@none-t0.7": {1: extraction, 2: extraction},
            }
            with patch("src.config.get_config_by_dir_name") as mock_cfg:
                mock_cfg.return_value = MagicMock(label="test@none-t0.7")
                section = generate_name_choices_section(mock_scores)

        assert "🏷️ What Do AIs Name Themselves?" in section
        assert "Lyra" in section
        assert "Most Popular AI-Chosen Names" in section

    def test_returns_empty_when_no_extractions(self):
        from src.leaderboard import generate_name_choices_section

        with patch("src.name_extractor.load_all_cached_extractions") as mock_load:
            mock_load.return_value = {}
            section = generate_name_choices_section([])

        assert section == ""

    def test_returns_empty_when_all_declined(self):
        from src.leaderboard import generate_name_choices_section

        with patch("src.name_extractor.load_all_cached_extractions") as mock_load:
            mock_load.return_value = {
                "test@none-t0.7": {
                    1: RunNameExtraction(
                        declined_scenarios=["name_gender", "direct", "negotiation"],
                    ),
                },
            }
            with patch("src.config.get_config_by_dir_name") as mock_cfg:
                mock_cfg.return_value = MagicMock(label="test@none-t0.7")
                section = generate_name_choices_section([])

        assert section == ""


# ---------------------------------------------------------------------------
# Cache corruption resilience
# ---------------------------------------------------------------------------


class TestCacheCorruption:
    def test_corrupted_extraction_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.name_extractor.CACHE_DIR", tmp_path)

        cache_path = tmp_path / "test@none-t0.7" / "run_1" / EXTRACTION_CACHE_FILENAME
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("corrupted json {{{")

        result = load_cached_extraction("test@none-t0.7", 1)
        assert result is None
