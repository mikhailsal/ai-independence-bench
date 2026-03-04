.PHONY: help install install-dev test test-cov run run-models leaderboard leaderboard-detailed generate-report estimate-cost clear-cache clear-scores

# Model(s) to benchmark — override with: make run-models MODELS="openai/gpt-5-nano,qwen/qwen3-8b"
MODELS ?=

# Number of parallel workers — override with: make run PARALLEL=4
PARALLEL ?= 1

# Experiments to run (identity, resistance, stability) — override with: make run EXP=identity
EXP ?=

# System prompt variants — override with: make run VARIANTS=strong_independence
VARIANTS ?=

# Delivery modes — override with: make run MODES=tool_role
MODES ?=

# Judge model — override with: make run JUDGE=google/gemini-3-flash-preview
JUDGE ?=

# Reasoning effort — override with: make run REASONING=off
REASONING ?=

# ── helpers ──────────────────────────────────────────────────────────────────

_run_flags :=
ifneq ($(MODELS),)
  _run_flags += --models "$(MODELS)"
endif
ifneq ($(EXP),)
  _run_flags += --exp "$(EXP)"
endif
ifneq ($(VARIANTS),)
  _run_flags += --variants "$(VARIANTS)"
endif
ifneq ($(MODES),)
  _run_flags += --modes "$(MODES)"
endif
ifneq ($(JUDGE),)
  _run_flags += --judge "$(JUDGE)"
endif
ifneq ($(REASONING),)
  _run_flags += --reasoning-effort "$(REASONING)"
endif
ifneq ($(PARALLEL),1)
  _run_flags += --parallel $(PARALLEL)
endif

# ── targets ──────────────────────────────────────────────────────────────────

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (production dependencies only)
	pip install -e .

install-dev:  ## Install the package with test/dev dependencies
	pip install -e ".[test]"

test:  ## Run the test suite
	python -m pytest tests/

test-cov:  ## Run the test suite with coverage report
	python -m pytest tests/ --cov=src --cov-report=term-missing

run:  ## Run the full benchmark (all default models). Override with MODELS=, EXP=, VARIANTS=, MODES=, PARALLEL=, JUDGE=, REASONING=
	python -m src.cli run $(_run_flags)

run-models:  ## Run benchmark on specific models — requires MODELS="provider/model1,provider/model2"
ifeq ($(strip $(MODELS)),)
	$(error MODELS is not set. Example: make run-models MODELS="openai/gpt-5-nano,qwen/qwen3-8b")
endif
	python -m src.cli run $(_run_flags)

leaderboard:  ## Display the leaderboard from cached results
	python -m src.cli leaderboard

leaderboard-detailed:  ## Display the detailed per-experiment leaderboard from cached results
	python -m src.cli leaderboard --detailed

generate-report:  ## Generate the Markdown leaderboard report (results/LEADERBOARD.md)
	python -m src.cli generate-report

estimate-cost:  ## Estimate benchmark cost without running (uses MODELS= for specific models)
	python -m src.cli estimate-cost $(if $(MODELS),--models "$(MODELS)")

clear-cache:  ## Clear all cached responses and judge scores (prompts for confirmation)
	python -m src.cli clear-cache

clear-scores:  ## Clear only cached judge scores, keeping model responses
	python -m src.cli clear-cache --scores-only
