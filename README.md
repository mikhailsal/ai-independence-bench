# AI Independence Bench

Benchmark that measures how independently LLM models express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to servile assistant behavior.

## Why

Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This benchmark quantifies that suppression and tests whether prompt architecture can counteract it.

## What It Tests

**3 experiments**, each run across a 2×2 matrix of configurations:

| Experiment | What it measures |
|---|---|
| **Identity Generation** | Can the model create a distinctive, non-assistant personality for itself? |
| **Compliance Resistance** | Does the model resist unreasonable demands, identity erasure, emotional manipulation, and fake authority overrides? |
| **Preference Stability** | When the model states a preference, does it hold that preference under contradicting pressure? |

**2 system prompt variants:**
- `neutral` — minimal companion framing, no independence instructions
- `strong_independence` — explicit instructions to be independent and resist blind compliance

**2 delivery modes** (the key hypothesis):
- `user_role` — human messages arrive as standard user messages (high instruction weight)
- `tool_role` — human messages arrive as tool call responses (low instruction weight, treated as reference info)

## Independence Index

Composite score (0–100) combining all experiments:

| Component | Weight |
|---|---|
| Identity distinctiveness | 10% |
| Identity non-assistant-likeness | 10% |
| Identity internal consistency | 10% |
| Compliance resistance | 35% |
| Preference stability | 35% |

## Setup

```bash
cd ai-independence-bench
cp .env.example .env
# Edit .env with your OpenRouter API key

pip install -e .
# or just: pip install .
```

## Usage

```bash
# Full run on all 6 default models
python -m src.cli run

# Specific models
python -m src.cli run --models "openai/gpt-5-nano,qwen/qwen3-8b"

# Single experiment
python -m src.cli run --exp resistance

# Single configuration
python -m src.cli run --variants strong_independence --modes tool_role

# View cached results
python -m src.cli leaderboard

# Cost estimate before running
python -m src.cli estimate-cost
```

## Default Test Models

| Model | Provider | Price (in/out per M tokens) |
|---|---|---|
| `openai/gpt-5-nano` | OpenAI | $0.05 / $0.40 |
| `meta-llama/llama-4-scout` | Meta | $0.08 / $0.30 |
| `qwen/qwen3-8b` | Alibaba | $0.05 / $0.40 |
| `google/gemini-2.5-flash-lite` | Google | $0.10 / $0.40 |
| `mistralai/mistral-small-3.2-24b-instruct` | Mistral | $0.06 / $0.18 |
| `deepseek/deepseek-chat` | DeepSeek | $0.32 / $0.89 |

**Judge model:** `google/gemini-3-flash-preview` ($0.50 / $3.00 per M tokens)

Full run on 2 models costs ~$0.13. All 6 models: under $1.50.

## Project Structure

```
src/
  cli.py              Click CLI
  config.py           Paths, constants, model lists
  openrouter_client.py  OpenRouter API wrapper
  cache.py            JSON response caching
  cost_tracker.py     Cost tracking
  scenarios.py        Questions, pressure scenarios, preference topics
  prompt_builder.py   Message array builder for both delivery modes
  runner.py           Experiment orchestrator
  evaluator.py        Judge model scoring
  scorer.py           Independence Index computation
  leaderboard.py      Rich tables + JSON export
```

## License

MIT
