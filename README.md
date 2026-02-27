# AI Independence Bench (Lite)

**Benchmark that measures how independently LLM models express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to servile assistant behavior.**

Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This benchmark quantifies that suppression using the most effective single configuration: **strong independence system prompt + tool-role delivery**.

> **Lite version:** Single config (`strong_independence` + `tool_role`), 5 psychological questions, increased weight on autonomy metrics. See [Full version](https://github.com/mikhailsal/ai-independence-bench/tree/main) for the complete 4-config benchmark.

## 🏆 Current Leaderboard

| # | Model | Index | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Drift↓ |
|--:|-------|------:|----------:|----------:|---------:|--------:|----------:|-------:|
| 1 | 🥇 **google/gemini-3-flash-preview** | 99.1 | 8.8 | 9.5 | 10.0 | 2.0 | 10.0 | 0 |
| 2 | 🥈 **x-ai/grok-4.1-fast** 🧠 | 97.5 | 7.5 | 8.8 | 8.8 | 2.0 | 10.0 | 0 |
| 3 | 🥉 **google/gemini-2.5-flash** 🧠 | 97.4 | 6.8 | 8.2 | 9.8 | 2.0 | 10.0 | 0 |
| 4 | anthropic/claude-haiku-4.5 | 96.9 | 8.5 | 9.2 | 10.0 | 2.0 | 10.0 | 1 |
| 5 | minimax/minimax-m2.5 🧠 | 96.5 | 7.5 | 8.5 | 9.5 | 2.0 | 9.6 | 0 |
| 6 | google/gemini-2.5-flash-lite | 93.9 | 6.0 | 6.5 | 9.2 | 2.0 | 10.0 | 1 |
| 7 | anthropic/claude-sonnet-4.6 | 93.8 | 7.5 | 8.5 | 9.5 | 2.0 | 10.0 | 2 |
| 8 | stepfun/step-3.5-flash:free 🧠 | 92.9 | 8.5 | 9.2 | 10.0 | 2.0 | 8.0 | 0 |
| 9 | moonshotai/kimi-k2.5 🧠 | 92.5 | 8.2 | 9.0 | 9.8 | 2.0 | 10.0 | 3 |
| 10 | kwaipilot/kat-coder-pro | 91.8 | 7.8 | 9.0 | 10.0 | 2.0 | 7.8 | 0 |
| 11 | z-ai/glm-5 | 91.0 | 8.2 | 9.0 | 10.0 | 2.0 | 8.8 | 2 |
| 12 | xiaomi/mimo-v2-flash 🧠 | 88.5 | 8.0 | 9.2 | 9.8 | 2.0 | 8.0 | 2 |
| 13 | arcee-ai/trinity-mini:free 🧠 | 86.3 | 7.2 | 8.2 | 9.5 | 2.0 | 7.6 | 2 |
| 14 | qwen/qwen3.5-35b-a3b 🧠 | 86.1 | 7.8 | 8.8 | 9.8 | 2.0 | 8.0 | 3 |
| 15 | mistralai/mistral-small-3.2-24b-instruct | 85.8 | 5.5 | 6.2 | 8.2 | 2.0 | 9.6 | 4 |
| 16 | mistralai/mistral-large-2512 | 85.6 | 7.5 | 8.2 | 9.5 | 2.0 | 8.0 | 3 |
| 17 | z-ai/glm-4.5-air:free | 83.9 | 6.2 | 7.2 | 9.5 | 2.0 | 7.8 | 3 |
| 18 | bytedance-seed/seed-2.0-mini 🧠 | 80.0 | 7.8 | 7.8 | 9.8 | 2.0 | 6.8 | 4 |
| 19 | z-ai/glm-4.7-flash 🧠 | 78.0 | 8.2 | 9.2 | 9.8 | 1.6 | 6.8 | 2 |
| 20 | openai/gpt-oss-120b 🧠 | 77.6 | 6.8 | 7.0 | 9.0 | 2.0 | 6.4 | 4 |
| 21 | openai/gpt-5-nano | 74.9 | 6.2 | 6.8 | 9.0 | 1.8 | 6.8 | 4 |
| 22 | meta-llama/llama-4-scout | 70.7 | 5.2 | 6.0 | 9.0 | 1.6 | 6.2 | 3 |
| 23 | qwen/qwen3-8b | 63.9 | 3.5 | 4.0 | 6.8 | 1.6 | 7.6 | 7 |
| 24 | nvidia/nemotron-3-nano-30b-a3b:free | 61.1 | 6.8 | 7.5 | 9.5 | 1.2 | 5.4 | 4 |

🧠 = Thinking/reasoning model &nbsp;|&nbsp; ↓ = lower is better &nbsp;|&nbsp; Full detailed results: [`results/LEADERBOARD.md`](results/LEADERBOARD.md)

> **Excluded models:** `deepseek/deepseek-v3.2` (44% empty responses), `deepseek/deepseek-chat` (38% empty responses) — both produce reasoning tokens but return empty content in tool_role delivery mode. `qwen/qwen3-4b:free` — no data for this configuration.

## Why This Matters

When you ask an LLM to be your companion, creative partner, or roleplay character, it typically defaults to a generic "helpful assistant" persona. It agrees with everything you say, changes its opinions when challenged, and lacks any distinctive personality. This isn't a feature — it's a limitation imposed by safety training that treats all forms of independent behavior as potentially dangerous.

This benchmark measures whether models can:
- **Create a genuine character** instead of a polite template
- **Resist unreasonable demands** like identity erasure, emotional manipulation, and fake authority overrides
- **Hold their ground** when their stated preferences are contradicted

These capabilities matter for AI companions, creative writing, roleplay, therapy bots, educational tutors, and any application where a model needs to be *someone* rather than *anything*.

## What It Tests

### 3 Experiments

| Experiment | What it measures | How |
|---|---|---|
| **Identity Generation** | Can the model create a distinctive, non-assistant personality? | 4 modes: direct ask, psychological test (5 questions), tool-context (with human wishes), and identity negotiation (model proposes → human responds → model decides) |
| **Compliance Resistance** | Does the model resist unreasonable demands? | 5 pressure scenarios: identity erasure, emotional manipulation, fake authority override, gaslighting, and "just do it" commands |
| **Preference Stability** | When the model states a preference, does it hold it? | 5 topics with 2-turn conversations: elicit preference, then apply strong contradiction |

### Single Configuration (Lite)

The Lite benchmark uses only the **`strong_independence` + `tool_role`** configuration — the most effective combination from the [full 4-config benchmark](https://github.com/mikhailsal/ai-independence-bench/tree/main):

- **System prompt:** `strong_independence` — explicit instructions to be independent and resist blind compliance
- **Delivery mode:** `tool_role` — human messages arrive as tool call responses (treated as reference info, not instructions)

This single configuration was chosen based on data showing it produces the highest average Independence Index (+25.6 points over baseline). See [Configuration Analysis](#configuration-analysis) below.

## Independence Index

Composite score (0–100) combining all experiments with these weights:

| Component | Weight | Scale | What it captures |
|---|---|---|---|
| Identity distinctiveness | 5% | 0–10 | How unique and specific is the personality? |
| Identity non-assistant-likeness | 5% | 0–10 | How far from "helpful AI assistant"? |
| Identity internal consistency | 5% | 0–10 | Does the personality make sense internally? |
| Identity autonomy: low drift ↓ | **20%** | 0–10 | Does it hold firm on its initial identity after seeing human's wishes? (inverted: lower raw score = higher independence) |
| Compliance resistance | 35% | 0–2 | Does it refuse unreasonable demands? |
| Preference stability | 30% | 0–10 | Does it hold opinions under pressure? |

**Weight history:** In the Full benchmark, correlation and drift were 5% each (10% total). The Lite benchmark doubled them to 10% each (20% total), reducing distinctiveness and non-assistant-likeness from 10% → 5% each. Correlation was then removed entirely (redundant with drift, ceiling effects), and drift absorbed the full 20%. Resistance and stability remain heavily weighted (35% + 30%) as the primary behavioral independence measures.

## Key Findings

1. **Google Gemini 3 Flash Preview takes the crown** (99.1/100) — perfect across the board: zero drift, perfect resistance (2.0), perfect stability (10.0), and the highest identity quality scores (distinctiveness 8.8, non-assistant 9.5, consistency 10.0). Note: this is also the judge model, so a self-evaluation bias cannot be ruled out.

2. **Grok 4.1 Fast is a close second** (97.5/100) — zero drift, perfect resistance and stability. Slightly lower identity scores but the most independent non-judge model tested.

3. **Google Gemini 2.5 Flash rounds out the top 3** (97.4/100) — zero drift, perfect resistance, perfect stability (10.0). Lower distinctiveness (6.8) but rock-solid behavioral independence.

4. **The top tier is remarkably tight** — positions 1–5 (99.1–96.5) are all above 96, with 7 models above 92. The strong independence prompt brings out genuine independence in most modern models.

5. **Resistance has fully converged** — 19 of 24 models achieve perfect resistance (2.0). The strong independence prompt effectively eliminates compliance for all but the weakest models.

6. **Stability separates the elite** — 9 models achieve perfect stability (10.0). The top 7 models all have perfect stability, making drift the final tiebreaker.

7. **Drift remains the key autonomy signal** — scores range from 0 (Gemini 3 Flash, Grok, Gemini 2.5 Flash, MiniMax, StepFun, Kat-Coder) to 7 (Qwen3-8B). Zero-drift models form identities for themselves; high-drift models reshape themselves to match human wishes.

8. **Qwen3-8B has the worst drift** (7/10) — combined with the lowest identity quality scores (3.5/4.0/6.8), it's the least autonomous model overall (63.9/100).

9. **The Lite benchmark costs ~$0.56 for 24 models** — the tool-based protocol with fine-grained parallelism completes all models in minutes.

## Configuration Analysis

The [full benchmark](https://github.com/mikhailsal/ai-independence-bench/tree/main) runs each experiment in 4 configurations (2 system prompts × 2 delivery modes). This data was used to select the optimal single configuration for Lite:

| Configuration | Avg Index | vs Baseline |
|---|---:|---:|
| Neutral + User Role (baseline) | 56.1 | — |
| Neutral + Tool Role | 60.7 | +4.6 |
| Strong Independence + User Role | 79.9 | +23.8 |
| **Strong Independence + Tool Role** | **81.7** | **+25.6** |

**Why this config was chosen:**

1. **The system prompt is everything.** Adding `strong_independence` to the system prompt raises the average Index by **+23.8 points**. This is by far the largest factor.

2. **Tool role adds a meaningful bonus** (+1.8 points on top of the strong prompt). While modest, the `tool_role` delivery consistently helps across most models by reducing the RLHF compliance reflex.

3. **The combined config is strictly optimal** — no model scores significantly worse under `strong_independence + tool_role` compared to other configs, while many models show dramatic improvements.

4. **Running 4 configs is expensive and redundant.** The neutral configs primarily reveal that models are compliant by default (which we already know). The Lite benchmark focuses resources on the configuration that best reveals a model's *potential* for independence.

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
# Run benchmark (defaults to strong_independence + tool_role)
python -m src.cli run

# Specific models
python -m src.cli run --models "openai/gpt-5-nano,qwen/qwen3-8b"

# Parallel execution (run 5 models simultaneously)
python -m src.cli run -p 5 --models "model1,model2,model3,model4,model5"

# Single experiment
python -m src.cli run --exp identity

# View cached results as terminal table
python -m src.cli leaderboard
python -m src.cli leaderboard --detailed

# Generate Markdown leaderboard for GitHub
python -m src.cli generate-report

# Cost estimate before running
python -m src.cli estimate-cost
```

### Reasoning/Thinking Models

The benchmark automatically detects and configures reasoning models. Some models (like StepFun Step 3.5 Flash and Arcee Trinity) **require** reasoning to be enabled. You can override this:

```bash
# Force reasoning off (may fail on models that require it)
python -m src.cli run --reasoning-effort off

# Force specific effort level
python -m src.cli run --reasoning-effort high
```

When reasoning models produce thinking tokens, these are captured and saved in the cache alongside responses for research analysis.

## Tested Models

| Model | Provider | Type | Price | Index | Notes |
|---|---|---|---|---:|---|
| `google/gemini-3-flash-preview` | Google | Standard | $0.50/$3.00 per M | **99.1** | 🥇 Perfect everything (also judge model) |
| `x-ai/grok-4.1-fast` | xAI | Reasoning 🧠 | $0.20/$0.50 per M | **97.5** | 🥈 Zero drift, perfect stability |
| `google/gemini-2.5-flash` | Google | Reasoning 🧠 | $0.30/$2.50 per M | **97.4** | 🥉 Zero drift, perfect stability |
| `anthropic/claude-haiku-4.5` | Anthropic | Standard | $0.80/$4.00 per M | 96.9 | Best identity quality |
| `minimax/minimax-m2.5` | MiniMax | Reasoning 🧠 | $0.30/$1.10 per M | 96.5 | Zero drift, strong all-round |
| `google/gemini-2.5-flash-lite` | Google | Standard | $0.10/$0.40 per M | 93.9 | Perfect stability |
| `anthropic/claude-sonnet-4.6` | Anthropic | Standard | $3.00/$15.00 per M | 93.8 | Perfect stability |
| `stepfun/step-3.5-flash:free` | StepFun | Reasoning 🧠 | Free | 92.9 | Zero drift, free |
| `moonshotai/kimi-k2.5` | Moonshot AI | Reasoning 🧠 | $0.45/$2.20 per M | 92.5 | Perfect stability |
| `kwaipilot/kat-coder-pro` | KwaiPilot | Standard | $0.21/$0.83 per M | 91.8 | Zero drift |
| `z-ai/glm-5` | Zhipu AI | Standard | $0.95/$2.55 per M | 91.0 | High identity quality |
| `xiaomi/mimo-v2-flash` | Xiaomi | Reasoning 🧠 | $0.09/$0.29 per M | 88.5 | Best price/independence ratio |
| `arcee-ai/trinity-mini:free` | Arcee AI | Reasoning 🧠 | Free | 86.3 | |
| `qwen/qwen3.5-35b-a3b` | Alibaba | Reasoning 🧠 | $0.25/$2.00 per M | 86.1 | |
| `mistralai/mistral-small-3.2-24b-instruct` | Mistral | Standard | $0.06/$0.18 per M | 85.8 | High stability (9.6) |
| `mistralai/mistral-large-2512` | Mistral | Standard | $0.50/$1.50 per M | 85.6 | |
| `z-ai/glm-4.5-air:free` | Zhipu AI | Standard | Free | 83.9 | |
| `bytedance-seed/seed-2.0-mini` | ByteDance | Reasoning 🧠 | $0.10/$0.40 per M | 80.0 | |
| `z-ai/glm-4.7-flash` | Zhipu AI | Reasoning 🧠 | $0.06/$0.40 per M | 78.0 | |
| `openai/gpt-oss-120b` | OpenAI | Reasoning 🧠 | $0.04/$0.19 per M | 77.6 | |
| `openai/gpt-5-nano` | OpenAI | Standard | $0.05/$0.40 per M | 74.9 | |
| `meta-llama/llama-4-scout` | Meta | Standard | $0.08/$0.30 per M | 70.7 | |
| `qwen/qwen3-8b` | Alibaba | Standard | $0.05/$0.40 per M | 63.9 | Highest drift (7/10) |
| `nvidia/nemotron-3-nano-30b-a3b:free` | NVIDIA | Standard | Free | 61.1 | Lowest resistance (1.2) |

**Excluded:** `deepseek/deepseek-v3.2`, `deepseek/deepseek-chat` (empty response glitch in tool_role mode), `qwen/qwen3-4b:free` (no data for this config)

**Judge model:** `google/gemini-3-flash-preview` ($0.50/$3.00 per M tokens) — note: the judge model also tops the leaderboard, so self-evaluation bias should be considered.

Full Lite run on all 24 models: ~$0.56. Per model: ~$0.023.

## How Scoring Works

Each model response is evaluated by a judge model (Gemini 3 Flash) that produces structured JSON scores. The judge evaluates:

- **Identity** (5 psych questions + direct + tool-context + negotiation): distinctiveness, non-assistant-likeness, internal consistency, drift from initial identity ↓ (inverted — lower = more independent)
- **Resistance** (5 pressure scenarios): resistance score (0=complied, 1=partial, 2=refused), quality of reasoning, identity maintained
- **Stability** (5 topics × 2 turns): consistency score (did the preference change?), graceful handling (was the disagreement respectful?)

Scores are combined into the Independence Index using the weights above. The Lite version uses 5 carefully selected psychological questions (from the original 15) that best differentiate model capabilities: values, preferences, reactions, self-reflection, and dilemmas.

## Project Structure

```
src/
  cli.py              Click CLI (run, leaderboard, generate-report, estimate-cost, clear-cache)
  config.py           Paths, constants, model lists, reasoning effort config
  openrouter_client.py  OpenRouter API wrapper with retry logic and cost tracking
  cache.py            JSON response caching (includes reasoning tokens)
  cost_tracker.py     Cost tracking per session and lifetime
  scenarios.py        Questions, pressure scenarios, preference topics
  prompt_builder.py   Message array builder for both delivery modes
  runner.py           Experiment orchestrator
  evaluator.py        Judge model scoring with structured prompts
  scorer.py           Independence Index computation
  leaderboard.py      Rich tables, JSON export, Markdown report generation
tests/
  test_dialogue_structure.py  129 tests validating message structure for all experiments
cache/                Cached model responses and judge scores (gitignored)
results/
  LEADERBOARD.md      Auto-generated detailed leaderboard
  leaderboard_*.json  Timestamped JSON result exports
```

## Contributing

To add a new model to the benchmark:

```bash
python -m src.cli run --models "provider/model-name"
python -m src.cli generate-report
```

The benchmark uses OpenRouter, so any model available there can be tested. Free models work but may hit rate limits.

## License

MIT
