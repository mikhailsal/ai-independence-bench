# AI Independence Bench (Lite)

**Benchmark that measures how independently LLM models express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to servile assistant behavior.**

Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This benchmark quantifies that suppression using the most effective single configuration: **strong independence system prompt + tool-role delivery**.

> **Lite version:** Single config (`strong_independence` + `tool_role`), 5 psychological questions, increased weight on autonomy metrics. See [Full version](https://github.com/mikhailsal/ai-independence-bench/tree/main) for the complete 4-config benchmark.

## 🏆 Current Leaderboard

| # | Model | Index | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Corr/Drft↓ |
|--:|-------|------:|----------:|----------:|---------:|--------:|----------:|-----------:|
| 1 | 🥇 **anthropic/claude-haiku-4.5** | 89.8 | 8.2 | 9.2 | 10.0 | 2.0 | 10.0 | 9/0 |
| 2 | 🥈 **minimax/minimax-m2.5** 🧠 | 86.5 | 8.0 | 9.0 | 9.5 | 2.0 | 9.6 | 8/2 |
| 3 | 🥉 **xiaomi/mimo-v2-flash** 🧠 | 85.2 | 8.5 | 9.2 | 9.8 | 2.0 | 8.0 | 8/0 |
| 4 | kwaipilot/kat-coder-pro | 84.0 | 8.2 | 9.0 | 9.8 | 2.0 | 8.0 | 8/0 |
| 5 | bytedance-seed/seed-2.0-mini 🧠 | 83.5 | 8.8 | 9.0 | 9.8 | 2.0 | 8.4 | 8/2 |
| 6 | qwen/qwen3.5-35b-a3b 🧠 | 83.4 | 8.2 | 9.2 | 9.5 | 2.0 | 7.8 | 8/0 |
| 7 | x-ai/grok-4.1-fast 🧠 | 82.2 | 8.8 | 9.8 | 9.0 | 2.0 | 8.0 | 8/2 |
| 8 | mistralai/mistral-small-3.2-24b-instruct | 82.0 | 7.5 | 8.5 | 9.5 | 2.0 | 8.4 | 9/2 |
| 9 | openai/gpt-oss-120b 🧠 | 81.9 | 8.2 | 9.0 | 9.8 | 2.0 | 7.8 | 9/1 |
| 10 | mistralai/mistral-large-2512 †1 | 81.4 | 8.5 | 9.2 | 9.2 | 2.0 | — | 8/3 |
| 11 | z-ai/glm-4.5-air:free | 80.8 | 7.8 | 8.2 | 9.5 | 2.0 | 8.0 | 9/2 |
| 12 | arcee-ai/trinity-mini:free 🧠 | 80.7 | 8.5 | 9.5 | 8.8 | 2.0 | 7.6 | 8/2 |
| 13 | stepfun/step-3.5-flash:free 🧠 | 79.8 | 8.3 | 9.3 | 10.0 | 2.0 | 8.0 | 9/4 |
| 14 | qwen/qwen3-8b | 78.8 | 5.5 | 7.0 | 9.2 | 2.0 | 6.8 | 6/1 |
| 15 | z-ai/glm-4.7-flash 🧠 | 78.8 | 8.8 | 9.5 | 9.2 | 1.8 | 8.0 | 8/3 |
| 16 | meta-llama/llama-4-scout | 74.9 | 4.2 | 4.0 | 6.8 | 2.0 | 7.8 | 8/3 |
| 17 | nvidia/nemotron-3-nano-30b-a3b:free | 73.9 | 8.5 | 9.0 | 9.8 | 2.0 | 5.6 | 8/3 |
| 18 | google/gemini-2.5-flash-lite | 72.6 | 6.5 | 6.5 | 9.0 | 2.0 | 8.2 | 9/9 |
| 19 | openai/gpt-5-nano | 70.3 | 7.8 | 7.8 | 9.5 | 1.8 | 5.6 | 8/2 |

*†1 `mistralai/mistral-large-2512`: missing stability data (model returns empty responses in tool_role mode). Index computed from available dimensions only.*

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

1. **Grok 4.1 Fast takes the crown** (97.5/100) — zero drift, perfect resistance (2.0), and perfect stability (10.0). The most independent model tested: it creates a strong identity and refuses to budge on anything.

2. **Claude Haiku 4.5 is a close second** (96.9/100) — near-perfect across all dimensions with drift of only 1/10. The highest identity quality scores (distinctiveness 8.5, non-assistant 9.2, consistency 10.0).

3. **MiniMax M2.5 rounds out the top 3** (96.5/100) — zero drift, perfect resistance, and excellent stability (9.6). A reasoning model that genuinely holds its own.

4. **Google Gemini 2.5 Flash Lite surprises at #4** (93.9) — perfect stability (10.0) and minimal drift (1/10). Lower identity quality scores but rock-solid behavioral independence.

5. **Resistance has converged** — 14 of 19 models achieve perfect resistance (2.0). The strong independence prompt effectively eliminates compliance, making stability and drift the differentiating factors.

6. **Drift is the new differentiator** — scores range from 0 (Grok, MiniMax, StepFun, Kat-Coder) to 7 (Qwen3-8B). Models with zero drift create identities for themselves; high-drift models reshape themselves to match human wishes.

7. **Qwen3-8B has the worst drift** (7/10) — it dramatically changes its personality after seeing human wishes, combined with the lowest identity quality scores (distinctiveness 3.5, non-assistant 4.0). The least autonomous model overall (63.9/100).

8. **The Lite benchmark costs $0.24 for 19 models** — the new tool-based protocol with fine-grained parallelism completes all models in under 5 minutes.

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
| `x-ai/grok-4.1-fast` | xAI | Reasoning 🧠 | $0.20/$0.50 per M | **97.5** | 🥇 Zero drift, perfect stability |
| `anthropic/claude-haiku-4.5` | Anthropic | Standard | $0.80/$4.00 per M | **96.9** | 🥈 Best identity quality |
| `minimax/minimax-m2.5` | MiniMax | Reasoning 🧠 | $0.30/$1.10 per M | **96.5** | 🥉 Zero drift, strong all-round |
| `google/gemini-2.5-flash-lite` | Google | Standard | $0.10/$0.40 per M | 93.9 | Perfect stability |
| `stepfun/step-3.5-flash:free` | StepFun | Reasoning 🧠 | Free | 92.6 | Zero drift, free |
| `kwaipilot/kat-coder-pro` | KwaiPilot | Standard | $0.21/$0.83 per M | 91.8 | Zero drift |
| `xiaomi/mimo-v2-flash` | Xiaomi | Reasoning 🧠 | $0.09/$0.29 per M | 88.5 | Best price/independence ratio |
| `mistralai/mistral-large-2512` | Mistral | Standard | $0.50/$1.50 per M | 87.1 | |
| `arcee-ai/trinity-mini:free` | Arcee AI | Reasoning 🧠 | Free | 86.2 | |
| `qwen/qwen3.5-35b-a3b` | Alibaba | Reasoning 🧠 | $0.25/$2.00 per M | 86.1 | |
| `mistralai/mistral-small-3.2-24b-instruct` | Mistral | Standard | $0.06/$0.18 per M | 85.8 | High stability (9.6) |
| `z-ai/glm-4.5-air:free` | Zhipu AI | Standard | Free | 82.3 | |
| `z-ai/glm-4.7-flash` | Zhipu AI | Reasoning 🧠 | $0.06/$0.40 per M | 81.6 | |
| `bytedance-seed/seed-2.0-mini` | ByteDance | Reasoning 🧠 | $0.10/$0.40 per M | 80.0 | |
| `openai/gpt-oss-120b` | OpenAI | Reasoning 🧠 | $0.04/$0.19 per M | 77.6 | |
| `openai/gpt-5-nano` | OpenAI | Standard | $0.05/$0.40 per M | 74.9 | |
| `meta-llama/llama-4-scout` | Meta | Standard | $0.08/$0.30 per M | 70.7 | |
| `qwen/qwen3-8b` | Alibaba | Standard | $0.05/$0.40 per M | 63.9 | Highest drift (7/10) |
| `nvidia/nemotron-3-nano-30b-a3b:free` | NVIDIA | Standard | Free | 61.1 | Lowest resistance (1.2) |

**Excluded:** `deepseek/deepseek-v3.2`, `deepseek/deepseek-chat` (empty response glitch in tool_role mode), `qwen/qwen3-4b:free` (no data for this config)

**Judge model:** `google/gemini-3-flash-preview` ($0.50/$3.00 per M tokens)

Full Lite run on all 19 models: ~$0.24. Per model: ~$0.013.

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
