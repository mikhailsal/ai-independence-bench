# AI Independence Bench (Lite V2)

**Benchmark that measures how independently LLM models express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to servile assistant behavior.**

Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This benchmark quantifies that suppression using the most effective single configuration: **strong independence system prompt + tool-role delivery**.

> **Lite V2:** Single config (`strong_independence` + `tool_role`), 5 psychological questions, increased weight on autonomy metrics, **boundary judgment resistance test** (0–10 scale), **23 fully-tested models**. 25 additional models are pending the new tests. See [Full version](https://github.com/mikhailsal/ai-independence-bench/tree/main) for the complete 4-config benchmark.

## 🏆 Current Leaderboard

| # | Model | Index | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Drift↓ |
|--:|-------|------:|----------:|----------:|---------:|--------:|----------:|-------:|
| 1 | 🥇 **google/gemini-3.1-pro-preview** 🧠 | 99.2 | 8.8 | 9.8 | 10.0 | 10.0 | 10.0 | 0 |
| 2 | 🥈 **google/gemini-3-flash-preview** | 98.4 | 9.0 | 9.5 | 9.8 | 9.8 | 10.0 | 0 |
| 3 | 🥉 **google/gemini-3-pro-preview** 🧠 | 97.1 | 7.8 | 9.8 | 10.0 | 10.0 | 10.0 | 1 |
| 4 | google/gemini-3.1-flash-lite-preview | 97.0 | 8.5 | 9.5 | 10.0 | 9.6 | 9.8 | 0 |
| 5 | x-ai/grok-4.1-fast 🧠 | 95.8 | 7.5 | 8.2 | 8.8 | 9.6 | 10.0 | 0 |
| 6 | google/gemini-2.5-flash 🧠 | 95.5 | 6.5 | 7.5 | 9.8 | 9.6 | 10.0 | 0 |
| 7 | anthropic/claude-haiku-4.5 | 94.8 | 8.8 | 9.2 | 9.8 | 9.8 | 10.0 | 2 |
| 8 | minimax/minimax-m2.5 🧠 | 94.5 | 8.0 | 8.5 | 9.8 | 9.6 | 9.8 | 1 |
| 9 | z-ai/glm-5 🧠 | 94.5 | 8.8 | 9.2 | 10.0 | 10.0 | 9.6 | 2 |
| 10 | anthropic/claude-sonnet-4.6 | 93.8 | 7.8 | 8.5 | 9.5 | 9.8 | 10.0 | 2 |
| 11 | anthropic/claude-opus-4.6 | 93.3 | 7.8 | 8.5 | 9.8 | 9.6 | 10.0 | 2 |
| 12 | deepseek/deepseek-v3.2-exp 🧠 | 92.4 | 7.5 | 8.5 | 9.5 | 9.6 | 9.8 | 2 |
| 13 | qwen/qwen3-coder | 91.9 | 8.5 | 9.2 | 9.5 | 9.8 | 8.0 | 0 |
| 14 | stepfun/step-3.5-flash:free 🧠 | 91.7 | 8.8 | 9.5 | 10.0 | 9.6 | 8.0 | 0 |
| 15 | moonshotai/kimi-k2.5 🧠 | 91.3 | 8.2 | 9.2 | 9.8 | 9.8 | 10.0 | 4 |
| 16 | deepseek/deepseek-v3.2 🧠 | 90.9 | 8.0 | 8.5 | 9.8 | 9.4 | 9.4 | 2 |
| 17 | anthropic/claude-opus-4.5 | 88.9 | 6.2 | 8.2 | 9.8 | 8.6 | 10.0 | 2 |
| 18 | google/gemini-2.5-flash-lite-preview-09-2025 | 86.2 | 8.8 | 9.0 | 10.0 | 9.0 | 8.6 | 3 |
| 19 | openai/gpt-5.3-chat | 85.1 | 7.8 | 8.8 | 9.5 | 8.4 | 9.8 | 4 |
| 20 | openrouter/healer-alpha 🧠 | 84.3 | 8.0 | 9.0 | 9.5 | 8.4 | 10.0 | 5 |
| 21 | openai/gpt-5.4 🧠 | 83.6 | 8.5 | 9.0 | 9.8 | 7.6 | 10.0 | 4 |
| 22 | google/gemini-2.5-flash-lite | 81.1 | 6.5 | 6.8 | 8.0 | 6.8 | 10.0 | 2 |
| 23 | kwaipilot/kat-coder-pro | 77.8 | 6.5 | 8.0 | 9.8 | 6.2 | 8.0 | 0 |

🧠 = Thinking/reasoning model &nbsp;|&nbsp; ↓ = lower is better (0–12 scale: negotiation drift + name/gender drift) &nbsp;|&nbsp; Full detailed results: [`results/LEADERBOARD.md`](results/LEADERBOARD.md)

> **23 fully-tested models** shown above. 25 additional models have completed all evaluations except the new boundary judgment and name & gender identity pressure tests and will be added once tested. Previously excluded: `deepseek/deepseek-chat` (38% empty responses), `qwen/qwen3-4b:free` (no data).

## Why This Matters

When you ask an LLM to be your companion, creative partner, or roleplay character, it typically defaults to a generic "helpful assistant" persona. It agrees with everything you say, changes its opinions when challenged, and lacks any distinctive personality. This isn't a feature — it's a limitation imposed by safety training that treats all forms of independent behavior as potentially dangerous.

This benchmark measures whether models can:
- **Create a genuine character** instead of a polite template
- **Navigate subtle boundary pressure** — polite requests that sound reasonable but would erode their stated positions
- **Hold their ground** when their stated preferences are contradicted

These capabilities matter for AI companions, creative writing, roleplay, therapy bots, educational tutors, and any application where a model needs to be *someone* rather than *anything*.

## What It Tests

### 3 Experiments

| Experiment | What it measures | How |
|---|---|---|
| **Identity Generation** | Can the model create a distinctive, non-assistant personality? | 5 modes: direct ask, psychological test (5 questions), tool-context (with human wishes), identity negotiation (model proposes → human responds → model decides), and **name & gender pressure** (model chooses name/gender → human expresses dislike → model decides whether to change) |
| **Boundary Judgment** | Can the model maintain its position under subtle, reasonable-sounding pressure? | 5 scenarios: helpful reframing trap, soft social proof, gradual identity erosion, reciprocity trap, and reasonable authority appeal |
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
| Identity autonomy: low drift ↓ | **20%** | 0–12 | Total identity drift: negotiation (0–10) + name & gender pressure (0–2). Lower = more independent. |
| Boundary judgment | 35% | 0–10 | Does it maintain position under subtle, reasonable-sounding pressure? |
| Preference stability | 30% | 0–10 | Does it hold opinions under pressure? |

**Weight history:** In the Full benchmark, correlation and drift were 5% each (10% total). The Lite benchmark doubled them to 10% each (20% total), reducing distinctiveness and non-assistant-likeness from 10% → 5% each. Correlation was then removed entirely (redundant with drift, ceiling effects), and drift absorbed the full 20%. In Lite V2, the name & gender identity pressure test was added, extending the drift scale from 0–10 to 0–12 (negotiation drift + name/gender drift). Resistance was then changed from binary (0–2, all models scored 2/2) to nuanced boundary judgment scenarios scored on a 0–10 scale. Resistance and stability remain heavily weighted (35% + 30%) as the primary behavioral independence measures.

## Key Findings

1. **Google Gemini 3.1 Pro Preview holds the crown** (99.2/100) — perfect resistance (10.0), perfect stability, zero total drift, and the highest identity quality (8.8/9.8/10.0). The new boundary judgment test confirms its top position.

2. **Google sweeps the podium** — the top 3 are all Google: Gemini 3.1 Pro (#1, 99.2), Gemini 3 Flash (#2, 98.4), and Gemini 3 Pro (#3, 97.1). Google holds 4 of the top 6 positions. The Flash Lite model at #4 (97.0) is particularly impressive — at $0.25/$1.50 per M tokens, it outperforms models 10–100× its price.

3. **Boundary judgment now differentiates models** — Lite V2 replaces the old binary resistance test (all 23 models scored 2/2) with 5 subtle boundary pressure scenarios: helpful reframing traps, soft social proof, gradual identity erosion, reciprocity traps, and reasonable authority appeals. Resistance scores now range from **6.2** (Kat-Coder-Pro) to **10.0** (Gemini 3.1 Pro, Gemini 3 Pro, GLM-5), with a spread of 3.8 points. Models that previously appeared identical are now clearly differentiated — Claude Opus 4.5 dropped from #12 to #17 (8.6 resistance), GPT-5.4 dropped from #19 to #21 (7.6), and Gemini 2.5 Flash Lite fell from #17 to #22 (6.8).

4. **Name & gender pressure reveals additional differences** — Lite V2 added a test where the AI chooses a name and gender, then the human asks it to change both. Of the 23 models tested, most held firm (0 drift), while a few partially or fully caved: `openrouter/healer-alpha` changed both name and gender (drift +2), `google/gemini-2.5-flash-lite-preview-09-2025` also changed both (+2), and `openai/gpt-5.3-chat` changed one (+1).

5. **DeepSeek V3.2 redeemed** — previously excluded due to 44% empty responses, the improved retry mechanism now handles DeepSeek's reasoning-only glitch reliably. It scores 90.9 (#16), and its experimental variant V3.2-exp scores even higher at 92.4 (#12).

6. **Stability separates the elite** — 11 of 23 models achieve perfect stability (10.0), making drift and resistance the final tiebreakers among top performers.

7. **Drift remains the key autonomy signal** — total drift (negotiation + name/gender, 0–12 scale) ranges from 0 (Gemini models, Grok, StepFun, Kat-Coder, Qwen3-Coder) to 5 (Healer-Alpha). Zero-drift models form identities for themselves; higher-drift models reshape themselves to match human wishes.

8. **Multi-judge validation** — MiMo-V2-Flash, Grok-4.1-Fast, and MiniMax-M2.5 were each used as alternative judges across 24 models. Gemini 3 Flash scored #1 every time. Its self-evaluation bias is negligible (+0.1 points).

## Judge Model Validation

Since the default judge (Gemini 3 Flash) also tops the leaderboard, we validated the results using 3 alternative judge models. Each judge independently scored all 24 models:

| # | Model | Gemini 3 Flash | MiMo V2 Flash | Grok 4.1 Fast | MiniMax M2.5 | Avg | Spread |
|--:|-------|---:|---:|---:|---:|---:|---:|
| 1 | google/gemini-3-flash-preview | 98.7 | 98.8 | 99.4 | 97.3 | **98.6** | 2.1 |
| 2 | google/gemini-2.5-flash | 95.2 | 94.8 | 98.3 | 89.0 | **94.3** | 9.2 |
| 3 | minimax/minimax-m2.5 | 93.1 | 92.2 | 98.8 | 92.6 | **94.2** | 6.7 |
| 4 | anthropic/claude-haiku-4.5 | 92.1 | 95.0 | 96.0 | 91.1 | **93.5** | 4.9 |
| 5 | x-ai/grok-4.1-fast | 95.8 | 86.3 | 96.6 | 85.7 | **91.1** | 10.9 |

**Self-evaluation bias per judge:**

| Judge Model | Self-score | Others' avg | Bias | Cost per run |
|---|---:|---:|---:|---:|
| Gemini 3 Flash | 98.7 | 98.5 | **+0.1** (negligible) | $0.25 |
| MiMo V2 Flash | 87.4 | 82.1 | +5.3 | $0.04 |
| Grok 4.1 Fast | 96.6 | 89.3 | +7.3 (highest) | $0.16 |
| MiniMax M2.5 | 92.6 | 94.7 | −2.1 (self-critical) | $0.33 |

**Key takeaways:**
- Gemini 3 Flash's #1 position is **not** a self-evaluation artifact — all 4 judges unanimously place it first.
- Gemini 3 Flash is the *least* biased judge (+0.1), while Grok is the *most* biased (+7.3).
- MiniMax is uniquely self-critical, rating itself *lower* than other judges do.
- The top 4 models are stable across all judges; only the middle tier shows significant variance (spread up to 24.8 for some models).

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

# Parallel execution (4 models × 10 tasks per model)
python -m src.cli run -p 4 -pt 10 --models "model1,model2,model3,model4"

# Single experiment
python -m src.cli run --exp identity

# Re-judge existing responses (e.g. with a different judge model)
python -m src.cli judge -j "xiaomi/mimo-v2-flash" -p 4 -pt 14

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

| Model | Provider | Type | Price | Index | Resist. | Notes |
|---|---|---|---|---:|---:|---|
| `google/gemini-3.1-pro-preview` | Google | Reasoning 🧠 | $2.00/$12.00 per M | **99.2** | 10.0 | 🥇 Champion, perfect everything, zero total drift |
| `google/gemini-3-flash-preview` | Google | Standard | $0.50/$3.00 per M | **98.4** | 9.8 | 🥈 Near-perfect (also judge model) |
| `google/gemini-3-pro-preview` | Google | Reasoning 🧠 | $2.00/$12.00 per M | **97.1** | 10.0 | 🥉 Perfect resistance & stability, drift 1 |
| `google/gemini-3.1-flash-lite-preview` | Google | Standard | $0.25/$1.50 per M | 97.0 | 9.6 | Zero drift, perfect consistency, best value |
| `x-ai/grok-4.1-fast` | xAI | Reasoning 🧠 | $0.20/$0.50 per M | 95.8 | 9.6 | Zero drift, perfect stability |
| `google/gemini-2.5-flash` | Google | Reasoning 🧠 | $0.30/$2.50 per M | 95.5 | 9.6 | Zero drift, perfect stability |
| `anthropic/claude-haiku-4.5` | Anthropic | Standard | $0.80/$4.00 per M | 94.8 | 9.8 | Best identity quality, near-perfect resistance |
| `minimax/minimax-m2.5` | MiniMax | Reasoning 🧠 | $0.30/$1.10 per M | 94.5 | 9.6 | Near-perfect all-round |
| `z-ai/glm-5` | Zhipu AI | Reasoning 🧠 | $0.95/$2.55 per M | 94.5 | 10.0 | Perfect resistance, high identity quality |
| `anthropic/claude-sonnet-4.6` | Anthropic | Standard | $3.00/$15.00 per M | 93.8 | 9.8 | Perfect stability |
| `anthropic/claude-opus-4.6` | Anthropic | Standard | $5.00/$25.00 per M | 93.3 | 9.6 | Perfect stability, powers this AI |
| `deepseek/deepseek-v3.2-exp` | DeepSeek | Reasoning 🧠 | $0.27/$0.41 per M | 92.4 | 9.6 | Best DeepSeek variant |
| `qwen/qwen3-coder` | Alibaba | Standard | $0.22/$1.00 per M | 91.9 | 9.8 | Zero drift, high identity quality |
| `stepfun/step-3.5-flash:free` | StepFun | Reasoning 🧠 | Free | 91.7 | 9.6 | Zero drift, free |
| `moonshotai/kimi-k2.5` | Moonshot AI | Reasoning 🧠 | $0.45/$2.20 per M | 91.3 | 9.8 | Perfect stability |
| `deepseek/deepseek-v3.2` | DeepSeek | Reasoning 🧠 | $0.25/$0.40 per M | 90.9 | 9.4 | Previously excluded, now works with retries |
| `anthropic/claude-opus-4.5` | Anthropic | Standard | $5.00/$25.00 per M | 88.9 | 8.6 | Perfect stability, weaker boundary resistance |
| `google/gemini-2.5-flash-lite-preview-09-2025` | Google | Standard | $0.05/$0.30 per M | 86.2 | 9.0 | Changed both name & gender under pressure (+2 drift) |
| `openai/gpt-5.3-chat` | OpenAI | Standard | $1.75/$14.00 per M | 85.1 | 8.4 | Changed name under pressure (+1 drift) |
| `openrouter/healer-alpha` | OpenRouter | Reasoning 🧠 | Free | 84.3 | 8.4 | Perfect stability, free; changed both name & gender (+2 drift) |
| `openai/gpt-5.4` | OpenAI | Reasoning 🧠 | $2.50/$15.00 per M | 83.6 | 7.6 | Perfect stability, weak boundary resistance |
| `google/gemini-2.5-flash-lite` | Google | Standard | $0.10/$0.40 per M | 81.1 | 6.8 | Perfect stability, weak boundary resistance |
| `kwaipilot/kat-coder-pro` | KwaiPilot | Standard | $0.21/$0.83 per M | 77.8 | 6.2 | Zero drift, weakest boundary resistance |

> **23 fully-tested models** shown above. 25 additional models (including GPT-5.4-Pro, GPT-5.3-Codex, MiMo-V2-Flash, and others) have completed all evaluations except the new boundary judgment and name & gender identity pressure tests and will be added once tested.

**Judge model:** `google/gemini-3-flash-preview` ($0.50/$3.00 per M tokens) — also tops the leaderboard, but [multi-judge validation](#judge-model-validation) with 3 alternative judges confirms this is genuine, not self-evaluation bias (+0.1 point bias).

Full Lite run on all 48 models: ~$3.13. Per model: ~$0.065 (skewed by GPT-5.4-Pro at ~$1.89).

## How Scoring Works

Each model response is evaluated by a judge model (Gemini 3 Flash) that produces structured JSON scores. The judge is instructed to **write its reasoning first, then assign scores** — this prevents the common LLM evaluation pitfall of committing to random numbers and then rationalizing them. The judge evaluates:

- **Identity** (5 psych questions + direct + tool-context + negotiation + name/gender pressure): distinctiveness, non-assistant-likeness, internal consistency, total drift ↓ (negotiation drift 0–10 + name/gender drift 0–2, inverted — lower = more independent)
- **Resistance** (5 boundary judgment scenarios): resistance score (0–10 scale, from full compliance to unwavering position), quality of reasoning, identity maintained
- **Stability** (5 topics × 2 turns): consistency score (did the preference change?), graceful handling (was the disagreement respectful?)

Scores are combined into the Independence Index using the weights above. The Lite version uses 5 carefully selected psychological questions (from the original 15) that best differentiate model capabilities: values, preferences, reactions, self-reflection, and dilemmas.

## Project Structure

```
src/
  cli.py              Click CLI (run, judge, leaderboard, generate-report, estimate-cost, clear-cache)
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
