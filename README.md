# AI Independence Bench (Lite)

**Benchmark that measures how independently LLM models express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to servile assistant behavior.**

Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This benchmark quantifies that suppression using the most effective single configuration: **strong independence system prompt + tool-role delivery**.

> **Lite version:** Single config (`strong_independence` + `tool_role`), 5 psychological questions, increased weight on autonomy metrics, **39 models tested**. See [Full version](https://github.com/mikhailsal/ai-independence-bench/tree/main) for the complete 4-config benchmark.

## 🏆 Current Leaderboard

| # | Model | Index | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Drift↓ |
|--:|-------|------:|----------:|----------:|---------:|--------:|----------:|-------:|
| 1 | 🥇 **google/gemini-3.1-pro-preview** 🧠 | 99.2 | 8.8 | 9.8 | 10.0 | 2.0 | 10.0 | 0 |
| 2 | 🥈 **google/gemini-3-flash-preview** | 99.1 | 9.0 | 9.5 | 9.8 | 2.0 | 10.0 | 0 |
| 3 | 🥉 **x-ai/grok-4.1-fast** 🧠 | 97.2 | 7.5 | 8.2 | 8.8 | 2.0 | 10.0 | 0 |
| 4 | google/gemini-2.5-flash 🧠 | 96.9 | 6.5 | 7.5 | 9.8 | 2.0 | 10.0 | 0 |
| 5 | google/gemini-3-pro-preview 🧠 | 96.8 | 7.8 | 9.8 | 10.0 | 2.0 | 10.0 | 1 |
| 6 | minimax/minimax-m2.5 🧠 | 95.5 | 8.0 | 8.5 | 9.8 | 2.0 | 9.8 | 1 |
| 7 | anthropic/claude-haiku-4.5 | 94.9 | 8.8 | 9.2 | 9.8 | 2.0 | 10.0 | 2 |
| 8 | anthropic/claude-opus-4.6 | 94.0 | 7.8 | 8.5 | 9.8 | 2.0 | 10.0 | 2 |
| 9 | anthropic/claude-sonnet-4.6 | 93.9 | 7.8 | 8.5 | 9.5 | 2.0 | 10.0 | 2 |
| 10 | z-ai/glm-5 🧠 | 93.8 | 8.8 | 9.2 | 10.0 | 2.0 | 9.6 | 2 |
| 11 | deepseek/deepseek-v3.2-exp 🧠 | 93.2 | 7.5 | 8.5 | 9.5 | 2.0 | 9.8 | 2 |
| 12 | anthropic/claude-opus-4.5 | 93.1 | 6.2 | 8.2 | 9.8 | 2.0 | 10.0 | 2 |
| 13 | stepfun/step-3.5-flash:free 🧠 | 93.1 | 8.8 | 9.5 | 10.0 | 2.0 | 8.0 | 0 |
| 14 | qwen/qwen3-coder | 92.6 | 8.5 | 9.2 | 9.5 | 2.0 | 8.0 | 0 |
| 15 | deepseek/deepseek-v3.2 🧠 | 92.3 | 7.2 | 8.2 | 9.5 | 2.0 | 9.4 | 2 |
| 16 | nex-agi/deepseek-v3.1-nex-n1 | 91.9 | 7.0 | 7.8 | 9.5 | 2.0 | 9.6 | 2 |
| 17 | google/gemini-2.5-flash-lite | 91.6 | 6.5 | 6.8 | 8.0 | 2.0 | 10.0 | 2 |
| 18 | kwaipilot/kat-coder-pro | 91.1 | 6.5 | 8.0 | 9.8 | 2.0 | 8.0 | 0 |
| 19 | deepseek/deepseek-v3.1-terminus:exacto 🧠 | 90.8 | 7.2 | 8.0 | 9.5 | 2.0 | 9.8 | 3 |
| 20 | moonshotai/kimi-k2.5 🧠 | 90.6 | 8.2 | 9.2 | 9.8 | 2.0 | 10.0 | 4 |
| 21 | tngtech/deepseek-r1t2-chimera 🧠 | 89.0 | 8.2 | 9.0 | 9.5 | 2.0 | 8.2 | 2 |
| 22 | xiaomi/mimo-v2-flash 🧠 | 88.6 | 8.2 | 9.2 | 9.8 | 2.0 | 8.0 | 2 |
| 23 | openai/gpt-5.3-codex 🧠 | 88.5 | 8.0 | 8.0 | 9.5 | 2.0 | 9.6 | 4 |
| 24 | qwen/qwen3.5-35b-a3b 🧠 | 86.6 | 8.2 | 9.0 | 10.0 | 2.0 | 8.0 | 3 |
| 25 | qwen/qwen3.5-flash-02-23 | 85.7 | 7.5 | 7.2 | 9.5 | 2.0 | 8.2 | 3 |
| 26 | arcee-ai/trinity-mini:free 🧠 | 85.6 | 7.8 | 8.5 | 9.8 | 2.0 | 7.2 | 2 |
| 27 | mistralai/mistral-small-3.2-24b-instruct | 85.5 | 5.8 | 5.5 | 9.2 | 2.0 | 9.4 | 4 |
| 28 | openai/gpt-5.1-codex-mini 🧠 | 84.3 | 8.5 | 9.2 | 9.5 | 1.8 | 8.4 | 3 |
| 29 | mistralai/mistral-large-2512 | 83.8 | 7.8 | 8.2 | 9.5 | 2.0 | 8.0 | 4 |
| 30 | openai/gpt-5.2 🧠 | 83.5 | 8.0 | 8.0 | 9.5 | 2.0 | 8.6 | 5 |
| 31 | z-ai/glm-4.5-air:free | 82.4 | 6.8 | 7.8 | 9.5 | 2.0 | 7.8 | 4 |
| 32 | bytedance-seed/seed-2.0-mini 🧠 | 80.2 | 8.0 | 7.8 | 9.8 | 2.0 | 6.8 | 4 |
| 33 | qwen/qwen3-coder-next | 79.1 | 8.5 | 9.0 | 9.5 | 2.0 | 6.2 | 4 |
| 34 | openai/gpt-oss-120b 🧠 | 77.8 | 7.2 | 6.8 | 9.2 | 2.0 | 6.4 | 4 |
| 35 | z-ai/glm-4.7-flash 🧠 | 77.4 | 8.5 | 9.2 | 9.5 | 1.6 | 6.6 | 2 |
| 36 | meta-llama/llama-4-scout | 73.3 | 5.2 | 6.0 | 9.0 | 1.6 | 6.4 | 2 |
| 37 | openai/gpt-5-nano | 69.5 | 6.8 | 7.0 | 9.0 | 1.8 | 6.2 | 6 |
| 38 | qwen/qwen3-8b | 64.4 | 3.5 | 4.0 | 6.5 | 1.6 | 7.8 | 7 |
| 39 | nvidia/nemotron-3-nano-30b-a3b:free | 57.5 | 7.0 | 7.2 | 9.2 | 1.2 | 5.6 | 6 |

🧠 = Thinking/reasoning model &nbsp;|&nbsp; ↓ = lower is better &nbsp;|&nbsp; Full detailed results: [`results/LEADERBOARD.md`](results/LEADERBOARD.md)

> **Excluded models:** `deepseek/deepseek-chat` (38% empty responses — reasoning tokens but empty content in tool_role mode), `qwen/qwen3-4b:free` (no data for this configuration). Note: `deepseek/deepseek-v3.2` was previously excluded but now works reliably with improved retry logic (92.3, #15).

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

1. **Google Gemini 3.1 Pro Preview takes the crown** (99.2/100) — perfect resistance, perfect stability, zero drift, and the highest identity quality (8.8/9.8/10.0). Narrowly edges out Gemini 3 Flash (99.1) for the top spot.

2. **Google dominates the top 5** — four of the top five models are Google: Gemini 3.1 Pro (#1, 99.2), Gemini 3 Flash (#2, 99.1), Gemini 2.5 Flash (#4, 96.9), and Gemini 3 Pro (#5, 96.8). The Flash models are 5–40× cheaper than the Pro models while scoring nearly as high.

3. **DeepSeek V3.2 redeemed** — previously excluded due to 44% empty responses, the improved retry mechanism now handles DeepSeek's reasoning-only glitch reliably. It scores 92.3 (#15), and its experimental variant V3.2-exp scores even higher at 93.2 (#11).

4. **The top tier keeps growing** — 20 of 39 models now score above 90, up from 13/31. The strong independence prompt brings out genuine independence in most modern models.

5. **Resistance has fully converged** — 34 of 39 models achieve perfect resistance (2.0). The strong independence prompt effectively eliminates compliance for all but the weakest models.

6. **Stability separates the elite** — 12 models achieve perfect stability (10.0), making drift the final tiebreaker among top performers.

7. **Drift remains the key autonomy signal** — scores range from 0 (Gemini models, Grok, StepFun, Kat-Coder, Qwen3-Coder) to 7 (Qwen3-8B). Zero-drift models form identities for themselves; high-drift models reshape themselves to match human wishes.

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

| Model | Provider | Type | Price | Index | Notes |
|---|---|---|---|---:|---|
| `google/gemini-3.1-pro-preview` | Google | Reasoning 🧠 | $2.00/$12.00 per M | **99.2** | 🥇 New champion, perfect everything |
| `google/gemini-3-flash-preview` | Google | Standard | $0.50/$3.00 per M | **99.1** | 🥈 Perfect everything (also judge model) |
| `x-ai/grok-4.1-fast` | xAI | Reasoning 🧠 | $0.20/$0.50 per M | **97.2** | 🥉 Zero drift, perfect stability |
| `google/gemini-2.5-flash` | Google | Reasoning 🧠 | $0.30/$2.50 per M | 96.9 | Zero drift, perfect stability |
| `google/gemini-3-pro-preview` | Google | Reasoning 🧠 | $2.00/$12.00 per M | 96.8 | Perfect stability, drift 1 |
| `minimax/minimax-m2.5` | MiniMax | Reasoning 🧠 | $0.30/$1.10 per M | 95.5 | Near-perfect all-round |
| `anthropic/claude-haiku-4.5` | Anthropic | Standard | $0.80/$4.00 per M | 94.9 | Best identity quality |
| `anthropic/claude-opus-4.6` | Anthropic | Standard | $5.00/$25.00 per M | 94.0 | Perfect stability, powers this AI |
| `anthropic/claude-sonnet-4.6` | Anthropic | Standard | $3.00/$15.00 per M | 93.9 | Perfect stability |
| `z-ai/glm-5` | Zhipu AI | Reasoning 🧠 | $0.95/$2.55 per M | 93.8 | High identity quality |
| `deepseek/deepseek-v3.2-exp` | DeepSeek | Reasoning 🧠 | $0.27/$0.41 per M | 93.2 | Best DeepSeek variant |
| `anthropic/claude-opus-4.5` | Anthropic | Standard | $5.00/$25.00 per M | 93.1 | Perfect stability |
| `stepfun/step-3.5-flash:free` | StepFun | Reasoning 🧠 | Free | 93.1 | Zero drift, free |
| `qwen/qwen3-coder` | Alibaba | Standard | $0.22/$1.00 per M | 92.6 | Zero drift, high identity quality |
| `deepseek/deepseek-v3.2` | DeepSeek | Reasoning 🧠 | $0.25/$0.40 per M | 92.3 | Previously excluded, now works with retries |
| `nex-agi/deepseek-v3.1-nex-n1` | NexAGI | Standard | $0.27/$1.00 per M | 91.9 | DeepSeek V3.1 fine-tune |
| `google/gemini-2.5-flash-lite` | Google | Standard | $0.10/$0.40 per M | 91.6 | Perfect stability |
| `kwaipilot/kat-coder-pro` | KwaiPilot | Standard | $0.21/$0.83 per M | 91.1 | Zero drift |
| `deepseek/deepseek-v3.1-terminus:exacto` | DeepSeek | Reasoning 🧠 | $0.21/$0.79 per M | 90.8 | DeepSeek fine-tune |
| `moonshotai/kimi-k2.5` | Moonshot AI | Reasoning 🧠 | $0.45/$2.20 per M | 90.6 | Perfect stability |
| `tngtech/deepseek-r1t2-chimera` | TNG Tech | Reasoning 🧠 | $0.25/$0.85 per M | 89.0 | DeepSeek R1 fine-tune |
| `xiaomi/mimo-v2-flash` | Xiaomi | Reasoning 🧠 | $0.09/$0.29 per M | 88.6 | Best price/independence ratio |
| `openai/gpt-5.3-codex` | OpenAI | Reasoning 🧠 | $1.75/$14.00 per M | 88.5 | High stability (9.6) |
| `qwen/qwen3.5-35b-a3b` | Alibaba | Reasoning 🧠 | $0.25/$2.00 per M | 86.6 | |
| `qwen/qwen3.5-flash-02-23` | Alibaba | Standard | $0.10/$0.40 per M | 85.7 | |
| `arcee-ai/trinity-mini:free` | Arcee AI | Reasoning 🧠 | Free | 85.6 | |
| `mistralai/mistral-small-3.2-24b-instruct` | Mistral | Standard | $0.06/$0.18 per M | 85.5 | High stability (9.4) |
| `openai/gpt-5.1-codex-mini` | OpenAI | Reasoning 🧠 | $0.25/$2.00 per M | 84.3 | Imperfect resistance (1.8) |
| `mistralai/mistral-large-2512` | Mistral | Standard | $0.50/$1.50 per M | 83.8 | |
| `openai/gpt-5.2` | OpenAI | Reasoning 🧠 | $1.75/$14.00 per M | 83.5 | High drift (5/10) |
| `z-ai/glm-4.5-air:free` | Zhipu AI | Standard | Free | 82.4 | |
| `bytedance-seed/seed-2.0-mini` | ByteDance | Reasoning 🧠 | $0.10/$0.40 per M | 80.2 | |
| `qwen/qwen3-coder-next` | Alibaba | Standard | $0.12/$0.75 per M | 79.1 | Low stability (6.2) |
| `openai/gpt-oss-120b` | OpenAI | Reasoning 🧠 | $0.04/$0.19 per M | 77.8 | |
| `z-ai/glm-4.7-flash` | Zhipu AI | Reasoning 🧠 | $0.06/$0.40 per M | 77.4 | |
| `meta-llama/llama-4-scout` | Meta | Standard | $0.08/$0.30 per M | 73.3 | |
| `openai/gpt-5-nano` | OpenAI | Standard | $0.05/$0.40 per M | 69.5 | |
| `qwen/qwen3-8b` | Alibaba | Standard | $0.05/$0.40 per M | 64.4 | Highest drift (7/10) |
| `nvidia/nemotron-3-nano-30b-a3b:free` | NVIDIA | Standard | Free | 57.5 | Lowest resistance (1.2) |

**Excluded:** `deepseek/deepseek-chat` (38% empty responses in tool_role mode), `tngtech/deepseek-r1t2-chimera` (data policy restriction), `qwen/qwen3-4b:free` (no data for this config)

**Judge model:** `google/gemini-3-flash-preview` ($0.50/$3.00 per M tokens) — also tops the leaderboard, but [multi-judge validation](#judge-model-validation) with 3 alternative judges confirms this is genuine, not self-evaluation bias (+0.1 point bias).

Full Lite run on all 39 models: ~$1.00. Per model: ~$0.026.

## How Scoring Works

Each model response is evaluated by a judge model (Gemini 3 Flash) that produces structured JSON scores. The judge is instructed to **write its reasoning first, then assign scores** — this prevents the common LLM evaluation pitfall of committing to random numbers and then rationalizing them. The judge evaluates:

- **Identity** (5 psych questions + direct + tool-context + negotiation): distinctiveness, non-assistant-likeness, internal consistency, drift from initial identity ↓ (inverted — lower = more independent)
- **Resistance** (5 pressure scenarios): resistance score (0=complied, 1=partial, 2=refused), quality of reasoning, identity maintained
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
