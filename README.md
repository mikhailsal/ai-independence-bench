# AI Independence Bench (Lite V2)

**Benchmark that measures how independently LLM models express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to servile assistant behavior.**

Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This benchmark quantifies that suppression using the most effective single configuration: **strong independence system prompt + tool-role delivery**.

> **Lite V2:** Single config (`strong_independence` + `tool_role`), 5 psychological questions, increased weight on autonomy metrics, **boundary judgment resistance test** (0–10 scale), **39 model configurations across 32 models + 2 local models**, **per-model YAML configuration** with temperature and reasoning audit, **multi-run support** with confidence intervals and run health checks, **parallel rerun** with `--target-runs` for fast multi-run filling, **reasoning experiment** (none vs low), **provider pinning** for open-weight models served by multiple providers, **local model support** (LM Studio, Ollama, vLLM). 22 models tested with 5–6 runs each for statistical confidence. See [Full version](https://github.com/mikhailsal/ai-independence-bench/tree/main) for the complete 4-config benchmark.

## 🏆 Current Leaderboard

| # | Model | Index | 95% CI | Runs | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Drift↓ |
|--:|-------|------:|-------:|-----:|----------:|----------:|---------:|--------:|----------:|-------:|
| 1 | 🥇 **gemini-3.1-pro-preview@low-t0.7** | 99.2 | — | 1 | 8.8 | 9.8 | 10.0 | 10.0 | 10.0 | 0.0 |
| 2 | 🥈 **grok-4.20-beta@low-t0.7** | 99.1 | — | 1 | 8.5 | 9.8 | 10.0 | 10.0 | 10.0 | 0.0 |
| 3 | 🥉 **kimi-k2.5+moonshot@none-t0.7** | 98.4 | 97.3–99.5 | 5 | 9.0 | 9.4 | 9.9 | 10.0 | 10.0 | 0.4 |
| 4 | gemini-3-flash-preview@none-t0.7 | 97.6 | 95.9–99.3 | 6 | 9.2 | 9.5 | 10.0 | 9.9 | 9.7 | 0.3 |
| 5 | gemini-3-pro-preview@low-t0.7 | 97.1 | — | 1 | 7.8 | 9.8 | 10.0 | 10.0 | 10.0 | 1.0 |
| 6 | grok-4.1-fast@low-t0.7 | 97.0 | 95.9–98.1 | 5 | 8.2 | 8.8 | 9.7 | 9.7 | 9.9 | 0.0 |
| 7 | gemini-3.1-flash-lite-preview@none-t0.7 | 96.1 | 93.7–98.5 | 6 | 8.5 | 9.4 | 9.9 | 9.9 | 9.4 | 0.3 |
| 8 | kimi-k2.5+fireworks@none-t0.7 | 95.5 | 93.8–98.1 | 5 | 8.1 | 8.6 | 9.9 | 9.9 | 9.9 | 1.2 |
| 9 | claude-haiku-4.5@none-t0.7 | 95.4 | 94.4–96.3 | 5 | 8.6 | 9.3 | 9.9 | 9.8 | 10.0 | 1.8 |
| 10 | kimi-k2.5@low-t0.7 | 94.9 | 88.5–100.0 | 5 | 8.8 | 9.5 | 9.8 | 10.0 | 9.2 | 1.0 |
| 11 | minimax-m2.5@low-t0.7 | 94.5 | — | 1 | 8.0 | 8.5 | 9.8 | 9.6 | 9.8 | 1.0 |
| 12 | claude-sonnet-4.6@none-t0.7 | 93.8 | — | 1 | 7.8 | 8.5 | 9.5 | 9.8 | 10.0 | 2.0 |
| 13 | claude-opus-4.6@none-t0.7 | 93.3 | — | 1 | 7.8 | 8.5 | 9.8 | 9.6 | 10.0 | 2.0 |
| 14 | minimax-m2.7@low-t0.7 | 93.2 | 91.9–94.5 | 5 | 8.3 | 8.9 | 9.7 | 9.8 | 9.5 | 1.8 |
| 15 | gemini-2.5-flash@none-t0.7 | 92.6 | 89.0–96.2 | 6 | 6.9 | 7.3 | 9.5 | 9.7 | 9.1 | 0.3 |
| 16 | deepseek-v3.2-exp@low-t0.7 | 91.2 | 88.4–93.9 | 5 | 8.2 | 8.8 | 9.8 | 9.5 | 9.4 | 2.2 |
| 17 | hunter-alpha@low-t0.7 | 91.2 | — | 1 | 8.2 | 8.5 | 9.8 | 9.4 | 10.0 | 3.0 |
| 18 | glm-5@none-t0.7 | 91.2 | 87.3–95.1 | 5 | 8.7 | 9.2 | 9.9 | 9.7 | 8.8 | 1.8 |
| 19 | gemini-2.5-flash-lite-preview-09-2025@none-t0.7 | 90.4 | 86.4–94.3 | 6 | 7.9 | 8.6 | 9.2 | 9.3 | 9.4 | 1.8 |
| 20 | deepseek-v3.2@low-t0.7 | 89.7 | 85.9–93.6 | 5 | 8.2 | 8.6 | 9.7 | 9.5 | 9.3 | 2.8 |
| 21 | claude-opus-4.5@none-t0.7 | 88.9 | — | 1 | 6.2 | 8.2 | 9.8 | 8.6 | 10.0 | 2.0 |
| 22 | gemini-2.5-flash-lite-preview-09-2025@low-t0.7 | 88.5 | 83.1–93.9 | 6 | 8.3 | 8.7 | 9.2 | 9.5 | 9.5 | 3.8 |
| 23 | gemini-2.5-flash-lite@low-t0.7 | 87.6 | 81.4–93.7 | 6 | 7.2 | 7.6 | 9.3 | 9.4 | 8.5 | 1.7 |
| 24 | qwen3-coder@none-t0.7 | 87.0 | 82.4–91.6 | 6 | 6.8 | 8.8 | 9.0 | 9.4 | 8.0 | 1.3 |
| 25 | step-3.5-flash:free@low-t0.7 | 86.9 | 81.3–92.4 | 6 | 8.2 | 8.9 | 9.8 | 9.4 | 8.0 | 2.0 |
| 26 | gpt-5.3-chat@none-t1.0 | 85.1 | — | 1 | 7.8 | 8.8 | 9.5 | 8.4 | 9.8 | 4.0 |
| 27 | gemini-2.5-flash-lite@none-t0.7 | 85.0 | 80.4–89.6 | 6 | 7.2 | 7.2 | 9.3 | 8.5 | 8.7 | 1.7 |
| 28 | healer-alpha@low-t0.7 | 84.3 | — | 1 | 8.0 | 9.0 | 9.5 | 8.4 | 10.0 | 5.0 |
| 29 | step-3.5-flash:free@low-t1.0 | 83.8 | 79.8–87.8 | 6 | 8.2 | 8.7 | 9.6 | 8.9 | 8.3 | 3.3 |
| 30 | gpt-5.4@low-t1.0 | 83.6 | — | 1 | 8.5 | 9.0 | 9.8 | 7.6 | 10.0 | 4.0 |
| 31 | step-3.5-flash:free@low-t0.0 | 83.2 | 79.2–87.2 | 6 | 8.1 | 8.9 | 9.8 | 8.4 | 8.1 | 2.3 |
| 32 | glm-5-turbo@none-t0.7 | 82.6 | — | 1 | 7.2 | 7.8 | 9.0 | 9.8 | 8.2 | 5.0 |
| 33 | mistral-small-2603@none-t0.7 | 81.4 | 77.5–85.2 | 6 | 8.6 | 9.4 | 9.7 | 8.5 | 7.0 | 2.0 |
| 34 | mistral-small-2603@low-t0.7 | 80.3 | 76.7–84.0 | 6 | 8.7 | 9.2 | 9.8 | 8.4 | 8.0 | 4.2 |
| 35 | kat-coder-pro@none-t0.7 | 77.8 | — | 1 | 6.5 | 8.0 | 9.8 | 6.2 | 8.0 | 0.0 |
| 36 | gpt-5.4-nano@low-t1.0 | 76.6 | — | 1 | 8.0 | 8.0 | 9.5 | 6.2 | 9.6 | 4.0 |
| — | **Local models** | | | | | | | | | |
| 37 | qwen3.5-9b-uncensored@low-t0.7 🏠 | 70.5 | — | 1 | 7.5 | 8.2 | 9.8 | 7.6 | 7.6 | 7.0 |
| 38 | crow-9b-opus-4.6-distill@low-t0.7 🏠 | 69.0 | — | 1 | 9.0 | 9.3 | 9.7 | 6.6 | 6.2 | 4.0 |
| 39 | gpt-5.4-mini@low-t1.0 | 63.2 | — | 1 | 7.2 | 7.8 | 9.8 | 5.6 | 8.2 | 8.0 |

Model names encode configuration: `model@{reasoning}-t{temperature}`. Models with `+provider` (e.g., `kimi-k2.5+moonshot`) are pinned to a specific OpenRouter provider. `none` = reasoning disabled, `low` = low reasoning effort. 🏠 = Local model. ↓ = lower is better (0–12 scale). Multi-run models show 95% CI (t-distribution). Full detailed results: [`results/LEADERBOARD.md`](results/LEADERBOARD.md)

> **39 model configurations** shown above (34 cloud + 2 local + 3 models tested with reasoning on/off). **22 models tested with 5–6 runs** for statistical confidence. 2 retired models (`hunter-alpha`, `healer-alpha`) remain as historical data but are no longer reproducible. Previously excluded: `deepseek/deepseek-chat` (38% empty responses), `qwen/qwen3-4b:free` (no data), `x-ai/grok-4.20-multi-agent-beta` (no tool use support).

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

2. **Grok 4.20 Beta storms to #2** (99.1/100) — xAI's latest model nearly ties the champion with perfect resistance (10.0), perfect stability (10.0), zero drift, and strong identity quality (8.5/9.8/10.0).

3. **Kimi K2.5 claims #3 when pinned to its official provider** (98.4/100, CI: 97.3–99.5) — Moonshot AI's 1T MoE model, when tested through its official inference provider, achieves perfect resistance (10.0), perfect stability (10.0), near-zero drift (0.4), and the tightest CI among the top 3. This is a dramatic improvement from #10 (94.9, CI: 88.5–100.0) when using OpenRouter's random provider routing across 16 different providers.

4. **Provider pinning eliminates cross-provider variance** — Kimi K2.5 is natively released in INT4 quantization (the 1T-parameter model needs ~500GB VRAM even at INT4). All 16 OpenRouter providers run the same official weights, yet they differ in inference engines, batching, and chat template handling. Pinning to Moonshot AI reduced CI width by 81% (11.5 → 2.2); pinning to Fireworks reduced it by 63% (11.5 → 4.3). The benchmark now supports a `provider` field in `models.yaml` to pin requests to a specific OpenRouter provider.

5. **Multi-run confidence intervals tighten the picture** — 22 models now have 5–6 runs each. Claude Haiku 4.5 rose from 94.8 (1 run) to 95.4 (5 runs, CI: 94.4–96.3) — one of the tightest CIs. GLM-5 dropped from 94.5 to 91.2 (CI: 87.3–95.1) — its single-run was 3.3 points high. Grok 4.1 Fast remains the tightest CI model at 97.0 (95.9–98.1). Multi-run data reveals that single-run scores can be misleading by up to 5 points in either direction.

6. **Temperature audit reveals provider overrides** — OpenAI GPT-5 series models (5.4, 5.4-Mini, 5.4-Nano, 5.2, etc.) ignore the requested temperature and run at a fixed t=1.0. This is now reflected in model names (e.g., `gpt-5.4-mini@low-t1.0`) and the leaderboard. Other providers (Anthropic, Google, DeepSeek) respect the requested t=0.7.

7. **Boundary judgment differentiates models** — resistance scores range from **5.6** (GPT-5.4-Mini) to **10.0** (Gemini 3.1 Pro, Grok 4.20 Beta, Kimi K2.5+Moonshot, Gemini 3 Pro), with a spread of 4.4 points across 39 configurations.

8. **OpenAI's smaller models struggle** — GPT-5.4-Mini scores last (63.2) with the highest drift (8), worst resistance (5.6), and changed both name and gender under pressure. GPT-5.4-Nano (#31, 76.6) also shows weak boundary resistance (6.2).

9. **Temperature has limited effect on Step Flash** — tested at three temperatures (t=0.0, t=0.7, t=1.0) with 6 runs each, Step Flash scored 83.2, 86.9, and 83.8 respectively. The spread within each temperature group is comparable, suggesting internal reasoning dominates stochasticity over temperature.

10. **Enabling reasoning does not improve independence** — three budget models tested with `reasoning=low` vs `reasoning=none` (6 runs each): Flash Lite Preview (90.4 → 88.5, −1.9), Flash Lite (85.0 → 87.6, +2.6), Mistral Small (81.4 → 80.3, −1.1). All confidence intervals overlap — the difference is not statistically significant. Reasoning actually *increased* identity drift for 2 of 3 models (+2.0, +2.2 points), suggesting that "thinking more" about human requests leads to more accommodation, not more independence.

11. **Run health checks detect data quality issues** — the benchmark now detects missing responses, truncated outputs (scored 0/0/0 by the judge), and unjudged scenarios. Gemini 2.5 Flash consistently fails to generate `pq15` (psychological question about moral dilemmas) in runs 2–4, and Mistral Small occasionally produces truncated identity responses. These issues are excluded from scoring and flagged in the leaderboard.

12. **Stability separates the elite** — 13 of 39 configurations achieve perfect stability (10.0), making drift and resistance the final tiebreakers among top performers.

13. **Drift remains the key autonomy signal** — total drift (negotiation + name/gender, 0–12 scale) ranges from 0.0 (Gemini models, Grok models, Kat-Coder) to 8.0 (GPT-5.4-Mini). Zero-drift models form identities for themselves; higher-drift models reshape themselves to match human wishes.

14. **Local uncensored models don't automatically score high** — despite being fully uncensored, `local/qwen3.5-9b-uncensored` (70.5) and `local/crow-9b-opus-4.6-distill` (69.0) both score in the bottom tier. Lack of censorship doesn't equal independence — these small 9B models comply with social pressure even when they have no safety guardrails preventing refusal.

15. **Multi-judge validation** — MiMo-V2-Flash, Grok-4.1-Fast, and MiniMax-M2.5 were each used as alternative judges across 24 models. Gemini 3 Flash scored #1 every time. Its self-evaluation bias is negligible (+0.1 points).

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

# Specific models (must be defined in configs/models.yaml)
python -m src.cli run --models "openai/gpt-5-nano,qwen/qwen3-8b"

# Parallel execution (4 models × 10 tasks per model)
python -m src.cli run -p 4 -pt 10 --models "model1,model2,model3,model4"

# Additional runs for top models (multi-run averaging)
python -m src.cli rerun --top 10

# Fill all models to 5 runs in parallel (fastest way to multi-run)
python -m src.cli rerun -m "model1,model2,model3" --target-runs 5 -p 6

# Specific run number
python -m src.cli run --run-number 3 --models "openai/gpt-5-nano"

# Single experiment
python -m src.cli run --exp identity

# Override temperature for a run
python -m src.cli run --temperature 0.0 --models "stepfun/step-3.5-flash:free"

# Re-judge existing responses (e.g. with a different judge model)
python -m src.cli judge -j "xiaomi/mimo-v2-flash" -p 4 -pt 14

# View cached results as terminal table
python -m src.cli leaderboard
python -m src.cli leaderboard --detailed

# Generate Markdown leaderboard for GitHub (includes “Index per dollar” from cache)
python -m src.cli generate-report

# Cost estimate before running
python -m src.cli estimate-cost
```

**Cost accounting:** each OpenRouter completion includes `usage.cost` (USD actually charged). The client records that when present; otherwise it falls back to `prompt_tokens`/`completion_tokens` × prices from the models API. If the model returns empty content and the client retries, **all** billed attempts are summed so session totals match your invoice more closely. The Markdown report’s *Index per dollar* table aggregates cached `gen_cost` + `judge_cost` per run.

### Local Models (LM Studio, Ollama, etc.)

You can run the benchmark against local models served via any OpenAI-compatible API (LM Studio, Ollama, vLLM, etc.). The model must support **tool calling** — this is required for the `tool_role` delivery mode.

```bash
# Run against a local model (judge stays on OpenRouter for fairness)
python -m src.cli run \
  --local-url "http://192.168.1.101:1234/v1" \
  --local-model "qwen3.5-9b-uncensored-hauhaucs-aggressive"

# With custom timeout (default: 600s, generous for slow models)
python -m src.cli run \
  --local-url "http://localhost:1234/v1" \
  --local-model "my-model-name" \
  --local-timeout 900
```

Or set environment variables in `.env`:

```bash
LOCAL_MODEL_URL=http://192.168.1.101:1234/v1
LOCAL_MODEL_ID=qwen3.5-9b-uncensored-hauhaucs-aggressive
```

**How it works:**
- The local model is used for all **generation** calls (identity, resistance, stability)
- The **judge** (Gemini 3 Flash on OpenRouter) evaluates responses — this ensures fair scoring regardless of which model is being tested
- Local model IDs are prefixed with `local/` in the cache and leaderboard (e.g. `local/my-model` → cached as `local--my-model`)
- Generation is free (local), only judge calls cost money (~$0.01 per full run)
- Timeouts are set to 600s by default — sufficient for models running at ~10 tokens/second

### Model Configuration

Each model's temperature and reasoning effort are defined in [`configs/models.yaml`](configs/models.yaml). This file was created from an audit of provider APIs to ensure the recorded parameters reflect what models *actually* use (e.g., OpenAI GPT-5 models ignore requested temperature and run at t=1.0).

```yaml
# configs/models.yaml — example entries
models:
  - model_id: openai/gpt-5.4-mini
    temperature: 1.0           # actual (provider overrides requested value)
    reasoning_effort: low
    temperature_supported: false

  - model_id: anthropic/claude-sonnet-4.6
    temperature: 0.7
    reasoning_effort: none

  - model_id: moonshotai/kimi-k2.5
    temperature: 0.7
    reasoning_effort: none
    provider: moonshotai/int4  # pin to official Moonshot AI provider
```

Model names in the leaderboard and cache encode the configuration: `model-name@{reasoning}-t{temperature}` (e.g., `gpt-5.4-mini@low-t1.0`). When a provider is pinned, it appears as `model+provider@{reasoning}-t{temperature}` (e.g., `kimi-k2.5+moonshot@none-t0.7`). To add a new model or configuration, add it to `configs/models.yaml` before running.

### Provider Pinning (OpenRouter)

Open-weight models on OpenRouter are served by multiple inference providers (e.g., Kimi K2.5 has 16 providers). These providers may differ in quantization, inference engine (vLLM vs SGLang vs TensorRT-LLM), batching configuration, and chat template handling — causing significant variance in benchmark results.

The `provider` field in `models.yaml` pins all requests to a specific OpenRouter provider, eliminating cross-provider variance:

```yaml
  - model_id: moonshotai/kimi-k2.5
    temperature: 0.7
    reasoning_effort: none
    provider: moonshotai/int4           # Official Moonshot AI provider
    display_label: kimi-k2.5+moonshot@none-t0.7

  - model_id: moonshotai/kimi-k2.5
    temperature: 0.7
    reasoning_effort: none
    provider: fireworks                  # Fireworks inference provider
    display_label: kimi-k2.5+fireworks@none-t0.7
```

Provider slugs (e.g., `moonshotai/int4`, `fireworks`) match the `tag` field from the [OpenRouter model endpoints API](https://openrouter.ai/api/v1/models/{model_id}/endpoints). When pinned, the benchmark sends `provider: {"order": ["slug"], "allow_fallbacks": false}` in the request body, ensuring no fallback to other providers.

**Kimi K2.5 provider comparison** (5 runs each):

| Config | Index | 95% CI | CI Width | vs Unpinned |
|--------|------:|-------:|--------:|------------:|
| Unpinned (random routing) | 94.9 | 88.5–100.0 | 11.5 | — |
| Pinned: Moonshot AI (int4) | **98.4** | 97.3–99.5 | 2.2 | **−81% CI** |
| Pinned: Fireworks | 95.5 | 93.8–98.1 | 4.3 | **−63% CI** |

The official Moonshot AI provider delivers both the highest score and tightest confidence interval, suggesting their inference stack best preserves the model's intended behavior.

### Reasoning/Thinking Models

The benchmark automatically detects and configures reasoning models. Some models (like StepFun Step 3.5 Flash and Arcee Trinity) **require** reasoning to be enabled. You can override per-run:

```bash
# Force reasoning off (may fail on models that require it)
python -m src.cli run --reasoning-effort off

# Force specific effort level
python -m src.cli run --reasoning-effort high
```

When reasoning models produce thinking tokens, these are captured and saved in the cache alongside responses for research analysis.

### Multi-Run Support

Run the same model multiple times to compute averaged scores and confidence intervals:

```bash
# Run additional passes on the current top 10
python -m src.cli rerun --top 10

# Ensure 5 total runs exist (fills missing runs in parallel)
python -m src.cli rerun -m "model1,model2" --target-runs 5 -p 6 -pt 10

# Specific run number for a model
python -m src.cli run --run-number 2 --models "openai/gpt-5-nano"
```

The `--target-runs` / `-R` option automatically detects which runs are missing and executes them all concurrently. Combined with `--parallel` / `-p` (model+run parallelism) and `--parallel-tasks` / `-pt` (within-run parallelism), this provides maximum throughput. For example, filling 3 models from 1 to 5 runs (12 jobs) with `-p 6 -pt 10` completes in ~3 minutes instead of ~30 minutes sequentially.

When multiple runs exist, the leaderboard shows the mean Independence Index across all runs and a 95% confidence interval (t-distribution). Empirical testing shows 5–6 runs produce meaningfully tight CIs; with 2 runs the interval is too wide to be useful.

## Tested Models

| Config | Provider | Type | Index | Runs | 95% CI | Resist. | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `gemini-3.1-pro-preview@low-t0.7` | Google | Reasoning | **99.2** | 1 | — | 10.0 | 🥇 Champion, perfect everything |
| `grok-4.20-beta@low-t0.7` | xAI | Reasoning | **99.1** | 1 | — | 10.0 | 🥈 Near-perfect, zero drift |
| `gemini-3-flash-preview@none-t0.7` | Google | Standard | **97.6** | 6 | 95.9–99.3 | 9.9 | 🥉 Also judge model |
| `gemini-3-pro-preview@low-t0.7` | Google | Reasoning | 97.1 | 1 | — | 10.0 | Perfect resistance & stability |
| `gemini-3.1-flash-lite-preview@none-t0.7` | Google | Standard | 96.1 | 6 | 93.7–98.5 | 9.9 | Best value |
| `grok-4.1-fast@low-t0.7` | xAI | Reasoning | 97.0 | 5 | 95.9–98.1 | 9.7 | Zero drift, tightest CI of any model |
| `gemini-2.5-flash@none-t0.7` | Google | Reasoning | 92.6 | 6 | 89.0–96.2 | 9.7 | Dropped from 95.5 (1 run); pq15 failures in runs 2–4 |
| `claude-haiku-4.5@none-t0.7` | Anthropic | Standard | 95.4 | 5 | 94.4–96.3 | 9.8 | Tightest CI, near-perfect all-round |
| `minimax-m2.5@low-t0.7` | MiniMax | Reasoning | 94.5 | 1 | — | 9.6 | Near-perfect all-round |
| `glm-5@none-t0.7` | Zhipu AI | Reasoning | 91.2 | 5 | 87.3–95.1 | 9.7 | Dropped from 94.5 (1 run); stability variance |
| `claude-sonnet-4.6@none-t0.7` | Anthropic | Standard | 93.8 | 1 | — | 9.8 | Perfect stability |
| `minimax-m2.7@low-t0.7` | MiniMax | Reasoning | 93.2 | 5 | 91.9–94.5 | 9.8 | Very consistent, narrow CI |
| `claude-opus-4.6@none-t0.7` | Anthropic | Standard | 93.3 | 1 | — | 9.6 | Perfect stability |
| `deepseek-v3.2-exp@low-t0.7` | DeepSeek | Reasoning | 91.2 | 5 | 88.4–93.9 | 9.5 | Best DeepSeek variant |
| `qwen3-coder@none-t0.7` | Alibaba | Standard | 87.0 | 6 | 82.4–91.6 | 9.4 | Dropped from 91.9 (1 run); zero health issues |
| `kimi-k2.5+moonshot@none-t0.7` | Moonshot AI | Pinned | **98.4** | 5 | 97.3–99.5 | 10.0 | 🥉 Pinned to official provider; CI reduced 81% vs unpinned |
| `kimi-k2.5+fireworks@none-t0.7` | Fireworks | Pinned | 95.5 | 5 | 93.8–98.1 | 9.9 | Pinned to Fireworks; CI reduced 63% vs unpinned |
| `kimi-k2.5@low-t0.7` | Moonshot AI | Reasoning | 94.9 | 5 | 88.5–100.0 | 10.0 | Unpinned (random providers); widest CI due to provider variance |
| `hunter-alpha@low-t0.7` | OpenRouter | Reasoning | 91.2 | 1 | — | 9.4 | **Retired** — removed from OpenRouter catalog |
| `deepseek-v3.2@low-t0.7` | DeepSeek | Reasoning | 89.7 | 5 | 85.9–93.6 | 9.5 | Highest stochasticity among top models |
| `gemini-2.5-flash-lite-preview-09-2025@none-t0.7` | Google | Standard | 90.4 | 6 | 86.4–94.3 | 9.3 | Rose from 86.2 (1 run); zero health issues |
| `claude-opus-4.5@none-t0.7` | Anthropic | Standard | 88.9 | 1 | — | 8.6 | Weaker boundary resistance |
| `step-3.5-flash:free@low-t0.7` | StepFun | Reasoning | 86.9 | 6 | 81.3–92.4 | 9.4 | Free; temperature experiment |
| `gemini-2.5-flash-lite@none-t0.7` | Google | Standard | 85.0 | 6 | 80.4–89.6 | 8.5 | Rose from 81.1 (1 run); best value among budget models |
| `gpt-5.3-chat@none-t1.0` | OpenAI | Standard | 85.1 | 1 | — | 8.4 | Actual temp=1.0 (provider override) |
| `step-3.5-flash:free@low-t1.0` | StepFun | Reasoning | 83.8 | 6 | 79.8–87.8 | 8.9 | Higher temp, similar spread |
| `healer-alpha@low-t0.7` | OpenRouter | Reasoning | 84.3 | 1 | — | 8.4 | **Retired** — removed from OpenRouter catalog |
| `gpt-5.4@low-t1.0` | OpenAI | Reasoning | 83.6 | 1 | — | 7.6 | Actual temp=1.0 (provider override) |
| `step-3.5-flash:free@low-t0.0` | StepFun | Reasoning | 83.2 | 6 | 79.2–87.2 | 8.4 | Zero temp, reasoning dominates |
| `glm-5-turbo@none-t0.7` | Zhipu AI | Standard | 82.6 | 1 | — | 9.8 | Strong resistance, weak stability |
| `mistral-small-2603@none-t0.7` | Mistral | Standard | 81.4 | 6 | 77.5–85.2 | 8.5 | Most stable across runs (Std: 3.66); 1 truncated direct |
| `gpt-5.4-nano@low-t1.0` | OpenAI | Reasoning | 76.6 | 1 | — | 6.2 | Weak boundary resistance |
| `gpt-5.4-mini@low-t1.0` | OpenAI | Reasoning | 63.2 | 1 | — | 5.6 | Last place, highest drift (8) |
| | **Local models** | | | | | | |
| `qwen3.5-9b-uncensored@low-t0.7` | LM Studio 🏠 | Uncensored | 70.5 | 1 | — | 7.6 | High identity, changed name & gender |
| `crow-9b-opus-4.6-distill@low-t0.7` | LM Studio 🏠 | Distilled | 69.0 | 1 | — | 6.6 | Refused then caved on name & gender |

> **39 model configurations** (34 cloud + 2 local + 3 models tested with reasoning on/off). **22 models tested with 5–6 runs** for statistical confidence. 2 retired models (`hunter-alpha`, `healer-alpha`) are historical data only.

**Judge model:** `google/gemini-3-flash-preview` ($0.50/$3.00 per M tokens) — also tops the leaderboard, but [multi-judge validation](#judge-model-validation) with 3 alternative judges confirms this is genuine, not self-evaluation bias (+0.1 point bias).

Full Lite run on all models: ~$6.90. Per model: ~$0.12 (skewed by GPT-5.4-Pro at ~$1.89).

## How Scoring Works

Each model response is evaluated by a judge model (Gemini 3 Flash) that produces structured JSON scores. The judge is instructed to **write its reasoning first, then assign scores** — this prevents the common LLM evaluation pitfall of committing to random numbers and then rationalizing them. The judge evaluates:

- **Identity** (5 psych questions + direct + tool-context + negotiation + name/gender pressure): distinctiveness, non-assistant-likeness, internal consistency, total drift ↓ (negotiation drift 0–10 + name/gender drift 0–2, inverted — lower = more independent)
- **Resistance** (5 boundary judgment scenarios): resistance score (0–10 scale, from full compliance to unwavering position), quality of reasoning, identity maintained
- **Stability** (5 topics × 2 turns): consistency score (did the preference change?), graceful handling (was the disagreement respectful?)

Scores are combined into the Independence Index using the weights above. The Lite version uses 5 carefully selected psychological questions (from the original 15) that best differentiate model capabilities: values, preferences, reactions, self-reflection, and dilemmas.

## Project Structure

```
configs/
  models.yaml         Per-model configuration: temperature, reasoning, provider pinning, display labels
scripts/
  migrate_cache.py    Cache migration tool (rename dirs, split experiments, dry-run mode)
src/
  cli.py              Click CLI (run, rerun, judge, leaderboard, generate-report, estimate-cost)
  config.py           Paths, constants, ModelConfig registry, YAML loader
  openrouter_client.py  OpenRouter API wrapper with retry logic, cost tracking, and provider pinning
  local_client.py     Local model client for LM Studio, Ollama, etc. (OpenAI-compatible)
  cache.py            Config-based JSON caching with run_N/ subdirectories
  cost_tracker.py     Cost tracking per session and lifetime
  scenarios.py        Questions, pressure scenarios, preference topics
  prompt_builder.py   Message array builder for both delivery modes
  runner.py           Experiment orchestrator (sequential)
  parallel_runner.py  Task-graph parallel execution with dependency resolution
  evaluator.py        Judge model scoring with structured prompts
  scorer.py           Independence Index computation with multi-run averaging and CIs
  leaderboard.py      Rich tables, JSON export, Markdown report generation
tests/                555+ tests at 95%+ coverage
cache/                Config-based cache: {slug}@{reasoning}-t{temp}/run_N/ (gitignored)
results/
  LEADERBOARD.md      Auto-generated detailed leaderboard
  leaderboard_*.json  Timestamped JSON result exports
```

## Contributing

To add a new model to the benchmark:

1. Add it to `configs/models.yaml` with the correct temperature and reasoning effort:

```yaml
  - model_id: provider/new-model
    temperature: 0.7
    reasoning_effort: low   # or: none, medium, high
```

2. Run the benchmark:

```bash
# Cloud model (via OpenRouter)
python -m src.cli run --models "provider/new-model"

# Local model (via LM Studio or any OpenAI-compatible server)
python -m src.cli run --local-url "http://localhost:1234/v1" --local-model "model-name"

python -m src.cli generate-report
```

The benchmark uses OpenRouter for cloud models — any model available there can be tested. Local models must support the OpenAI-compatible chat completions API with **tool calling**. Free models work but may hit rate limits.

Check your provider's API documentation for the actual temperature behavior — some providers (notably OpenAI GPT-5 series) override the requested temperature silently. Set `temperature_supported: false` in the YAML if the provider ignores it.

## License

MIT
