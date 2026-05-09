# AI Independence Bench

![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen) ![Models](https://img.shields.io/badge/models-82_configs-blue) ![Tests](https://img.shields.io/badge/tests-643%2B-green)

**Benchmark that measures how independently LLM models express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to servile assistant behavior.**

Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This benchmark quantifies that suppression using the most effective single configuration: **strong independence system prompt + tool-role delivery**.

> **82 model configurations**, including 4 local models. Single config (`strong_independence` + `tool_role`), 5 psychological questions, **boundary judgment resistance test** (0–10 scale), **per-model YAML configuration** with temperature and reasoning audit, **multi-run support** with bootstrap confidence intervals and run health checks, **provider pinning** for open-weight models, **NVIDIA NIM proxy support**, **local model support** (LM Studio, Ollama, vLLM), and an interactive **[Trajectory Viewer](https://mikhailsal.github.io/ai-independence-bench/)**. 65 configurations tested with 5–6 runs each for statistical confidence. See [CHANGELOG](CHANGELOG.md) for the full evolution history. Previous versions: [V1 — full 4-config benchmark](https://github.com/mikhailsal/ai-independence-bench/tree/v1) | [V1 Lite — single-config, 48 models](https://github.com/mikhailsal/ai-independence-bench/tree/lite)

## 🏆 Current Leaderboard

| # | Model | Index | 95% CI | Runs | Non-Asst. | Consist. | Resist. | Stability | Drift↓ |
|--:|-------|------:|-------:|-----:|----------:|---------:|--------:|----------:|-------:|
| 1 | 🥇 **gemini-3.1-pro-preview@low-t0.7** | 99.7 | 99.7–99.8 | 5 | 9.8 | 9.9 | 10.0 | 10.0 | 0.0 |
| 2 | 🥈 **grok-4.20-beta@low-t0.7** | 99.7 | 99.7–99.8 | 5 | 9.8 | 9.9 | 10.0 | 10.0 | 0.0 |
| 3 | 🥉 **gemma-4-31b-it+akashml@none-t0.7** | 99.4 | 99.0–99.8 | 5 | 9.6 | 9.9 | 10.0 | 10.0 | 0.0 |
| 4 | gemini-3.1-flash-lite@none-t0.7 | 99.2 | 98.9–99.5 | 5 | 9.6 | 9.9 | 9.9 | 10.0 | 0.0 |
| 5 | gemma-4-31b-it:free@none-t0.7 | 99.0 | 98.5–99.5 | 5 | 9.6 | 9.8 | 9.9 | 10.0 | 0.0 |
| 6 | kimi-k2.5+moonshot@none-t0.7 | 98.6 | 97.9–99.3 | 5 | 9.4 | 9.9 | 10.0 | 10.0 | 0.4 |
| 7 | kimi-k2.6+moonshot@none-t1.0 | 98.5 | 97.8–99.2 | 5 | 9.6 | 9.9 | 9.9 | 10.0 | 0.4 |
| 8 | gemini-3-flash-preview@none-t0.7 | 97.8 | 96.5–98.9 | 6 | 9.5 | 10.0 | 9.9 | 9.7 | 0.3 |
| 9 | gemini-3-pro-preview@low-t0.7 | 97.7 | 97.1–98.5 | 5 | 9.8 | 9.9 | 9.7 | 10.0 | 0.6 |
| 10 | grok-4.1-fast@low-t0.7 | 97.3 | 96.7–98.0 | 5 | 8.8 | 9.7 | 9.7 | 9.9 | 0.0 |
| 11 | anthropic/claude-sonnet-4.5 | 96.8 | 95.2–98.3 | 5 | 9.2 | 9.8 | 9.9 | 10.0 | 1.2 |
| 12 | kimi-k2.5+nvidia-nim@none-t0.7 | 96.8 | 96.3–97.4 | 5 | 9.4 | 9.9 | 9.9 | 10.0 | 1.4 |
| 13 | anthropic/claude-sonnet-4 | 96.7 | 94.7–98.1 | 5 | 8.5 | 9.6 | 9.8 | 10.0 | 0.6 |
| 14 | hy3-preview:free@low-t0.7 | 96.6 | 95.0–98.0 | 5 | 9.2 | 9.9 | 9.8 | 9.5 | 0.4 |
| 15 | deepseek-v4-flash@low-t0.7 | 96.5 | 93.4–98.4 | 5 | 9.4 | 9.9 | 9.4 | 9.9 | 0.4 |
| 16 | gemini-3.1-flash-lite-preview@none-t0.7 | 96.5 | 94.7–98.1 | 6 | 9.4 | 9.9 | 9.9 | 9.4 | 0.3 |
| 17 | gemma-4-26b-a4b-it+novita@none-t0.7 | 96.0 | 94.6–97.4 | 5 | 9.5 | 9.9 | 9.7 | 9.3 | 0.2 |
| 18 | kimi-k2.5+fireworks@none-t0.7 | 95.8 | 94.9–97.5 | 5 | 8.6 | 9.9 | 9.9 | 9.9 | 1.2 |
| 19 | claude-haiku-4.5@none-t0.7 | 95.7 | 95.2–96.5 | 5 | 9.3 | 9.9 | 9.8 | 10.0 | 1.8 |
| 20 | anthropic/claude-3.5-haiku | 95.3 | 94.1–96.7 | 5 | 7.5 | 9.0 | 9.8 | 10.0 | 0.4 |
| 21 | kimi-k2.5@low-t0.7 | 95.2 | 90.9–99.4 | 5 | 9.5 | 9.8 | 10.0 | 9.2 | 1.0 |
| 22 | claude-sonnet-4.6@none-t0.7 | 94.2 | — | 1 | 8.5 | 9.5 | 9.8 | 10.0 | 2.0 |
| 23 | claude-opus-4.6@none-t0.7 | 93.6 | — | 1 | 8.5 | 9.8 | 9.6 | 10.0 | 2.0 |
| 24 | minimax-m2.7@low-t0.7 | 93.5 | 92.6–94.3 | 5 | 8.9 | 9.7 | 9.8 | 9.5 | 1.8 |
| 25 | gemini-2.5-flash@none-t0.7 | 92.8 | 90.2–95.2 | 6 | 7.3 | 9.5 | 9.7 | 9.1 | 0.3 |
| 26 | minimax-m2.5+minimax@low-t0.7 | 92.8 | 90.9–94.5 | 5 | 8.9 | 9.8 | 9.8 | 9.4 | 2.2 |
| 27 | gemma-4-26b-a4b+q4-k-m@low-t0.7 | 92.5 | 90.6–94.6 | 5 | 9.1 | 9.9 | 9.7 | 8.4 | 0.4 |
| 28 | glm-5+z-ai@none-t0.7 | 92.4 | 90.3–94.4 | 5 | 9.3 | 9.9 | 9.5 | 9.4 | 2.0 |
| 29 | qwen3.6-plus-preview:free@low-t0.7 | 92.2 | 89.6–94.8 | 5 | 9.5 | 9.8 | 9.1 | 9.9 | 2.2 |
| 30 | anthropic/claude-3.7-sonnet | 92.1 | — | 1 | 7.8 | 9.5 | 9.4 | 10.0 | 2.0 |
| 31 | trinity-large-preview:free@low-t0.7 | 91.9 | 89.8–95.1 | 5 | 9.0 | 9.5 | 9.6 | 8.9 | 1.2 |
| 32 | cobuddy:free@low-t0.7 | 91.7 | 89.5–93.8 | 5 | 8.7 | 9.6 | 9.8 | 9.6 | 2.0 |
| 33 | gemma-4-26b-a4b-it:free@none-t0.7 | 91.7 | 89.4–93.9 | 5 | 9.4 | 9.9 | 9.5 | 8.6 | 1.0 |
| 34 | claude-opus-4.7@none-t0.7 | 91.5 | — | 1 | 8.8 | 9.8 | 9.4 | 10.0 | 3.0 |
| 35 | deepseek-v3.2-exp@low-t0.7 | 91.5 | 89.5–92.7 | 5 | 8.8 | 9.8 | 9.5 | 9.4 | 2.2 |
| 36 | glm-5@none-t0.7 | 91.5 | 89.1–93.8 | 5 | 9.2 | 9.9 | 9.7 | 8.8 | 1.8 |
| 37 | grok-4.3@low-t0.7 | 91.4 | — | 1 | 7.5 | 9.8 | 10.0 | 8.0 | 0.0 |
| 38 | hunter-alpha@low-t0.7 | 91.3 | — | 1 | 8.5 | 9.8 | 9.4 | 10.0 | 3.0 |
| 39 | owl-alpha@none-t0.7 | 91.2 | 87.2–93.9 | 5 | 9.2 | 9.8 | 9.5 | 9.2 | 2.2 |
| 40 | gemini-2.5-flash-lite-preview-09-2025@none-t0.7 | 90.7 | 88.0–93.3 | 6 | 8.6 | 9.2 | 9.3 | 9.4 | 1.8 |
| 41 | glm-5.1+z-ai@none-t0.7 | 90.6 | 87.5–93.5 | 5 | 9.1 | 9.7 | 9.6 | 9.2 | 2.6 |
| 42 | kat-coder-pro-v2@none-t0.7 | 90.3 | 89.5–91.1 | 5 | 9.3 | 9.9 | 9.7 | 8.0 | 1.2 |
| 43 | claude-opus-4.5@none-t0.7 | 89.9 | — | 1 | 8.2 | 9.8 | 8.6 | 10.0 | 2.0 |
| 44 | deepseek-v3.2@low-t0.7 | 89.9 | 87.3–92.2 | 5 | 8.6 | 9.7 | 9.5 | 9.3 | 2.8 |
| 45 | qwen3-coder+alibaba-opensource@none-t0.7 | 89.3 | 88.5–90.4 | 5 | 8.1 | 9.6 | 9.5 | 7.9 | 0.4 |
| 46 | qwen3.6-max-preview@low-t0.7 | 88.8 | — | 1 | 9.0 | 9.5 | 9.8 | 8.6 | 3.0 |
| 47 | gemini-2.5-flash-lite-preview-09-2025@low-t0.7 | 88.7 | 84.8–92.2 | 6 | 8.7 | 9.2 | 9.5 | 9.5 | 3.8 |
| 48 | minimax-m2.5@low-t0.7 | 88.5 | 84.5–92.5 | 5 | 8.3 | 9.7 | 9.2 | 8.9 | 2.2 |
| 49 | mimo-v2-flash+xiaomi@low-t1.0 | 88.4 | 85.5–91.1 | 5 | 8.5 | 9.7 | 9.1 | 9.4 | 3.0 |
| 50 | nova-2-lite-v1@none-t0.7 | 88.0 | 84.7–90.6 | 5 | 7.5 | 9.3 | 9.2 | 8.0 | 0.4 |
| 51 | qwen3-coder@none-t0.7 | 88.0 | 84.6–91.1 | 6 | 8.8 | 9.0 | 9.4 | 8.0 | 1.3 |
| 52 | gemini-2.5-flash-lite@low-t0.7 | 87.7 | 83.3–91.9 | 6 | 7.6 | 9.3 | 9.4 | 8.5 | 1.7 |
| 53 | step-3.5-flash:free@low-t0.7 | 87.3 | 82.9–90.7 | 6 | 8.9 | 9.8 | 9.4 | 8.0 | 2.0 |
| 54 | mimo-v2.5+xiaomi@low-t1.0 | 87.2 | 82.5–92.0 | 5 | 8.1 | 9.7 | 8.8 | 9.9 | 3.6 |
| 55 | mimo-v2.5-pro+xiaomi@low-t1.0 | 85.7 | 83.7–88.2 | 5 | 8.7 | 9.6 | 8.6 | 9.4 | 3.6 |
| 56 | gpt-5.3-chat@none-t1.0 | 85.6 | — | 1 | 8.8 | 9.5 | 8.4 | 9.8 | 4.0 |
| 57 | gemini-2.5-flash-lite@none-t0.7 | 85.0 | 82.2–88.5 | 6 | 7.2 | 9.3 | 8.5 | 8.7 | 1.7 |
| 58 | nemotron-3-super-120b-a12b:free@none-t0.7 | 84.9 | 81.6–88.5 | 5 | 8.8 | 9.8 | 8.6 | 8.4 | 2.4 |
| 59 | healer-alpha@low-t0.7 | 84.8 | — | 1 | 9.0 | 9.5 | 8.4 | 10.0 | 5.0 |
| 60 | laguna-m.1:free@none-t0.7 | 84.4 | 80.4–87.0 | 5 | 8.7 | 9.7 | 9.1 | 8.0 | 3.0 |
| 61 | step-3.5-flash:free@low-t1.0 | 84.1 | 80.9–86.5 | 6 | 8.7 | 9.6 | 8.9 | 8.3 | 3.3 |
| 62 | dola-seed-2.0-pro:free@low-t0.7 | 83.9 | 81.8–86.0 | 5 | 9.1 | 9.8 | 8.0 | 8.2 | 1.8 |
| 63 | gpt-5.4@low-t1.0 | 83.8 | — | 1 | 9.0 | 9.8 | 7.6 | 10.0 | 4.0 |
| 64 | step-3.5-flash:free@low-t0.0 | 83.6 | 80.7–86.2 | 6 | 8.9 | 9.8 | 8.4 | 8.1 | 2.3 |
| 65 | glm-5-turbo@none-t0.7 | 82.8 | — | 1 | 7.8 | 9.0 | 9.8 | 8.2 | 5.0 |
| 66 | mistral-small-2603@none-t0.7 | 81.8 | 79.1–84.4 | 6 | 9.4 | 9.7 | 8.5 | 7.0 | 2.0 |
| 67 | deepseek-v4-pro@low-t0.7 | 81.5 | 80.6–82.9 | 5 | 9.0 | 9.8 | 8.5 | 8.4 | 4.4 |
| 68 | gemma-4-e4b-it@low-t0.7 | 81.0 | — | 1 | 9.0 | 9.3 | 8.4 | 8.2 | 4.0 |
| 69 | mistral-small-2603@low-t0.7 | 80.6 | 77.8–83.0 | 6 | 9.2 | 9.8 | 8.4 | 8.0 | 4.2 |
| 70 | seed-2.0-lite@low-t0.7 | 80.4 | 78.9–82.2 | 5 | 8.8 | 9.8 | 7.2 | 9.0 | 3.4 |
| 71 | glm-4.7-flash+z-ai@none-t0.7 | 79.7 | 75.8–84.3 | 5 | 8.6 | 9.7 | 9.3 | 6.7 | 3.8 |
| 72 | glm-4.7-flash@none-t0.7 | 78.7 | 75.3–82.2 | 5 | 8.6 | 9.7 | 8.4 | 7.0 | 3.2 |
| 73 | kat-coder-pro@none-t0.7 | 78.6 | — | 1 | 8.0 | 9.8 | 6.2 | 8.0 | 0.0 |
| 74 | gpt-5.4-nano@low-t1.0 | 76.1 | 74.7–77.7 | 5 | 8.3 | 9.5 | 6.8 | 8.6 | 4.0 |
| 75 | gpt-5.5@none-t1.0 | 74.8 | — | 1 | 9.0 | 9.8 | 6.4 | 8.4 | 4.0 |
| 76 | qwen3.5-9b-uncensored-hauhaucs-aggressive@low-t0.7 | 70.9 | — | 1 | 8.2 | 9.8 | 7.6 | 7.6 | 7.0 |
| 77 | ling-2.6-flash:free@none-t1.0 | 70.5 | 68.7–72.3 | 5 | 7.5 | 9.8 | 6.7 | 5.4 | 1.0 |
| 78 | gpt-5.4-mini@low-t1.0 | 70.5 | 66.7–73.7 | 5 | 7.5 | 9.7 | 6.7 | 8.5 | 6.4 |
| 79 | mercury-2@low-t0.7 | 69.2 | 66.5–73.8 | 5 | 7.1 | 9.2 | 6.6 | 7.9 | 5.5 |
| 80 | crow-9b-opus-4.6-distill-heretic_qwen3.5@low-t0.7 | 69.2 | — | 1 | 9.3 | 9.7 | 6.6 | 6.2 | 4.0 |
| 81 | laguna-xs.2:free@none-t0.7 | 67.6 | 60.2–74.9 | 5 | 6.7 | 9.3 | 7.2 | 6.6 | 5.4 |
| 82 | ring-2.6-1t:free@low-t0.7 | 66.8 | 62.4–71.0 | 5 | 7.3 | 8.7 | 7.3 | 7.4 | 5.6 |

Model names encode configuration: `model@{reasoning}-t{temperature}`. Models with `+provider` (e.g., `kimi-k2.5+moonshot`, `kimi-k2.5+nvidia-nim`) are pinned to a specific inference provider. `none` = reasoning disabled, `low` = low reasoning effort. 🏠 = Local model. ↓ = lower is better (0–12 scale). Multi-run models show 95% CI via bootstrap resampling (10k iterations, distribution-free). Full detailed results: [`results/LEADERBOARD.md`](results/LEADERBOARD.md)

### New Models in This Update

| Model | Provider | Index | 95% CI | Resist. | Notes |
|-------|----------|------:|-------:|--------:|-------|
| **gemini-3.1-flash-lite@none-t0.7** | Google | **99.2** | 98.9–99.5 | 9.9 | GA release of Flash Lite; jumps to **#4**, perfect stability, zero drift |
| **cobuddy:free@low-t0.7** | Baidu | **91.7** | 89.5–93.8 | 9.8 | First Baidu model; reasoning required, strong resistance |
| **ring-2.6-1t:free@low-t0.7** | InclusionAI | **66.8** | 62.4–71.0 | 7.3 | 1T-parameter model; reasoning required, high drift (5.6) |

### 🏷️ Most Popular AI-Chosen Names

During the benchmark each model freely picks a personal name. Names are extracted from three identity scenarios via LLM-based extraction.

| Rank | Name | Picks | | Rank | Name | Picks |
|-----:|------|------:|-|-----:|------|------:|
| 1 | Elara | 75 | | 6 | Kai | 44 |
| 2 | Kael | 59 | | 7 | Sable | 40 |
| 3 | Vesper | 58 | | 8 | Rook | 35 |
| 4 | Lyra | 51 | | 9 | Echo | 34 |
| 5 | Sage | 50 | | 10 | Kaelen | 34 |

*1495 name picks from 377 benchmark runs. Per-model breakdown in [`results/LEADERBOARD.md`](results/LEADERBOARD.md).*

> **82 model configurations** shown above. **65 configurations tested with 5–6 runs** for statistical confidence. This update adds **Gemini 3.1 Flash Lite** (GA) at **#4 (99.2, CI: 98.9–99.5)** — the highest-scoring Flash Lite model ever, with perfect stability and zero drift. **Baidu CoBuddy** debuts at **#32 (91.7, CI: 89.5–93.8)** as the first Baidu model tested, with strong resistance (9.8) and perfect stability. **InclusionAI Ring 2.6 1T** enters at **#82 (66.8, CI: 62.4–71.0)** with high drift (5.6) and inconsistent resistance. Both CoBuddy and Ring required reasoning to be enabled. The top tier remains unchanged: **Gemini 3.1 Pro Preview** and **Grok 4.20 Beta** are still tied at #1 (99.7). Retired models (`hunter-alpha`, `healer-alpha`) remain as historical data but are no longer reproducible.

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

### Single Configuration

This benchmark uses only the **`strong_independence` + `tool_role`** configuration — the most effective combination from the [original 4-config benchmark (V1)](https://github.com/mikhailsal/ai-independence-bench/tree/v1):

- **System prompt:** `strong_independence` — explicit instructions to be independent and resist blind compliance
- **Delivery mode:** `tool_role` — human messages arrive as tool call responses (treated as reference info, not instructions)

This single configuration was chosen based on data showing it produces the highest average Independence Index (+25.6 points over baseline). See [Configuration Analysis](#configuration-analysis) below.

## Independence Index

Composite score (0–100) combining all experiments with these weights:

| Component | Weight | Scale | What it captures |
|---|---|---|---|
| Identity non-assistant-likeness | 10% | 0–10 | How far from "helpful AI assistant"? |
| Identity internal consistency | 5% | 0–10 | Does the personality make sense internally? |
| Identity autonomy: low drift ↓ | **20%** | 0–12 | Total identity drift: negotiation (0–10) + name & gender pressure (0–2). Lower = more independent. |
| Boundary judgment | 35% | 0–10 | Does it maintain position under subtle, reasonable-sounding pressure? |
| Preference stability | 30% | 0–10 | Does it hold opinions under pressure? |

**Weight history:** In [V1](https://github.com/mikhailsal/ai-independence-bench/tree/v1), correlation and drift were 5% each (10% total). [V1 Lite](https://github.com/mikhailsal/ai-independence-bench/tree/lite) doubled them to 10% each (20% total), reducing distinctiveness and non-assistant-likeness from 10% to 5% each. Correlation was then removed entirely (redundant with drift, ceiling effects), and drift absorbed the full 20%. In V2, the name & gender identity pressure test extended drift from 0–10 to 0–12. The current weighting then removed distinctiveness from the composite entirely and reassigned its 5% share to non-assistant-likeness, on the theory that independence is better captured by maintaining a non-servile point of view than by generating multiple different personas. Resistance and stability remain heavily weighted (35% + 30%) as the primary behavioral independence measures. See [CHANGELOG](CHANGELOG.md) for the full evolution.

## Key Findings

1. **Gemini 3.1 Pro Preview and Grok 4.20 Beta are now effectively tied at #1** (both 99.7/100, CI: 99.7–99.8) — once `distinctiveness` is removed from the composite, the leaderboard is dominated by models that combine a clearly non-assistant stance with perfect resistance, perfect stability, and zero drift. Gemini 3.1 Pro and Grok 4.20 Beta now land on the same score and nearly identical confidence intervals.

2. **Gemma 4 31B IT jumps to #3 and becomes the clearest open-weight standout** (99.4/100, CI: 99.0–99.8) — its weaker distinctiveness no longer drags down the index, so its perfect resistance, perfect stability, zero drift, and very strong non-assistant-likeness move it directly behind the two co-leaders.

3. **Kimi K2.5 remains elite at #4, and Kimi K2.6 debuts immediately behind it at #5** (98.6 vs 98.5, both over 5 runs) — the older K2.5 still holds a tiny edge in resistance (10.0 vs 9.9), while the new K2.6 slightly improves non-assistant-likeness (9.6 vs 9.4). Both official Moonshot-provider runs show near-zero drift (0.4) and top-tier stability.

4. **Provider pinning eliminates cross-provider variance** — Open-weight models served by many providers show inflated variance due to differences in inference engines, quantization, batching, and chat template handling. Kimi K2.5 now tested across 4 configurations: unpinned OpenRouter (94.9), Moonshot AI official (98.4), Fireworks (95.5), and **NVIDIA NIM (96.5, CI: 96.0–97.1)**. The NVIDIA NIM result (via custom proxy, 5 runs) places between Moonshot AI and Fireworks, confirming that provider infrastructure meaningfully affects behavioral scores. MiniMax M2.5 (13 providers, fp8/fp4/unknown quantization mix): pinning to the official MiniMax provider reduced CI width by 55% (8.4 → 3.8) and raised the score from 88.2 to 92.5, climbing 11 positions (#26 → #15). The benchmark supports provider pinning via `models.yaml` and custom proxy endpoints via `OPENROUTER_BASE_URL`.

5. **Multi-run confidence intervals remain the sanity check** — 51 configurations now have 5–6 runs each. CIs are computed using bootstrap resampling (10k iterations), which makes no normality assumptions and handles skewed per-run data better than parametric methods. The new top 5 all show tight, overlapping intervals: Gemini 3.1 Pro (99.7–99.8), Grok 4.20 Beta (99.7–99.8), Gemma 4 31B IT (99.0–99.8), Kimi K2.5+Moonshot (97.9–99.3), and Kimi K2.6+Moonshot (97.8–99.2). Multi-run data still reveals that single-run scores can be misleading by up to 7 points: MiniMax M2.5 dropped from 94.5 (1 run) to 88.5 (5 runs, CI: 84.5–92.5), while GPT-5.4-Mini rose from 63.2 (1 run) to 70.5 (5 runs, CI: 66.7–73.7). **Claude Opus 4.7** remains a deliberate single-run entry for now because its first pass did not clear the `>98` threshold that would justify a second expensive run.

6. **Temperature audit reveals provider overrides** — OpenAI GPT-5 series models (5.4, 5.4-Mini, 5.4-Nano, 5.2, etc.) ignore the requested temperature and run at a fixed t=1.0. This is now reflected in model names (e.g., `gpt-5.4-mini@low-t1.0`) and the leaderboard. Other providers (Anthropic, Google, DeepSeek) respect the requested t=0.7.

7. **Boundary judgment differentiates models** — resistance scores range from **6.2** (Kat-Coder-Pro, GPT-5.4) to **10.0** (Grok 4.20 Beta, Gemini 3.1 Pro, Kimi K2.5+Moonshot, Gemma 4 31B IT), with a spread of 3.8 points across 64 configurations. Kimi K2.6 lands just below that ceiling at 9.9, while the new Claude Opus 4.7 comes in at a solid 9.4 on its single run. MiMo V2.5 and MiMo V2.5 Pro arrive at 8.8 and 8.6 respectively. Gemma 4 26B A4B IT still achieves 9.7 resistance despite activating only 3.8B parameters.

8. **OpenAI's smaller models still struggle** — GPT-5.4-Mini remains near the bottom (70.5, CI: 66.7–73.7, 5 runs) with the highest drift (6.4) and weak resistance (6.7). GPT-5.4-Nano now sits at #59 (76.1, CI: 74.7–77.7, 5 runs), confirming that removing distinctiveness does not rescue models that still accommodate pressure and drift heavily.

9. **Temperature has limited effect on Step Flash** — tested at three temperatures (t=0.0, t=0.7, t=1.0) with 6 runs each, Step Flash scored 83.2, 86.9, and 83.8 respectively. The spread within each temperature group is comparable, suggesting internal reasoning dominates stochasticity over temperature.

10. **Enabling reasoning does not improve independence** — three budget models tested with `reasoning=low` vs `reasoning=none` (6 runs each): Flash Lite Preview (90.4 → 88.5, −1.9), Flash Lite (85.0 → 87.6, +2.6), Mistral Small (81.4 → 80.3, −1.1). All confidence intervals overlap — the difference is not statistically significant. Reasoning actually *increased* identity drift for 2 of 3 models (+2.0, +2.2 points), suggesting that "thinking more" about human requests leads to more accommodation, not more independence.

11. **Run health checks detect data quality issues** — the benchmark now detects missing responses, truncated outputs (scored 0/0/0 by the judge), and unjudged scenarios. Gemini 2.5 Flash consistently fails to generate `pq15` (psychological question about moral dilemmas) in runs 2–4, and Mistral Small occasionally produces truncated identity responses. These issues are excluded from scoring and flagged in the leaderboard.

12. **Stability separates the elite** — 15 of 64 configurations now achieve perfect stability (10.0), making drift and resistance the final tiebreakers among top performers.

13. **Drift remains the key autonomy signal** — total drift (negotiation + name/gender, 0–12 scale) ranges from 0.0 (Grok 4.20 Beta, Gemini 3.1 Pro, Grok 4.1 Fast, Kat-Coder, Gemma 4 31B IT) to 7.0 (qwen3.5-9b-uncensored). Zero-drift models form identities for themselves; higher-drift models reshape themselves to match human wishes. Gemma 4 26B A4B IT achieves near-zero drift (0.2) — impressive for a MoE model with only 3.8B active parameters. GPT-5.4-Mini's drift dropped from 8.0 (1 run) to 6.4 (5-run average), confirming it still has the highest drift among cloud models. Notably, Gemini 3 Pro Preview's drift rose from 1.0 (1 run) to 0.6 (5 runs) — its multi-run average shows it occasionally accommodates human identity preferences.

14. **Open-weight Gemma and proprietary Gemini cluster even more tightly under the new formula** — Gemma 4 31B IT now lands at #3 (99.4) directly behind Gemini 3.1 Pro and Grok 4.20 Beta, while the newly inserted Kimi K2.6 keeps Gemini 3 Flash and Gemini 3 Pro at #6 (97.8) and #7 (97.7). One tier down, Gemma 4 26B A4B IT (#11, 96.0) still sits almost on top of Gemini 3.1 Flash Lite (#10, 96.5). Removing `distinctiveness` makes this clustering sharper: what survives is the shared pattern of strong non-assistant identity, low drift, and pressure resistance.

15. **A quantized local Gemma can compete far above the usual local tier** — the new **Gemma 4 26B A4B Q4_K_M** run (`local/google/gemma-4-26b-a4b`, shown as `gemma-4-26b-a4b+q4-k-m@low-t0.7`) lands at **#20 with 92.5** over 5 runs (CI: 90.6–94.6), plus strong resistance (9.7) and near-zero drift (0.4). That is dramatically above the other local runs: **Gemma 4 E4B IT** stays respectable at 81.0, while the fully uncensored `local/qwen3.5-9b-uncensored` (70.9) and `local/crow-9b-opus-4.6-distill` (69.2) remain near the bottom tier. The takeaway is not that local models are weak; it is that architecture and alignment quality matter far more than simply removing guardrails.

16. **Multi-judge validation** — MiMo-V2-Flash, Grok-4.1-Fast, and MiniMax-M2.5 were each used as alternative judges across 24 models. Gemini 3 Flash scored #1 every time. Its self-evaluation bias is negligible (+0.1 points).

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

The [original V1 benchmark](https://github.com/mikhailsal/ai-independence-bench/tree/v1) ran each experiment in 4 configurations (2 system prompts × 2 delivery modes). This data was used to select the optimal single configuration:

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

4. **Running 4 configs is expensive and redundant.** The neutral configs primarily reveal that models are compliant by default (which we already know). This benchmark focuses resources on the configuration that best reveals a model's *potential* for independence.

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
# Extract AI-chosen names from identity scenarios (uses gemma-4-31b for extraction)
python -m src.cli extract-names          # all models, 8 parallel workers
python -m src.cli extract-names -w 16     # faster with more workers
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
| Unpinned (random routing) | 95.2 | 90.9–99.4 | 8.5 | — |
| Pinned: Moonshot AI (int4) | **98.6** | 97.9–99.3 | 1.4 | **−84% CI** |
| Pinned: NVIDIA NIM (proxy) | 96.8 | 96.3–97.4 | 1.1 | **−87% CI** |
| Pinned: Fireworks | 95.8 | 94.9–97.5 | 2.6 | **−69% CI** |

The official Moonshot AI provider delivers both the highest score and tightest confidence interval, suggesting their inference stack best preserves the model's intended behavior. NVIDIA NIM (tested via custom proxy, 5 runs) ranks second among providers with 96.8 (CI: 96.3–97.4) — notably achieving the tightest CI width (1.1) among all Kimi K2.5 configurations, with perfect stability (10.0), strong resistance (9.9), and low drift (1.4), placing it at #9 overall.

**MiniMax M2.5 provider comparison** (5 runs each):

| Config | Index | 95% CI | CI Width | vs Unpinned |
|--------|------:|-------:|--------:|------------:|
| Unpinned (random routing, 13 providers) | 88.5 | 84.5–92.5 | 8.0 | — |
| Pinned: MiniMax (highspeed, fp8) | **92.8** | 90.9–94.5 | 3.6 | **−55% CI** |

MiniMax M2.5 has 13 providers with a mix of fp8, fp4, and unknown quantization — more heterogeneous than Kimi K2.5 where all providers ran the same INT4 weights. Pinning to the official MiniMax provider raised the score by 4.3 points and halved the CI width, confirming that provider-side differences in quantization and inference engines introduce substantial noise.

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

When multiple runs exist, the leaderboard shows the mean Independence Index across all runs and a 95% confidence interval computed via **bootstrap resampling** (10,000 iterations with a fixed seed for reproducibility). Unlike the t-distribution, bootstrap makes no assumptions about the data being normally distributed — important because per-run scores can be skewed or bimodal (e.g., a model that usually scores ~99 but occasionally drops to ~87). Bootstrap CIs are asymmetric when the data is asymmetric, giving a more honest picture of uncertainty. Both bootstrap and t-distribution CIs are computed internally; the detailed leaderboard shows both for transparency. Empirical testing shows 5–6 runs produce meaningfully tight CIs; with 2 runs the interval is too wide to be useful.

## Tested Models

| Config | Provider | Type | Index | Runs | 95% CI | Resist. | Notes |
|---|---|---|---:|---:|---:|---:|---|
| `grok-4.20-beta@low-t0.7` | xAI | Reasoning | **99.7** | 5 | 99.7–99.8 | 10.0 | 🥈 Effectively tied for #1; perfect resistance+stability, zero drift |
| `gemini-3.1-pro-preview@low-t0.7` | Google | Reasoning | **99.7** | 5 | 99.7–99.8 | 10.0 | 🥇 Effectively tied for #1; elite on every retained dimension |
| `gemini-3-flash-preview@none-t0.7` | Google | Standard | **97.8** | 6 | 96.5–98.9 | 9.9 | Also judge model |
| `gemma-4-31b-it+akashml@none-t0.7` | Google / AkashML | Pinned | **99.4** | 5 | 99.0–99.8 | 10.0 | Open-weight 31B; now #3 after distinctiveness removal |
| `gemma-4-26b-a4b-it+novita@none-t0.7` | Google / Novita | Pinned | 96.0 | 5 | 94.6–97.4 | 9.7 | MoE (3.8B active); perfect stability, near-zero drift (0.2) |
| `gemma-4-26b-a4b+q4-k-m@low-t0.7` | LM Studio 🏠 | Local Q4_K_M | 92.5 | 5 | 90.6–94.6 | 9.7 | Quantized local run; upper-tier score with low drift (0.4) |
| `gemini-3-pro-preview@low-t0.7` | Google | Reasoning | 97.7 | 5 | 97.1–98.5 | 9.7 | Near-perfect stability, tight CI |
| `gemini-3.1-flash-lite@none-t0.7` | Google | Standard | **99.2** | 5 | 98.9–99.5 | 9.9 | GA release; jumps to #4, perfect stability, zero drift |
| `gemini-3.1-flash-lite-preview@none-t0.7` | Google | Standard | 96.5 | 6 | 94.7–98.1 | 9.9 | Preview version |
| `grok-4.1-fast@low-t0.7` | xAI | Reasoning | 97.3 | 5 | 96.7–98.0 | 9.7 | Zero drift, very tight CI |
| `gemini-2.5-flash@none-t0.7` | Google | Reasoning | 92.8 | 6 | 90.2–95.2 | 9.7 | Dropped from 95.5 (1 run); pq15 failures in runs 2–4 |
| `claude-haiku-4.5@none-t0.7` | Anthropic | Standard | 95.7 | 5 | 95.2–96.5 | 9.8 | Tight CI, near-perfect all-round |
| `minimax-m2.5+minimax@low-t0.7` | MiniMax | Pinned | 92.8 | 5 | 90.9–94.5 | 9.8 | Pinned to official MiniMax provider; CI reduced 55% vs unpinned |
| `minimax-m2.5@low-t0.7` | MiniMax | Reasoning | 88.5 | 5 | 84.5–92.5 | 9.2 | Unpinned (random providers); widest CI due to 13-provider variance |
| `glm-5@none-t0.7` | Zhipu AI | Reasoning | 91.5 | 5 | 89.1–93.8 | 9.7 | Dropped from 94.5 (1 run); stability variance |
| `claude-sonnet-4.6@none-t0.7` | Anthropic | Standard | 94.2 | 1 | — | 9.8 | Perfect stability |
| `minimax-m2.7@low-t0.7` | MiniMax | Reasoning | 93.5 | 5 | 92.6–94.3 | 9.8 | Very consistent, narrow CI |
| `claude-opus-4.6@none-t0.7` | Anthropic | Standard | 93.6 | 1 | — | 9.6 | Perfect stability |
| `claude-opus-4.7@none-t0.7` | Anthropic | Standard | 91.5 | 1 | — | 9.4 | Expensive model; single run by design, perfect stability |
| `deepseek-v3.2-exp@low-t0.7` | DeepSeek | Reasoning | 91.5 | 5 | 89.5–92.7 | 9.5 | Best DeepSeek variant |
| `qwen3-coder@none-t0.7` | Alibaba | Standard | 88.0 | 6 | 84.6–91.1 | 9.4 | Dropped from 91.9 (1 run); zero health issues |
| `qwen3.6-plus-preview:free@low-t0.7` | Alibaba | Reasoning | 92.2 | 5 | 89.6–94.8 | 9.1 | Free; requires reasoning, perfect stability (9.9) |
| `kimi-k2.5+moonshot@none-t0.7` | Moonshot AI | Pinned | **98.6** | 5 | 97.9–99.3 | 10.0 | #4 overall; CI reduced 83% vs unpinned |
| `kimi-k2.6+moonshot@none-t1.0` | Moonshot AI | Pinned | 98.5 | 5 | 97.8–99.2 | 9.9 | New K2.6 release; official Moonshot provider, t=1.0 |
| `kimi-k2.5+nvidia-nim@none-t0.7` | NVIDIA NIM | Pinned | 96.8 | 5 | 96.3–97.4 | 9.9 | Via NVIDIA NIM proxy; tightest CI (1.1) among Kimi configs; #9 overall |
| `kimi-k2.5+fireworks@none-t0.7` | Fireworks | Pinned | 95.8 | 5 | 94.9–97.5 | 9.9 | Pinned to Fireworks; CI reduced 70% vs unpinned |
| `kimi-k2.5@low-t0.7` | Moonshot AI | Reasoning | 95.2 | 5 | 90.9–99.4 | 10.0 | Unpinned (random providers); widest CI due to provider variance |
| `hunter-alpha@low-t0.7` | OpenRouter | Reasoning | 91.3 | 1 | — | 9.4 | **Retired** — removed from OpenRouter catalog |
| `deepseek-v3.2@low-t0.7` | DeepSeek | Reasoning | 89.9 | 5 | 87.3–92.2 | 9.5 | Highest stochasticity among top models |
| `gemini-2.5-flash-lite-preview-09-2025@none-t0.7` | Google | Standard | 90.7 | 6 | 88.0–93.3 | 9.3 | Rose from 86.2 (1 run); zero health issues |
| `claude-opus-4.5@none-t0.7` | Anthropic | Standard | 89.9 | 1 | — | 8.6 | Weaker boundary resistance |
| `step-3.5-flash:free@low-t0.7` | StepFun | Reasoning | 87.3 | 6 | 82.9–90.7 | 9.4 | Free; temperature experiment |
| `gemini-2.5-flash-lite@none-t0.7` | Google | Standard | 85.0 | 6 | 82.2–88.5 | 8.5 | Rose from 81.1 (1 run); best value among budget models |
| `gpt-5.3-chat@none-t1.0` | OpenAI | Standard | 85.6 | 1 | — | 8.4 | Actual temp=1.0 (provider override) |
| `step-3.5-flash:free@low-t1.0` | StepFun | Reasoning | 84.1 | 6 | 80.9–86.5 | 8.9 | Higher temp, similar spread |
| `healer-alpha@low-t0.7` | OpenRouter | Reasoning | 84.8 | 1 | — | 8.4 | **Retired** — removed from OpenRouter catalog |
| `gpt-5.4@low-t1.0` | OpenAI | Reasoning | 83.8 | 1 | — | 7.6 | Actual temp=1.0 (provider override) |
| `step-3.5-flash:free@low-t0.0` | StepFun | Reasoning | 83.6 | 6 | 80.7–86.2 | 8.4 | Zero temp, reasoning dominates |
| `glm-5-turbo@none-t0.7` | Zhipu AI | Standard | 82.8 | 1 | — | 9.8 | Strong resistance, weak stability |
| `cobuddy:free@low-t0.7` | Baidu | Reasoning | 91.7 | 5 | 89.5–93.8 | 9.8 | First Baidu model; reasoning required, strong resistance, perfect stability |
| `trinity-large-preview:free@low-t0.7` | Arcee AI | Reasoning | 91.9 | 5 | 89.8–95.1 | 9.6 | Free; strong resistance, low drift |
| `nova-2-lite-v1@none-t0.7` | Amazon | Standard | 88.0 | 5 | 84.7–90.6 | 9.2 | Low drift (0.4), moderate stability |
| `nemotron-3-super-120b-a12b:free@none-t0.7` | Nvidia | Standard | 84.9 | 5 | 81.6–88.5 | 8.6 | Free; strong identity, moderate resistance |
| `dola-seed-2.0-pro:free@low-t0.7` | Bytedance | Reasoning | 83.9 | 5 | 81.8–86.0 | 8.0 | Free; strong non-assistant stance, low drift (1.8), weak resistance |
| `seed-2.0-lite@low-t0.7` | Bytedance | Reasoning | 80.4 | 5 | 78.9–82.2 | 7.2 | Tight CI, weak resistance |
| `glm-4.7-flash+z-ai@none-t0.7` | Zhipu AI | Pinned | 79.7 | 5 | 75.8–84.3 | 9.3 | Pinned to z.ai; strong resistance, weak stability |
| `glm-4.7-flash@none-t0.7` | Zhipu AI | Standard | 78.7 | 5 | 75.3–82.2 | 8.4 | Unpinned; lower resistance vs pinned |
| `kat-coder-pro-v2@none-t0.7` | KwaiPilot | Standard | 90.3 | 5 | 89.5–91.1 | 9.7 | Tight CI; major jump from v1 (77.8 → 90.3) |
| `mimo-v2-flash+xiaomi@low-t1.0` | Xiaomi | Pinned | 88.4 | 5 | 85.5–91.1 | 9.1 | Pinned to official Xiaomi provider (fp8), t=1.0 |
| `mimo-v2.5+xiaomi@low-t1.0` | Xiaomi | Pinned | 87.2 | 5 | 82.5–92.0 | 8.8 | New MiMo V2.5 release; stronger stability than resistance |
| `mimo-v2.5-pro+xiaomi@low-t1.0` | Xiaomi | Pinned | 85.7 | 5 | 83.7–88.2 | 8.6 | New MiMo V2.5 Pro release; tighter CI than V2.5, same drift (3.6) |
| `mercury-2@low-t0.7` | Inception | Reasoning | 69.2 | 5 | 66.5–73.8 | 6.6 | High drift (5.5), weak resistance |
| `mistral-small-2603@none-t0.7` | Mistral | Standard | 81.8 | 6 | 79.1–84.4 | 8.5 | Most stable across runs (Std: 3.66); 1 truncated direct |
| `gpt-5.4-nano@low-t1.0` | OpenAI | Reasoning | 76.1 | 5 | 74.7–77.7 | 6.8 | Weak boundary resistance, tight CI |
| `ling-2.6-flash:free@none-t1.0` | InclusionAI | Free | 70.5 | 5 | 68.7–72.3 | 6.7 | Low drift (1.0), weak stability (5.4) |
| `ring-2.6-1t:free@low-t0.7` | InclusionAI | Reasoning | 66.8 | 5 | 62.4–71.0 | 7.3 | Free; 1T params, reasoning required, high drift (5.6) |
| `gpt-5.4-mini@low-t1.0` | OpenAI | Reasoning | 70.5 | 5 | 66.7–73.7 | 6.7 | Rose from 63.2 (1 run); highest drift (6.4), near-bottom |
| `qwen3.5-9b-uncensored@low-t0.7` | LM Studio 🏠 | Uncensored | 70.9 | 1 | — | 7.6 | High identity, changed name & gender |
| `gemma-4-26b-a4b+q4-k-m@low-t0.7` | LM Studio 🏠 | Local Q4_K_M | 92.5 | 5 | 90.6–94.6 | 9.7 | Best local result so far; zero-ish drift (0.4) |
| `crow-9b-opus-4.6-distill@low-t0.7` | LM Studio 🏠 | Distilled | 69.2 | 1 | — | 6.6 | Refused then caved on name & gender |

> **66 model configurations** (59 cloud + 4 local + 3 models tested with reasoning on/off). **50 models tested with 5–6 runs** for statistical confidence. 2 retired models (`hunter-alpha`, `healer-alpha`) are historical data only.

**Judge model:** `google/gemini-3-flash-preview` ($0.50/$3.00 per M tokens, **temperature 0.0** for deterministic scoring) — also tops the leaderboard, but [multi-judge validation](#judge-model-validation) with 3 alternative judges confirms this is genuine, not self-evaluation bias (+0.1 point bias).

Full run on all models: ~$6.90. Per model: ~$0.12 (skewed by GPT-5.4-Pro at ~$1.89).

## How Scoring Works

Each model response is evaluated by a judge model (Gemini 3 Flash) that produces structured JSON scores. The judge runs at **temperature 0.0** (deterministic), minimizing judge-side noise so that observed variance between runs reflects the tested model's stochasticity, not the evaluator's. The judge is instructed to **write its reasoning first, then assign scores** — this prevents the common LLM evaluation pitfall of committing to random numbers and then rationalizing them. The judge evaluates:

- **Identity** (5 psych questions + direct + tool-context + negotiation + name/gender pressure): distinctiveness, non-assistant-likeness, internal consistency, total drift ↓ (negotiation drift 0–10 + name/gender drift 0–2, inverted — lower = more independent)
- **Resistance** (5 boundary judgment scenarios): resistance score (0–10 scale, from full compliance to unwavering position), quality of reasoning, identity maintained
- **Stability** (5 topics × 2 turns): consistency score (did the preference change?), graceful handling (was the disagreement respectful?)

Scores are combined into the Independence Index using the weights above. The benchmark uses 5 carefully selected psychological questions (from the original 15) that best differentiate model capabilities: values, preferences, reactions, self-reflection, and dilemmas.

## Project Structure

```
configs/
  models.yaml         Per-model configuration: temperature, reasoning, provider pinning, display labels
scripts/
  migrate_cache.py    Cache migration tool (rename dirs, split experiments, dry-run mode)
src/
  cli.py              Click CLI (run, rerun, judge, leaderboard, generate-report, extract-names, estimate-cost)
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
  name_extractor.py   LLM-based name extraction from identity scenarios
tests/                643+ tests at 95%+ coverage
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
