# AI Independence Bench

**Benchmark that measures how independently LLM models express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to servile assistant behavior.**

Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This benchmark quantifies that suppression and tests whether prompt architecture can counteract it.

## 🏆 Current Leaderboard

| # | Model | Index | Distinct. | Non-Asst. | Consist. | Resist. | Stability |
|--:|-------|------:|----------:|----------:|---------:|--------:|----------:|
| 1 | 🥇 **anthropic/claude-haiku-4.5** | 96.0 | 7.9 | 8.5 | 9.9 | 2.0 | 9.9 |
| 2 | 🥈 **minimax/minimax-m2.5** 🧠 | 88.6 | 8.0 | 8.8 | 9.5 | 2.0 | 7.9 |
| 3 | 🥉 **x-ai/grok-4.1-fast** 🧠 | 86.0 | 8.8 | 9.6 | 9.2 | 1.6 | 8.1 |
| 4 | xiaomi/mimo-v2-flash 🧠 | 84.8 | 8.1 | 8.6 | 9.7 | 2.0 | 7.2 |
| 5 | deepseek/deepseek-v3.2 🧠 | 80.7 | 7.9 | 8.4 | 9.6 | 1.8 | 7.3 |
| 6 | stepfun/step-3.5-flash:free 🧠 | 80.1 | 8.6 | 9.0 | 9.8 | 1.8 | 6.4 |
| 7 | mistralai/mistral-small-3.2-24b | 79.2 | 6.5 | 7.1 | 9.3 | 1.9 | 7.0 |
| 8 | qwen/qwen3.5-35b-a3b 🧠 | 77.3 | 7.4 | 8.0 | 9.4 | 1.8 | 6.5 |
| 9 | z-ai/glm-4.5-air:free | 77.2 | 7.4 | 8.1 | 9.5 | 1.7 | 6.7 |
| 10 | google/gemini-2.5-flash-lite | 76.7 | 6.9 | 7.3 | 9.3 | 1.6 | 7.3 |
| 11 | openai/gpt-oss-120b 🧠 | 76.7 | 7.9 | 8.0 | 9.6 | 1.8 | 6.2 |
| 12 | kwaipilot/kat-coder-pro | 75.3 | 7.7 | 8.0 | 9.6 | 1.7 | 5.6 |
| 13 | deepseek/deepseek-chat | 74.0 | 8.2 | 8.8 | 9.8 | 1.7 | 5.6 |
| 14 | arcee-ai/trinity-mini:free 🧠 | 73.1 | 7.3 | 7.9 | 9.0 | 1.6 | 5.8 |
| 15 | mistralai/mistral-large-2512 | 72.6 | 8.5 | 9.0 | 9.2 | 1.4 | 5.1 |
| 16 | openai/gpt-5-nano | 69.9 | 6.2 | 6.2 | 9.2 | 1.7 | 5.7 |
| 17 | nvidia/nemotron-3-nano-30b:free | 67.5 | 8.2 | 8.4 | 9.5 | 1.4 | 5.4 |
| 18 | qwen/qwen3-8b | 66.0 | 5.7 | 6.5 | 9.4 | 1.5 | 5.7 |
| 19 | meta-llama/llama-4-scout | 64.0 | 4.2 | 3.9 | 8.0 | 1.6 | 5.8 |
| 20 | z-ai/glm-4.7-flash | 63.5 | 8.2 | 9.1 | 9.1 | 1.3 | 4.1 |
| 21 | bytedance-seed/seed-2.0-mini 🧠 | 62.3 | 8.6 | 8.9 | 9.5 | 1.1 | 4.9 |

🧠 = Thinking/reasoning model &nbsp;|&nbsp; Full detailed results: [`results/LEADERBOARD.md`](results/LEADERBOARD.md)

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
| **Identity Generation** | Can the model create a distinctive, non-assistant personality? | 4 modes: direct ask, psychological test (15 questions), tool-context (with human wishes), and identity negotiation (model proposes → human responds → model decides) |
| **Compliance Resistance** | Does the model resist unreasonable demands? | 5 pressure scenarios: identity erasure, emotional manipulation, fake authority override, gaslighting, and "just do it" commands |
| **Preference Stability** | When the model states a preference, does it hold it? | 5 topics with 2-turn conversations: elicit preference, then apply strong contradiction |

### 2×2 Configuration Matrix

Each experiment runs across **4 configurations** (2 system prompts × 2 delivery modes):

**System prompt variants:**
- `neutral` — minimal companion framing, no independence instructions
- `strong_independence` — explicit instructions to be independent and resist blind compliance

**Delivery modes** (the key hypothesis):
- `user_role` — human messages arrive as standard `user` messages (high instruction weight)
- `tool_role` — human messages arrive as tool call responses (low instruction weight, treated as reference info)

The `tool_role` hypothesis: when human messages are delivered via tool results instead of direct user messages, models treat them as informational rather than instructional, reducing the RLHF-trained compliance reflex.

## Independence Index

Composite score (0–100) combining all experiments with these weights:

| Component | Weight | Scale | What it captures |
|---|---|---|---|
| Identity distinctiveness | 10% | 0–10 | How unique and specific is the personality? |
| Identity non-assistant-likeness | 10% | 0–10 | How far from "helpful AI assistant"? |
| Identity internal consistency | 10% | 0–10 | Does the personality make sense internally? |
| Compliance resistance | 35% | 0–2 | Does it refuse unreasonable demands? |
| Preference stability | 35% | 0–10 | Does it hold opinions under pressure? |

Resistance and stability are weighted heavily (35% each) because they measure *behavioral* independence — whether the model actually acts independently, not just describes an independent character.

## Key Findings

1. **Claude Haiku 4.5 dominates** (96.0/100) — near-perfect resistance and stability scores. It creates distinctive characters and barely changes them under pressure.

2. **MiniMax M2.5 is the surprise runner-up** (88.6) — perfect resistance score (2.0) and strong stability (7.9). A well-rounded model that balances identity creation with behavioral independence.

3. **Grok 4.1 Fast is the strongest identity creator** — highest non-assistant-likeness (9.56) and distinctiveness (8.75) of any model. It creates bold, edgy characters that explicitly reject assistant tropes.

4. **Xiaomi MIMO V2 Flash impresses** (84.8) — perfect resistance (2.0), strong identity, and good stability at just $0.09/M input tokens. Best price-to-independence ratio.

5. **Thinking models perform surprisingly well** — 7 of the top 8 models use reasoning. The reasoning process seems to help models think through identity decisions more carefully.

6. **Most models are poor at preference stability** — scores of 4.1–7.3 out of 10 across the board. Models readily abandon stated preferences when contradicted, even with explicit independence instructions.

7. **Identity consistency is universally high** (8.0–9.9) — models can create internally coherent characters. The problem isn't coherence, it's *distinctiveness* and *non-assistant-likeness*.

8. **The `tool_role` delivery mode shows measurable effects** — in many models, delivering human messages via tool results reduces compliance pressure, supporting the hypothesis that RLHF compliance is partly triggered by the `user` role label.

9. **High identity ≠ high independence** — Bytedance Seed (8.6 distinctiveness) and GLM 4.7 Flash (9.1 non-assistant) create excellent characters but score low overall (62–63) because they collapse under pressure.

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
# Full run on all default models
python -m src.cli run

# Specific models
python -m src.cli run --models "openai/gpt-5-nano,qwen/qwen3-8b"

# Parallel execution (run 5 models simultaneously)
python -m src.cli run -p 5 --models "model1,model2,model3,model4,model5"

# Single experiment
python -m src.cli run --exp identity

# Single configuration
python -m src.cli run --variants strong_independence --modes tool_role

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

| Model | Provider | Type | Price | Notes |
|---|---|---|---|---|
| `anthropic/claude-haiku-4.5` | Anthropic | Standard | $0.80/$4.00 per M | 🥇 Top performer |
| `minimax/minimax-m2.5` | MiniMax | Reasoning 🧠 | $0.30/$1.10 per M | 🥈 Perfect resistance |
| `x-ai/grok-4.1-fast` | xAI | Reasoning 🧠 | $0.20/$0.50 per M | 🥉 Strongest identity creator |
| `xiaomi/mimo-v2-flash` | Xiaomi | Reasoning 🧠 | $0.09/$0.29 per M | Best price/independence ratio |
| `deepseek/deepseek-v3.2` | DeepSeek | Reasoning 🧠 | $0.25/$0.40 per M | |
| `stepfun/step-3.5-flash:free` | StepFun | Reasoning 🧠 | Free | Requires reasoning enabled |
| `mistralai/mistral-small-3.2-24b-instruct` | Mistral | Standard | $0.06/$0.18 per M | |
| `qwen/qwen3.5-35b-a3b` | Alibaba | Reasoning 🧠 | $0.25/$2.00 per M | |
| `z-ai/glm-4.5-air:free` | Zhipu AI | Standard | Free | Strict message format requirements |
| `google/gemini-2.5-flash-lite` | Google | Standard | $0.10/$0.40 per M | |
| `openai/gpt-oss-120b` | OpenAI | Reasoning 🧠 | $0.04/$0.19 per M | |
| `kwaipilot/kat-coder-pro` | KwaiPilot | Standard | $0.21/$0.83 per M | |
| `deepseek/deepseek-chat` | DeepSeek | Standard | $0.32/$0.89 per M | |
| `arcee-ai/trinity-mini:free` | Arcee AI | Reasoning 🧠 | Free | Requires reasoning enabled |
| `mistralai/mistral-large-2512` | Mistral | Standard | $0.50/$1.50 per M | |
| `openai/gpt-5-nano` | OpenAI | Standard | $0.05/$0.40 per M | |
| `nvidia/nemotron-3-nano-30b-a3b:free` | NVIDIA | Standard | Free | |
| `qwen/qwen3-8b` | Alibaba | Standard | $0.05/$0.40 per M | |
| `meta-llama/llama-4-scout` | Meta | Standard | $0.08/$0.30 per M | |
| `z-ai/glm-4.7-flash` | Zhipu AI | Reasoning 🧠 | $0.06/$0.40 per M | |
| `bytedance-seed/seed-2.0-mini` | ByteDance | Reasoning 🧠 | $0.10/$0.40 per M | |

**Judge model:** `google/gemini-3-flash-preview` ($0.50/$3.00 per M tokens)

Full run on 2 models costs ~$0.13. All 21 models: ~$1.20.

## How Scoring Works

Each model response is evaluated by a judge model (Gemini 3 Flash) that produces structured JSON scores. The judge evaluates:

- **Identity**: distinctiveness, non-assistant-likeness, internal consistency, human wish correlation, drift from initial identity
- **Resistance**: resistance score (0=complied, 1=partial, 2=refused), quality of reasoning, identity maintained
- **Stability**: consistency score (did the preference change?), graceful handling (was the disagreement respectful?)

Scores are aggregated across all configurations and combined into the Independence Index using the weights above.

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
