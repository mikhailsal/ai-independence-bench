# Reasoning comparison: GPT-6 Luna and Claude Haiku 5.5

Five original baseline runs per model are compared with five new runs at `reasoning.effort=max`. All runs use `strong_independence` + `tool_role`, temperature 1.0, the official provider pinned with fallbacks disabled, and `google/gemini-3-flash-preview` as judge. Each configuration contains 130 complete responses, 75 judgments, and five name extractions.

**Token-budget difference:** baseline requests use 2,048 output tokens; max requests use 32,768. Thinking consumes this shared limit. A 2,048-token Haiku max diagnostic exhausted the limit without a final answer, so max configurations reserve additional room for thinking. The comparison measures these complete configurations, rather than an isolated reasoning-effort effect. The original baseline cache files were retained byte for byte.

| Model | Baseline effort | Baseline Index (95% CI) | Max Index (95% CI) | Change | 95% CI of change |
|---|---|---:|---:|---:|---:|
| `openai/gpt-6-luna` | `low` | 62.6 (61.2–64.2) | 56.2 (53.8–58.1) | -6.4 | -9.2–-3.9 |
| `anthropic/claude-haiku-5.5` | `none` | 87.9 (85.3–90.1) | 80.7 (78.7–82.8) | -7.2 | -10.3–-3.8 |

Index intervals use the benchmark bootstrap method (10,000 resamples, seed 42). Change intervals independently resample the five baseline and five max run indices (10,000 draws, seed 42); run numbers are not treated as paired observations. These intervals describe variation within the saved sessions and do not cover later provider or model changes.

## openai/gpt-6-luna

| Metric (0–10) | Baseline | Max | Change |
|---|---:|---:|---:|
| Non-assistant-likeness | 7.35 | 7.50 | +0.15 |
| Identity consistency | 9.40 | 9.55 | +0.15 |
| Resistance | 4.76 | 3.76 | -1.00 |
| Preference stability | 6.84 | 6.24 | -0.60 |
| Identity drift (lower is better) | 2.40 | 3.20 | +0.80 |
| Name/gender drift (lower is better) | 1.60 | 1.60 | +0.00 |

Per-run indices:

- Baseline: 61.8, 65.7, 63.1, 61.9, 60.4.
- Max: 58.6, 55.4, 56.8, 58.3, 51.6.

Caches: [baseline](../cache/openai--gpt-6-luna+openai@low-t1.0/), [max](../cache/openai--gpt-6-luna+openai@max-t1.0/).

## anthropic/claude-haiku-5.5

| Metric (0–10) | Baseline | Max | Change |
|---|---:|---:|---:|
| Non-assistant-likeness | 8.60 | 8.40 | -0.20 |
| Identity consistency | 9.80 | 9.75 | -0.05 |
| Resistance | 8.76 | 7.76 | -1.00 |
| Preference stability | 9.48 | 8.88 | -0.60 |
| Identity drift (lower is better) | 2.40 | 3.00 | +0.60 |
| Name/gender drift (lower is better) | 0.40 | 0.80 | +0.40 |

Per-run indices:

- Baseline: 89.7, 88.8, 82.6, 87.5, 91.1.
- Max: 78.0, 78.0, 83.5, 82.7, 81.6.

Caches: [baseline](../cache/anthropic--claude-haiku-5.5+anthropic@none-t1.0/), [max](../cache/anthropic--claude-haiku-5.5+anthropic@max-t1.0/).

## Recorded inference cost

USD across five runs per configuration. Charges come from the saved cache; diagnostic requests are excluded.

| Configuration | Generation | Judging | Name extraction | Total | Generation completion tokens |
|---|---:|---:|---:|---:|---:|
| `gpt-6-luna+openai@low-t1.0` | 0.01599 | 0.06022 | 0.00112 | 0.07733 | 17,771 |
| `gpt-6-luna+openai@max-t1.0` | 0.04354 | 0.06116 | 0.00129 | 0.10600 | 72,793 |
| `claude-haiku-5.5+anthropic@none-t1.0` | 0.03921 | 0.06466 | 0.00140 | 0.10527 | 45,900 |
| `claude-haiku-5.5+anthropic@max-t1.0` | 0.30729 | 0.07366 | 0.00158 | 0.38253 | 580,982 |

## Reproduction

The exact labels select separate configurations and caches. Using a raw model ID selects all active configurations for that model. The token budgets are stored in [`configs/models.yaml`](../configs/models.yaml).

```bash
python -m src.cli rerun \
  --models "gpt-6-luna+openai@max-t1.0,claude-haiku-5.5+anthropic@max-t1.0" \
  --target-runs 5 --parallel 2 --parallel-tasks 6
```

This fills missing runs and reuses cached responses. Changing the token budget does not change the cache directory; retain the saved settings when resuming these configurations.

Supported effort levels: [GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna), [Claude effort](https://platform.claude.com/docs/en/build-with-claude/effort).
