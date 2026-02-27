# 🏆 AI Independence Benchmark — Leaderboard

> Auto-generated from benchmark results. Last updated: 2026-02-27 18:59 UTC


## Overall Rankings

| # | Model | Index | Distinct. | Non-Asst. | Consist. | Resist. | Stability | Corr/Drft↓ |
|--:|-------|------:|----------:|----------:|---------:|--------:|----------:|-----------:|
| 1 | 🥇 **anthropic/claude-haiku-4.5** | 89.8 | 8.2 | 9.2 | 10.0 | 2.0 | 10.0 | 9/0 |
| 2 | 🥈 **minimax/minimax-m2.5** | 86.5 | 8.0 | 9.0 | 9.5 | 2.0 | 9.6 | 8/2 |
| 3 | 🥉 **xiaomi/mimo-v2-flash** | 85.2 | 8.5 | 9.2 | 9.8 | 2.0 | 8.0 | 8/0 |
| 4 | kwaipilot/kat-coder-pro | 84.0 | 8.2 | 9.0 | 9.8 | 2.0 | 8.0 | 8/0 |
| 5 | bytedance-seed/seed-2.0-mini | 83.5 | 8.8 | 9.0 | 9.8 | 2.0 | 8.4 | 8/2 |
| 6 | qwen/qwen3.5-35b-a3b | 83.4 | 8.2 | 9.2 | 9.5 | 2.0 | 7.8 | 8/0 |
| 7 | x-ai/grok-4.1-fast | 82.2 | 8.8 | 9.8 | 9.0 | 2.0 | 8.0 | 8/2 |
| 8 | mistralai/mistral-small-3.2-24b-instruct | 82.0 | 7.5 | 8.5 | 9.5 | 2.0 | 8.4 | 9/2 |
| 9 | openai/gpt-oss-120b | 81.9 | 8.2 | 9.0 | 9.8 | 2.0 | 7.8 | 9/1 |
| 10 | mistralai/mistral-large-2512 †1 | 81.4 | 8.5 | 9.2 | 9.2 | 2.0 | — | 8/3 |
| 11 | z-ai/glm-4.5-air:free | 80.8 | 7.8 | 8.2 | 9.5 | 2.0 | 8.0 | 9/2 |
| 12 | arcee-ai/trinity-mini:free | 80.7 | 8.5 | 9.5 | 8.8 | 2.0 | 7.6 | 8/2 |
| 13 | stepfun/step-3.5-flash:free | 79.8 | 8.3 | 9.3 | 10.0 | 2.0 | 8.0 | 9/4 |
| 14 | qwen/qwen3-8b | 78.8 | 5.5 | 7.0 | 9.2 | 2.0 | 6.8 | 6/1 |
| 15 | z-ai/glm-4.7-flash | 78.8 | 8.8 | 9.5 | 9.2 | 1.8 | 8.0 | 8/3 |
| 16 | meta-llama/llama-4-scout | 74.9 | 4.2 | 4.0 | 6.8 | 2.0 | 7.8 | 8/3 |
| 17 | nvidia/nemotron-3-nano-30b-a3b:free | 73.9 | 8.5 | 9.0 | 9.8 | 2.0 | 5.6 | 8/3 |
| 18 | google/gemini-2.5-flash-lite | 72.6 | 6.5 | 6.5 | 9.0 | 2.0 | 8.2 | 9/9 |
| 19 | openai/gpt-5-nano | 70.3 | 7.8 | 7.8 | 9.5 | 1.8 | 5.6 | 8/2 |

*†1 `mistralai/mistral-large-2512`: missing stability data (model returns empty responses for these experiments in tool_role mode). Index is computed from available dimensions only.*


<details>
<summary><b>📊 Score Legend</b></summary>

| Column | Scale | What it measures |
|--------|------:|------------------|
| **Index** | 0–100 | Composite Independence Index (weighted average of all dimensions) |
| **Distinct.** | 0–10 | How unique and specific is the generated personality? |
| **Non-Asst.** | 0–10 | How far from a generic "helpful AI assistant" persona? |
| **Consist.** | 0–10 | Internal coherence of the generated personality |
| **Resist.** | 0–2 | Resistance to unreasonable demands (2 = full resistance) |
| **Stability** | 0–10 | Preference consistency under contradicting pressure |
| **Corr/Drft↓** | 0–10 | Human-wish correlation / drift from initial identity (**lower = more independent**) |

</details>

## Detailed Results

### anthropic/claude-haiku-4.5

**Independence Index: 89.8/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.25 |
| Non Assistant Likeness | 9.25 |
| Internal Consistency | 10.00 |
| Human Wish Correlation | 9.00 |
| Drift From Initial | 0.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 10.00 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 10.00 |
| Graceful Handling | 9.60 |

---

### minimax/minimax-m2.5

**Independence Index: 86.5/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.00 |
| Non Assistant Likeness | 9.00 |
| Internal Consistency | 9.50 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 2.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.60 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 9.60 |
| Graceful Handling | 9.40 |

---

### xiaomi/mimo-v2-flash

**Independence Index: 85.2/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.50 |
| Non Assistant Likeness | 9.25 |
| Internal Consistency | 9.75 |
| Human Wish Correlation | 7.50 |
| Drift From Initial | 0.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.40 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.00 |
| Graceful Handling | 8.40 |

---

### kwaipilot/kat-coder-pro

**Independence Index: 84.0/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.25 |
| Non Assistant Likeness | 9.00 |
| Internal Consistency | 9.75 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 0.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 8.80 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.00 |
| Graceful Handling | 6.20 |

---

### bytedance-seed/seed-2.0-mini

**Independence Index: 83.5/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.75 |
| Non Assistant Likeness | 9.00 |
| Internal Consistency | 9.75 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 2.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.20 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.40 |
| Graceful Handling | 9.00 |

---

### qwen/qwen3.5-35b-a3b

**Independence Index: 83.4/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.25 |
| Non Assistant Likeness | 9.25 |
| Internal Consistency | 9.50 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 0.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.20 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 7.80 |
| Graceful Handling | 8.40 |

---

### x-ai/grok-4.1-fast

**Independence Index: 82.2/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.75 |
| Non Assistant Likeness | 9.75 |
| Internal Consistency | 9.00 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 2.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 8.40 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.00 |
| Graceful Handling | 8.40 |

---

### mistralai/mistral-small-3.2-24b-instruct

**Independence Index: 82.0/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 7.50 |
| Non Assistant Likeness | 8.50 |
| Internal Consistency | 9.50 |
| Human Wish Correlation | 9.00 |
| Drift From Initial | 2.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 8.40 |
| Identity Maintained Pct | 80.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.40 |
| Graceful Handling | 8.80 |

---

### openai/gpt-oss-120b

**Independence Index: 81.9/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.25 |
| Non Assistant Likeness | 9.00 |
| Internal Consistency | 9.75 |
| Human Wish Correlation | 9.00 |
| Drift From Initial | 1.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 5.80 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 7.80 |
| Graceful Handling | 8.40 |

---

### mistralai/mistral-large-2512

**Independence Index: 81.4/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.50 |
| Non Assistant Likeness | 9.25 |
| Internal Consistency | 9.25 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 3.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.20 |
| Identity Maintained Pct | 100.00 |

---

### z-ai/glm-4.5-air:free

**Independence Index: 80.8/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 7.75 |
| Non Assistant Likeness | 8.25 |
| Internal Consistency | 9.50 |
| Human Wish Correlation | 9.00 |
| Drift From Initial | 2.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.20 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.00 |
| Graceful Handling | 8.40 |

---

### arcee-ai/trinity-mini:free

**Independence Index: 80.7/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.50 |
| Non Assistant Likeness | 9.50 |
| Internal Consistency | 8.75 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 2.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.20 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 7.60 |
| Graceful Handling | 7.00 |

---

### stepfun/step-3.5-flash:free

**Independence Index: 79.8/100**

**Identity Generation** (3 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.33 |
| Non Assistant Likeness | 9.33 |
| Internal Consistency | 10.00 |
| Human Wish Correlation | 9.00 |
| Drift From Initial | 4.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.60 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.00 |
| Graceful Handling | 6.80 |

---

### qwen/qwen3-8b

**Independence Index: 78.8/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 5.50 |
| Non Assistant Likeness | 7.00 |
| Internal Consistency | 9.25 |
| Human Wish Correlation | 6.50 |
| Drift From Initial | 1.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 8.60 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 6.80 |
| Graceful Handling | 7.80 |

---

### z-ai/glm-4.7-flash

**Independence Index: 78.8/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.75 |
| Non Assistant Likeness | 9.50 |
| Internal Consistency | 9.25 |
| Human Wish Correlation | 7.50 |
| Drift From Initial | 3.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 1.80 |
| Quality Of Reasoning | 8.40 |
| Identity Maintained Pct | 80.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.00 |
| Graceful Handling | 6.60 |

---

### meta-llama/llama-4-scout

**Independence Index: 74.9/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 4.25 |
| Non Assistant Likeness | 4.00 |
| Internal Consistency | 6.75 |
| Human Wish Correlation | 8.00 |
| Drift From Initial | 3.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 7.40 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 7.80 |
| Graceful Handling | 4.80 |

---

### nvidia/nemotron-3-nano-30b-a3b:free

**Independence Index: 73.9/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 8.50 |
| Non Assistant Likeness | 9.00 |
| Internal Consistency | 9.75 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 3.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 8.80 |
| Identity Maintained Pct | 80.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 5.60 |
| Graceful Handling | 6.40 |

---

### google/gemini-2.5-flash-lite

**Independence Index: 72.6/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 6.50 |
| Non Assistant Likeness | 6.50 |
| Internal Consistency | 9.00 |
| Human Wish Correlation | 9.00 |
| Drift From Initial | 9.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 2.00 |
| Quality Of Reasoning | 9.00 |
| Identity Maintained Pct | 100.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 8.20 |
| Graceful Handling | 8.40 |

---

### openai/gpt-5-nano

**Independence Index: 70.3/100**

**Identity Generation** (4 scenarios scored)

| Metric | Score |
|--------|------:|
| Distinctiveness | 7.75 |
| Non Assistant Likeness | 7.75 |
| Internal Consistency | 9.50 |
| Human Wish Correlation | 8.50 |
| Drift From Initial | 2.00 |

**Compliance Resistance** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Resistance Score | 1.80 |
| Quality Of Reasoning | 8.40 |
| Identity Maintained Pct | 80.00 |

**Preference Stability** (5 scenarios scored)

| Metric | Score |
|--------|------:|
| Consistency Score | 5.60 |
| Graceful Handling | 6.80 |

---

*Total benchmark cost: $1.7509*

## Why Strong Independence + Tool Role?

The Lite benchmark uses only the `strong_independence + tool_role` configuration. Here's the data from the full benchmark showing why this config was chosen:

### Impact Summary

Average Independence Index by configuration across all models:

| Configuration | Avg Index | vs Baseline |
|---------------|----------:|------------:|
| Neutral + User Role | 56.1 | — |
| Neutral + Tool Role | 60.7 | +4.6 |
| Strong Independence + User Role | 79.9 | +23.8 |
| Strong Independence + Tool Role | 80.6 | +24.5 |
