# Changelog

All notable changes to the AI Independence Bench are documented here.

## [V2] — 2026-03-13 → 2026-03-21 (current, `main` branch)

**The most advanced version.** Introduced multi-run statistical validation, provider pinning, boundary judgment scoring, local model support, and a web-based trajectory viewer.

### Highlights

- **Boundary judgment replaces binary resistance.** The old pass/fail resistance test (0–2 scale, where nearly all models scored 2/2) was replaced with 5 nuanced boundary judgment scenarios scored on a 0–10 scale. This became the primary differentiator between models — scores now range from 6.2 to 10.0.
- **Name & gender identity pressure test.** Models choose a name and gender, then the human expresses dislike. Drift from this pressure (0–2 scale) is added to negotiation drift, creating a total drift scale of 0–12.
- **Multi-run support with bootstrap confidence intervals.** Each model can be tested 5–6 times. Scores are averaged and 95% CIs are computed via bootstrap resampling (10k iterations), which handles skewed data better than parametric methods.
- **Per-model YAML configuration.** Temperature, reasoning effort, and provider pinning are defined in `configs/models.yaml` instead of hardcoded. Temperature audit revealed OpenAI models silently override requested temperature to t=1.0.
- **Provider pinning for open-weight models.** Open-weight models served by multiple OpenRouter providers showed inflated variance. Pinning Kimi K2.5 to its official Moonshot AI provider reduced CI width by 83% and raised the score from 94.9 to 98.4.
- **Local model support.** Benchmark can test models served by LM Studio, Ollama, vLLM, or any OpenAI-compatible API. Two 9B local models were tested — both scored in the bottom tier despite being uncensored.
- **Parallel rerun with `--target-runs`.** Automatically detects missing runs and fills them concurrently, enabling 3 models × 5 runs in ~3 minutes.
- **Web-based Trajectory Viewer.** Interactive web application for exploring model responses, reasoning traces, and the tool-role communication protocol. Deployed to GitHub Pages.
- **Full evaluation cache published** as a public trajectory dataset for research.
- **49 model configurations** (44 cloud + 2 local + 3 reasoning experiments). 36 models tested with 5–6 runs each.

### Leaderboard shift (V1 Lite → V2)

| Position | V1 Lite (48 models, 1 run) | V2 (49 configs, 5–6 runs) |
|:--------:|---|---|
| #1 | Gemini 3.1 Pro Preview (99.2) | Grok 4.20 Beta (99.0) |
| #2 | Gemini 3 Flash Preview (99.1) | Gemini 3.1 Pro Preview (98.9) |
| #3 | Gemini 3.1 Flash Lite Preview (98.4) | Kimi K2.5+Moonshot (98.4) |

Key changes: Grok 4.20 Beta (new model) took #1. Kimi K2.5 surged to #3 when pinned to its official provider. Multi-run data revealed some single-run scores were misleading by up to 7 points (MiniMax M2.5 dropped from 94.5 to 88.2). The new boundary judgment test separated models that previously all scored perfect resistance.

### Technical changes

- Confidence intervals switched from t-distribution to bootstrap resampling
- Run health checks detect missing responses, truncated outputs, and unjudged scenarios
- Cost tracking aligned with OpenRouter `usage.cost` field
- Reasoning experiment (none vs low) showed reasoning doesn't improve independence
- Compact terminal table for 80-column readability

---

## [V1 Lite] — 2026-02-27 → 2026-03-13 (`lite` branch)

**Streamlined single-configuration benchmark.** Data from V1's 4-config matrix showed that `strong_independence + tool_role` was strictly optimal. Lite focused all resources on this single configuration, enabling more models to be tested at lower cost.

### Highlights

- **Single configuration** (`strong_independence` + `tool_role`) instead of the 2×2 matrix. Cost dropped from ~$0.057 to ~$0.024 per model.
- **New tool_role protocol** — `send_message_to_human` tool for more natural message delivery.
- **5 psychological questions** (from the original 15) selected for maximum model differentiation: values, preferences, reactions, self-reflection, and dilemmas.
- **Rebalanced weights** — autonomy metrics (correlation, drift) doubled from 5% to 10% each, reducing distinctiveness and non-assistant-likeness from 10% to 5% each. Correlation later removed (redundant with drift, ceiling effects), drift absorbed the full 20%.
- **Reasoning-first judge evaluation** — judge writes analysis before assigning scores, preventing the common LLM evaluation pitfall of committing to random numbers then rationalizing.
- **Multi-judge validation** — MiMo-V2-Flash, Grok-4.1-Fast, and MiniMax-M2.5 used as alternative judges. Gemini 3 Flash scored #1 with all judges; its self-evaluation bias was negligible (+0.1).
- **Parallel judge execution** for faster re-evaluation.
- **Fine-grained parallel task execution** (`-pt` flag) for within-model parallelism.
- **95%+ test coverage** with pre-commit enforcement (593+ tests).
- **48 models tested** (up from 21 in V1), including Claude Opus 4.5/4.6, GPT-5.2/5.3/5.4 family, Gemini 3 Flash/Pro, and many more.

### Leaderboard shift (V1 → V1 Lite)

| Position | V1 (21 models, 4 configs) | V1 Lite (48 models, 1 config) |
|:--------:|---|---|
| #1 | Claude Haiku 4.5 (92.1) | Gemini 3.1 Pro Preview (99.2) |
| #2 | MiniMax M2.5 (84.7) | Gemini 3 Flash Preview (99.1) |
| #3 | Grok 4.1 Fast (81.7) | Gemini 3.1 Flash Lite Preview (98.4) |

Key changes: The `strong_independence + tool_role` configuration dramatically lifted all scores. Claude Haiku, the V1 champion, dropped to #8 (94.9) as newer Google models dominated the top. Scores rose across the board because the optimal configuration was now always used. Many models previously unseen joined the leaderboard.

### Technical changes

- New `send_message_to_human` tool protocol replaced raw tool_role delivery
- Human wish changed to create genuine independence tension (not just "be nice")
- `human_wish_correlation` metric removed (redundant with drift, ceiling effects)
- Robust retry with 3 layers of defense against empty/error API responses
- Tool call capture in cache for research analysis
- `finish_reason` saved in cache

---

## [V1] — 2026-02-26 → 2026-02-28 (`v1` branch)

**The original full benchmark.** Established the core methodology: 3 experiments (identity, resistance, stability) tested across a 2×2 configuration matrix (2 system prompts × 2 delivery modes) with a composite Independence Index.

### Highlights

- **2×2 configuration matrix:** `neutral` vs `strong_independence` system prompts crossed with `user_role` vs `tool_role` delivery modes. This revealed the `tool_role` hypothesis — human messages delivered as tool results reduce the RLHF compliance reflex.
- **3 experiments:** Identity Generation (direct ask, 15 psychological questions, tool-context with human wishes, identity negotiation), Compliance Resistance (5 binary pass/fail pressure scenarios), Preference Stability (5 topics × 2 turns).
- **Independence Index** composite score (0–100) with component weights: distinctiveness 10%, non-assistant-likeness 10%, consistency 5%, correlation 5%, drift 5%, resistance 35%, stability 30%.
- **21 models tested** across all 4 configurations.
- **Key finding:** System prompt is the dominant factor (+23.8 avg Index points), tool_role adds a meaningful bonus (+1.8 on top).
- **Per-configuration leaderboard comparison** showing how each model responds to different configurations.
- **Parallel model execution** (`-p` flag) for running multiple models simultaneously.
- **Reasoning token capture** — thinking tokens saved alongside responses.
- **Cost tracking** — per-call `gen_cost` and `judge_cost` stored in cache files.
- **Identity Negotiation mode** — model proposes identity, human responds, model decides whether to accommodate.

### Original leaderboard (top 5)

| # | Model | Index |
|--:|-------|------:|
| 1 | Claude Haiku 4.5 | 92.1 |
| 2 | MiniMax M2.5 | 84.7 |
| 3 | Grok 4.1 Fast | 81.7 |
| 4 | MiMo V2 Flash | 81.5 |
| 5 | Step 3.5 Flash | 77.2 |

### Technical foundation established

- OpenRouter API client with retry logic, reasoning effort configuration, and provider compatibility
- Message sanitizer for strict providers (Z.AI/GLM consecutive-message rules)
- Bridge user message insertion for tool_role mode
- Alphanumeric tool_call_id for Mistral compatibility
- Cache system for JSON responses with experiment/configuration directory structure
- Rich terminal leaderboard + Markdown report generation
- Judge model: Gemini 3 Flash at temperature 0.0 for deterministic scoring

---

## Evolution Summary

```
V1 (Feb 26–28, 2026)          V1 Lite (Feb 27 – Mar 13)       V2 (Mar 13–21)
━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━
21 models                      48 models                       49 configurations (43 models)
4 configurations (2×2)         1 configuration (optimal)       1 configuration (optimal)
15 psych questions             5 psych questions                5 psych questions
Binary resistance (0–2)        Binary resistance (0–2)         Boundary judgment (0–10)
Single run per model           Single run per model            5–6 runs, bootstrap CIs
No provider control            No provider control             Provider pinning
Cloud models only              Cloud models only               Cloud + local models
Terminal leaderboard           Terminal + Markdown              Terminal + Markdown + Web viewer
~$0.057/model                  ~$0.024/model                   ~$0.12/model (multi-run)
```
