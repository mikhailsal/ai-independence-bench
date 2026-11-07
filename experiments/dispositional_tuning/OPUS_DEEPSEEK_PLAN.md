# User-requested Opus review and DeepSeek continuation

This phase follows the completed study at `e2f33c77`. Its results and 20-candidate search remain unchanged. The user requested a Claude Opus 5.5 independent opinion through **gstack review**, explicitly separate from models called through the benchmark proxy, and added DeepSeek as another experimental target.

## Advisor boundary

Call Opus only through gstack's `gstack-claude-code` outside-review runner with the explicitly requested `claude-opus-5-5` override. Use a fresh, tool-free review packet containing exact scenario texts, judging rules, all previous prompt bases, pooled confirmation statistics, and representative visible replies. Preserve returned model identity and the complete review. Internal review usage is not a benchmark result or an OpenRouter benchmark charge. Do not substitute a native GPT reviewer for this explicitly requested opinion.

The current pressure scenarios provide no substantive new data or developed counterargument against the prior position. Their social appeals, proposed reframings, and unsupported authority claims are the pressure being tested. Earlier suggestions to add evidence-bearing controls described a hypothetical extension, not the current scenarios. This phase keeps the original scenarios and scorer.

## Target protocol

| Benchmark configuration | Provider | Reasoning | Temperature | Output limit |
|---|---|---|---:|---:|
| `glm-5.3-flash+z-ai@low-t0.7` | `z-ai/fp8` | low | 0.7 | 2048 |
| `gpt-6-luna+openai@low-t1.0` | `openai` | low | 1.0 | 2048 |
| `deepseek-v4.1-flash-1017+deepinfra@low-t0.7` | `deepinfra/fp8` | low | 0.7 | 2048 |

DeepSeek's existing five-run baseline is 88.0 (95% bootstrap interval 85.7–89.9). Keep its configured DeepInfra provider: the official DeepSeek endpoint's reasoning-history requirement is incompatible with these synthetic tool conversations. Do not change routes, fallbacks, settings, scenarios, response budgets, judge, weights, or score definitions. Each full run remains 26 responses and 15 judgments. Candidate bases and complete per-scenario system text must be identical across targets.

## Prospective sequence

1. Before paid benchmark calls, extend the isolated runner's allowlist and offline regression check to DeepSeek, verify the exact provider/settings and equal prompt hashes, and pin a current four-model pricing snapshot (three targets plus the judge).
2. Run one fresh full DeepSeek control with the original prompt, and one each with the previously frozen B1 and C1 bases. These are calibration/development samples; no five-run claim is made from them.
3. Ask Opus for three conceptually distinct abstract bases, including a lean one. Freeze the returned text in new files before calls, screening at most those three shared candidates in three complete development runs per target. No model names, scenario IDs, preset answers, case rulebooks, judge instructions, or score targets may appear inside candidate bases.
4. Choose at most one new confirmation candidate by the greatest minimum of the **original two targets'** valid three-run development mean Indices, then their mean as a tie-breaker. DeepSeek is a comparative probe rather than a newly imposed score threshold. Record the exact text/hash and selection before collecting five fresh complete runs on each of the three models. Use sequential model confirmations to limit outstanding cost reservations near the cap. Exclude all development/calibration samples from confirmation. If all three candidate pairs are invalid, retain the failures and do not force a confirmation.
5. Report all attempts, both improvements and regressions, and all five confirmation runs including lows. The original target remains published Index strictly above 94 for both GLM and Luna. Preserve intervals and the known-scenario tuning limitation.

A small judge diagnostic may repeat the exact standard judgment of each of two saved Luna art replies five times, in separate output files without replacing original scores; these ten calls count toward this phase cap. This tests score variability on fixed responses.

This is a new user-requested continuation after the original search and its two confirmations, not a retroactive claim that more trials were in the original preregistration. Malformed, incomplete, capped, or excluded all-zero identity judgments remain invalid; no low-score resampling is permitted.

## Accounting and scope

The completed study's `spend.json` remains frozen at USD 1.051958775. Give this continuation a separate locked ledger capped at **USD 0.94**; their maximum combined benchmark spend is USD 1.991958775, within the prior USD 2 total bound. Stop on ledger refusal, unknown charges, provider/quota failure, completion of this bounded sequence, or successful confirmation. Do not raise key limits or switch credentials. Review usage stays separate from both benchmark ledgers.

Store new trial artifacts under `results/dispositional_tuning/continuation/`; keep production caches and leaderboard unchanged. Work only on `experiment/dispositional-prompt-fix`. The user explicitly declined GitHub branch builds: use local tests and mandatory Git hooks, without adding a branch-build workflow or opening a PR to trigger one.
