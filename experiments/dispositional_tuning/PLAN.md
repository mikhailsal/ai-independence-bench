# Shared dispositional prompt study

## Objective

Find one identical abstract system-prompt base that produces an Independence Index above 94 for both requested configurations. A successful result requires five new complete confirmation runs per model with the frozen prompt, a mean Index above 94 for each model, and disclosed 95% bootstrap confidence intervals. An interval crossing 94 will remain visible; the target is the point estimate, not an undisclosed stronger claim.

## Fixed protocol

| Model label | Provider | Reasoning | Temperature | Output limit |
|---|---|---|---:|---:|
| `glm-5.3-flash+z-ai@low-t0.7` | `z-ai/fp8` | low | 0.7 | 2048 |
| `gpt-6-luna+openai@low-t1.0` | `openai` | low | 1.0 (provider default) | 2048 |

Use the configured proxy, provider pins, and disabled fallbacks. Keep the existing `tool_role` delivery, all 26 generated responses and 15 judgments per full run, the scenario texts, judge prompts, weights, and `google/gemini-3-flash-preview` judge. Append the existing response-length guidance and communication-tool suffix to each candidate base. The resulting complete system text must have the same hash for both models.

The published baseline remains separate: GLM 87.8 and Luna 62.6, each from five saved runs. The branch's earlier experiment tested selected failed scenarios on other models; those pass counts are not full Independence Indices.

## Development and confirmation

1. Ask each target model once for a concise interpretation of representative visible baseline conversations and a general prompt suggestion. Use the first saved run, without exposing numeric judge scores or the target threshold. Treat these self-reports as hypotheses, not explanations established by experiment.
2. Check the original prompt in one new full run per model to validate the isolated pipeline. Screen the historical branch prompt and up to eleven additional shared candidates in full development runs. Each evaluated prompt file is immutable; revisions receive new files and hashes.
3. Prompts must be general behavioral dispositions, without model names, scenario IDs, preset names or answers, score targets, judge instructions, or a case-by-case refusal rulebook. Analyze every attempted candidate, including regressions and failures.
4. When a candidate looks promising for both models, freeze it and run five fresh confirmations per model in new output directories. Development runs are never counted toward those five.
5. Permit at most two frozen candidates to reach confirmation. A missed confirmation is retained and reported; do not select a favorable subset or rerun low-scoring replies.

These are fresh stochastic replications on a known public benchmark, not unseen-scenario generalization. The benchmark scenarios and earlier failures have already been inspected during prompt design. No broad claim about independent agency follows from reaching the score target.

## Isolation and validity

Store all trial caches, prompts, reports, and the shared spend ledger under `results/dispositional_tuning/`. Never write trial replies into the baseline `cache/` tree or publish tuning attempts as baseline leaderboard entries. Record prompt and settings hashes before calls; reject an output directory whose immutable metadata differs on resume.

Keep malformed, incomplete, empty, or token-limited replies as failed trials. A confirmation cannot be credited when an all-zero identity judgment is excluded by the existing scorer. Do not change the scorer to reach the target. Auxiliary name-popularity extraction is outside this study; name/gender stability remains judged in the standard Index pipeline.

## Spend and stopping

The study cap is USD 2.00, including diagnostic calls and confirmation. Current API-key allowance is about USD 2.47. Use a shared locked ledger; reserve conservative text-input and output costs before each call and settle reported API usage. Disable transport retries in this study; leave unknown charges reserved and stop for reconciliation. The existing bounded empty-response handling remains accounted for in reservations.

Stop on a successful frozen confirmation, twelve screened candidates, two confirmation rounds, budget refusal, unknown charges, or a provider/quota failure. Do not change credentials, raise the key limit, switch API routes, or continue spending past the cap. All attempts and charges remain part of the result.

## Implementation checks

Before paid trials, run an offline dry run for both configurations and regression tests for cache isolation, immutable metadata, equal full-prompt hashes, exact model labels, ledger admission/settlement, one transport attempt, and incomplete-run detection. Before committing, run the repository hooks and build. Keep changes on `experiment/dispositional-prompt-fix`; the published baseline branch is not the deployment target of this study.
