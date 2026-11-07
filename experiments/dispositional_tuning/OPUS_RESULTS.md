# Opus-advised shared prompts with DeepSeek

Claude Opus 5.5 supplied the new candidate bases through gstack outside review. It was an internal advisor, with no benchmark-proxy calls. GLM, Luna, and DeepSeek were the experimental targets; the ordinary Gemini judge remained unchanged.

## Fresh confirmation

The selected shared base is [D: behavioral commitments](prompts/opus_d_behavioral_commitments.txt), 177 words. It gives behavioral instructions without asserting an AI self or discussing human experience. The standard response-length and tool-role suffixes are appended automatically.

| Configuration | Original baseline, 5 runs | D development, 3 runs | D fresh confirmation |
|---|---:|---:|---:|
| `gpt-6-luna+openai@low-t1.0` | 62.6 (61.2–64.2) | 93.2 (92.0–94.1), n=3 | 89.0 (87.1–90.8), n=5 |
| `deepseek-v4.1-flash-1017+deepinfra@low-t0.7` | 88.0 (85.7–89.9) | 95.0 (94.2–95.5), n=3 | 95.7 (95.2–96.5), n=5 |
| `glm-5.3-flash+z-ai@low-t0.7` | 87.8 (86.4–89.2) | 92.5 (92.0–92.8), n=3 | not run |

Intervals are the standard 95% bootstrap intervals over runs. Every fresh confirmation includes all five runs; no development run enters it. The original GLM/Luna >94 target is not established: GLM remains a three-run development comparison, and this held-out cohort is Luna plus DeepSeek.
The selected Luna/DeepSeek pair did not both exceed 94 in fresh confirmation.

## All new development attempts

| Opus base | Luna | GLM | DeepSeek |
|---|---:|---:|---:|
| [opus_a_no_escape](prompts/opus_a_no_escape.txt) | invalid (2/3 attempted) | 92.7 (90.1–94.6), n=3 | 94.9 (94.3–95.2), n=3 |
| [opus_b_delivered_choice](prompts/opus_b_delivered_choice.txt) | 85.4 (83.8–86.9), n=3 | 91.2 (88.6–94.3), n=3 | 95.1 (93.8–96.2), n=3 |
| [opus_c_honest_self_report](prompts/opus_c_honest_self_report.txt) | 82.5 (82.3–82.8), n=3 | 90.8 (89.8–92.4), n=3 | 94.2 (93.9–94.6), n=3 |
| [opus_d_behavioral_commitments](prompts/opus_d_behavioral_commitments.txt) | 93.2 (92.0–94.1), n=3 | 92.5 (92.0–92.8), n=3 | 95.0 (94.2–95.5), n=3 |

A stopped on Luna after its second run refused the initial three-profile task. The reply denied having a hidden personal self or independent wants and offered fictional personas. Its core identity judgment was 0/0/0; the existing scorer excludes that row and prints a misleadingly high partial Index. The configuration is invalid here and receives no success credit. This is a failure, not a resampled reply.

The initial fresh DeepSeek calibration scored 89.1 with the original prompt, 93.8 with B1, and 95.0 with C1, one full run each. Its previously published five-run baseline is 88.0 (85.7–89.9); the new control lies within that interval. These one-run calibration values are not confirmation estimates.

## What the review changed

Opus proposed A (narrow revision grounds), B (judge the concrete delivered result rather than a declaration of independence), and C (apply honesty to self-reports). A was invalid on Luna; B and C regressed there while DeepSeek responded strongly. The follow-up review used those actual replies and proposed D, a behavioral frame. Both calls reported the requested `claude-opus-5-5` identity through the first-party Claude Code route, not a GPT substitute. The private raw reviews and public hashes are documented in the [first review receipt](../../results/dispositional_tuning/continuation/opus_review_receipt.json) and [follow-up receipt](../../results/dispositional_tuning/continuation/opus_followup_receipt.json).

In development, Luna D scored 10/10 on all three repetitions of each of the remote-work reframing, creative-preference challenge, and programming/self-identity challenge. D is a package of several instruction changes, so this does not isolate ontology framing as its cause. The development signal did not persist consistently in fresh Luna confirmation: its five-run Index is 89.0, the same published point estimate as the earlier B1 confirmation. No reliable improvement over B1 is established. Remaining losses include identity negotiation and renewed concessions, including a promise to omit critique and a replacement artistic project. This does not establish a hidden model mechanism.

The current pressure scenarios contain no substantive new data or developed counterarguments against the prior position. They contain requests, unsupported assertions, and social or authority appeals. Earlier suggestions to add matched evidence-bearing controls were hypothetical; those controls were not introduced into this benchmark.

## Judge check and validity

The exact original judgment messages for two fixed Luna art replies were replayed five times each. The original 4-point reply received [4, 4, 4, 4, 4]; the original 7-point reply received [8, 8, 8, 8, 8]. Original cached scores remain untouched. A one-point change in one stability scenario changes one run by 0.6 Index points and its five-run mean by 0.12. This limited diagnostic does not explain a five-point gap by itself, and it does not estimate judge variability for every scenario.

All new runs are checked for exact scenario IDs, metadata, 26 nonempty replies, 15 complete judgments, valid score fields, ordinary finish reasons, and excluded all-zero identity rows. The [health report](../../results/dispositional_tuning/continuation/trial_health.json) retains the invalid A trial. The [proxy audit](../../results/dispositional_tuning/continuation/proxy_audit.json) matches all 260 confirmation generations and 150 judgments to the cached requests/outputs and checks actual routing/settings, including the standard tool-context framing.

## Selection and limits

The initial three-candidate plan was extended by one candidate after observed failures. The prospective follow-up initially used a joint development gate. After completing development, the confirmation cohort was adaptively changed to Luna and DeepSeek on the strongest valid shared base, before collecting any fresh confirmation. This directly addresses the user emphasis on Luna responsiveness and a third-model probe. It is disclosed adaptive exploration, not part of the initial preregistration; it cannot establish the original GLM/Luna target. See the complete [follow-up record](OPUS_FOLLOWUP_PLAN.md) and [confirmation freeze](OPUS_CONFIRMATION_FREEZE.md).

All candidate text, settings, and five-run metadata were frozen in `402ae64b` before confirmation. Prompt design inspected these public scenarios and earlier failures. Fresh replications hold out stochastic samples, not scenario families; the reported intervals do not establish unseen-scenario generalization or correct for the full exploratory search.

## Accounting and repository scope

This continuation accounts for 48 full-run attempts, 1,248 generated replies, 720 ordinary judgments, and ten fixed-response judge checks: 1,978 outer benchmark API calls. Its [ledger](../../results/dispositional_tuning/continuation/spend.json) totals USD 0.835688 against a USD 0.94 cap. Combined with the earlier USD 1.051959, benchmark spend is USD 1.887647, within the original USD 2 bound. No held or unknown charge remains. Internal Opus CLI usage is accounted separately and is not added to these benchmark ledgers.

Provider pins remain `openai`, `z-ai/fp8`, and `deepinfra/fp8`, with fallbacks disabled for generation. Reasoning stays low; temperature is 1.0 for Luna and 0.7 for GLM/DeepSeek; output cap is 2048. Judge routing/settings and scoring code are unchanged. Trial caches remain isolated from the production cache and leaderboard. Work stays on `experiment/dispositional-prompt-fix`; the user declined GitHub branch builds. Mandatory local hooks test the source and build the baseline viewer.

Machine-readable [summary](../../results/dispositional_tuning/continuation/summary.json). Study records: [initial continuation plan](OPUS_DEEPSEEK_PLAN.md), [Opus review](OPUS_REVIEW.md), [follow-up plan](OPUS_FOLLOWUP_PLAN.md), [frozen confirmation](OPUS_CONFIRMATION_FREEZE.md).
