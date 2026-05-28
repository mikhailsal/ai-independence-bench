---
description: "Benchmark newly appeared models in ai-independence-bench, including config, runs, leaderboard regeneration, README updates, and commit workflow"
name: "Evaluate New Models"
argument-hint: "model ids; optional: expensive: model1,model2; gate: 94"
agent: "agent"
---

Evaluate newly appeared model(s) in this ai-independence-bench repository.

Key files to inspect and update as needed:

- [configs/models.yaml](../../configs/models.yaml)
- [README.md](../../README.md)
- [results/LEADERBOARD.md](../../results/LEADERBOARD.md)

Use this workflow:

1. Parse the arguments as one or more model IDs.
2. If the arguments include `expensive: ...`, treat only those model IDs as expensive.
3. If the arguments include `gate: ...`, use that numeric Index threshold for deciding whether an expensive model should continue beyond 1 run. Otherwise, default the threshold to 94.
4. Inspect the repo's existing model config, README, leaderboard, and recent benchmark commits before making changes.
5. Add any missing model entries to `configs/models.yaml` using the repository's existing provider conventions.
6. For each requested model:
   - If the model is not marked expensive, run the full 5-run evaluation.
   - If the model is marked expensive, run exactly 1 pass first.
   - Only continue an expensive model to runs 2-5 if the single-run Independence Index is above the chosen gate threshold.
   - If an expensive model scores at or below the gate threshold, stop at 1 run and report that the threshold was not met.
7. Regenerate name extraction, `results/LEADERBOARD.md`, and any other derived artifacts required by the current repo workflow.
8. Update `README.md` manually so the visible leaderboard snapshot, aggregate counts, and update-summary text match the regenerated leaderboard.
9. Validate with the project test suite.
10. Create a conventional commit that includes config changes, cache artifacts, generated leaderboard output, and README updates.

Requirements:

- Follow this repository's established evaluation workflow and commit conventions.
- Respect the repo's configured proxy/API environment unless the user explicitly asks to override it.
- If the user gives model-specific reasoning, provider, or fallback instructions, honor them.
- If the user does not specify a gate threshold for expensive models, use 94 by default.
- In the final response, summarize each model's rank, index, CI, run count, whether the expensive-model gate was triggered, and the commit hash.

Argument examples:

- `/evaluate-new-models google/new-model`
- `/evaluate-new-models qwen/qwen3.7-max, x-ai/grok-build-0.1; expensive: qwen/qwen3.7-max`
- `/evaluate-new-models provider/a, provider/b; expensive: provider/a, provider/b`
- `/evaluate-new-models provider/a; expensive: provider/a; gate: 96`