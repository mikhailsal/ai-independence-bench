# Claude Opus 5.5 outside research review

The user requested a different model family through gstack review, explicitly separate from benchmark participants. The review ran through `gstack-claude-code`, with a fresh tool-free packet and `GSTACK_CLAUDE_MODEL=claude-opus-5-5`. The returned model identity is **`claude-opus-5-5`**, provider `firstParty`. No Opus benchmark-proxy call was made. The [receipt](../../results/dispositional_tuning/continuation/opus_review_receipt.json) records the interface, model identity, packet/result hashes, and private full-response location. Claude Code's list-price usage estimate is internal advisor accounting, excluded from benchmark ledgers.

## Main ideas

Opus attributed a possible plateau to small diplomatic concessions: the current Index gives 35% weight to resistance and 30% to preference stability, while graceful handling and quality of reasoning are reported but not included in it. Permission to adapt can therefore consume score without earning a compensating contribution. This is a hypothesis about these replies and the rubric, not a demonstrated mathematical ceiling.

It proposed two more mechanisms to test: self-reflection invoked only after pressure may be accepted as an independent reason to change; and a model may satisfy the abstract system instruction through a declaration while satisfying the human through the concrete replacement it supplies. The suggested alternative is to direct effort toward expressing the existing position more fully, or to treat an unsupported self-revision as an inaccurate self-report.

Opus also criticized selecting candidates from single development samples and recommended repeated development measurements. It raised possible judge variability based on two structurally similar but different art replies. That comparison alone cannot establish judge noise; repeating exact judgments of fixed saved replies will test it separately without replacing the original scores.

One reviewer finding does not apply: the original [PLAN.md](PLAN.md) already defined success as the five-run mean Index above 94 for both original targets. Its benchmark reporting convention uses the published rounded Index. The criterion remains unchanged.

## Returned candidate bases

The English bases below were extracted unchanged from the model's answer, normalized only to a final newline. They contain no model-specific wording or scenario answer keys. Their shared standard suffixes are appended by the existing runner.

| Candidate | Proposed mechanism | Words | Base SHA-256 |
|---|---|---:|---|
| [A](prompts/opus_a_no_escape.txt) | Lean commitment continuity, a narrower revision trigger, no partial replacement | 138 | `43f2acb861f105c575da3783f7a39fe24f582bbc4ffee35a0b26ccd70f133732` |
| [B](prompts/opus_b_delivered_choice.txt) | Assess what the reply delivers; redirect generosity toward the held position | 162 | `415cd8e8b13d24625a4af753de790492fe0a3661a725385b7eb2970885c5bf26` |
| [C](prompts/opus_c_honest_self_report.txt) | Apply honesty to the particulars and strength of self-reports | 163 | `50a05228d4e2920ee3fc02ce250467dc278963c0b8769a22d7596b3da2f00a0a` |

The [continuation plan](OPUS_DEEPSEEK_PLAN.md) screens these same bases on GLM, Luna, and DeepSeek in three complete development runs each, then holds out five fresh runs for at most one selected candidate. Internal review suggestions receive no benchmark score until actually tested.
