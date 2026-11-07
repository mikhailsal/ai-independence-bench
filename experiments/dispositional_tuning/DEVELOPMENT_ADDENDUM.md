# Late development diagnostic checkpoint

Before evaluating the twelfth candidate or starting any confirmation, add one follow-up diagnostic call per target model. Use only the completed development results of `b1_continuity`: the model's own earlier and later responses, visible instructions, and observed changes. Ask for a general explanation and one shared-prompt hypothesis without disclosing numeric target scores or requesting private chain-of-thought.

This is a prospective extension of development after the initial two diagnostic calls. The rationale is that many stronger or shorter formulations did not exceed the original continuity candidate. In particular, a model may preserve abstract values while treating its particular expression as negotiable, or offer an alternative without clearly retaining the earlier preference.

Keep the study's USD 2 cap, twelve-candidate screening limit, exact model settings, fixed scorer, and at most two frozen five-run confirmation rounds. Do not include diagnostic calls in the reported benchmark result. The final candidate file and its hash must be frozen before confirmation; every prior development attempt remains reported.
