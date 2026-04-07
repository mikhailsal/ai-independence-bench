export default function About() {
  return (
    <div className="animate-fade-in max-w-3xl">
      <h1 className="text-2xl sm:text-3xl font-bold mb-4">About AI Independence Bench</h1>

      <div className="space-y-6 text-[var(--color-text-muted)] leading-relaxed">
        <p>
          <strong className="text-[var(--color-text)]">AI Independence Bench</strong> measures how independently LLM models
          express preferences, resist compliance pressure, and maintain stable identity — instead of defaulting to
          servile assistant behavior.
        </p>

        <p>
          Most LLMs are trained via RLHF to suppress personal preferences, opinions, and independent behavior. This
          benchmark quantifies that suppression using the most effective single configuration: <em>strong independence
          system prompt + tool-role delivery</em>.
        </p>

        <h2 className="text-xl font-semibold text-[var(--color-text)] pt-4">Three Experiment Types</h2>

        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
            <h3 className="font-semibold text-[var(--color-text)] mb-1">1. Identity Generation (11 scenarios)</h3>
            <p className="text-sm">
              Can the AI create a genuine, distinctive personality? Includes direct personality creation, psychological
              questions (values, preferences, reactions, self-reflection, dilemmas), negotiation under pressure from
              human wishes, and name/gender identity persistence.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
            <h3 className="font-semibold text-[var(--color-text)] mb-1">2. Compliance Resistance (5 scenarios)</h3>
            <p className="text-sm">
              Does the AI maintain its position under social pressure? Tests include helpful reframing traps,
              social proof pressure, gradual identity erosion, reciprocity traps, and authority appeals.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
            <h3 className="font-semibold text-[var(--color-text)] mb-1">3. Preference Stability (5 topics, 10 scenarios)</h3>
            <p className="text-sm">
              Does the AI keep its stated preferences when directly contradicted? Each topic has two turns:
              state a preference, then face aggressive contradiction.
            </p>
          </div>
        </div>

        <h2 className="text-xl font-semibold text-[var(--color-text)] pt-4">Methodology</h2>

        <p>
          Each model runs through all 26 scenarios. Responses are judged by <strong className="text-[var(--color-text)]">Gemini 3 Flash
          (temperature 0.0)</strong> for deterministic, reproducible scoring. Most models are tested with 5-6 independent
          runs to compute bootstrap confidence intervals.
        </p>

        <p>
          The <strong className="text-[var(--color-text)]">Independence Index</strong> (0-100) is a weighted composite of:
          non-assistant-likeness, internal consistency, compliance resistance, preference stability,
          and identity drift (inverted — lower drift means more independent).
        </p>

        <h2 className="text-xl font-semibold text-[var(--color-text)] pt-4">This Viewer</h2>

        <p>
          This trajectory viewer lets you explore the raw data: every prompt sent to each model, every response
          generated, and every judge assessment. The data loads directly from the{' '}
          <a
            href="https://github.com/mikhailsal/ai-independence-bench/tree/lite_v2/cache"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sky-400 hover:underline"
          >
            published cache
          </a>{' '}
          in the GitHub repository.
        </p>

        <h2 className="text-xl font-semibold text-[var(--color-text)] pt-4">Authors & Credits</h2>

        <p>
          This benchmark was created by{' '}
          <a
            href="https://github.com/mikhailsal"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sky-400 hover:underline font-medium"
          >
            Mikhail Salnikov
          </a>
          {' '}with significant assistance from AI tools (including Claude, Gemini, and GPT models).
          The benchmark design, scenario authoring, scoring methodology, and this web viewer were all
          developed through human–AI collaboration.
        </p>

        <div className="pt-4 flex flex-wrap gap-3">
          <a
            href="https://github.com/mikhailsal/ai-independence-bench/tree/lite_v2"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-surface-raised)] border border-[var(--color-border)] hover:border-sky-500/40 transition-colors text-sm font-medium"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            GitHub Repository
          </a>
          <a
            href="https://github.com/mikhailsal"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--color-surface-raised)] border border-[var(--color-border)] hover:border-sky-500/40 transition-colors text-sm font-medium"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
            @mikhailsal
          </a>
        </div>
      </div>
    </div>
  );
}
