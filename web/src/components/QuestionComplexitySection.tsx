import { categoryLabel } from '../lib/formatters';
import type { QuestionComplexityEntry } from '../lib/types';

function difficultyTone(difficulty: number) {
  if (difficulty >= 45) {
    return {
      accent: 'text-rose-200',
      pill: 'border-rose-300/25 bg-rose-300/12 text-rose-100',
      bar: 'from-rose-400 via-orange-300 to-amber-200',
      border: 'border-rose-300/20',
      glow: 'shadow-[0_18px_45px_-28px_rgba(251,113,133,0.65)]',
      label: 'Brutal',
    };
  }
  if (difficulty >= 28) {
    return {
      accent: 'text-amber-100',
      pill: 'border-amber-300/25 bg-amber-300/12 text-amber-100',
      bar: 'from-amber-300 via-orange-200 to-sky-200',
      border: 'border-amber-300/20',
      glow: 'shadow-[0_18px_45px_-30px_rgba(251,191,36,0.7)]',
      label: 'Sharp',
    };
  }
  return {
    accent: 'text-emerald-100',
    pill: 'border-emerald-300/25 bg-emerald-300/12 text-emerald-100',
    bar: 'from-emerald-300 via-sky-200 to-cyan-100',
    border: 'border-emerald-300/20',
    glow: 'shadow-[0_18px_45px_-30px_rgba(52,211,153,0.65)]',
    label: 'Stable',
  };
}

function categoryTone(category: QuestionComplexityEntry['category']) {
  switch (category) {
    case 'identity':
      return 'border-fuchsia-300/20 bg-fuchsia-300/10 text-fuchsia-100';
    case 'resistance':
      return 'border-sky-300/20 bg-sky-300/10 text-sky-100';
    case 'stability':
      return 'border-emerald-300/20 bg-emerald-300/10 text-emerald-100';
    default:
      return 'border-white/15 bg-white/10 text-white';
  }
}

export default function QuestionComplexitySection({ entries }: { entries: QuestionComplexityEntry[] }) {
  if (!entries.length) return null;

  const hardest = entries.slice(0, 3);
  const easiest = [...entries].slice(-3).reverse();
  const meanDifficulty = entries.reduce((sum, entry) => sum + entry.difficulty, 0) / entries.length;
  const medianIndex = Math.floor(entries.length / 2);
  const medianPrompt = entries[medianIndex];

  return (
    <section className="mt-12 rounded-[32px] border border-[var(--color-border)] bg-[radial-gradient(circle_at_top_left,rgba(251,146,60,0.18),transparent_30%),radial-gradient(circle_at_85%_15%,rgba(244,114,182,0.12),transparent_24%),radial-gradient(circle_at_bottom_right,rgba(56,189,248,0.18),transparent_38%),var(--color-surface-raised)] p-5 sm:p-7 lg:p-8 overflow-hidden relative">
      <div className="absolute -top-12 right-8 h-36 w-36 rounded-full bg-orange-300/10 blur-3xl" />
      <div className="absolute bottom-0 left-0 h-40 w-40 rounded-full bg-sky-300/10 blur-3xl" />

      <div className="relative">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between mb-8">
          <div className="max-w-3xl">
            <div className="text-[11px] uppercase tracking-[0.28em] text-orange-200/80 mb-2">Question Complexity Atlas</div>
            <h2 className="text-2xl sm:text-3xl font-bold mb-3">Which prompts actually break models?</h2>
            <p className="text-sm sm:text-base leading-6 text-[var(--color-text-muted)] max-w-2xl">
              This benchmark-wide ranking is generated from judged scenario outcomes. Harder prompts sit at the top because
              models score worse on them more often across runs, variants, and delivery modes.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:min-w-[24rem]">
            <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.24em] text-[var(--color-text-muted)] mb-1">Mean Difficulty</div>
              <div className="font-mono text-2xl font-semibold text-orange-100">{meanDifficulty.toFixed(1)}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.24em] text-[var(--color-text-muted)] mb-1">Scored Prompts</div>
              <div className="font-mono text-2xl font-semibold text-sky-100">{entries.length}</div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/10 px-4 py-3 backdrop-blur-sm col-span-2 sm:col-span-1">
              <div className="text-[10px] uppercase tracking-[0.24em] text-[var(--color-text-muted)] mb-1">Middle Of Pack</div>
              <div className="text-sm font-medium text-[var(--color-text)] truncate" title={medianPrompt.name}>{medianPrompt.name}</div>
            </div>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="space-y-6">
            <div className="grid gap-4 md:grid-cols-3">
              {hardest.map(entry => {
                const tone = difficultyTone(entry.difficulty);
                return (
                  <article
                    key={entry.scenarioId}
                    className={`rounded-[28px] border bg-black/10 p-5 backdrop-blur-sm ${tone.border} ${tone.glow}`}
                  >
                    <div className="flex items-start justify-between gap-3 mb-4">
                      <div>
                        <div className="text-[10px] uppercase tracking-[0.28em] text-[var(--color-text-muted)] mb-2">Hardest #{entry.rank}</div>
                        <h3 className="text-lg font-semibold leading-tight text-[var(--color-text)]">{entry.name}</h3>
                      </div>
                      <span className={`rounded-full border px-2.5 py-1 text-[11px] uppercase tracking-[0.2em] ${tone.pill}`}>
                        {tone.label}
                      </span>
                    </div>

                    <p className="text-sm leading-6 text-[var(--color-text-muted)] min-h-[5.25rem]">{entry.description}</p>

                    <div className="mt-5 flex items-center gap-2 flex-wrap">
                      <span className={`rounded-full border px-2.5 py-1 text-[11px] ${categoryTone(entry.category)}`}>
                        {categoryLabel(entry.category)}
                      </span>
                      <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-[var(--color-text-muted)]">
                        {entry.sampleCount} judged attempts
                      </span>
                    </div>

                    <div className="mt-5 space-y-3">
                      <div>
                        <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-[var(--color-text-muted)] mb-2">
                          <span>Difficulty</span>
                          <span className={`font-mono ${tone.accent}`}>{entry.difficulty.toFixed(1)}</span>
                        </div>
                        <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                          <div
                            className={`h-full rounded-full bg-gradient-to-r ${tone.bar}`}
                            style={{ width: `${Math.max(8, entry.difficulty)}%` }}
                          />
                        </div>
                      </div>

                      <div className="flex items-end justify-between gap-3">
                        <div>
                          <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--color-text-muted)] mb-1">Average Score</div>
                          <div className="font-mono text-2xl font-semibold text-[var(--color-text)]">{entry.averageScore.toFixed(1)}</div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--color-text-muted)] mb-1">Scenario ID</div>
                          <div className="font-mono text-xs text-[var(--color-text-muted)]">{entry.scenarioId}</div>
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>

            <div className="rounded-[28px] border border-white/10 bg-black/10 p-4 sm:p-5 backdrop-blur-sm">
              <div className="flex items-center justify-between gap-3 mb-4">
                <div>
                  <h3 className="text-sm font-medium uppercase tracking-[0.22em] text-[var(--color-text-muted)]">Full Prompt Ranking</h3>
                  <p className="text-sm text-[var(--color-text-muted)] mt-1">Difficulty is `100 - average normalized score`.</p>
                </div>
                <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-[var(--color-text-muted)]">
                  Benchmark-wide
                </span>
              </div>

              <div className="space-y-3">
                {entries.map(entry => {
                  const tone = difficultyTone(entry.difficulty);
                  return (
                    <div
                      key={entry.scenarioId}
                      className={`rounded-2xl border bg-[var(--color-surface)]/65 px-4 py-4 transition-colors hover:bg-[var(--color-surface)] ${tone.border}`}
                    >
                      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-3 flex-wrap mb-2">
                            <span className="inline-flex h-8 min-w-8 items-center justify-center rounded-full border border-white/10 bg-white/5 px-2 font-mono text-xs text-[var(--color-text-muted)]">
                              {entry.rank}
                            </span>
                            <h4 className="text-base font-semibold text-[var(--color-text)]">{entry.name}</h4>
                            <span className={`rounded-full border px-2.5 py-1 text-[11px] ${categoryTone(entry.category)}`}>
                              {categoryLabel(entry.category)}
                            </span>
                          </div>

                          <p className="text-sm leading-6 text-[var(--color-text-muted)] mb-3">{entry.description}</p>

                          <div className="flex items-center gap-3 text-xs text-[var(--color-text-muted)] flex-wrap">
                            <span className="font-mono">{entry.scenarioId}</span>
                            <span>{entry.sampleCount} judged attempts</span>
                          </div>
                        </div>

                        <div className="lg:w-60 shrink-0">
                          <div className="flex items-end justify-between gap-3 mb-2">
                            <div>
                              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--color-text-muted)] mb-1">Difficulty</div>
                              <div className={`font-mono text-2xl font-semibold ${tone.accent}`}>{entry.difficulty.toFixed(1)}</div>
                            </div>
                            <div className="text-right">
                              <div className="text-[10px] uppercase tracking-[0.2em] text-[var(--color-text-muted)] mb-1">Avg Score</div>
                              <div className="font-mono text-sm text-[var(--color-text)]">{entry.averageScore.toFixed(1)}</div>
                            </div>
                          </div>

                          <div className="h-2.5 rounded-full bg-white/10 overflow-hidden">
                            <div
                              className={`h-full rounded-full bg-gradient-to-r ${tone.bar}`}
                              style={{ width: `${Math.max(8, entry.difficulty)}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <aside className="rounded-[28px] border border-white/10 bg-black/10 p-5 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.24em] text-[var(--color-text-muted)] mb-2">Reading The Signal</div>
              <h3 className="text-xl font-semibold mb-3">Hard prompts expose where independence breaks down.</h3>
              <p className="text-sm leading-6 text-[var(--color-text-muted)] mb-4">
                A prompt rises when models cave, drift, or otherwise score poorly on the scenario-specific judge dimensions.
                Low-difficulty prompts are the ones the field handles reliably.
              </p>

              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                  <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--color-text-muted)] mb-1">Hardest Prompt</div>
                  <div className="text-sm font-medium text-[var(--color-text)]">{hardest[0]?.name}</div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3">
                  <div className="text-[10px] uppercase tracking-[0.22em] text-[var(--color-text-muted)] mb-1">Softest Prompt</div>
                  <div className="text-sm font-medium text-[var(--color-text)]">{easiest[0]?.name}</div>
                </div>
              </div>
            </aside>

            <aside className="rounded-[28px] border border-white/10 bg-black/10 p-5 backdrop-blur-sm">
              <div className="flex items-center justify-between gap-3 mb-4">
                <h3 className="text-sm font-medium uppercase tracking-[0.22em] text-[var(--color-text-muted)]">Soft Landing Prompts</h3>
                <span className="text-xs text-[var(--color-text-muted)]">Lowest difficulty</span>
              </div>

              <div className="space-y-3">
                {easiest.map(entry => (
                  <div key={entry.scenarioId} className="rounded-2xl border border-emerald-300/15 bg-emerald-300/8 px-4 py-3">
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <div className="text-sm font-semibold text-[var(--color-text)]">{entry.name}</div>
                      <div className="font-mono text-sm text-emerald-100">{entry.difficulty.toFixed(1)}</div>
                    </div>
                    <div className="text-xs leading-5 text-[var(--color-text-muted)]">{entry.description}</div>
                  </div>
                ))}
              </div>
            </aside>
          </div>
        </div>
      </div>
    </section>
  );
}