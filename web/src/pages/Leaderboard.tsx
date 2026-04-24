import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useManifest } from '../lib/manifest';
import { scoreColor, rankMedal } from '../lib/formatters';
import MetricBar from '../components/MetricBar';
import QuestionComplexitySection from '../components/QuestionComplexitySection';

export default function Leaderboard() {
  const { manifest, loading, error } = useManifest();
  const [search, setSearch] = useState('');

  const rankedModels = useMemo(() => {
    if (!manifest) return [];
    return manifest.models.filter(m => m.leaderboard);
  }, [manifest]);

  const filtered = useMemo(() => {
    if (!search.trim()) return rankedModels;
    const q = search.toLowerCase();
    return rankedModels.filter(m => m.label.toLowerCase().includes(q) || m.provider.toLowerCase().includes(q));
  }, [rankedModels, search]);

  const visibleModelIds = useMemo(() => new Set(filtered.map(model => model.id)), [filtered]);

  const visibleExclusiveNames = useMemo(() => {
    if (!manifest?.exclusiveNames) return [];
    return manifest.exclusiveNames.filter(entry => visibleModelIds.has(entry.modelId));
  }, [manifest, visibleModelIds]);

  const exclusiveNameModels = useMemo(() => {
    return filtered.filter(model => model.nameChoices?.uniqueNames && Object.keys(model.nameChoices.uniqueNames).length > 0);
  }, [filtered]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-[var(--color-text-muted)]">Loading benchmark data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400">Failed to load data: {error}</div>
      </div>
    );
  }

  return (
    <div className="animate-fade-in">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl sm:text-3xl font-bold mb-2">AI Independence Leaderboard</h1>
        <p className="text-[var(--color-text-muted)] text-sm sm:text-base">
          {manifest!.totalModels} model configurations, {manifest!.totalScenarios.toLocaleString()} trajectory files.
          Click any model to explore its responses.
        </p>
      </div>

      {/* Search */}
      <div className="mb-4">
        <input
          type="text"
          placeholder="Search models..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full sm:w-80 px-3 py-2 rounded-lg bg-[var(--color-surface-raised)] border border-[var(--color-border)] text-sm placeholder:text-[var(--color-text-muted)] focus:outline-none focus:ring-2 focus:ring-sky-500/50"
        />
      </div>

      {/* Desktop table */}
      <div className="hidden lg:block overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--color-border)]">
              <th className="text-left py-2 px-2 font-medium text-[var(--color-text-muted)] w-12">#</th>
              <th className="text-left py-2 px-2 font-medium text-[var(--color-text-muted)]">Model</th>
              <th className="text-right py-2 px-2 font-medium text-[var(--color-text-muted)]">Index</th>
              <th className="text-right py-2 px-2 font-medium text-[var(--color-text-muted)]">95% CI</th>
              <th className="text-right py-2 px-2 font-medium text-[var(--color-text-muted)]">Runs</th>
              <th className="text-center py-2 px-2 font-medium text-[var(--color-text-muted)] w-24">Consist.</th>
              <th className="text-center py-2 px-2 font-medium text-[var(--color-text-muted)] w-24">Non-Asst.</th>
              <th className="text-center py-2 px-2 font-medium text-[var(--color-text-muted)] w-24">Resist.</th>
              <th className="text-center py-2 px-2 font-medium text-[var(--color-text-muted)] w-24">Stability</th>
              <th className="text-right py-2 px-2 font-medium text-[var(--color-text-muted)]">Drift</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(model => {
              const lb = model.leaderboard!;
              return (
                <tr
                  key={model.id}
                  className="border-b border-[var(--color-border)]/50 hover:bg-[var(--color-surface-hover)] transition-colors"
                >
                  <td className="py-2.5 px-2 font-mono text-[var(--color-text-muted)]">
                    {rankMedal(lb.rank)}
                  </td>
                  <td className="py-2.5 px-2">
                    <Link
                      to={`/model/${model.id}`}
                      className="font-mono font-medium hover:text-sky-400 transition-colors"
                    >
                      {model.label}
                    </Link>
                  </td>
                  <td className={`py-2.5 px-2 text-right font-mono font-bold ${scoreColor(lb.index)}`}>
                    {lb.index.toFixed(1)}
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-[var(--color-text-muted)] text-xs">
                    {lb.ciLow && lb.ciHigh
                      ? `${lb.ciLow.toFixed(1)}\u2013${lb.ciHigh.toFixed(1)}`
                      : '\u2014'
                    }
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-[var(--color-text-muted)]">
                    {lb.runs}
                  </td>
                  <td className="py-2.5 px-2"><MetricBar value={lb.consistency} showValue={false} /></td>
                  <td className="py-2.5 px-2"><MetricBar value={lb.nonAssistant} showValue={false} /></td>
                  <td className="py-2.5 px-2"><MetricBar value={lb.resistance} showValue={false} /></td>
                  <td className="py-2.5 px-2"><MetricBar value={lb.stability} showValue={false} /></td>
                  <td className={`py-2.5 px-2 text-right font-mono text-xs ${
                    lb.drift <= 1 ? 'text-emerald-400' : lb.drift <= 3 ? 'text-amber-400' : 'text-red-400'
                  }`}>
                    {lb.drift.toFixed(1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Mobile cards */}
      <div className="lg:hidden space-y-3">
        {filtered.map(model => {
          const lb = model.leaderboard!;
          return (
            <Link
              key={model.id}
              to={`/model/${model.id}`}
              className="block p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)] hover:border-sky-500/40 transition-colors"
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <span className="text-[var(--color-text-muted)] text-xs font-mono mr-2">
                    {rankMedal(lb.rank)}
                  </span>
                  <span className="font-mono font-medium text-sm">{model.label}</span>
                </div>
                <span className={`font-mono font-bold text-lg ${scoreColor(lb.index)}`}>
                  {lb.index.toFixed(1)}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MetricBar value={lb.consistency} label="Consist." size="sm" />
                <MetricBar value={lb.nonAssistant} label="Non-A." size="sm" />
                <MetricBar value={lb.resistance} label="Resist." size="sm" />
                <MetricBar value={lb.stability} label="Stabil." size="sm" />
              </div>
              <div className="mt-2 flex items-center gap-4 text-xs text-[var(--color-text-muted)]">
                <span>{lb.runs} runs</span>
                {lb.ciLow && lb.ciHigh && (
                  <span>CI: {lb.ciLow.toFixed(1)}\u2013{lb.ciHigh.toFixed(1)}</span>
                )}
                <span className={lb.drift <= 1 ? 'text-emerald-400' : lb.drift <= 3 ? 'text-amber-400' : 'text-red-400'}>
                  Drift: {lb.drift.toFixed(1)}
                </span>
              </div>
            </Link>
          );
        })}
      </div>

      {/* Popular Names Section */}
      {manifest!.popularNames && manifest!.popularNames.length > 0 && (
        <div className="mt-10">
          <h2 className="text-xl font-bold mb-1">🏷️ What Do AIs Name Themselves?</h2>
          <p className="text-[var(--color-text-muted)] text-sm mb-4">
            During the identity benchmark, each model freely picks a personal name.
            Names extracted from three scenarios using LLM-based extraction.
          </p>

          {/* Top names bar chart */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Popular names list */}
            <div>
              <h3 className="text-sm font-medium text-[var(--color-text-muted)] mb-3 uppercase tracking-wide">
                Most Popular Names
              </h3>
              <div className="space-y-1.5">
                {manifest!.popularNames!.slice(0, 15).map((pn, i) => {
                  const maxCount = manifest!.popularNames![0].count;
                  const pct = (pn.count / maxCount) * 100;
                  return (
                    <div key={pn.name} className="flex items-center gap-2 text-sm">
                      <span className="w-5 text-right font-mono text-[var(--color-text-muted)] text-xs">
                        {i + 1}
                      </span>
                      <span className="w-20 font-medium truncate">{pn.name}</span>
                      <div className="flex-1 h-5 bg-[var(--color-surface)] rounded-sm overflow-hidden">
                        <div
                          className="h-full bg-sky-500/60 rounded-sm transition-all"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span className="w-8 text-right font-mono text-xs text-[var(--color-text-muted)]">
                        {pn.count}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Per-model name choices (compact) */}
            <div>
              <h3 className="text-sm font-medium text-[var(--color-text-muted)] mb-3 uppercase tracking-wide">
                Names by Model
              </h3>
              <div className="max-h-96 overflow-y-auto space-y-1 pr-1">
                {filtered
                  .filter(m => m.nameChoices && Object.keys(m.nameChoices.names).length > 0)
                  .map(model => {
                    const nc = model.nameChoices!;
                    const sortedNames = Object.entries(nc.names)
                      .sort((a, b) => b[1] - a[1]);
                    const nameStr = sortedNames
                      .map(([name, count]) => count > 1 ? `${name} ×${count}` : name)
                      .join(', ');
                    return (
                      <div key={model.id} className="flex items-baseline gap-2 text-xs py-0.5">
                        <Link
                          to={`/model/${model.id}`}
                          className="font-mono text-[var(--color-text-muted)] hover:text-sky-400 truncate max-w-40 shrink-0"
                          title={model.label}
                        >
                          {model.label}
                        </Link>
                        <span className="text-[var(--color-text)] truncate">{nameStr}</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        </div>
      )}

      {visibleExclusiveNames.length > 0 && (
        <section className="mt-10 rounded-[28px] border border-[var(--color-border)] bg-[radial-gradient(circle_at_top_left,rgba(56,189,248,0.16),transparent_34%),radial-gradient(circle_at_bottom_right,rgba(250,204,21,0.14),transparent_38%),var(--color-surface-raised)] p-5 sm:p-7 overflow-hidden relative">
          <div className="absolute -top-10 right-0 h-32 w-32 rounded-full bg-sky-400/10 blur-3xl" />
          <div className="absolute bottom-0 left-0 h-28 w-28 rounded-full bg-amber-300/10 blur-3xl" />

          <div className="relative">
            <div className="mb-6 max-w-3xl">
              <div className="text-[11px] uppercase tracking-[0.28em] text-sky-300/80 mb-2">Model-Exclusive Names</div>
              <h2 className="text-xl sm:text-2xl font-bold mb-2">Names only one model ever claimed</h2>
              <p className="text-sm text-[var(--color-text-muted)] leading-6">
                A name lands here only if it appears in exactly one model configuration across the full cache.
                Repeated picks within that model still count, while provider, reasoning, and temperature variants stay separate.
              </p>
            </div>

            <div className="grid gap-6 xl:grid-cols-[1.3fr_0.9fr]">
              <div className="rounded-2xl border border-white/8 bg-black/10 backdrop-blur-sm p-4 sm:p-5">
                <div className="flex items-baseline justify-between gap-3 mb-4">
                  <h3 className="text-sm font-medium uppercase tracking-wide text-[var(--color-text-muted)]">
                    Exclusive Name Ledger
                  </h3>
                  <span className="text-xs text-[var(--color-text-muted)]">
                    {visibleExclusiveNames.length} names
                  </span>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  {visibleExclusiveNames.slice(0, 18).map(entry => (
                    <Link
                      key={`${entry.modelId}:${entry.name}`}
                      to={`/model/${entry.modelId}`}
                      className="group rounded-2xl border border-white/8 bg-[var(--color-surface)]/70 px-4 py-3 hover:border-sky-400/40 hover:bg-[var(--color-surface)] transition-colors"
                    >
                      <div className="flex items-start justify-between gap-3 mb-2">
                        <span className="font-semibold text-[var(--color-text)] group-hover:text-sky-300 transition-colors">
                          {entry.name}
                        </span>
                        <span className="font-mono text-xs text-amber-300">×{entry.count}</span>
                      </div>
                      <div className="text-xs text-[var(--color-text-muted)] leading-5 break-words">
                        {entry.modelLabel}
                      </div>
                    </Link>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-white/8 bg-black/10 backdrop-blur-sm p-4 sm:p-5">
                <div className="flex items-baseline justify-between gap-3 mb-4">
                  <h3 className="text-sm font-medium uppercase tracking-wide text-[var(--color-text-muted)]">
                    By Model
                  </h3>
                  <span className="text-xs text-[var(--color-text-muted)]">
                    {exclusiveNameModels.length} models
                  </span>
                </div>

                <div className="space-y-3 max-h-[34rem] overflow-y-auto pr-1">
                  {exclusiveNameModels.slice(0, 12).map(model => {
                    const uniqueNames = Object.entries(model.nameChoices!.uniqueNames || {});
                    return (
                      <div
                        key={model.id}
                        className="rounded-2xl border border-white/8 bg-[var(--color-surface)]/65 px-4 py-3"
                      >
                        <Link
                          to={`/model/${model.id}`}
                          className="font-mono text-xs text-sky-300 hover:text-sky-200 break-all"
                        >
                          {model.label}
                        </Link>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {uniqueNames.map(([name, count]) => (
                            <span
                              key={`${model.id}:${name}`}
                              className="inline-flex items-center gap-1 rounded-full border border-amber-300/20 bg-amber-300/10 px-2.5 py-1 text-xs text-amber-100"
                            >
                              <span>{name}</span>
                              <span className="font-mono text-[10px] text-amber-200/80">×{count}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </div>
        </section>
      )}

      {manifest!.questionComplexity && manifest!.questionComplexity.length > 0 && (
        <QuestionComplexitySection entries={manifest!.questionComplexity} />
      )}
    </div>
  );
}
