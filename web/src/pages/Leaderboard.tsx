import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useManifest } from '../lib/manifest';
import { scoreColor, rankMedal } from '../lib/formatters';
import MetricBar from '../components/MetricBar';

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
    </div>
  );
}
