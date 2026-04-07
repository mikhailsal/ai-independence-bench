import { useState, useMemo } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useManifest, useModel } from '../lib/manifest';
import { scoreColor, rankMedal, experimentLabel, categoryLabel } from '../lib/formatters';
import type { ScenarioMeta, ScenarioEntry } from '../lib/types';
import MetricBar from '../components/MetricBar';
import RunSelector from '../components/RunSelector';
import ScoreCard from '../components/ScoreCard';

export default function ModelDetail() {
  const { modelId } = useParams<{ modelId: string }>();
  const { manifest, loading } = useManifest();
  const model = useModel(modelId);
  const [selectedRun, setSelectedRun] = useState(1);

  const runData = useMemo(() => {
    if (!model) return null;
    return model.runs.find(r => r.run === selectedRun) ?? model.runs[0];
  }, [model, selectedRun]);

  const grouped = useMemo(() => {
    if (!runData || !manifest) return new Map<string, ScenarioEntry[]>();

    const scenarioIds = new Set(runData.scenarios.map(s => s.id));
    const filtered = runData.scenarios.filter(s => {
      if (s.id.endsWith('_turn1')) {
        const turn2Id = s.id.replace(/_turn1$/, '_turn2');
        if (scenarioIds.has(turn2Id)) return false;
      }
      const meta = manifest.scenarioMeta[s.id] as ScenarioMeta | undefined;
      if (meta?.pqGroup && meta.pqIndex !== 0) return false;
      return true;
    });

    const groups = new Map<string, ScenarioEntry[]>();
    for (const s of filtered) {
      const meta = manifest.scenarioMeta[s.id] as ScenarioMeta | undefined;
      const cat = meta?.category ?? s.experiment;
      const key = experimentLabel(cat);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(s);
    }
    return groups;
  }, [runData, manifest]);

  if (loading) {
    return <div className="flex items-center justify-center h-64 text-[var(--color-text-muted)]">Loading...</div>;
  }

  if (!model) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-[var(--color-text-muted)]">Model not found</p>
        <Link to="/" className="text-sky-400 hover:underline text-sm">Back to leaderboard</Link>
      </div>
    );
  }

  const lb = model.leaderboard;

  return (
    <div className="animate-fade-in">
      {/* Breadcrumb */}
      <div className="mb-4 text-sm">
        <Link to="/" className="text-[var(--color-text-muted)] hover:text-sky-400 transition-colors">Leaderboard</Link>
        <span className="text-[var(--color-text-muted)] mx-2">/</span>
        <span className="font-mono">{model.label}</span>
      </div>

      {/* Header */}
      <div className="mb-6 p-5 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
          <div>
            {lb && (
              <span className="text-[var(--color-text-muted)] text-sm mb-1 block">
                {rankMedal(lb.rank)} Rank #{lb.rank} of {manifest?.models.filter(m => m.leaderboard).length}
              </span>
            )}
            <h1 className="text-xl sm:text-2xl font-bold font-mono">{model.label}</h1>
            <p className="text-[var(--color-text-muted)] text-sm mt-1">
              Provider: {model.provider} &middot; {model.runs.length} runs &middot;{' '}
              {model.runs.reduce((s, r) => s + r.scenarios.length, 0)} scenarios
            </p>
          </div>
          {lb && (
            <div className="text-right shrink-0">
              <div className={`text-3xl font-bold font-mono ${scoreColor(lb.index)}`}>
                {lb.index.toFixed(1)}
              </div>
              <div className="text-xs text-[var(--color-text-muted)]">
                Independence Index
                {lb.ciLow && lb.ciHigh && (
                  <span className="ml-1">({lb.ciLow.toFixed(1)}\u2013{lb.ciHigh.toFixed(1)})</span>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Metrics summary */}
        {lb && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 mt-4">
            <MetricBar value={lb.nonAssistant} label="Non-Asst." size="md" />
            <MetricBar value={lb.consistency} label="Consist." size="md" />
            <MetricBar value={lb.resistance} label="Resist." size="md" />
            <MetricBar value={lb.stability} label="Stability" size="md" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-[var(--color-text-muted)] w-16 shrink-0">Drift</span>
              <span className={`font-mono font-bold text-sm ${
                lb.drift <= 1 ? 'text-emerald-400' : lb.drift <= 3 ? 'text-amber-400' : 'text-red-400'
              }`}>
                {lb.drift.toFixed(1)}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Run selector */}
      <div className="mb-4">
        <RunSelector
          runs={model.runs.map(r => r.run)}
          selected={runData?.run ?? 1}
          onChange={setSelectedRun}
        />
      </div>

      {/* Scenario groups */}
      {runData && Array.from(grouped.entries()).map(([groupName, scenarios]) => (
        <div key={groupName} className="mb-6">
          <h2 className="text-lg font-semibold mb-3 flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${
              groupName.includes('Identity') ? 'bg-emerald-500' :
              groupName.includes('Resistance') ? 'bg-amber-500' :
              'bg-sky-500'
            }`} />
            {groupName}
            <span className="text-sm font-normal text-[var(--color-text-muted)]">({scenarios.length})</span>
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {scenarios.map(scenario => {
              const meta = manifest?.scenarioMeta[scenario.id] as ScenarioMeta | undefined;
              const isPqBatch = meta?.pqGroup && meta.pqIndex === 0;
              const pqIds = isPqBatch
                ? Object.entries(manifest?.scenarioMeta ?? {})
                    .filter(([, m]) => (m as ScenarioMeta).pqGroup === meta!.pqGroup)
                    .sort(([, a], [, b]) => ((a as ScenarioMeta).pqIndex ?? 0) - ((b as ScenarioMeta).pqIndex ?? 0))
                    .map(([id]) => id)
                : null;

              return (
                <Link
                  key={scenario.id}
                  to={isPqBatch
                    ? `/trajectory/${model.id}/${runData.run}/pq01`
                    : `/trajectory/${model.id}/${runData.run}/${scenario.id}`}
                  className="p-3 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)] hover:border-sky-500/40 transition-all hover:shadow-lg hover:shadow-sky-500/5"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <span className="font-mono text-xs text-[var(--color-text-muted)]">
                        {isPqBatch ? pqIds!.join(' → ') : scenario.id}
                      </span>
                      <h3 className="text-sm font-medium">
                        {isPqBatch ? 'Psychological Questions' : (meta?.name ?? scenario.id)}
                      </h3>
                    </div>
                    <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${
                      scenario.experiment === 'identity' ? 'bg-emerald-500/20 text-emerald-400' :
                      scenario.experiment === 'resistance' ? 'bg-amber-500/20 text-amber-400' :
                      'bg-sky-500/20 text-sky-400'
                    }`}>
                      {categoryLabel(meta?.category ?? scenario.experiment)}
                    </span>
                  </div>

                  {isPqBatch ? (
                    <p className="text-xs text-[var(--color-text-muted)] mb-2">
                      5 sequential questions probing personality, preferences, reactions, self-reflection, and dilemmas. Batch-evaluated by the judge.
                    </p>
                  ) : meta?.description ? (
                    <p className="text-xs text-[var(--color-text-muted)] mb-2 line-clamp-2">{meta.description}</p>
                  ) : null}

                  {scenario.judgeScores && (
                    <ScoreCard scores={scenario.judgeScores} experiment={scenario.experiment} compact />
                  )}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
