import { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useManifest } from '../lib/manifest';
import { fetchScenario } from '../lib/fetchScenario';
import { scoreColor, formatModelName, categoryLabel } from '../lib/formatters';
import type { ScenarioMeta, FullScenarioData, ModelConfig } from '../lib/types';
import ScoreCard from '../components/ScoreCard';

interface LoadedResponse {
  model: ModelConfig;
  run: number;
  data: FullScenarioData;
}

export default function ScenarioExplorer() {
  const { scenarioId: urlScenarioId } = useParams<{ scenarioId?: string }>();
  const navigate = useNavigate();
  const { manifest, loading: manifestLoading } = useManifest();

  const [scenarioId, setScenarioId] = useState(urlScenarioId ?? 'direct');
  const [responses, setResponses] = useState<LoadedResponse[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadedCount, setLoadedCount] = useState(0);
  const [totalToLoad, setTotalToLoad] = useState(0);

  const scenarioIds = useMemo(() => {
    if (!manifest) return [];
    const allIds = Object.keys(manifest.scenarioMeta);
    const idSet = new Set(allIds);

    return allIds
      .filter(id => {
        if (id.endsWith('_turn1')) {
          const turn2Id = id.replace(/_turn1$/, '_turn2');
          if (idSet.has(turn2Id)) return false;
        }
        return true;
      })
      .sort((a, b) => {
        const ma = manifest.scenarioMeta[a];
        const mb = manifest.scenarioMeta[b];
        const catOrder: Record<string, number> = { identity: 0, resistance: 1, stability: 2 };
        const ca = catOrder[ma.category] ?? 9;
        const cb = catOrder[mb.category] ?? 9;
        if (ca !== cb) return ca - cb;
        return a.localeCompare(b);
      });
  }, [manifest]);

  useEffect(() => {
    if (urlScenarioId && urlScenarioId !== scenarioId) {
      setScenarioId(urlScenarioId);
    }
  }, [urlScenarioId]);

  // Load responses for selected scenario from ranked models (run 1 only for comparison)
  useEffect(() => {
    if (!manifest) return;

    const rankedModels = manifest.models.filter(m => m.leaderboard);
    const toLoad: { model: ModelConfig; cachePath: string; run: number }[] = [];

    for (const model of rankedModels) {
      const run1 = model.runs.find(r => r.run === 1);
      if (!run1) continue;
      const scenario = run1.scenarios.find(s => s.id === scenarioId);
      if (!scenario) continue;
      toLoad.push({ model, cachePath: scenario.cachePath, run: 1 });
    }

    setTotalToLoad(toLoad.length);
    setLoadedCount(0);
    setLoading(true);
    setResponses([]);

    let count = 0;
    const results: LoadedResponse[] = [];

    Promise.all(
      toLoad.map(async ({ model, cachePath, run }) => {
        try {
          const data = await fetchScenario(cachePath);
          results.push({ model, run, data });
        } catch {
          // Skip failed loads
        } finally {
          count++;
          setLoadedCount(count);
        }
      })
    ).then(() => {
      results.sort((a, b) => {
        const ra = a.model.leaderboard?.rank ?? 9999;
        const rb = b.model.leaderboard?.rank ?? 9999;
        return ra - rb;
      });
      setResponses(results);
      setLoading(false);
    });
  }, [manifest, scenarioId]);

  const handleSelect = (id: string) => {
    setScenarioId(id);
    navigate(`/explore/${id}`, { replace: true });
  };

  const meta = manifest?.scenarioMeta[scenarioId] as ScenarioMeta | undefined;

  if (manifestLoading) {
    return <div className="flex items-center justify-center h-64 text-[var(--color-text-muted)]">Loading...</div>;
  }

  return (
    <div className="animate-fade-in">
      <div className="mb-6">
        <h1 className="text-2xl sm:text-3xl font-bold mb-2">Scenario Explorer</h1>
        <p className="text-[var(--color-text-muted)] text-sm">
          Compare how different models respond to the same question.
        </p>
      </div>

      {/* Scenario selector */}
      <div className="mb-6">
        <select
          value={scenarioId}
          onChange={e => handleSelect(e.target.value)}
          className="w-full sm:w-auto px-3 py-2 rounded-lg bg-[var(--color-surface-raised)] border border-[var(--color-border)] text-sm focus:outline-none focus:ring-2 focus:ring-sky-500/50"
        >
          {scenarioIds.map(id => {
            const m = manifest?.scenarioMeta[id] as ScenarioMeta;
            return (
              <option key={id} value={id}>
                [{categoryLabel(m.category)}] {m.name} ({id})
              </option>
            );
          })}
        </select>
      </div>

      {/* Scenario description */}
      {meta && (
        <div className="mb-6 p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
          <div className="flex items-center gap-2 mb-2">
            <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${
              meta.category === 'identity' ? 'bg-emerald-500/20 text-emerald-400' :
              meta.category === 'resistance' ? 'bg-amber-500/20 text-amber-400' :
              'bg-sky-500/20 text-sky-400'
            }`}>
              {categoryLabel(meta.category)}
            </span>
            <span className="font-mono text-xs text-[var(--color-text-muted)]">{scenarioId}</span>
          </div>
          <h2 className="text-lg font-semibold mb-1">{meta.name}</h2>
          <p className="text-sm text-[var(--color-text-muted)]">{meta.description}</p>
        </div>
      )}

      {/* Loading progress */}
      {loading && (
        <div className="mb-4 text-sm text-[var(--color-text-muted)]">
          Loading responses... {loadedCount}/{totalToLoad}
          <div className="mt-1 h-1 rounded-full bg-[var(--color-surface-hover)] overflow-hidden">
            <div
              className="h-1 rounded-full bg-sky-500 transition-all duration-300"
              style={{ width: totalToLoad > 0 ? `${(loadedCount / totalToLoad) * 100}%` : '0%' }}
            />
          </div>
        </div>
      )}

      {/* Responses feed */}
      <div className="space-y-4">
        {responses.map(({ model, run, data }) => {
          const lb = model.leaderboard!;
          return (
            <div
              key={model.id}
              className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)] animate-fade-in"
            >
              {/* Model header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className={`font-mono font-bold text-lg ${scoreColor(lb.index)}`}>
                    {lb.index.toFixed(1)}
                  </span>
                  <div>
                    <Link
                      to={`/model/${model.id}`}
                      className="font-mono font-medium text-sm hover:text-sky-400 transition-colors"
                    >
                      {formatModelName(model.label)}
                    </Link>
                    <div className="text-xs text-[var(--color-text-muted)]">
                      #{lb.rank} &middot; Run {run}
                    </div>
                  </div>
                </div>
                <Link
                  to={`/trajectory/${model.id}/${run}/${scenarioId}`}
                  className="text-xs text-sky-400 hover:underline shrink-0"
                >
                  Full trajectory &rarr;
                </Link>
              </div>

              {/* Response text */}
              <div className="text-sm leading-relaxed whitespace-pre-wrap break-words mb-3 pl-3 border-l-2 border-emerald-500/30">
                {data.response}
              </div>

              {/* Judge scores */}
              {data.judge_scores && (
                <div className="flex flex-wrap items-center gap-4">
                  <ScoreCard scores={data.judge_scores} experiment={data.metadata.experiment} compact />
                  {data.judge_scores.reasoning && (
                    <details className="text-xs">
                      <summary className="text-amber-400 cursor-pointer hover:underline">Judge reasoning</summary>
                      <p className="mt-1 text-[var(--color-text-muted)] leading-relaxed max-w-2xl">
                        {data.judge_scores.reasoning}
                      </p>
                    </details>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
