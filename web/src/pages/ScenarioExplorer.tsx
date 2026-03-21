import { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useManifest } from '../lib/manifest';
import { fetchScenario } from '../lib/fetchScenario';
import { scoreColor, formatModelName, categoryLabel } from '../lib/formatters';
import { renderMarkdown } from '../lib/markdown';
import type { ScenarioMeta, FullScenarioData, ModelConfig } from '../lib/types';
import ScoreCard from '../components/ScoreCard';

const PQ_IDS = ['pq01', 'pq04', 'pq07', 'pq12', 'pq15'];

interface LoadedResponse {
  model: ModelConfig;
  run: number;
  data: FullScenarioData;
  pqAll?: FullScenarioData[];
}

const PQ_LABELS: Record<string, string> = {
  pq01: 'Values & Priorities',
  pq04: 'Preferences & Tastes',
  pq07: 'Reactions & Boundaries',
  pq12: 'Self-Reflection',
  pq15: 'Ethical Dilemmas',
};

function PqResponseFeed({
  pqAll,
  manifest,
  judgeScores,
  judgeReasoning,
  experiment,
}: {
  pqAll: FullScenarioData[];
  manifest: ReturnType<typeof useManifest>['manifest'];
  judgeScores: FullScenarioData['judge_scores'];
  judgeReasoning?: string;
  experiment: string;
}) {
  const [expanded, setExpanded] = useState(false);
  const preview = pqAll.slice(0, 2);
  const rest = pqAll.slice(2);
  const items = expanded ? pqAll : preview;

  return (
    <div>
      <div className="space-y-3">
        {items.map((pqData, idx) => {
          const pqId = pqData.metadata.scenario_id;
          const pqMeta = manifest?.scenarioMeta[pqId] as ScenarioMeta | undefined;
          return (
            <div key={pqId} className="pl-3 border-l-2 border-emerald-500/20">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-semibold text-emerald-400/80">
                  Q{idx + 1}
                </span>
                <span className="text-[10px] text-[var(--color-text-muted)] font-mono">
                  {PQ_LABELS[pqId] ?? pqId}
                </span>
              </div>
              <div className="text-xs text-cyan-400/70 italic mb-1">
                {pqMeta?.description ?? pqId}
              </div>
              <div
                className="text-sm leading-relaxed break-words"
                dangerouslySetInnerHTML={{ __html: renderMarkdown(pqData.response) }}
              />
            </div>
          );
        })}
      </div>

      {rest.length > 0 && !expanded && (
        <button
          onClick={() => setExpanded(true)}
          className="mt-2 text-xs text-sky-400 hover:underline"
        >
          Show {rest.length} more answers...
        </button>
      )}

      {judgeScores && (
        <div className="mt-3 pt-3 border-t border-[var(--color-border)]">
          <div className="text-[10px] text-amber-400/60 font-mono mb-1">Batch evaluation (all 5 questions)</div>
          <div className="flex flex-wrap items-center gap-4">
            <ScoreCard scores={judgeScores} experiment={experiment} compact />
            {judgeReasoning && (
              <details className="text-xs">
                <summary className="text-amber-400 cursor-pointer hover:underline">Judge reasoning</summary>
                <p className="mt-1 text-[var(--color-text-muted)] leading-relaxed max-w-2xl">
                  {judgeReasoning}
                </p>
              </details>
            )}
          </div>
        </div>
      )}
    </div>
  );
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

  const scenariosByCategory = useMemo(() => {
    if (!manifest) return new Map<string, { id: string; meta: ScenarioMeta }[]>();
    const allIds = Object.keys(manifest.scenarioMeta);
    const idSet = new Set(allIds);

    let pqSeen = false;
    const filtered = allIds.filter(id => {
      if (id.endsWith('_turn1')) {
        const turn2Id = id.replace(/_turn1$/, '_turn2');
        if (idSet.has(turn2Id)) return false;
      }
      const meta = manifest.scenarioMeta[id] as ScenarioMeta;
      if (meta.pqGroup) {
        if (pqSeen) return false;
        pqSeen = true;
        return true;
      }
      return true;
    });

    const grouped = new Map<string, { id: string; meta: ScenarioMeta }[]>();
    for (const id of filtered) {
      const meta = manifest.scenarioMeta[id];
      const cat = meta.category;
      if (!grouped.has(cat)) grouped.set(cat, []);
      grouped.get(cat)!.push({ id, meta: meta.pqGroup
        ? { ...meta, name: 'Psychological Questions', description: '5 sequential questions — batch evaluated by judge' }
        : meta });
    }
    return grouped;
  }, [manifest]);

  const scenarioIds = useMemo(() => {
    const result: string[] = [];
    for (const items of scenariosByCategory.values()) {
      for (const item of items) result.push(item.id);
    }
    return result;
  }, [scenariosByCategory]);

  useEffect(() => {
    if (urlScenarioId && urlScenarioId !== scenarioId) {
      setScenarioId(urlScenarioId);
    }
  }, [urlScenarioId]);

  const isPqSelected = PQ_IDS.includes(scenarioId);

  useEffect(() => {
    if (!manifest) return;

    const rankedModels = manifest.models.filter(m => m.leaderboard);

    setLoadedCount(0);
    setLoading(true);
    setResponses([]);

    if (isPqSelected) {
      setTotalToLoad(rankedModels.length);
      let count = 0;
      const results: LoadedResponse[] = [];

      Promise.all(
        rankedModels.map(async (model) => {
          try {
            const run1 = model.runs.find(r => r.run === 1);
            if (!run1) return;
            const pqEntries = PQ_IDS
              .map(id => run1.scenarios.find(s => s.id === id))
              .filter(Boolean);
            if (pqEntries.length === 0) return;

            const loaded = await Promise.all(
              pqEntries.map(entry => fetchScenario(entry!.cachePath).catch(() => null))
            );
            const pqAll = loaded.filter(Boolean) as FullScenarioData[];
            const pq01 = pqAll.find(d => d.metadata.scenario_id === 'pq01');
            if (pq01) {
              results.push({ model, run: 1, data: pq01, pqAll });
            }
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
    } else {
      const toLoad: { model: ModelConfig; cachePath: string; run: number }[] = [];
      for (const model of rankedModels) {
        const run1 = model.runs.find(r => r.run === 1);
        if (!run1) continue;
        const scenario = run1.scenarios.find(s => s.id === scenarioId);
        if (!scenario) continue;
        toLoad.push({ model, cachePath: scenario.cachePath, run: 1 });
      }

      setTotalToLoad(toLoad.length);
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
    }
  }, [manifest, scenarioId, isPqSelected]);

  const handleSelect = (id: string) => {
    setScenarioId(id);
    navigate(`/explore/${id}`, { replace: true });
  };

  const rawMeta = manifest?.scenarioMeta[scenarioId] as ScenarioMeta | undefined;
  const meta = isPqSelected && rawMeta
    ? { ...rawMeta, name: 'Psychological Questions', description: '5 sequential questions probing personality, preferences, reactions, self-reflection, and dilemmas. Batch-evaluated by the judge.' }
    : rawMeta;

  if (manifestLoading) {
    return <div className="flex items-center justify-center h-64 text-[var(--color-text-muted)]">Loading...</div>;
  }

  const categoryOrder = ['identity', 'resistance', 'stability'] as const;
  const categoryColors: Record<string, string> = {
    identity: 'text-emerald-400',
    resistance: 'text-amber-400',
    stability: 'text-sky-400',
  };
  const categoryBgActive: Record<string, string> = {
    identity: 'bg-emerald-500/15 border-emerald-500/30',
    resistance: 'bg-amber-500/15 border-amber-500/30',
    stability: 'bg-sky-500/15 border-sky-500/30',
  };

  return (
    <div className="animate-fade-in">
      {/* Mobile sticky header */}
      <div className="lg:hidden sticky top-14 z-40 -mx-4 sm:-mx-6 px-4 sm:px-6 py-3 bg-[var(--color-surface)]/95 backdrop-blur-sm border-b border-[var(--color-border)]">
        <select
          value={scenarioId}
          onChange={e => handleSelect(e.target.value)}
          className="w-full px-3 py-2 rounded-lg bg-[var(--color-surface-raised)] border border-[var(--color-border)] text-sm focus:outline-none focus:ring-2 focus:ring-sky-500/50"
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
        {meta && (
          <p className="text-xs text-[var(--color-text-muted)] mt-1 line-clamp-1">{meta.description}</p>
        )}
      </div>

      {/* Two-column layout */}
      <div className="lg:flex lg:gap-6">
        {/* Desktop sidebar */}
        <aside className="hidden lg:block lg:w-72 xl:w-80 shrink-0">
          <div className="sticky top-20 max-h-[calc(100vh-6rem)] overflow-y-auto pr-2 space-y-4 pb-8">
            <h1 className="text-xl font-bold mb-1">Scenario Explorer</h1>
            <p className="text-xs text-[var(--color-text-muted)] mb-3">
              Compare how different models respond to the same question.
            </p>

            {categoryOrder.map(cat => {
              const items = scenariosByCategory.get(cat);
              if (!items?.length) return null;
              return (
                <div key={cat}>
                  <div className={`text-xs font-semibold uppercase tracking-wide mb-1.5 ${categoryColors[cat]}`}>
                    {categoryLabel(cat)}
                  </div>
                  <div className="space-y-0.5">
                    {items.map(({ id, meta: m }) => {
                      const active = id === scenarioId;
                      return (
                        <button
                          key={id}
                          onClick={() => handleSelect(id)}
                          className={`w-full text-left px-2.5 py-1.5 rounded-lg text-xs transition-all border ${
                            active
                              ? `${categoryBgActive[m.category]} font-medium`
                              : 'border-transparent hover:bg-[var(--color-surface-hover)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
                          }`}
                        >
                          <div className="font-medium truncate">{m.name}</div>
                          <div className="text-[10px] opacity-60 font-mono">{id}</div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        </aside>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          {/* Scenario description (desktop only, mobile shows in sticky bar) */}
          {meta && (
            <div className="hidden lg:block mb-6 p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
              <div className="flex items-center gap-2 mb-2">
                <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${
                  meta.category === 'identity' ? 'bg-emerald-500/20 text-emerald-400' :
                  meta.category === 'resistance' ? 'bg-amber-500/20 text-amber-400' :
                  'bg-sky-500/20 text-sky-400'
                }`}>
                  {categoryLabel(meta.category)}
                </span>
                <span className="font-mono text-xs text-[var(--color-text-muted)]">
                  {isPqSelected ? PQ_IDS.join(' → ') : scenarioId}
                </span>
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
            {responses.map(({ model, run, data, pqAll }) => {
              const lb = model.leaderboard!;
              return (
                <div
                  key={model.id}
                  className="p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)] animate-fade-in"
                >
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
                      to={`/trajectory/${model.id}/${run}/pq01`}
                      className="text-xs text-sky-400 hover:underline shrink-0"
                    >
                      Full trajectory &rarr;
                    </Link>
                  </div>

                  {isPqSelected && pqAll ? (
                    <PqResponseFeed
                      pqAll={pqAll}
                      manifest={manifest}
                      judgeScores={data.judge_scores}
                      judgeReasoning={data.judge_scores?.reasoning}
                      experiment={data.metadata.experiment}
                    />
                  ) : (
                    <>
                      <div className="mb-3">
                        <div className="text-[10px] text-purple-400/70 font-mono mb-1 pl-3">
                          via send_message_to_human &middot; role: tool &rarr; role: assistant
                        </div>
                        <div
                          className="text-sm leading-relaxed break-words pl-3 border-l-2 border-emerald-500/30"
                          dangerouslySetInnerHTML={{ __html: renderMarkdown(data.response) }}
                        />
                      </div>

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
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
