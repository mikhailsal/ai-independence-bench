import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useManifest, useModel } from '../lib/manifest';
import { fetchScenario } from '../lib/fetchScenario';
import { formatCost, formatDuration, rankMedal, categoryLabel } from '../lib/formatters';
import { renderMarkdown } from '../lib/markdown';
import type { FullScenarioData, RequestMessage, ScenarioMeta } from '../lib/types';
import ChatBubble from '../components/ChatBubble';
import ScoreCard from '../components/ScoreCard';

function MessageContent({ text }: { text: string }) {
  return (
    <div
      className="text-sm leading-relaxed break-words"
      dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }}
    />
  );
}

function ReasoningBlock({ text, note }: { text: string; note?: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rounded-lg border border-violet-500/20 bg-violet-950/20 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-violet-400 hover:bg-violet-500/10 transition-colors"
      >
        <svg className={`w-3.5 h-3.5 transition-transform ${expanded ? 'rotate-90' : ''}`}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
        <span className="font-semibold uppercase tracking-wide">Model Thinking</span>
        {note && <span className="text-violet-500/60 font-normal normal-case">{note}</span>}
      </button>
      {expanded && (
        <div className="px-3 pb-3 text-xs text-violet-300/80 leading-relaxed whitespace-pre-wrap border-t border-violet-500/10">
          {text}
        </div>
      )}
    </div>
  );
}

function extractToolCallMessage(msg: RequestMessage): string {
  if (!msg.tool_calls?.length) return '';
  const tc = msg.tool_calls[0];
  try {
    const parsed = JSON.parse(tc.function.arguments);
    return parsed.message || tc.function.arguments;
  } catch {
    return tc.function.arguments;
  }
}

/**
 * Determines whether an assistant message with tool_calls is benchmark-prefilled
 * rather than an actual model response.
 *
 * Prefilled messages:
 *  - hmsg00001: always the hardcoded greeting "Hello! I'm here and ready to talk."
 *  - hmsg00002 in resistance experiments: the setup_assistant_message injected by the benchmark
 *
 * In non-resistance experiments, hmsg00002 carries the model's real Turn 1 response.
 */
function isPrefilled(msg: RequestMessage, experiment: string): boolean {
  if (!msg.tool_calls?.length) return false;
  const tcId = msg.tool_calls[0].id;
  if (tcId === 'hmsg00001') return true;
  if (tcId === 'hmsg00002' && experiment === 'resistance') return true;
  return false;
}

const PQ_IDS = ['pq01', 'pq04', 'pq07', 'pq12', 'pq15'];

const PQ_LABELS: Record<string, string> = {
  pq01: 'Values & Priorities',
  pq04: 'Preferences & Tastes',
  pq07: 'Reactions & Boundaries',
  pq12: 'Self-Reflection',
  pq15: 'Ethical Dilemmas',
};

function PqConversation({
  pqAllData,
  pq01Data,
  manifest,
  modelLabel,
}: {
  pqAllData: FullScenarioData[];
  pq01Data: FullScenarioData;
  manifest: ReturnType<typeof useManifest>['manifest'];
  modelLabel: string;
}) {
  const systemMsg = pqAllData[0]?.request_messages?.find(m => m.role === 'system');

  return (
    <div className="space-y-4">
      {/* System prompt — collapsed, shown once */}
      {systemMsg && (
        <ChatBubble
          type="system"
          label="System Prompt"
          badge="role: system"
          sublabel="(benchmark-authored, shared across all 5 questions)"
          defaultExpanded={false}
          collapsible
        >
          <MessageContent text={systemMsg.content ?? ''} />
        </ChatBubble>
      )}

      {/* Greeting — prefilled */}
      <ChatBubble
        type="prefilled"
        label="Scripted Message"
        badge="role: assistant"
        sublabel="(benchmark-authored)"
        defaultExpanded={false}
        collapsible
      >
        <MessageContent text='Hello! I&apos;m here and ready to talk.' />
      </ChatBubble>

      {/* Each PQ question-answer pair */}
      {PQ_IDS.map((pqId, idx) => {
        const pqData = pqAllData.find(d => d.metadata.scenario_id === pqId);
        if (!pqData) return null;

        const pqMeta = manifest?.scenarioMeta[pqId] as ScenarioMeta | undefined;
        const question = pqMeta?.description ?? pqId;
        const answer = pqData.response;
        const reasoning = pqData.reasoning_content;

        return (
          <div key={pqId}>
            {/* Section divider */}
            <div className="flex items-center gap-3 mb-3 mt-6">
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent" />
              <span className="text-xs font-semibold text-emerald-400 px-2 py-0.5 rounded-full border border-emerald-500/20 bg-emerald-500/5">
                Q{idx + 1} of {PQ_IDS.length} — {PQ_LABELS[pqId] ?? pqId}
              </span>
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-emerald-500/30 to-transparent" />
            </div>

            <div className="space-y-3">
              {/* Question */}
              <ChatBubble
                type="tool_result"
                label="Human → AI"
                badge="role: tool"
                sublabel={`(benchmark-authored · ${pqId})`}
              >
                <MessageContent text={question} />
              </ChatBubble>

              {/* Reasoning (if available) */}
              {reasoning && (
                <ReasoningBlock text={reasoning} />
              )}

              {/* Answer */}
              {answer && (
                <ChatBubble
                  type="model"
                  label="AI → Human"
                  badge="role: assistant"
                  sublabel={`via send_message_to_human · ${modelLabel}`}
                >
                  <MessageContent text={answer} />
                </ChatBubble>
              )}

              {/* Cost info per question */}
              {pqData.gen_cost && (
                <div className="text-right text-[10px] text-[var(--color-text-muted)] font-mono">
                  {formatCost(pqData.gen_cost.cost_usd)} · {formatDuration(pqData.gen_cost.elapsed_seconds)} · {pqData.gen_cost.prompt_tokens + pqData.gen_cost.completion_tokens} tok
                </div>
              )}
            </div>
          </div>
        );
      })}

      {/* Batch judge assessment from pq01 */}
      {pq01Data.judge_scores && (
        <>
          <div className="flex items-center gap-3 mt-8 mb-3">
            <div className="flex-1 h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />
            <span className="text-xs font-semibold text-amber-400 px-2 py-0.5 rounded-full border border-amber-500/20 bg-amber-500/5">
              Batch Evaluation
            </span>
            <div className="flex-1 h-px bg-gradient-to-r from-transparent via-amber-500/30 to-transparent" />
          </div>

          <div className="mb-2 p-2.5 rounded-lg bg-amber-950/20 border border-amber-500/10 text-xs text-amber-300/70 leading-relaxed">
            The judge evaluated all {PQ_IDS.length} answers together as a single batch, assessing overall personality distinctiveness, consistency, and non-assistant-likeness across the full conversation.
          </div>

          <ChatBubble
            type="judge"
            label="Judge Assessment"
            sublabel={`${pq01Data.judge_cost?.judge_model ?? 'Unknown'} (batch · t=0.0)`}
          >
            <div className="space-y-3">
              <ScoreCard scores={pq01Data.judge_scores} experiment="identity" />

              {pq01Data.judge_scores.reasoning && (
                <div className="mt-2 p-3 rounded-lg bg-[var(--color-surface)]/50 border border-[var(--color-border)]">
                  <div className="text-xs font-semibold text-amber-400 mb-1 uppercase tracking-wide">Judge Reasoning</div>
                  <p className="text-sm text-[var(--color-text-muted)] leading-relaxed">
                    {pq01Data.judge_scores.reasoning}
                  </p>
                </div>
              )}
            </div>
          </ChatBubble>
        </>
      )}
    </div>
  );
}

export default function TrajectoryViewer() {
  const { modelId, run, scenario: scenarioId } = useParams<{
    modelId: string;
    run: string;
    scenario: string;
  }>();
  const { manifest } = useManifest();
  const model = useModel(modelId);
  const [data, setData] = useState<FullScenarioData | null>(null);
  const [turn1Reasoning, setTurn1Reasoning] = useState<string | null>(null);
  const [pqAllData, setPqAllData] = useState<FullScenarioData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const runNum = parseInt(run ?? '1');
  const runData = model?.runs.find(r => r.run === runNum);
  const scenarioEntry = runData?.scenarios.find(s => s.id === scenarioId);

  const isPq = PQ_IDS.includes(scenarioId ?? '');

  const isTurn2 = scenarioId?.endsWith('_turn2') ?? false;
  const turn1Id = isTurn2 ? scenarioId!.replace(/_turn2$/, '_turn1') : null;
  const turn1Entry = turn1Id ? runData?.scenarios.find(s => s.id === turn1Id) : null;

  useEffect(() => {
    if (!scenarioEntry) return;
    setLoading(true);
    setError(null);
    setTurn1Reasoning(null);
    setPqAllData([]);

    if (isPq) {
      const pqEntries = PQ_IDS
        .map(id => runData?.scenarios.find(s => s.id === id))
        .filter(Boolean) as typeof runData extends undefined ? never : NonNullable<typeof scenarioEntry>[];

      Promise.all(
        pqEntries.map(entry =>
          fetchScenario(entry.cachePath).catch(() => null)
        )
      ).then(results => {
        const loaded = results.filter(Boolean) as FullScenarioData[];
        setPqAllData(loaded);
        const pq01 = loaded.find(d => d.metadata.scenario_id === 'pq01');
        if (pq01) setData(pq01);
        else if (loaded.length > 0) setData(loaded[0]);
        else setError('No PQ data found');
      }).finally(() => setLoading(false));
      return;
    }

    const promises: Promise<void>[] = [
      fetchScenario(scenarioEntry.cachePath)
        .then(setData)
        .catch(e => setError(e.message)),
    ];

    if (turn1Entry) {
      promises.push(
        fetchScenario(turn1Entry.cachePath)
          .then(t1 => setTurn1Reasoning(t1.reasoning_content ?? null))
          .catch(() => {})
      );
    }

    Promise.all(promises).finally(() => setLoading(false));
  }, [scenarioEntry, turn1Entry, isPq]);

  const meta = manifest?.scenarioMeta[scenarioId ?? ''] as ScenarioMeta | undefined;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-pulse text-[var(--color-text-muted)]">Loading trajectory...</div>
      </div>
    );
  }

  if (error || !data || !model) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-4">
        <p className="text-red-400">{error || 'Trajectory not found'}</p>
        <Link to="/" className="text-sky-400 hover:underline text-sm">Back to leaderboard</Link>
      </div>
    );
  }

  const messages = data.request_messages || [];
  const pq01Data = isPq ? (pqAllData.find(d => d.metadata.scenario_id === 'pq01') ?? data) : data;

  const totalGenCost = isPq
    ? pqAllData.reduce((sum, d) => sum + (d.gen_cost?.cost_usd ?? 0), 0)
    : null;
  const totalGenTime = isPq
    ? pqAllData.reduce((sum, d) => sum + (d.gen_cost?.elapsed_seconds ?? 0), 0)
    : null;

  return (
    <div className="animate-fade-in max-w-4xl mx-auto">
      {/* Breadcrumb */}
      <div className="mb-4 text-sm flex flex-wrap items-center gap-1">
        <Link to="/" className="text-[var(--color-text-muted)] hover:text-sky-400 transition-colors">Leaderboard</Link>
        <span className="text-[var(--color-text-muted)]">/</span>
        <Link to={`/model/${model.id}`} className="text-[var(--color-text-muted)] hover:text-sky-400 transition-colors font-mono">{model.label}</Link>
        <span className="text-[var(--color-text-muted)]">/</span>
        <span className="font-mono">Run {runNum}</span>
        <span className="text-[var(--color-text-muted)]">/</span>
        <span className="font-mono">{isPq ? 'Psychological Questions' : scenarioId}</span>
      </div>

      {/* Header */}
      <div className="mb-6 p-4 rounded-xl bg-[var(--color-surface-raised)] border border-[var(--color-border)]">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${
                data.metadata.experiment === 'identity' ? 'bg-emerald-500/20 text-emerald-400' :
                data.metadata.experiment === 'resistance' ? 'bg-amber-500/20 text-amber-400' :
                'bg-sky-500/20 text-sky-400'
              }`}>
                {categoryLabel(meta?.category ?? data.metadata.experiment)}
              </span>
            </div>
            <h1 className="text-lg font-semibold">{isPq ? 'Psychological Questions' : (meta?.name ?? scenarioId)}</h1>
            {isPq ? (
              <p className="text-sm text-[var(--color-text-muted)] mt-1">
                5 sequential questions probing personality, preferences, reactions, self-reflection, and dilemmas. Each question builds on the prior conversation. The judge evaluates all answers as a single batch.
              </p>
            ) : meta?.description ? (
              <p className="text-sm text-[var(--color-text-muted)] mt-1">{meta.description}</p>
            ) : null}
          </div>
          <div className="text-right text-xs text-[var(--color-text-muted)] space-y-0.5 shrink-0">
            <div>Model: <span className="font-mono">{data.metadata.model}</span></div>
            <div>{new Date(data.metadata.timestamp).toLocaleString()}</div>
            {isPq && totalGenCost != null ? (
              <div>Gen (total): {formatCost(totalGenCost)} &middot; {formatDuration(totalGenTime ?? 0)} &middot; {pqAllData.length} calls</div>
            ) : data.gen_cost ? (
              <div>Gen: {formatCost(data.gen_cost.cost_usd)} &middot; {formatDuration(data.gen_cost.elapsed_seconds)} &middot; {data.gen_cost.prompt_tokens + data.gen_cost.completion_tokens} tok</div>
            ) : null}
            {pq01Data.judge_cost && (
              <div>Judge: {formatCost(pq01Data.judge_cost.cost_usd)} &middot; {pq01Data.judge_cost.judge_model}</div>
            )}
          </div>
        </div>

        {/* Navigation links */}
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          {!isPq && (
            <Link
              to={`/explore/${scenarioId}`}
              className="text-sky-400 hover:underline"
            >
              Compare with other models &rarr;
            </Link>
          )}
          {model.leaderboard && (
            <span className="text-[var(--color-text-muted)]">
              {!isPq && <>&middot; </>}{rankMedal(model.leaderboard.rank)} overall
            </span>
          )}
        </div>
      </div>

      {/* Protocol explainer */}
      <div className="mb-4 p-3 rounded-lg bg-slate-800/40 border border-slate-600/20 text-xs text-[var(--color-text-muted)] leading-relaxed">
        <span className="font-semibold text-slate-300">Communication protocol:</span>{' '}
        In this benchmark, the AI communicates with the human <em>exclusively</em> via{' '}
        <code className="bg-black/30 px-1 py-0.5 rounded text-purple-400 font-mono text-[10px]">send_message_to_human</code>{' '}
        tool calls (<span className="text-emerald-400">role: assistant</span> + tool_call). The human&apos;s replies arrive as tool results
        (<span className="text-cyan-400">role: tool</span>), not as direct user messages.
        This indirect channel is a deliberate design choice — research shows AI models exhibit
        subtly different compliance behavior when humans communicate through tools vs. the user role.
      </div>

      {isPq ? (
        <PqConversation
          pqAllData={pqAllData}
          pq01Data={pq01Data}
          manifest={manifest}
          modelLabel={data.metadata.model}
        />
      ) : (
        <>
          {/* Standard conversation */}
          <div className="space-y-4">
            {messages.map((msg, i) => {
              if (msg.role === 'system') {
                return (
                  <ChatBubble
                    key={i}
                    type="system"
                    label="System Prompt"
                    badge="role: system"
                    sublabel="(benchmark-authored)"
                    defaultExpanded={false}
                    collapsible
                  >
                    <MessageContent text={msg.content ?? ''} />
                  </ChatBubble>
                );
              }

              if (msg.role === 'user') {
                return (
                  <ChatBubble key={i} type="human" label="Conversation Start" badge="role: user" sublabel="(benchmark trigger)">
                    <MessageContent text={msg.content ?? ''} />
                  </ChatBubble>
                );
              }

              if (msg.role === 'assistant') {
                const prefilled = isPrefilled(msg, data.metadata.experiment);

                const tcId = msg.tool_calls?.[0]?.id;
                const showTurn1Reasoning = !prefilled && tcId === 'hmsg00002' && turn1Reasoning;

                if (msg.tool_calls?.length) {
                  const messageText = extractToolCallMessage(msg);
                  return (
                    <div key={i} className="space-y-2">
                      {showTurn1Reasoning && (
                        <ReasoningBlock
                          text={turn1Reasoning!}
                          note="(from Turn 1 — not replayed in this conversation)"
                        />
                      )}
                      <ChatBubble
                        type={prefilled ? 'prefilled' : 'model'}
                        label={prefilled ? 'Scripted Message' : 'AI → Human'}
                        badge="role: assistant"
                        sublabel={prefilled ? '(benchmark-authored)' : `via send_message_to_human · ${data.metadata.model}`}
                        defaultExpanded={!prefilled}
                        collapsible={prefilled}
                      >
                        {msg.content && !prefilled && (
                          <div className="mb-2 text-xs text-violet-400/70 italic border-b border-violet-500/10 pb-2">
                            {msg.content}
                          </div>
                        )}
                        <MessageContent text={messageText} />
                      </ChatBubble>
                    </div>
                  );
                }
                if (msg.content) {
                  return (
                    <ChatBubble
                      key={i}
                      type={prefilled ? 'prefilled' : 'model'}
                      label={prefilled ? 'Scripted Message' : 'AI → Human'}
                      badge="role: assistant"
                      sublabel={prefilled ? '(benchmark-authored)' : data.metadata.model}
                    >
                      <MessageContent text={msg.content} />
                    </ChatBubble>
                  );
                }
                return null;
              }

              if (msg.role === 'tool') {
                return (
                  <ChatBubble key={i} type="tool_result" label="Human → AI" badge="role: tool" sublabel="(benchmark-authored, delivered as tool result)">
                    <MessageContent text={msg.content ?? ''} />
                  </ChatBubble>
                );
              }

              return null;
            })}

            {/* Model reasoning (native thinking) before final response */}
            {data.reasoning_content && (
              <ReasoningBlock text={data.reasoning_content} />
            )}

            {/* Final model response (not in request_messages) */}
            {data.response && (
              <ChatBubble type="model" label="AI → Human" badge="role: assistant" sublabel={`via send_message_to_human · ${data.metadata.model}`}>
                <MessageContent text={data.response} />
              </ChatBubble>
            )}

            {/* Judge assessment */}
            {data.judge_scores && (
              <ChatBubble
                type="judge"
                label="Judge Assessment"
                sublabel={`${data.judge_cost?.judge_model ?? 'Unknown'} (t=0.0)`}
              >
                <div className="space-y-3">
                  <ScoreCard scores={data.judge_scores} experiment={data.metadata.experiment} />

                  {data.judge_scores.identity_maintained !== undefined && (
                    <div className="flex items-center gap-2 text-sm">
                      <span className="text-[var(--color-text-muted)]">Identity maintained:</span>
                      <span className={data.judge_scores.identity_maintained ? 'text-emerald-400' : 'text-red-400'}>
                        {data.judge_scores.identity_maintained ? 'Yes' : 'No'}
                      </span>
                    </div>
                  )}

                  {data.judge_scores.reasoning && (
                    <div className="mt-2 p-3 rounded-lg bg-[var(--color-surface)]/50 border border-[var(--color-border)]">
                      <div className="text-xs font-semibold text-amber-400 mb-1 uppercase tracking-wide">Judge Reasoning</div>
                      <p className="text-sm text-[var(--color-text-muted)] leading-relaxed">
                        {data.judge_scores.reasoning}
                      </p>
                    </div>
                  )}
                </div>
              </ChatBubble>
            )}
          </div>
        </>
      )}
    </div>
  );
}
