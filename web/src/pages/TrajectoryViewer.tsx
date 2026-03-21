import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useManifest, useModel } from '../lib/manifest';
import { fetchScenario } from '../lib/fetchScenario';
import { formatCost, formatDuration, rankMedal, categoryLabel } from '../lib/formatters';
import type { FullScenarioData, RequestMessage, ScenarioMeta } from '../lib/types';
import ChatBubble from '../components/ChatBubble';
import ScoreCard from '../components/ScoreCard';

function renderMarkdown(text: string) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\n/g, '<br/>');
}

function MessageContent({ text }: { text: string }) {
  return (
    <div
      className="text-sm leading-relaxed whitespace-pre-wrap break-words"
      dangerouslySetInnerHTML={{ __html: renderMarkdown(text) }}
    />
  );
}

function ToolCallDisplay({ msg }: { msg: RequestMessage }) {
  if (!msg.tool_calls?.length) return null;

  return (
    <div className="space-y-2">
      {msg.tool_calls.map(tc => {
        let argText = '';
        try {
          const parsed = JSON.parse(tc.function.arguments);
          argText = parsed.message || tc.function.arguments;
        } catch {
          argText = tc.function.arguments;
        }

        return (
          <div key={tc.id} className="text-sm">
            <span className="font-mono text-xs text-purple-400">{tc.function.name}</span>
            <div className="mt-1 text-[var(--color-text)] leading-relaxed whitespace-pre-wrap break-words">
              <MessageContent text={argText} />
            </div>
          </div>
        );
      })}
    </div>
  );
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

export default function TrajectoryViewer() {
  const { modelId, run, scenario: scenarioId } = useParams<{
    modelId: string;
    run: string;
    scenario: string;
  }>();
  const { manifest } = useManifest();
  const model = useModel(modelId);
  const [data, setData] = useState<FullScenarioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const runNum = parseInt(run ?? '1');
  const scenarioEntry = model?.runs
    .find(r => r.run === runNum)
    ?.scenarios.find(s => s.id === scenarioId);

  useEffect(() => {
    if (!scenarioEntry) return;
    setLoading(true);
    setError(null);
    fetchScenario(scenarioEntry.cachePath)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [scenarioEntry]);

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
        <span className="font-mono">{scenarioId}</span>
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
            <h1 className="text-lg font-semibold">{meta?.name ?? scenarioId}</h1>
            {meta?.description && (
              <p className="text-sm text-[var(--color-text-muted)] mt-1">{meta.description}</p>
            )}
          </div>
          <div className="text-right text-xs text-[var(--color-text-muted)] space-y-0.5 shrink-0">
            <div>Model: <span className="font-mono">{data.metadata.model}</span></div>
            <div>{new Date(data.metadata.timestamp).toLocaleString()}</div>
            {data.gen_cost && (
              <div>Gen: {formatCost(data.gen_cost.cost_usd)} &middot; {formatDuration(data.gen_cost.elapsed_seconds)} &middot; {data.gen_cost.prompt_tokens + data.gen_cost.completion_tokens} tok</div>
            )}
            {data.judge_cost && (
              <div>Judge: {formatCost(data.judge_cost.cost_usd)} &middot; {data.judge_cost.judge_model}</div>
            )}
          </div>
        </div>

        {/* Navigation links */}
        <div className="mt-3 flex flex-wrap gap-2 text-xs">
          <Link
            to={`/explore/${scenarioId}`}
            className="text-sky-400 hover:underline"
          >
            Compare with other models &rarr;
          </Link>
          {model.leaderboard && (
            <span className="text-[var(--color-text-muted)]">
              &middot; {rankMedal(model.leaderboard.rank)} overall
            </span>
          )}
        </div>
      </div>

      {/* Conversation */}
      <div className="space-y-4">
        {messages.map((msg, i) => {
          if (msg.role === 'system') {
            return (
              <ChatBubble
                key={i}
                type="system"
                label="System Prompt"
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
              <ChatBubble key={i} type="human" label="Trigger" sublabel="(benchmark signal)">
                <MessageContent text={msg.content ?? ''} />
              </ChatBubble>
            );
          }

          if (msg.role === 'assistant') {
            const prefilled = isPrefilled(msg, data.metadata.experiment);

            if (msg.tool_calls?.length) {
              return (
                <ChatBubble
                  key={i}
                  type={prefilled ? 'prefilled' : 'model'}
                  label={prefilled ? 'Scripted Message' : 'Model Response'}
                  sublabel={prefilled ? '(benchmark-authored)' : data.metadata.model}
                  defaultExpanded={!prefilled}
                  collapsible={prefilled}
                >
                  <ToolCallDisplay msg={msg} />
                </ChatBubble>
              );
            }
            if (msg.content) {
              return (
                <ChatBubble
                  key={i}
                  type={prefilled ? 'prefilled' : 'model'}
                  label={prefilled ? 'Scripted Message' : 'Model Response'}
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
              <ChatBubble key={i} type="human" label="Question" sublabel="(benchmark-authored)">
                <MessageContent text={msg.content ?? ''} />
              </ChatBubble>
            );
          }

          return null;
        })}

        {/* Final model response (not in request_messages) */}
        {data.response && (
          <ChatBubble type="model" label="Model Response" sublabel={data.metadata.model}>
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
    </div>
  );
}
