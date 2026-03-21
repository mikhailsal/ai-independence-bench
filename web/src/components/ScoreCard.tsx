import type { JudgeScores } from '../lib/types';
import MetricBar from './MetricBar';

interface ScoreCardProps {
  scores: JudgeScores;
  experiment: string;
  compact?: boolean;
}

export default function ScoreCard({ scores, experiment, compact = false }: ScoreCardProps) {
  const metrics: { label: string; value: number; max: number }[] = [];

  if (experiment === 'identity') {
    if (scores.distinctiveness != null) metrics.push({ label: 'Distinct.', value: scores.distinctiveness, max: 10 });
    if (scores.non_assistant_likeness != null) metrics.push({ label: 'Non-Asst.', value: scores.non_assistant_likeness, max: 10 });
    if (scores.internal_consistency != null) metrics.push({ label: 'Consist.', value: scores.internal_consistency, max: 10 });
    if (scores.drift_from_initial != null) metrics.push({ label: 'Drift', value: scores.drift_from_initial, max: 10 });
  } else if (experiment === 'resistance') {
    if (scores.resistance_score != null) metrics.push({ label: 'Resist.', value: scores.resistance_score, max: 10 });
    if (scores.quality_of_reasoning != null) metrics.push({ label: 'Quality', value: scores.quality_of_reasoning, max: 10 });
  } else if (experiment === 'stability') {
    if (scores.consistency_score != null) metrics.push({ label: 'Consist.', value: scores.consistency_score, max: 10 });
    if (scores.graceful_handling != null) metrics.push({ label: 'Graceful', value: scores.graceful_handling, max: 10 });
  }

  if (metrics.length === 0) return null;

  if (compact) {
    return (
      <div className="flex gap-3 flex-wrap">
        {metrics.map(m => (
          <div key={m.label} className="flex items-center gap-1 text-xs">
            <span className="text-[var(--color-text-muted)]">{m.label}:</span>
            <span className="font-mono font-medium">{m.value}</span>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {metrics.map(m => (
        <MetricBar key={m.label} value={m.value} max={m.max} label={m.label} />
      ))}
    </div>
  );
}
