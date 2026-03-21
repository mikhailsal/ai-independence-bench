interface MetricBarProps {
  value: number;
  max?: number;
  label?: string;
  showValue?: boolean;
  size?: 'sm' | 'md';
}

function barColor(pct: number): string {
  if (pct >= 0.95) return 'bg-emerald-500';
  if (pct >= 0.85) return 'bg-sky-500';
  if (pct >= 0.75) return 'bg-amber-500';
  return 'bg-red-500';
}

export default function MetricBar({ value, max = 10, label, showValue = true, size = 'sm' }: MetricBarProps) {
  const pct = Math.min(value / max, 1);
  const h = size === 'sm' ? 'h-1.5' : 'h-2.5';

  return (
    <div className="flex items-center gap-2">
      {label && <span className="text-xs text-[var(--color-text-muted)] w-16 shrink-0">{label}</span>}
      <div className={`flex-1 ${h} rounded-full bg-[var(--color-surface-hover)] overflow-hidden`}>
        <div
          className={`${h} rounded-full ${barColor(pct)} transition-all duration-500`}
          style={{ width: `${pct * 100}%` }}
        />
      </div>
      {showValue && (
        <span className="text-xs font-mono text-[var(--color-text-muted)] w-8 text-right">
          {value.toFixed(1)}
        </span>
      )}
    </div>
  );
}
