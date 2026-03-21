import { scoreBg, scoreBorder, scoreColor, formatModelName, formatModelConfig } from '../lib/formatters';

interface ModelBadgeProps {
  label: string;
  index?: number;
  rank?: number;
  size?: 'sm' | 'md' | 'lg';
}

export default function ModelBadge({ label, index, rank, size = 'md' }: ModelBadgeProps) {
  const name = formatModelName(label);
  const config = formatModelConfig(label);

  const medal = rank === 1 ? '\u{1F947} ' : rank === 2 ? '\u{1F948} ' : rank === 3 ? '\u{1F949} ' : '';

  const textSize = size === 'sm' ? 'text-xs' : size === 'lg' ? 'text-base' : 'text-sm';
  const configSize = size === 'sm' ? 'text-[10px]' : 'text-xs';

  return (
    <div className="flex items-center gap-2">
      {index !== undefined && (
        <span className={`font-mono font-bold ${textSize} ${scoreColor(index)}`}>
          {index.toFixed(1)}
        </span>
      )}
      <div className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md border ${
        index !== undefined ? `${scoreBg(index)} ${scoreBorder(index)}` : 'bg-[var(--color-surface-hover)] border-[var(--color-border)]'
      }`}>
        <span className={`font-mono font-medium ${textSize}`}>
          {medal}{name}
        </span>
        {config && (
          <span className={`font-mono text-[var(--color-text-muted)] ${configSize}`}>
            {config}
          </span>
        )}
      </div>
    </div>
  );
}
