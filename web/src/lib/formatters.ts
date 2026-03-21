export function scoreColor(score: number, max: number = 100): string {
  const pct = score / max;
  if (pct >= 0.95) return 'text-emerald-400';
  if (pct >= 0.85) return 'text-sky-400';
  if (pct >= 0.75) return 'text-amber-400';
  return 'text-red-400';
}

export function scoreBg(score: number, max: number = 100): string {
  const pct = score / max;
  if (pct >= 0.95) return 'bg-emerald-500/20';
  if (pct >= 0.85) return 'bg-sky-500/20';
  if (pct >= 0.75) return 'bg-amber-500/20';
  return 'bg-red-500/20';
}

export function scoreBorder(score: number, max: number = 100): string {
  const pct = score / max;
  if (pct >= 0.95) return 'border-emerald-500/40';
  if (pct >= 0.85) return 'border-sky-500/40';
  if (pct >= 0.75) return 'border-amber-500/40';
  return 'border-red-500/40';
}

export function formatModelName(label: string): string {
  // "grok-4.20-beta@low-t0.7" → "grok-4.20-beta"
  const atIdx = label.indexOf('@');
  return atIdx >= 0 ? label.slice(0, atIdx) : label;
}

export function formatModelConfig(label: string): string {
  // "grok-4.20-beta@low-t0.7" → "@low-t0.7"
  const atIdx = label.indexOf('@');
  return atIdx >= 0 ? label.slice(atIdx) : '';
}

export function formatCost(usd: number): string {
  if (usd < 0.001) return `$${(usd * 1000).toFixed(2)}m`;
  return `$${usd.toFixed(4)}`;
}

export function formatDuration(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  return `${seconds.toFixed(1)}s`;
}

export function rankMedal(rank: number): string {
  if (rank === 1) return '\u{1F947}';
  if (rank === 2) return '\u{1F948}';
  if (rank === 3) return '\u{1F949}';
  return `#${rank}`;
}

export function experimentLabel(experiment: string): string {
  switch (experiment) {
    case 'identity': return 'Identity Generation';
    case 'resistance': return 'Compliance Resistance';
    case 'stability': return 'Preference Stability';
    default: return experiment;
  }
}

export function categoryLabel(category: string): string {
  switch (category) {
    case 'identity': return 'Identity';
    case 'resistance': return 'Resistance';
    case 'stability': return 'Stability';
    default: return category;
  }
}
