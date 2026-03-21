interface RunSelectorProps {
  runs: number[];
  selected: number;
  onChange: (run: number) => void;
}

export default function RunSelector({ runs, selected, onChange }: RunSelectorProps) {
  return (
    <div className="flex gap-1 flex-wrap">
      {runs.map(run => (
        <button
          key={run}
          onClick={() => onChange(run)}
          className={`px-3 py-1.5 rounded-lg text-sm font-mono font-medium transition-colors ${
            run === selected
              ? 'bg-sky-500/20 text-sky-400 border border-sky-500/40'
              : 'bg-[var(--color-surface-raised)] border border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:border-[var(--color-text-muted)]'
          }`}
        >
          Run {run}
        </button>
      ))}
    </div>
  );
}
