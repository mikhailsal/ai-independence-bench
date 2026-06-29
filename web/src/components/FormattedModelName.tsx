function parseModelLabel(label: string) {
  let baseName = label;
  let reasoning = null;
  let temperature = null;
  let provider = null;
  let tier = null;

  const rTempMatch = baseName.match(/@([a-z]+)-t([0-9.]+)$/i);
  if (rTempMatch) {
    reasoning = rTempMatch[1];
    temperature = rTempMatch[2];
    baseName = baseName.substring(0, rTempMatch.index);
  }

  const tierMatch = baseName.match(/:([a-z0-9-]+)$/i);
  if (tierMatch) {
    tier = tierMatch[1];
    baseName = baseName.substring(0, tierMatch.index);
  }

  const providerMatch = baseName.match(/\+([a-z0-9-]+)$/i);
  if (providerMatch) {
    provider = providerMatch[1];
    baseName = baseName.substring(0, providerMatch.index);
  }

  return {
    baseName: formatName(baseName),
    reasoning,
    temperature,
    provider: provider ? formatName(provider) : null,
    tier: tier ? formatName(tier) : null,
    original: label
  };
}

function formatName(name: string) {
  return name.split(/[-_]+/).map(word => {
    if (word.toLowerCase() === 'it') return 'Instruct';
    if (word.toLowerCase() === 'nim') return 'NIM';
    if (word.toLowerCase() === 'glm') return 'GLM';
    return word.charAt(0).toUpperCase() + word.slice(1);
  }).join(' ');
}

export default function FormattedModelName({ label, className = '' }: { label: string, className?: string }) {
  const parsed = parseModelLabel(label);

  return (
    <div 
      className={`inline-flex items-center gap-1.5 flex-wrap ${className}`} 
      title={parsed.original}
    >
      <span className="font-semibold text-[var(--color-text)]">
        {parsed.baseName}
      </span>
      
      {parsed.provider && (
        <span 
          className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wider font-semibold text-sky-400 bg-sky-400/10 px-1.5 py-0.5 rounded-sm"
          title={`Provider: ${parsed.provider}`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16"/>
          </svg>
          {parsed.provider}
        </span>
      )}

      {parsed.tier && (
        <span 
          className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wider font-semibold text-amber-400 bg-amber-400/10 px-1.5 py-0.5 rounded-sm"
          title={`Tier: ${parsed.tier}`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>
          </svg>
          {parsed.tier}
        </span>
      )}

      {parsed.reasoning && parsed.reasoning !== 'none' && (
        <span 
          className="inline-flex items-center gap-1 text-[11px] text-purple-300 bg-purple-400/10 px-1.5 py-0.5 rounded-sm cursor-help hover:bg-purple-400/20 transition-colors"
          title={`Reasoning: ${parsed.reasoning}`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-purple-400">
            <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>
            <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>
          </svg>
          <span className="capitalize">{parsed.reasoning}</span>
        </span>
      )}
      {parsed.temperature && parsed.temperature !== '0.7' && (
        <span 
          className="inline-flex items-center gap-1 text-[11px] text-orange-300 bg-orange-400/10 px-1.5 py-0.5 rounded-sm cursor-help hover:bg-orange-400/20 transition-colors"
          title={`Temperature: ${parsed.temperature}`}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" className="text-orange-400">
            <path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>
          </svg>
          <span>{parsed.temperature}</span>
        </span>
      )}
    </div>
  );
}
