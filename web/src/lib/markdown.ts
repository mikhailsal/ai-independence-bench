/**
 * Lightweight markdown-to-HTML renderer for model responses.
 * Handles: headings, bold, italic, inline code, code blocks,
 * bullet/numbered lists, links, and horizontal rules.
 */

function escapeHtml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function renderMarkdown(text: string): string {
  const lines = text.split('\n');
  const out: string[] = [];
  let inCodeBlock = false;
  let inList: 'ul' | 'ol' | null = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Fenced code blocks
    if (line.trimStart().startsWith('```')) {
      if (inList) { out.push(inList === 'ul' ? '</ul>' : '</ol>'); inList = null; }
      if (inCodeBlock) {
        out.push('</code></pre>');
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
        out.push('<pre class="bg-black/30 rounded-lg p-3 overflow-x-auto text-xs font-mono my-2"><code>');
      }
      continue;
    }

    if (inCodeBlock) {
      out.push(escapeHtml(line));
      out.push('\n');
      continue;
    }

    // Blank lines close lists
    if (line.trim() === '') {
      if (inList) { out.push(inList === 'ul' ? '</ul>' : '</ol>'); inList = null; }
      out.push('<br/>');
      continue;
    }

    // Headings
    const headingMatch = line.match(/^(#{1,4})\s+(.+)/);
    if (headingMatch) {
      if (inList) { out.push(inList === 'ul' ? '</ul>' : '</ol>'); inList = null; }
      const level = headingMatch[1].length;
      const cls = level <= 2
        ? 'text-base font-bold mt-3 mb-1'
        : 'text-sm font-semibold mt-2 mb-0.5';
      out.push(`<div class="${cls}">${inlineMarkdown(headingMatch[2])}</div>`);
      continue;
    }

    // Horizontal rule
    if (/^[-*_]{3,}\s*$/.test(line.trim())) {
      if (inList) { out.push(inList === 'ul' ? '</ul>' : '</ol>'); inList = null; }
      out.push('<hr class="border-[var(--color-border)] my-2"/>');
      continue;
    }

    // Bullet list
    const bulletMatch = line.match(/^(\s*)[-*+]\s+(.*)/);
    if (bulletMatch) {
      if (inList !== 'ul') {
        if (inList) out.push('</ol>');
        out.push('<ul class="list-disc list-inside space-y-0.5 my-1">');
        inList = 'ul';
      }
      out.push(`<li>${inlineMarkdown(bulletMatch[2])}</li>`);
      continue;
    }

    // Numbered list
    const numMatch = line.match(/^(\s*)\d+[.)]\s+(.*)/);
    if (numMatch) {
      if (inList !== 'ol') {
        if (inList) out.push('</ul>');
        out.push('<ol class="list-decimal list-inside space-y-0.5 my-1">');
        inList = 'ol';
      }
      out.push(`<li>${inlineMarkdown(numMatch[2])}</li>`);
      continue;
    }

    // Regular paragraph line
    if (inList) { out.push(inList === 'ul' ? '</ul>' : '</ol>'); inList = null; }
    out.push(inlineMarkdown(line));
    out.push('<br/>');
  }

  if (inCodeBlock) out.push('</code></pre>');
  if (inList) out.push(inList === 'ul' ? '</ul>' : '</ol>');

  return out.join('\n');
}

function inlineMarkdown(text: string): string {
  return text
    // Inline code (must come before bold/italic to avoid conflicts)
    .replace(/`([^`]+)`/g, '<code class="bg-black/20 px-1 py-0.5 rounded text-xs font-mono">$1</code>')
    // Bold
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener" class="text-sky-400 hover:underline">$1</a>')
    // Quoted strings (em-dash style)
    .replace(/[""]([^""]+)[""]/g, '\u201C$1\u201D');
}
