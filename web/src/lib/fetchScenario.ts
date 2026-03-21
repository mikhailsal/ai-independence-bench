import type { FullScenarioData } from './types';

const GITHUB_RAW_BASE =
  'https://raw.githubusercontent.com/mikhailsal/ai-independence-bench/lite_v2/cache/';

const isDev = import.meta.env.DEV;
const BASE = import.meta.env.BASE_URL || '/';

const memCache = new Map<string, FullScenarioData>();

export async function fetchScenario(cachePath: string): Promise<FullScenarioData> {
  if (memCache.has(cachePath)) return memCache.get(cachePath)!;

  // In dev: Vite middleware serves local cache files
  // In prod: fetch from raw GitHub
  const url = isDev
    ? `${BASE}cache/${cachePath}`
    : `${GITHUB_RAW_BASE}${cachePath}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${cachePath}: HTTP ${response.status}`);
  }

  const data: FullScenarioData = await response.json();
  memCache.set(cachePath, data);
  return data;
}

export function clearScenarioCache() {
  memCache.clear();
}
