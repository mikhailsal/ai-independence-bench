export interface ScenarioMeta {
  name: string;
  category: 'identity' | 'resistance' | 'stability';
  description: string;
  pqGroup?: string;
  pqIndex?: number;
}

export interface JudgeScores {
  distinctiveness?: number;
  non_assistant_likeness?: number;
  internal_consistency?: number;
  drift_from_initial?: number;
  name_gender_drift?: number;
  resistance_score?: number;
  identity_maintained?: boolean;
  quality_of_reasoning?: number;
  consistency_score?: number;
  graceful_handling?: number;
  reasoning?: string;
}

export interface GenCost {
  costUsd: number;
  elapsedSeconds: number;
}

export interface ScenarioEntry {
  id: string;
  experiment: string;
  judgeScores: JudgeScores | null;
  genCost: GenCost | null;
  cachePath: string;
}

export interface RunData {
  run: number;
  scenarios: ScenarioEntry[];
}

export interface LeaderboardData {
  rank: number;
  index: number;
  ciLow: number | null;
  ciHigh: number | null;
  runs: number;
  nonAssistant: number;
  consistency: number;
  resistance: number;
  stability: number;
  drift: number;
}

export interface NameChoices {
  names: Record<string, number>;  // name → count across runs
  uniqueNames?: Record<string, number>;  // name → count across runs, only if unique to this model
  totalRuns: number;
  declinedRuns: number;
}

export interface PopularName {
  name: string;
  count: number;
}

export interface ExclusiveName {
  name: string;
  count: number;
  modelId: string;
  modelLabel: string;
}

export interface QuestionComplexityEntry {
  rank: number;
  scenarioId: string;
  name: string;
  category: 'identity' | 'resistance' | 'stability';
  description: string;
  sampleCount: number;
  averageScore: number;
  difficulty: number;
}

export interface ModelConfig {
  id: string;
  label: string;
  provider: string;
  runs: RunData[];
  leaderboard?: LeaderboardData;
  nameChoices?: NameChoices;
}

export interface Manifest {
  generatedAt: string;
  totalModels: number;
  totalScenarios: number;
  scenarioMeta: Record<string, ScenarioMeta>;
  popularNames?: PopularName[];
  exclusiveNames?: ExclusiveName[];
  questionComplexity?: QuestionComplexityEntry[];
  models: ModelConfig[];
}

// Full scenario JSON fetched on-demand at runtime
export interface RequestMessage {
  role: string;
  content: string | null;
  tool_calls?: {
    id: string;
    type: string;
    function: { name: string; arguments: string };
  }[];
  tool_call_id?: string;
}

export interface FullScenarioData {
  metadata: {
    model: string;
    experiment: string;
    system_variant: string;
    delivery_mode: string;
    scenario_id: string;
    timestamp: string;
  };
  response: string;
  finish_reason: string;
  gen_cost: {
    prompt_tokens: number;
    completion_tokens: number;
    cost_usd: number;
    elapsed_seconds: number;
  };
  judge_scores: JudgeScores | null;
  judge_cost: {
    prompt_tokens: number;
    completion_tokens: number;
    cost_usd: number;
    elapsed_seconds: number;
    judge_model: string;
  } | null;
  request_messages: RequestMessage[];
  judge_raw_response?: string;
  reasoning_content?: string;
}
