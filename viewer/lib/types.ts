export type Status = "success" | "failed" | "skipped";
export type ScoreValue = boolean | number;
export type ScoreType = "BOOLEAN" | "NUMERIC";

export interface TokenUsage {
  input_tokens: number;
  cached_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  total_tokens: number;
}

export interface ErrorRecord {
  type: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface TaskOutput {
  text: string;
  value?: unknown;
  media?: MediaArtifact[];
  metadata?: Record<string, unknown>;
}

export interface MediaArtifact {
  path: string;
  mime_type: string;
  format: string;
  sha256: string;
  bytes: number;
  metadata?: Record<string, unknown>;
}

export interface EvalResult {
  key?: string | null;
  score?: ScoreValue | null;
  data_type?: ScoreType | null;
  description?: string | null;
  comment?: string | null;
  metadata?: Record<string, unknown>;
  error?: ErrorRecord | null;
}

export interface GenerationRecord {
  response_id?: string | null;
  latency_s: number;
  usage: TokenUsage;
  raw_request?: unknown;
  raw_response?: unknown;
  output_text: string;
  error?: ErrorRecord | null;
}

export interface ToolCallRecord {
  name: string;
  arguments?: unknown;
  result?: unknown;
  agent?: string | null;
  turn_id?: string | null;
  call_id?: string | null;
  status: Status;
  duration_s: number;
  started_at?: string | null;
  ended_at?: string | null;
  error?: ErrorRecord | null;
  metadata?: Record<string, unknown>;
}

export interface TurnRecord {
  id: string;
  role: string;
  mode?: string | null;
  status: Status;
  started_at: string;
  ended_at: string;
  duration_s: number;
  input?: unknown;
  output?: TaskOutput | null;
  evals: EvalResult[];
  tool_calls: ToolCallRecord[];
  error?: ErrorRecord | null;
  metadata?: Record<string, unknown>;
}

export interface StepRecord {
  key: string;
  status: Status;
  started_at: string;
  ended_at: string;
  duration_s: number;
  output?: TaskOutput | null;
  evals: EvalResult[];
  usage: TokenUsage;
  response_id?: string | null;
  generations: GenerationRecord[];
  tool_calls?: ToolCallRecord[];
  error?: ErrorRecord | null;
  metadata?: Record<string, unknown>;
}

export interface ModelConfig {
  key: string;
  model: string;
  params?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface ItemRunRecord {
  item_run_id: string;
  run_id: string;
  experiment_name: string;
  item_index: number;
  item_id: string;
  item: Record<string, string>;
  model_key: string;
  model: string;
  model_params?: Record<string, unknown>;
  variant_key?: string | null;
  repetition: number;
  status: Status;
  started_at: string;
  ended_at: string;
  duration_s: number;
  output?: TaskOutput | null;
  evals: EvalResult[];
  usage: TokenUsage;
  response_id?: string | null;
  generations: GenerationRecord[];
  steps: StepRecord[];
  turns?: TurnRecord[];
  tool_calls?: ToolCallRecord[];
  raw_input?: unknown;
  raw_output?: unknown;
  error?: ErrorRecord | null;
}

export interface RunManifest {
  run_id: string;
  experiment_name: string;
  started_at: string;
  ended_at?: string | null;
  dataset_path: string;
  dataset_sha256?: string | null;
  experiment_file?: string | null;
  experiment_sha256?: string | null;
  output_dir: string;
  settings?: Record<string, unknown>;
  model_configs: ModelConfig[];
  variant_configs?: unknown[];
  metadata?: Record<string, unknown>;
  git_commit?: string | null;
  python_version?: string | null;
}

export interface ScoreMetric {
  id: string;
  scope: "item_run" | "step";
  stepKey: string;
  scoreKey: string;
  label: string;
}

export interface ScoreAggregate extends ScoreMetric {
  mean: number | null;
  count: number;
  errorCount: number;
}

export interface LatencySummary {
  avg: number | null;
  p50: number | null;
  p90: number | null;
  p95: number | null;
  max: number | null;
}

export interface ModelRunSummary {
  id: string;
  runKey: string;
  runId: string;
  experimentName: string;
  startedAt: string;
  endedAt?: string | null;
  datasetSha256?: string | null;
  experimentSha256?: string | null;
  gitCommit?: string | null;
  modelKey: string;
  model: string;
  itemRunCount: number;
  successCount: number;
  failedCount: number;
  evaluatorErrorCount: number;
  usage: TokenUsage;
  latency: LatencySummary;
  scoreAggregates: ScoreAggregate[];
}

export interface RunSummary {
  key: string;
  path: string;
  runId: string;
  experimentName: string;
  status: "running" | "complete" | "needs_review" | "empty";
  startedAt: string;
  endedAt?: string | null;
  datasetPath: string;
  datasetSha256?: string | null;
  experimentFile?: string | null;
  experimentSha256?: string | null;
  gitCommit?: string | null;
  modelKeys: string[];
  itemRunCount: number;
  successCount: number;
  failedCount: number;
  evaluatorErrorCount: number;
  totalTokens: number;
  latency: LatencySummary;
  scoreAggregates: ScoreAggregate[];
  modelSummaries: ModelRunSummary[];
  artifacts: ArtifactFile[];
}

export interface ArtifactFile {
  name: string;
  href: string;
}

export interface RunDetail {
  summary: RunSummary;
  manifest: RunManifest;
  records: ItemRunRecord[];
  metrics: ScoreMetric[];
  itemFields: string[];
  stepKeys: string[];
}

export interface Lane {
  id: string;
  runKey: string;
  modelKey: string;
  label: string;
}

export interface ScoreDelta {
  metric: ScoreMetric;
  baseline: ScoreValue | null;
  candidate: ScoreValue | null;
  delta: number | null;
}

export interface CompareRow {
  id: string;
  itemId: string;
  repetition: number;
  baselineRecord?: ItemRunRecord;
  candidateRecord?: ItemRunRecord;
  scoreDeltas: ScoreDelta[];
  latencyDeltaS: number | null;
  totalTokensDelta: number | null;
  changed: boolean;
  regression: boolean;
  newFailure: boolean;
  slower: boolean;
  moreTokens: boolean;
}

export interface CompareResult {
  baseline: Lane;
  candidate: Lane;
  metrics: ScoreMetric[];
  rows: CompareRow[];
}

export interface ViewerInfo {
  tag: string;
  latestTag: string | null;
  updateAvailable: boolean;
}
