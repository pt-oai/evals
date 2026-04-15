import type {
  CompareResult,
  CompareRow,
  EvalResult,
  ItemRunRecord,
  Lane,
  LatencySummary,
  RunDetail,
  RunManifest,
  RunSummary,
  ScoreAggregate,
  ScoreDelta,
  ScoreMetric,
  ScoreValue,
  StepRecord,
  TokenUsage,
} from "./types";

const emptyUsage: TokenUsage = {
  input_tokens: 0,
  cached_tokens: 0,
  output_tokens: 0,
  reasoning_tokens: 0,
  total_tokens: 0,
};

export function scoreNumber(score: ScoreValue | null | undefined): number | null {
  if (typeof score === "boolean") {
    return score ? 1 : 0;
  }
  if (typeof score === "number" && Number.isFinite(score)) {
    return score;
  }
  return null;
}

export function metricId(scope: "item_run" | "step", stepKey: string, scoreKey: string): string {
  return [scope, stepKey, scoreKey].join("::");
}

export function metricLabel(metric: ScoreMetric): string {
  return metric.scope === "step" ? `${metric.stepKey} / ${metric.scoreKey}` : metric.scoreKey;
}

export function collectMetrics(records: ItemRunRecord[]): ScoreMetric[] {
  const metrics = new Map<string, ScoreMetric>();
  for (const record of records) {
    for (const result of record.evals ?? []) {
      addMetric(metrics, "item_run", "", result);
    }
    for (const step of record.steps ?? []) {
      for (const result of step.evals ?? []) {
        addMetric(metrics, "step", step.key, result);
      }
    }
  }
  return [...metrics.values()].sort((a, b) => a.label.localeCompare(b.label));
}

function addMetric(
  metrics: Map<string, ScoreMetric>,
  scope: "item_run" | "step",
  stepKey: string,
  result: EvalResult,
) {
  if (!result.key) {
    return;
  }
  const id = metricId(scope, stepKey, result.key);
  if (!metrics.has(id)) {
    const metric = { id, scope, stepKey, scoreKey: result.key, label: "" };
    metric.label = metricLabel(metric);
    metrics.set(id, metric);
  }
}

export function aggregateScores(records: ItemRunRecord[]): ScoreAggregate[] {
  const metrics = collectMetrics(records);
  return metrics.map((metric) => {
    const values: number[] = [];
    let errorCount = 0;
    for (const result of iterMetricResults(records, metric)) {
      if (result.error) {
        errorCount += 1;
      }
      const value = scoreNumber(result.score);
      if (value !== null && !result.error) {
        values.push(value);
      }
    }
    return {
      ...metric,
      mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
      count: values.length,
      errorCount,
    };
  });
}

export function buildRunSummary(
  key: string,
  runPath: string,
  manifest: RunManifest,
  records: ItemRunRecord[],
): RunSummary {
  const failedCount = records.filter((record) => record.status === "failed").length;
  const successCount = records.filter((record) => record.status === "success").length;
  const evaluatorErrorCount = records.reduce(
    (count, record) => count + evalErrorCount(record.evals) + stepEvalErrorCount(record.steps),
    0,
  );
  const modelKeys = [
    ...new Set([
      ...(manifest.model_configs ?? []).map((model) => model.key),
      ...records.map((record) => record.model_key),
    ]),
  ].sort();
  const totalTokens = records.reduce((total, record) => total + usage(record).total_tokens, 0);
  return {
    key,
    path: runPath,
    runId: manifest.run_id,
    experimentName: manifest.experiment_name,
    status: runStatus(manifest, records, failedCount, evaluatorErrorCount),
    startedAt: manifest.started_at,
    endedAt: manifest.ended_at,
    datasetPath: manifest.dataset_path,
    datasetSha256: manifest.dataset_sha256,
    experimentFile: manifest.experiment_file,
    experimentSha256: manifest.experiment_sha256,
    gitCommit: manifest.git_commit,
    modelKeys,
    itemRunCount: records.length,
    successCount,
    failedCount,
    evaluatorErrorCount,
    totalTokens,
    latency: summarizeLatency(records.map((record) => record.duration_s).filter(isFiniteNumber)),
    scoreAggregates: aggregateScores(records),
    artifacts: [],
  };
}

export function buildRunDetail(
  key: string,
  runPath: string,
  manifest: RunManifest,
  records: ItemRunRecord[],
): RunDetail {
  const metrics = collectMetrics(records);
  const itemFields = [...new Set(records.flatMap((record) => Object.keys(record.item ?? {})))].sort();
  const stepKeys = [...new Set(records.flatMap((record) => (record.steps ?? []).map((step) => step.key)))].sort();
  return {
    summary: buildRunSummary(key, runPath, manifest, records),
    manifest,
    records,
    metrics,
    itemFields,
    stepKeys,
  };
}

export function buildLanes(summaries: RunSummary[]): Lane[] {
  return summaries.flatMap((summary) =>
    summary.modelKeys.map((modelKey) => ({
      id: laneId(summary.key, modelKey),
      runKey: summary.key,
      modelKey,
      label: `${summary.experimentName} - ${summary.key} - ${modelKey}`,
    })),
  );
}

export function laneId(runKey: string, modelKey: string): string {
  return `${encodeURIComponent(runKey)}::${encodeURIComponent(modelKey)}`;
}

export function parseLaneId(id: string): { runKey: string; modelKey: string } | null {
  const [runKey, modelKey] = id.split("::");
  if (!runKey || !modelKey) {
    return null;
  }
  return { runKey: decodeURIComponent(runKey), modelKey: decodeURIComponent(modelKey) };
}

export function buildCompareResult(
  baseline: Lane,
  candidate: Lane,
  baselineRecords: ItemRunRecord[],
  candidateRecords: ItemRunRecord[],
): CompareResult {
  const filteredBaseline = baselineRecords.filter((record) => record.model_key === baseline.modelKey);
  const filteredCandidate = candidateRecords.filter((record) => record.model_key === candidate.modelKey);
  const baselineByKey = mapByItemRepetition(filteredBaseline);
  const candidateByKey = mapByItemRepetition(filteredCandidate);
  const keys = [...new Set([...baselineByKey.keys(), ...candidateByKey.keys()])].sort();
  const metrics = mergeMetrics(collectMetrics(filteredBaseline), collectMetrics(filteredCandidate));
  const rows: CompareRow[] = keys.map((key) => {
    const baselineRecord = baselineByKey.get(key);
    const candidateRecord = candidateByKey.get(key);
    const scoreDeltas = metrics.map((metric) => buildScoreDelta(metric, baselineRecord, candidateRecord));
    const latencyDeltaS =
      baselineRecord && candidateRecord ? candidateRecord.duration_s - baselineRecord.duration_s : null;
    const totalTokensDelta =
      baselineRecord && candidateRecord
        ? usage(candidateRecord).total_tokens - usage(baselineRecord).total_tokens
        : null;
    const changed = scoreDeltas.some((delta) => delta.delta !== null && delta.delta !== 0);
    const regression = scoreDeltas.some((delta) => delta.delta !== null && delta.delta < 0);
    const newFailure = baselineRecord?.status !== "failed" && candidateRecord?.status === "failed";
    return {
      id: key,
      itemId: baselineRecord?.item_id ?? candidateRecord?.item_id ?? key,
      repetition: baselineRecord?.repetition ?? candidateRecord?.repetition ?? 0,
      baselineRecord,
      candidateRecord,
      scoreDeltas,
      latencyDeltaS,
      totalTokensDelta,
      changed,
      regression,
      newFailure,
      slower: latencyDeltaS !== null && latencyDeltaS > 0,
      moreTokens: totalTokensDelta !== null && totalTokensDelta > 0,
    };
  });
  return { baseline, candidate, metrics, rows };
}

export function recordScore(record: ItemRunRecord | undefined, metric: ScoreMetric): ScoreValue | null {
  if (!record) {
    return null;
  }
  const result = [...iterMetricResults([record], metric)][0];
  return result?.score ?? null;
}

export function recordHasEvaluatorError(record: ItemRunRecord): boolean {
  return evalErrorCount(record.evals) + stepEvalErrorCount(record.steps) > 0;
}

export function outputPreview(record: ItemRunRecord | undefined, maxLength = 180): string {
  const text = record?.output?.text?.trim() ?? "";
  if (text.length <= maxLength) {
    return text;
  }
  return `${text.slice(0, maxLength - 3)}...`;
}

export function formatNumber(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

export function statusLabel(status: RunSummary["status"] | ItemRunRecord["status"]): string {
  switch (status) {
    case "needs_review":
      return "Needs review";
    case "running":
      return "Running";
    case "complete":
      return "Complete";
    case "empty":
      return "Empty";
    case "success":
      return "Passed";
    case "failed":
      return "Failed";
    case "skipped":
      return "Skipped";
    default:
      return String(status);
  }
}

function buildScoreDelta(
  metric: ScoreMetric,
  baselineRecord: ItemRunRecord | undefined,
  candidateRecord: ItemRunRecord | undefined,
): ScoreDelta {
  const baselineScore = recordScore(baselineRecord, metric);
  const candidateScore = recordScore(candidateRecord, metric);
  const baselineNumber = scoreNumber(baselineScore);
  const candidateNumber = scoreNumber(candidateScore);
  return {
    metric,
    baseline: baselineScore,
    candidate: candidateScore,
    delta: baselineNumber !== null && candidateNumber !== null ? candidateNumber - baselineNumber : null,
  };
}

function mergeMetrics(left: ScoreMetric[], right: ScoreMetric[]): ScoreMetric[] {
  const metrics = new Map<string, ScoreMetric>();
  for (const metric of [...left, ...right]) {
    metrics.set(metric.id, metric);
  }
  return [...metrics.values()].sort((a, b) => a.label.localeCompare(b.label));
}

function mapByItemRepetition(records: ItemRunRecord[]): Map<string, ItemRunRecord> {
  const mapped = new Map<string, ItemRunRecord>();
  for (const record of records) {
    mapped.set(itemRepetitionKey(record), record);
  }
  return mapped;
}

function itemRepetitionKey(record: ItemRunRecord): string {
  return `${record.item_id}\u0000${record.repetition}`;
}

function* iterMetricResults(records: ItemRunRecord[], metric: ScoreMetric): Iterable<EvalResult> {
  for (const record of records) {
    if (metric.scope === "item_run") {
      for (const result of record.evals ?? []) {
        if (result.key === metric.scoreKey) {
          yield result;
        }
      }
    } else {
      for (const step of record.steps ?? []) {
        if (step.key !== metric.stepKey) {
          continue;
        }
        for (const result of step.evals ?? []) {
          if (result.key === metric.scoreKey) {
            yield result;
          }
        }
      }
    }
  }
}

function evalErrorCount(evals: EvalResult[] | undefined): number {
  return (evals ?? []).filter((result) => result.error).length;
}

function stepEvalErrorCount(steps: StepRecord[] | undefined): number {
  return (steps ?? []).reduce((count, step) => count + evalErrorCount(step.evals), 0);
}

function runStatus(
  manifest: RunManifest,
  records: ItemRunRecord[],
  failedCount: number,
  evaluatorErrorCount: number,
): RunSummary["status"] {
  if (records.length === 0) {
    return "empty";
  }
  if (!manifest.ended_at) {
    return "running";
  }
  if (failedCount > 0 || evaluatorErrorCount > 0) {
    return "needs_review";
  }
  return "complete";
}

function summarizeLatency(values: number[]): LatencySummary {
  const sorted = values.filter(isFiniteNumber).sort((a, b) => a - b);
  if (!sorted.length) {
    return { avg: null, p50: null, p95: null, max: null };
  }
  return {
    avg: sorted.reduce((sum, value) => sum + value, 0) / sorted.length,
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
    max: sorted[sorted.length - 1],
  };
}

function percentile(values: number[], percent: number): number {
  if (values.length === 1) {
    return values[0];
  }
  const index = (values.length - 1) * (percent / 100);
  const lower = Math.floor(index);
  const upper = Math.min(lower + 1, values.length - 1);
  const weight = index - lower;
  return values[lower] * (1 - weight) + values[upper] * weight;
}

function usage(record: ItemRunRecord): TokenUsage {
  return record.usage ?? emptyUsage;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
