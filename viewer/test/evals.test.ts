import { mkdtemp, mkdir, writeFile } from "fs/promises";
import os from "os";
import path from "path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { aggregateScores, buildCompareResult, buildLanes, buildRunSummary } from "../lib/evals";
import { loadCompare, loadRunSummaries } from "../lib/server/runs";
import type { ItemRunRecord, RunManifest, TokenUsage } from "../lib/types";

const usage: TokenUsage = {
  input_tokens: 10,
  cached_tokens: 2,
  output_tokens: 5,
  reasoning_tokens: 1,
  total_tokens: 15,
};

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("artifact summaries", () => {
  it("aggregates item and step scores", () => {
    const records = [
      record("item-1", "m1", { exact: true }, { draft: { clear: 0.5 } }),
      record("item-2", "m1", { exact: false }, { draft: { clear: 1 } }),
    ];

    const scores = aggregateScores(records);

    expect(scores.find((score) => score.label === "exact")?.mean).toBe(0.5);
    expect(scores.find((score) => score.label === "draft / clear")?.mean).toBe(0.75);
  });

  it("builds one lane per run and model", () => {
    const summary = buildRunSummary("run-a", "/tmp/run-a", manifest("run-a", ["m1", "m2"]), [
      record("item-1", "m1", { exact: true }),
    ]);

    expect(buildLanes([summary]).map((lane) => lane.modelKey)).toEqual(["m1", "m2"]);
    expect(summary.modelSummaries.map((model) => model.modelKey)).toEqual(["m1", "m2"]);
  });

  it("summarizes scores, latency, and tokens by model", () => {
    const summary = buildRunSummary("run-a", "/tmp/run-a", manifest("run-a", ["m1", "m2", "m3"]), [
      record("item-1", "m1", { exact: true }, {}, 10),
      record("item-2", "m1", { exact: false }, {}, 30, "failed"),
      record("item-1", "m2", { exact: true }, {}, 20),
    ]);

    const m1 = summary.modelSummaries.find((model) => model.modelKey === "m1");
    const m3 = summary.modelSummaries.find((model) => model.modelKey === "m3");

    expect(m1?.itemRunCount).toBe(2);
    expect(m1?.failedCount).toBe(1);
    expect(m1?.usage.total_tokens).toBe(40);
    expect(m1?.latency.avg).toBe(2);
    expect(m1?.latency.p50).toBe(2);
    expect(m1?.latency.p90).toBeCloseTo(2.8);
    expect(m1?.scoreAggregates.find((score) => score.label === "exact")?.mean).toBe(0.5);
    expect(m3?.itemRunCount).toBe(0);
    expect(m3?.usage.total_tokens).toBe(0);
  });

  it("joins compare rows by item and repetition", () => {
    const baseline = { id: "a", runKey: "run-a", modelKey: "m1", label: "A" };
    const candidate = { id: "b", runKey: "run-b", modelKey: "m2", label: "B" };
    const comparison = buildCompareResult(
      baseline,
      candidate,
      [record("item-1", "m1", { exact: true }, {}, 12)],
      [record("item-1", "m2", { exact: false }, {}, 20, "failed")],
    );

    expect(comparison.rows).toHaveLength(1);
    expect(comparison.rows[0].regression).toBe(true);
    expect(comparison.rows[0].newFailure).toBe(true);
    expect(comparison.rows[0].totalTokensDelta).toBe(8);
  });
});

describe("server artifact loading", () => {
  it("discovers immediate run folders and ignores other folders", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "pt-evals-viewer-"));
    await writeRun(root, "run-a", manifest("run-a", ["m1"]), [record("item-1", "m1", { exact: true })]);
    await mkdir(path.join(root, "notes"));
    vi.stubEnv("PT_EVALS_RUNS_DIR", root);

    const summaries = await loadRunSummaries();

    expect(summaries).toHaveLength(1);
    expect(summaries[0].key).toBe("run-a");
    expect(summaries[0].scoreAggregates[0].mean).toBe(1);
  });

  it("loads comparisons across runs", async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), "pt-evals-viewer-"));
    await writeRun(root, "run-a", manifest("run-a", ["m1"]), [record("item-1", "m1", { exact: true })]);
    await writeRun(root, "run-b", manifest("run-b", ["m2"]), [record("item-1", "m2", { exact: false })]);
    vi.stubEnv("PT_EVALS_RUNS_DIR", root);

    const comparison = await loadCompare({
      baselineRun: "run-a",
      baselineModel: "m1",
      candidateRun: "run-b",
      candidateModel: "m2",
    });

    expect(comparison.rows[0].scoreDeltas[0].delta).toBe(-1);
  });
});

function manifest(key: string, models: string[]): RunManifest {
  return {
    run_id: key,
    experiment_name: "demo",
    started_at: key === "run-b" ? "2026-04-15T00:02:00Z" : "2026-04-15T00:01:00Z",
    ended_at: "2026-04-15T00:03:00Z",
    dataset_path: "datasets/qa.csv",
    dataset_sha256: "dataset",
    experiment_file: "qa.py",
    experiment_sha256: "experiment",
    output_dir: key,
    model_configs: models.map((model) => ({ key: model, model: "gpt-test" })),
    git_commit: "abcdef1234",
  };
}

function record(
  itemId: string,
  modelKey: string,
  scores: Record<string, boolean | number>,
  stepScores: Record<string, Record<string, boolean | number>> = {},
  totalTokens = 15,
  status: "success" | "failed" = "success",
): ItemRunRecord {
  return {
    item_run_id: `${itemId}-${modelKey}`,
    run_id: "run",
    experiment_name: "demo",
    item_index: itemId === "item-1" ? 0 : 1,
    item_id: itemId,
    item: { id: itemId, question: `Question ${itemId}` },
    model_key: modelKey,
    model: "gpt-test",
    model_params: {},
    repetition: 0,
    status,
    started_at: "2026-04-15T00:00:00Z",
    ended_at: "2026-04-15T00:00:01Z",
    duration_s: totalTokens / 10,
    output: { text: `Answer ${itemId}` },
    evals: Object.entries(scores).map(([key, score]) => ({ key, score })),
    usage: { ...usage, total_tokens: totalTokens },
    generations: [],
    steps: Object.entries(stepScores).map(([stepKey, values]) => ({
      key: stepKey,
      status: "success",
      started_at: "2026-04-15T00:00:00Z",
      ended_at: "2026-04-15T00:00:01Z",
      duration_s: 1,
      output: { text: `${stepKey} output` },
      evals: Object.entries(values).map(([key, score]) => ({ key, score })),
      usage,
      generations: [],
    })),
    error: status === "failed" ? { type: "RuntimeError", message: "failed" } : null,
  };
}

async function writeRun(root: string, name: string, runManifest: RunManifest, records: ItemRunRecord[]) {
  const dir = path.join(root, name);
  await mkdir(dir);
  await writeFile(path.join(dir, "manifest.json"), `${JSON.stringify(runManifest, null, 2)}\n`);
  await writeFile(path.join(dir, "results.jsonl"), `${records.map((item) => JSON.stringify(item)).join("\n")}\n`);
}
