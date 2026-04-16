import { promises as fs } from "fs";
import path from "path";

import { buildCompareResult, buildLanes, buildRunDetail, buildRunSummary } from "../evals";
import type { ArtifactFile, CompareResult, ItemRunRecord, Lane, RunDetail, RunManifest, RunSummary } from "../types";

const artifactNames = ["manifest.json", "results.jsonl", "results.csv", "scores.csv", "steps.csv"];

export function runsRoot(): string {
  const root = process.env.PRISM_RUNS_DIR;
  if (!root) {
    throw new Error("No runs directory selected.");
  }
  return path.resolve(root);
}

export async function loadRunSummaries(): Promise<RunSummary[]> {
  const entries = await discoverRunEntries();
  const summaries = await Promise.all(
    entries.map(async ({ key, dir }) => {
      const manifest = await readManifest(dir);
      const records = await readRecords(dir);
      const summary = buildRunSummary(key, dir, manifest, records);
      summary.artifacts = await availableArtifacts(key, dir);
      return summary;
    }),
  );
  return summaries.sort((a, b) => {
    const left = Date.parse(a.startedAt) || 0;
    const right = Date.parse(b.startedAt) || 0;
    return right - left || b.key.localeCompare(a.key);
  });
}

export async function loadRunDetail(runKey: string): Promise<RunDetail> {
  const dir = await resolveRunDir(runKey);
  const manifest = await readManifest(dir);
  const records = await readRecords(dir);
  const detail = buildRunDetail(runKey, dir, manifest, records);
  detail.summary.artifacts = await availableArtifacts(runKey, dir);
  return detail;
}

export async function loadLanes(): Promise<Lane[]> {
  return buildLanes(await loadRunSummaries());
}

export async function loadCompare(params: {
  baselineRun: string;
  baselineModel: string;
  candidateRun: string;
  candidateModel: string;
}): Promise<CompareResult> {
  const [summaries, baselineDetail, candidateDetail] = await Promise.all([
    loadRunSummaries(),
    loadRunDetail(params.baselineRun),
    loadRunDetail(params.candidateRun),
  ]);
  const lanes = buildLanes(summaries);
  const baseline = lanes.find(
    (lane) => lane.runKey === params.baselineRun && lane.modelKey === params.baselineModel,
  );
  const candidate = lanes.find(
    (lane) => lane.runKey === params.candidateRun && lane.modelKey === params.candidateModel,
  );
  if (!baseline || !candidate) {
    throw new Error("Selected run or model was not found.");
  }
  return buildCompareResult(baseline, candidate, baselineDetail.records, candidateDetail.records);
}

export async function resolveArtifact(runKey: string, artifactName: string): Promise<string> {
  if (!artifactNames.includes(artifactName)) {
    throw new Error("Artifact was not found.");
  }
  const dir = await resolveRunDir(runKey);
  const target = path.join(dir, artifactName);
  await fs.access(target);
  return target;
}

async function discoverRunEntries(): Promise<Array<{ key: string; dir: string }>> {
  const root = runsRoot();
  const entries = await fs.readdir(root, { withFileTypes: true });
  const runs: Array<{ key: string; dir: string }> = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue;
    }
    const dir = path.join(root, entry.name);
    try {
      await fs.access(path.join(dir, "manifest.json"));
      runs.push({ key: entry.name, dir });
    } catch {
      // Non-run folders are ignored.
    }
  }
  return runs;
}

async function resolveRunDir(runKey: string): Promise<string> {
  if (path.basename(runKey) !== runKey || runKey.includes("..")) {
    throw new Error("Run was not found.");
  }
  const root = runsRoot();
  const dir = path.join(root, runKey);
  await fs.access(path.join(dir, "manifest.json"));
  const [realRoot, realDir] = await Promise.all([fs.realpath(root), fs.realpath(dir)]);
  if (realDir !== realRoot && !realDir.startsWith(`${realRoot}${path.sep}`)) {
    throw new Error("Run was not found.");
  }
  return realDir;
}

async function readManifest(dir: string): Promise<RunManifest> {
  const text = await fs.readFile(path.join(dir, "manifest.json"), "utf8");
  return JSON.parse(text) as RunManifest;
}

async function readRecords(dir: string): Promise<ItemRunRecord[]> {
  const file = path.join(dir, "results.jsonl");
  let text = "";
  try {
    text = await fs.readFile(file, "utf8");
  } catch {
    return [];
  }
  const records: ItemRunRecord[] = [];
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }
    try {
      records.push(JSON.parse(trimmed) as ItemRunRecord);
    } catch {
      // A run can be inspected while a line is still being appended.
    }
  }
  return records;
}

async function availableArtifacts(runKey: string, dir: string): Promise<ArtifactFile[]> {
  const files: ArtifactFile[] = [];
  for (const name of artifactNames) {
    try {
      await fs.access(path.join(dir, name));
      files.push({ name, href: `/artifacts/${encodeURIComponent(runKey)}/${encodeURIComponent(name)}` });
    } catch {
      // Optional artifact.
    }
  }
  return files;
}
