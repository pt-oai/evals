import type { ColumnOrderState, SortingState, VisibilityState } from "@tanstack/react-table";

const version = 1;
const runsKey = "pt-evals.viewer.runs";
const compareKey = "pt-evals.viewer.compare";
const runDetailKeyPrefix = "pt-evals.viewer.run-detail";

export type StorageLike = Pick<Storage, "getItem" | "removeItem" | "setItem">;

export type SavedReference = {
  id: string;
  label?: string;
  group?: string;
};

export type SavedChart = {
  id: string;
  metricId: string;
  label?: string;
  group?: string;
};

export type RunsPreferences = {
  version: typeof version;
  hiddenModelKeys: string[];
  chartsCustomized: boolean;
  charts: SavedChart[];
  sorting: SortingState;
  columnVisibility: VisibilityState;
  columnOrder: SavedReference[];
};

export type RunDetailPreferences = {
  version: typeof version;
  query: string;
  modelFilter: SavedReference | null;
  scoreFilter: SavedReference | null;
  visibleScoreIds: SavedReference[];
  stepFilter: SavedReference | null;
  threshold: string;
  failedOnly: boolean;
  evalErrorsOnly: boolean;
  sorting: SortingState;
};

export type ComparePreferences = {
  version: typeof version;
  baselineLane: SavedReference | null;
  candidateLane: SavedReference | null;
  query: string;
  scoreFilter: SavedReference | null;
  regressionsOnly: boolean;
  newFailuresOnly: boolean;
  changedOnly: boolean;
  slowerOnly: boolean;
  moreTokensOnly: boolean;
  failedOnly: boolean;
  evalErrorsOnly: boolean;
  sorting: SortingState;
};

export const defaultRunsPreferences: RunsPreferences = {
  version,
  hiddenModelKeys: [],
  chartsCustomized: false,
  charts: [],
  sorting: [{ id: "started", desc: true }],
  columnVisibility: {},
  columnOrder: [],
};

export const defaultRunDetailPreferences: RunDetailPreferences = {
  version,
  query: "",
  modelFilter: null,
  scoreFilter: null,
  visibleScoreIds: [],
  stepFilter: null,
  threshold: "",
  failedOnly: false,
  evalErrorsOnly: false,
  sorting: [{ id: "item_index", desc: false }],
};

export const defaultComparePreferences: ComparePreferences = {
  version,
  baselineLane: null,
  candidateLane: null,
  query: "",
  scoreFilter: null,
  regressionsOnly: false,
  newFailuresOnly: false,
  changedOnly: false,
  slowerOnly: false,
  moreTokensOnly: false,
  failedOnly: false,
  evalErrorsOnly: false,
  sorting: [],
};

export function readRunsPreferences(storage = browserStorage()): RunsPreferences {
  return readPreference(runsKey, defaultRunsPreferences, normalizeRunsPreferences, storage);
}

export function writeRunsPreferences(preferences: RunsPreferences, storage = browserStorage()) {
  writePreference(runsKey, preferences, storage);
}

export function readRunDetailPreferences(runKey: string, storage = browserStorage()): RunDetailPreferences {
  return readPreference(runDetailKey(runKey), defaultRunDetailPreferences, normalizeRunDetailPreferences, storage);
}

export function writeRunDetailPreferences(runKey: string, preferences: RunDetailPreferences, storage = browserStorage()) {
  writePreference(runDetailKey(runKey), preferences, storage);
}

export function readComparePreferences(storage = browserStorage()): ComparePreferences {
  return readPreference(compareKey, defaultComparePreferences, normalizeComparePreferences, storage);
}

export function writeComparePreferences(preferences: ComparePreferences, storage = browserStorage()) {
  writePreference(compareKey, preferences, storage);
}

export function clearRunsPreferences(storage = browserStorage()) {
  removePreference(runsKey, storage);
}

export function savedReference(id: string, label?: string, group?: string): SavedReference {
  return cleanReference({ id, label, group }) ?? { id };
}

export function mergeSavedReferences(saved: SavedReference[], available: SavedReference[]): SavedReference[] {
  if (!saved.length) {
    return available;
  }
  const availableById = new Map(available.map((reference) => [reference.id, reference]));
  const seen = new Set<string>();
  const merged: SavedReference[] = [];
  for (const reference of saved) {
    if (seen.has(reference.id)) {
      continue;
    }
    seen.add(reference.id);
    merged.push(availableById.get(reference.id) ?? reference);
  }
  for (const reference of available) {
    if (!seen.has(reference.id)) {
      merged.push(reference);
    }
  }
  return merged;
}

export function availableReferenceIds(references: SavedReference[], availableIds: string[]): string[] {
  const available = new Set(availableIds);
  return references.map((reference) => reference.id).filter((id) => available.has(id));
}

export function missingReferences(references: SavedReference[], availableIds: string[]): SavedReference[] {
  const available = new Set(availableIds);
  return references.filter((reference) => !available.has(reference.id));
}

export function effectiveId(reference: SavedReference | null, availableIds: string[]): string {
  return reference && availableIds.includes(reference.id) ? reference.id : "";
}

export function missingReference(reference: SavedReference | null, availableIds: string[]): SavedReference | null {
  return reference && !availableIds.includes(reference.id) ? reference : null;
}

export function effectiveSorting(sorting: SortingState, columnIds: string[]): SortingState {
  const available = new Set(columnIds);
  return sorting.filter((item) => available.has(item.id));
}

export function effectiveColumnOrder(columnOrder: ColumnOrderState, columnIds: string[]): ColumnOrderState {
  const available = new Set(columnIds);
  return columnOrder.filter((id) => available.has(id));
}

export function effectiveColumnVisibility(columnVisibility: VisibilityState, columnIds: string[]): VisibilityState {
  const available = new Set(columnIds);
  const next: VisibilityState = {};
  for (const [id, visible] of Object.entries(columnVisibility)) {
    if (available.has(id)) {
      next[id] = visible;
    }
  }
  return next;
}

export function referenceLabels(references: SavedReference[]): Record<string, string> {
  const labels: Record<string, string> = {};
  for (const reference of references) {
    if (reference.label) {
      labels[reference.id] = reference.label;
    }
  }
  return labels;
}

export function referencesFromIds(ids: string[], labels: Record<string, string> = {}): SavedReference[] {
  const seen = new Set<string>();
  const references: SavedReference[] = [];
  for (const id of ids) {
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    references.push(savedReference(id, labels[id]));
  }
  return references;
}

function readPreference<T>(
  key: string,
  defaults: T,
  normalize: (value: unknown) => T | null,
  storage: StorageLike | null,
): T {
  if (!storage) {
    return defaults;
  }
  try {
    const raw = storage.getItem(key);
    if (!raw) {
      return defaults;
    }
    return normalize(JSON.parse(raw)) ?? defaults;
  } catch {
    return defaults;
  }
}

function writePreference(key: string, preferences: unknown, storage: StorageLike | null) {
  if (!storage) {
    return;
  }
  try {
    storage.setItem(key, JSON.stringify(preferences));
  } catch {
    // Preferences should never break the viewer.
  }
}

function removePreference(key: string, storage: StorageLike | null) {
  if (!storage) {
    return;
  }
  try {
    storage.removeItem(key);
  } catch {
    // Preferences should never break the viewer.
  }
}

function browserStorage(): StorageLike | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function runDetailKey(runKey: string): string {
  return `${runDetailKeyPrefix}.${encodeURIComponent(runKey)}`;
}

function normalizeRunsPreferences(value: unknown): RunsPreferences | null {
  if (!isRecord(value) || value.version !== version) {
    return null;
  }
  return {
    version,
    hiddenModelKeys: stringArray(value.hiddenModelKeys),
    chartsCustomized: value.chartsCustomized === true,
    charts: chartArray(value.charts),
    sorting: sortingArray(value.sorting),
    columnVisibility: booleanRecord(value.columnVisibility),
    columnOrder: referenceArray(value.columnOrder),
  };
}

function normalizeRunDetailPreferences(value: unknown): RunDetailPreferences | null {
  if (!isRecord(value) || value.version !== version) {
    return null;
  }
  return {
    version,
    query: stringValue(value.query),
    modelFilter: cleanReference(value.modelFilter),
    scoreFilter: cleanReference(value.scoreFilter),
    visibleScoreIds: referenceArray(value.visibleScoreIds),
    stepFilter: cleanReference(value.stepFilter),
    threshold: stringValue(value.threshold),
    failedOnly: value.failedOnly === true,
    evalErrorsOnly: value.evalErrorsOnly === true,
    sorting: sortingArray(value.sorting),
  };
}

function normalizeComparePreferences(value: unknown): ComparePreferences | null {
  if (!isRecord(value) || value.version !== version) {
    return null;
  }
  return {
    version,
    baselineLane: cleanReference(value.baselineLane),
    candidateLane: cleanReference(value.candidateLane),
    query: stringValue(value.query),
    scoreFilter: cleanReference(value.scoreFilter),
    regressionsOnly: value.regressionsOnly === true,
    newFailuresOnly: value.newFailuresOnly === true,
    changedOnly: value.changedOnly === true,
    slowerOnly: value.slowerOnly === true,
    moreTokensOnly: value.moreTokensOnly === true,
    failedOnly: value.failedOnly === true,
    evalErrorsOnly: value.evalErrorsOnly === true,
    sorting: sortingArray(value.sorting),
  };
}

function chartArray(value: unknown): SavedChart[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const charts: SavedChart[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    if (!isRecord(item) || typeof item.id !== "string" || typeof item.metricId !== "string" || !item.id || !item.metricId) {
      continue;
    }
    if (seen.has(item.id)) {
      continue;
    }
    seen.add(item.id);
    charts.push({
      id: item.id,
      metricId: item.metricId,
      label: optionalString(item.label),
      group: optionalString(item.group),
    });
  }
  return charts;
}

function referenceArray(value: unknown): SavedReference[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const references: SavedReference[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    const reference = cleanReference(item);
    if (!reference || seen.has(reference.id)) {
      continue;
    }
    seen.add(reference.id);
    references.push(reference);
  }
  return references;
}

function cleanReference(value: unknown): SavedReference | null {
  if (typeof value === "string") {
    return value ? { id: value } : null;
  }
  if (!isRecord(value) || typeof value.id !== "string" || !value.id) {
    return null;
  }
  return {
    id: value.id,
    label: optionalString(value.label),
    group: optionalString(value.group),
  };
}

function sortingArray(value: unknown): SortingState {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((item) => {
    if (!isRecord(item) || typeof item.id !== "string" || !item.id) {
      return [];
    }
    return [{ id: item.id, desc: item.desc === true }];
  });
}

function stringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return [...new Set(value.filter((item): item is string => typeof item === "string" && Boolean(item)))];
}

function booleanRecord(value: unknown): Record<string, boolean> {
  if (!isRecord(value)) {
    return {};
  }
  const record: Record<string, boolean> = {};
  for (const [key, item] of Object.entries(value)) {
    if (typeof item === "boolean") {
      record[key] = item;
    }
  }
  return record;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
