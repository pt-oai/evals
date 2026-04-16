import { describe, expect, it } from "vitest";

import {
  availableReferenceIds,
  defaultRunsPreferences,
  effectiveColumnOrder,
  effectiveColumnVisibility,
  effectiveId,
  effectiveSorting,
  mergeSavedReferences,
  missingReference,
  missingReferences,
  readRunsPreferences,
  savedReference,
  writeRunsPreferences,
  type StorageLike,
} from "../lib/preferences";

class MemoryStorage implements StorageLike {
  private readonly values = new Map<string, string>();

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  removeItem(key: string) {
    this.values.delete(key);
  }

  setItem(key: string, value: string) {
    this.values.set(key, value);
  }
}

describe("viewer preferences", () => {
  it("round-trips all-runs preferences", () => {
    const storage = new MemoryStorage();
    const preferences = {
      ...defaultRunsPreferences,
      hiddenModelKeys: ["gpt-5-low"],
      chartsCustomized: true,
      charts: [{ id: "score:item_run::exact", metricId: "score:item_run::exact", label: "exact", group: "Scores" }],
      sorting: [{ id: "totalTokens", desc: true }],
      columnVisibility: { hashes: false, totalTokens: true },
      columnOrder: [savedReference("run", "Run"), savedReference("totalTokens", "Total tokens")],
    };

    writeRunsPreferences(preferences, storage);

    expect(readRunsPreferences(storage)).toEqual(preferences);
  });

  it("falls back to defaults for corrupt or unknown stored values", () => {
    const storage = new MemoryStorage();
    storage.setItem("prism-evals.viewer.runs", "{broken");

    expect(readRunsPreferences(storage)).toEqual(defaultRunsPreferences);

    storage.setItem("prism-evals.viewer.runs", JSON.stringify({ version: 999, hiddenModelKeys: ["old"] }));

    expect(readRunsPreferences(storage)).toEqual(defaultRunsPreferences);
  });

  it("keeps stale references visible but removes them from effective ids", () => {
    const references = [
      savedReference("score:item_run::old", "Old score", "Scores"),
      savedReference("latency:avg", "Avg time", "Latency"),
    ];

    expect(availableReferenceIds(references, ["latency:avg"])).toEqual(["latency:avg"]);
    expect(missingReferences(references, ["latency:avg"])).toEqual([
      savedReference("score:item_run::old", "Old score", "Scores"),
    ]);
    expect(effectiveId(savedReference("missing", "Missing"), ["present"])).toBe("");
    expect(missingReference(savedReference("missing", "Missing"), ["present"])).toEqual(savedReference("missing", "Missing"));
  });

  it("restores saved order when stale references reappear and appends new references", () => {
    const saved = [savedReference("score:old", "Old score"), savedReference("latency:avg", "Avg time")];
    const available = [
      savedReference("latency:avg", "Avg time"),
      savedReference("score:old", "Old score, current"),
      savedReference("tokens:total", "Total tokens"),
    ];

    expect(mergeSavedReferences(saved, available)).toEqual([
      savedReference("score:old", "Old score, current"),
      savedReference("latency:avg", "Avg time"),
      savedReference("tokens:total", "Total tokens"),
    ]);
  });

  it("filters table-only state before passing it to TanStack Table", () => {
    expect(effectiveSorting([{ id: "missing", desc: true }, { id: "run", desc: false }], ["run"])).toEqual([
      { id: "run", desc: false },
    ]);
    expect(effectiveColumnOrder(["missing", "run", "model"], ["run", "model"])).toEqual(["run", "model"]);
    expect(effectiveColumnVisibility({ missing: false, hashes: false, run: true }, ["run", "hashes"])).toEqual({
      hashes: false,
      run: true,
    });
  });
});
