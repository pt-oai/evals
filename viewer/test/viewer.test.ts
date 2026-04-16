import { afterEach, describe, expect, it, vi } from "vitest";

import { compareVersionTags, loadViewerInfo, versionTag } from "../lib/server/viewer";

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("viewer info", () => {
  it("shows the current package tag", () => {
    vi.stubEnv("PRISM_VIEWER_VERSION", "0.5.8");

    expect(loadViewerInfo()).toMatchObject({ tag: "v0.5.8", updateAvailable: false });
  });

  it("marks newer tags as available", () => {
    vi.stubEnv("PRISM_VIEWER_TAG", "v0.5.8");
    vi.stubEnv("PRISM_VIEWER_LATEST_TAG", "v0.5.9");

    expect(loadViewerInfo()).toEqual({ tag: "v0.5.8", latestTag: "v0.5.9", updateAvailable: true });
  });

  it("does not mark the same tag as available", () => {
    vi.stubEnv("PRISM_VIEWER_TAG", "v0.5.8");
    vi.stubEnv("PRISM_VIEWER_LATEST_TAG", "v0.5.8");

    expect(loadViewerInfo()).toEqual({ tag: "v0.5.8", latestTag: "v0.5.8", updateAvailable: false });
  });

  it("compares version tags numerically", () => {
    expect(versionTag("0.5.8")).toBe("v0.5.8");
    expect(compareVersionTags("v0.5.10", "v0.5.9")).toBeGreaterThan(0);
  });
});
