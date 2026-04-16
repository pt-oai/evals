import packageJson from "../../package.json";
import type { ViewerInfo } from "../types";

export function loadViewerInfo(): ViewerInfo {
  const tag = process.env.PT_EVALS_VIEWER_TAG ?? versionTag(process.env.PT_EVALS_VIEWER_VERSION ?? packageJson.version);
  const latestTag = process.env.PT_EVALS_VIEWER_LATEST_TAG ?? null;
  const updateAvailable = latestTag ? compareVersionTags(latestTag, tag) > 0 : false;
  return {
    tag,
    latestTag,
    updateAvailable,
  };
}

export function versionTag(version: string): string {
  return version.startsWith("v") ? version : `v${version}`;
}

export function compareVersionTags(left: string, right: string): number {
  const leftParts = versionParts(left);
  const rightParts = versionParts(right);
  const length = Math.max(leftParts.length, rightParts.length);
  for (let index = 0; index < length; index += 1) {
    const difference = (leftParts[index] ?? 0) - (rightParts[index] ?? 0);
    if (difference !== 0) {
      return difference;
    }
  }
  return 0;
}

function versionParts(tag: string): number[] {
  return tag
    .replace(/^refs\/tags\//, "")
    .replace(/^v/, "")
    .split(".")
    .map((part) => Number.parseInt(part, 10))
    .filter((part) => Number.isFinite(part));
}
