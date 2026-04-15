import type { RunSummary, ScoreValue } from "../lib/types";

export function formatDate(value?: string | null): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export function formatShortHash(value?: string | null): string {
  if (!value) {
    return "-";
  }
  return value.slice(0, 10);
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  return new Intl.NumberFormat().format(value);
}

export function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  return `${value.toFixed(2)}s`;
}

export function formatScore(value: ScoreValue | null | undefined): string {
  if (value === null || value === undefined) {
    return "-";
  }
  if (typeof value === "boolean") {
    return value ? "pass" : "fail";
  }
  return value.toFixed(3);
}

export function statusTone(status: RunSummary["status"] | "success" | "failed" | "skipped") {
  switch (status) {
    case "complete":
    case "success":
      return "border-leaf/30 bg-leaf/10 text-leaf";
    case "needs_review":
    case "failed":
      return "border-coral/30 bg-coral/10 text-coral";
    case "running":
      return "border-gold/30 bg-gold/10 text-gold";
    default:
      return "border-line bg-mist text-ink";
  }
}
