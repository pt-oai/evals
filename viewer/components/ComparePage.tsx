"use client";

import {
  createColumnHelper,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { outputPreview, recordHasEvaluatorError, statusLabel } from "../lib/evals";
import type { CompareResult, CompareRow, Lane } from "../lib/types";
import { formatInt, formatScore, formatSeconds } from "./format";
import {
  DataTable,
  EmptyState,
  ErrorState,
  LoadingState,
  PageTitle,
  SearchInput,
  SelectInput,
  Stat,
  StatGrid,
  Toggle,
  Toolbar,
} from "./ui";

const columnHelper = createColumnHelper<CompareRow>();

export function ComparePage() {
  const [lanes, setLanes] = useState<Lane[]>([]);
  const [baselineId, setBaselineId] = useState("");
  const [candidateId, setCandidateId] = useState("");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [query, setQuery] = useState("");
  const [scoreFilter, setScoreFilter] = useState("");
  const [regressionsOnly, setRegressionsOnly] = useState(false);
  const [newFailuresOnly, setNewFailuresOnly] = useState(false);
  const [changedOnly, setChangedOnly] = useState(false);
  const [slowerOnly, setSlowerOnly] = useState(false);
  const [moreTokensOnly, setMoreTokensOnly] = useState(false);
  const [failedOnly, setFailedOnly] = useState(false);
  const [evalErrorsOnly, setEvalErrorsOnly] = useState(false);
  const [sorting, setSorting] = useState<SortingState>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/compare")
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error ?? "Compare options could not be loaded.");
        }
        setLanes(payload.lanes ?? []);
      })
      .catch((caught: Error) => setError(caught.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!lanes.length || baselineId || candidateId) {
      return;
    }
    setBaselineId(lanes[0].id);
    setCandidateId(lanes[Math.min(1, lanes.length - 1)].id);
  }, [baselineId, candidateId, lanes]);

  useEffect(() => {
    const baseline = lanes.find((lane) => lane.id === baselineId);
    const candidate = lanes.find((lane) => lane.id === candidateId);
    if (!baseline || !candidate) {
      setResult(null);
      return;
    }
    const params = new URLSearchParams({
      baselineRun: baseline.runKey,
      baselineModel: baseline.modelKey,
      candidateRun: candidate.runKey,
      candidateModel: candidate.modelKey,
    });
    fetch(`/api/compare?${params.toString()}`)
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error ?? "Runs could not be compared.");
        }
        setResult(payload);
      })
      .catch((caught: Error) => setError(caught.message));
  }, [baselineId, candidateId, lanes]);

  const filteredRows = useMemo(() => {
    if (!result) {
      return [];
    }
    const needle = query.trim().toLowerCase();
    return result.rows.filter((row) => {
      if (regressionsOnly && !row.regression) {
        return false;
      }
      if (newFailuresOnly && !row.newFailure) {
        return false;
      }
      if (changedOnly && !row.changed) {
        return false;
      }
      if (slowerOnly && !row.slower) {
        return false;
      }
      if (moreTokensOnly && !row.moreTokens) {
        return false;
      }
      if (failedOnly && row.baselineRecord?.status !== "failed" && row.candidateRecord?.status !== "failed") {
        return false;
      }
      if (
        evalErrorsOnly &&
        !(row.baselineRecord && recordHasEvaluatorError(row.baselineRecord)) &&
        !(row.candidateRecord && recordHasEvaluatorError(row.candidateRecord))
      ) {
        return false;
      }
      if (scoreFilter && !row.scoreDeltas.some((delta) => delta.metric.id === scoreFilter && delta.delta !== null)) {
        return false;
      }
      if (!needle) {
        return true;
      }
      return [
        row.itemId,
        row.baselineRecord?.model_key ?? "",
        row.candidateRecord?.model_key ?? "",
        row.baselineRecord?.output?.text ?? "",
        row.candidateRecord?.output?.text ?? "",
        ...Object.values(row.baselineRecord?.item ?? {}),
        ...Object.values(row.candidateRecord?.item ?? {}),
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });
  }, [
    changedOnly,
    evalErrorsOnly,
    failedOnly,
    moreTokensOnly,
    newFailuresOnly,
    query,
    regressionsOnly,
    result,
    scoreFilter,
    slowerOnly,
  ]);

  const columns = useMemo(
    () => [
      columnHelper.accessor("itemId", {
        header: "Item",
        cell: ({ row }) => (
          <div className="min-w-32">
            <div className="font-semibold text-ink">{row.original.itemId}</div>
            <div className="mt-1 text-xs text-slate-500">rep {row.original.repetition}</div>
          </div>
        ),
      }),
      columnHelper.display({
        id: "status",
        header: "Status",
        cell: ({ row }) => (
          <div className="min-w-32 text-xs leading-5 text-slate-600">
            <div>Base: {row.original.baselineRecord ? statusLabel(row.original.baselineRecord.status) : "Missing"}</div>
            <div>Cand: {row.original.candidateRecord ? statusLabel(row.original.candidateRecord.status) : "Missing"}</div>
          </div>
        ),
      }),
      columnHelper.display({
        id: "scores",
        header: "Score changes",
        cell: ({ row }) => (
          <div className="flex min-w-64 flex-wrap gap-1">
            {row.original.scoreDeltas
              .filter((delta) => delta.delta !== null || delta.baseline !== null || delta.candidate !== null)
              .slice(0, 5)
              .map((delta) => (
                <span
                  key={delta.metric.id}
                  className={`rounded-md border px-2 py-1 text-xs ${
                    delta.delta !== null && delta.delta < 0
                      ? "border-coral/30 bg-coral/10 text-coral"
                      : delta.delta !== null && delta.delta > 0
                        ? "border-leaf/30 bg-leaf/10 text-leaf"
                        : "border-line bg-mist text-ink"
                  }`}
                >
                  {delta.metric.label}: {formatScore(delta.baseline)} to {formatScore(delta.candidate)}
                </span>
              ))}
          </div>
        ),
      }),
      columnHelper.display({
        id: "baseline",
        header: "Baseline output",
        cell: ({ row }) => (
          <p className="max-h-36 min-w-80 overflow-hidden whitespace-pre-wrap text-sm">
            {outputPreview(row.original.baselineRecord, 420) || "-"}
          </p>
        ),
      }),
      columnHelper.display({
        id: "candidate",
        header: "Candidate output",
        cell: ({ row }) => (
          <p className="max-h-36 min-w-80 overflow-hidden whitespace-pre-wrap text-sm">
            {outputPreview(row.original.candidateRecord, 420) || "-"}
          </p>
        ),
      }),
      columnHelper.accessor("latencyDeltaS", {
        header: "Time",
        cell: ({ getValue }) => signedSeconds(getValue()),
      }),
      columnHelper.accessor("totalTokensDelta", {
        header: "Tokens",
        cell: ({ getValue }) => signedInt(getValue()),
      }),
    ],
    [],
  );

  const table = useReactTable({
    data: filteredRows,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const stats = useMemo(() => {
    const rows = result?.rows ?? [];
    return {
      compared: rows.length,
      regressions: rows.filter((row) => row.regression).length,
      newFailures: rows.filter((row) => row.newFailure).length,
      changed: rows.filter((row) => row.changed).length,
    };
  }, [result]);

  if (loading) {
    return <LoadingState />;
  }
  if (error) {
    return <ErrorState message={error} />;
  }
  if (lanes.length < 2) {
    return <EmptyState title="Not enough lanes" body="Run at least two models or two runs, then compare them here." />;
  }

  return (
    <>
      <PageTitle eyebrow="Compare" title="Compare run lanes">
        <Link href="/" className="rounded-md border border-line bg-white px-3 py-2 text-sm font-semibold text-ink hover:border-ink">
          All runs
        </Link>
      </PageTitle>

      <Toolbar>
        <SelectInput value={baselineId} onChange={(event) => setBaselineId(event.target.value)} aria-label="Baseline">
          {lanes.map((lane) => (
            <option key={lane.id} value={lane.id}>
              {lane.label}
            </option>
          ))}
        </SelectInput>
        <SelectInput value={candidateId} onChange={(event) => setCandidateId(event.target.value)} aria-label="Candidate">
          {lanes.map((lane) => (
            <option key={lane.id} value={lane.id}>
              {lane.label}
            </option>
          ))}
        </SelectInput>
      </Toolbar>

      <StatGrid>
        <Stat label="Compared" value={formatInt(stats.compared)} detail="Joined items" />
        <Stat label="Regressions" value={formatInt(stats.regressions)} detail="Lower scores" />
        <Stat label="New failures" value={formatInt(stats.newFailures)} detail="Candidate only" />
        <Stat label="Changed" value={formatInt(stats.changed)} detail="Score movement" />
      </StatGrid>

      <Toolbar>
        <SearchInput value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search items" />
        <SelectInput value={scoreFilter} onChange={(event) => setScoreFilter(event.target.value)} aria-label="Score">
          <option value="">All scores</option>
          {(result?.metrics ?? []).map((metric) => (
            <option key={metric.id} value={metric.id}>
              {metric.label}
            </option>
          ))}
        </SelectInput>
        <Toggle checked={regressionsOnly} onChange={setRegressionsOnly} label="Regressions" />
        <Toggle checked={newFailuresOnly} onChange={setNewFailuresOnly} label="New failures" />
        <Toggle checked={changedOnly} onChange={setChangedOnly} label="Changed" />
        <Toggle checked={slowerOnly} onChange={setSlowerOnly} label="Slower" />
        <Toggle checked={moreTokensOnly} onChange={setMoreTokensOnly} label="More tokens" />
        <Toggle checked={failedOnly} onChange={setFailedOnly} label="Failed" />
        <Toggle checked={evalErrorsOnly} onChange={setEvalErrorsOnly} label="Eval errors" />
      </Toolbar>

      <DataTable table={table} empty="No matching comparisons." />
    </>
  );
}

function signedSeconds(value: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${formatSeconds(value)}`;
}

function signedInt(value: number | null): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${formatInt(value)}`;
}
