"use client";

import { createColumnHelper, getCoreRowModel, getSortedRowModel, useReactTable, type SortingState } from "@tanstack/react-table";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { formatNumber } from "../lib/evals";
import type { RunSummary } from "../lib/types";
import { formatDate, formatInt, formatSeconds, formatShortHash } from "./format";
import { DataTable, EmptyState, ErrorState, LoadingState, PageTitle } from "./ui";

const columnHelper = createColumnHelper<RunSummary>();

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sorting, setSorting] = useState<SortingState>([{ id: "startedAt", desc: true }]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/runs")
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error ?? "Runs could not be loaded.");
        }
        setRuns(payload);
      })
      .catch((caught: Error) => setError(caught.message))
      .finally(() => setLoading(false));
  }, []);

  const columns = useMemo(
    () => [
      columnHelper.accessor("experimentName", {
        header: "Run",
        cell: ({ row }) => (
          <div className="min-w-56">
            <Link href={`/runs/${encodeURIComponent(row.original.key)}`} className="font-semibold text-ink hover:text-leaf">
              {row.original.experimentName}
            </Link>
            <div className="mt-1 text-xs text-slate-500">{row.original.key}</div>
          </div>
        ),
      }),
      columnHelper.accessor("startedAt", {
        header: "Started",
        cell: ({ getValue }) => <span className="block min-w-40">{formatDate(getValue())}</span>,
      }),
      columnHelper.accessor("modelKeys", {
        header: "Models",
        cell: ({ getValue }) => <span className="block min-w-32">{getValue().join(", ") || "-"}</span>,
      }),
      columnHelper.accessor("itemRunCount", {
        header: "Items",
        cell: ({ getValue }) => formatInt(getValue()),
      }),
      columnHelper.accessor("failedCount", {
        header: "Failed",
        cell: ({ getValue }) => <span className={getValue() ? "font-semibold text-coral" : ""}>{formatInt(getValue())}</span>,
      }),
      columnHelper.accessor("evaluatorErrorCount", {
        header: "Eval errors",
        cell: ({ getValue }) => <span className={getValue() ? "font-semibold text-coral" : ""}>{formatInt(getValue())}</span>,
      }),
      columnHelper.display({
        id: "scores",
        header: "Scores",
        cell: ({ row }) => (
          <div className="flex min-w-56 flex-wrap gap-1">
            {row.original.scoreAggregates.slice(0, 3).map((score) => (
              <span key={score.id} className="rounded-md border border-line bg-mist px-2 py-1 text-xs text-ink">
                {score.label}: {formatNumber(score.mean)}
              </span>
            ))}
            {!row.original.scoreAggregates.length ? <span>-</span> : null}
          </div>
        ),
      }),
      columnHelper.accessor((row) => row.latency.avg, {
        id: "latency",
        header: "Avg time",
        cell: ({ getValue }) => formatSeconds(getValue()),
      }),
      columnHelper.accessor("totalTokens", {
        header: "Tokens",
        cell: ({ getValue }) => formatInt(getValue()),
      }),
      columnHelper.display({
        id: "hashes",
        header: "Hashes",
        cell: ({ row }) => (
          <div className="min-w-40 text-xs leading-5 text-slate-500">
            <div>Data {formatShortHash(row.original.datasetSha256)}</div>
            <div>Code {formatShortHash(row.original.experimentSha256)}</div>
            <div>Git {formatShortHash(row.original.gitCommit)}</div>
          </div>
        ),
      }),
    ],
    [],
  );

  const table = useReactTable({
    data: runs,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  if (loading) {
    return <LoadingState />;
  }
  if (error) {
    return <ErrorState message={error} />;
  }
  if (!runs.length) {
    return <EmptyState title="No runs found" body="Choose a folder with completed eval runs, then refresh this page." />;
  }

  return (
    <>
      <PageTitle eyebrow="All runs" title="Review eval runs">
        <Link href="/compare" className="rounded-md bg-ink px-4 py-2 text-sm font-semibold text-white hover:bg-leaf">
          Compare runs
        </Link>
      </PageTitle>
      <DataTable table={table} empty="No matching runs." />
    </>
  );
}
