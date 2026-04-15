"use client";

import {
  createColumnHelper,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnOrderState,
  type Row,
  type SortingState,
  type Table,
  type VisibilityState,
} from "@tanstack/react-table";
import Link from "next/link";
import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react";

import { formatNumber } from "../lib/evals";
import type { ModelRunSummary, RunSummary, ScoreAggregate } from "../lib/types";
import { formatDate, formatInt, formatSeconds, formatShortHash } from "./format";
import { DataTable, EmptyState, ErrorState, LoadingState, PageTitle } from "./ui";

type ModelRunRow = ModelRunSummary & { runGroupIndex: number; runItemsLabel: string };

const columnHelper = createColumnHelper<ModelRunRow>();
const runTones = ["bg-white", "bg-slate-50"];

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sorting, setSorting] = useState<SortingState>([{ id: "started", desc: true }]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>([]);
  const [hiddenModelKeys, setHiddenModelKeys] = useState<string[]>([]);
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

  const modelRows = useMemo(
    () =>
      runs.flatMap((run, runGroupIndex) =>
        run.modelSummaries.map((modelSummary) => ({
          ...modelSummary,
          runGroupIndex,
          runItemsLabel: itemLabelForRun(run.modelSummaries),
        })),
      ),
    [runs],
  );

  const modelOptions = useMemo(() => [...new Set(modelRows.map((row) => row.modelKey))].sort(), [modelRows]);
  const modelOptionsKey = modelOptions.join("|");

  useEffect(() => {
    setHiddenModelKeys((current) => current.filter((modelKey) => modelOptions.includes(modelKey)));
  }, [modelOptionsKey, modelOptions]);

  const scoreMetrics = useMemo(() => {
    const metrics = new Map<string, ScoreAggregate>();
    for (const row of modelRows) {
      for (const score of row.scoreAggregates) {
        if (!metrics.has(score.id)) {
          metrics.set(score.id, score);
        }
      }
    }
    return [...metrics.values()].sort((left, right) => left.label.localeCompare(right.label));
  }, [modelRows]);

  const filteredRows = useMemo(
    () => modelRows.filter((row) => !hiddenModelKeys.includes(row.modelKey)),
    [hiddenModelKeys, modelRows],
  );

  const columns = useMemo(
    () => [
      columnHelper.accessor("experimentName", {
        id: "run",
        header: "Run",
        enableHiding: false,
        cell: ({ row }) => (
          <div className="min-w-48 leading-tight">
            <Link href={`/runs/${encodeURIComponent(row.original.runKey)}`} className="font-semibold text-ink hover:text-leaf">
              {row.original.experimentName}
            </Link>
            <div className="mt-0.5 text-[11px] text-slate-500">{row.original.runKey}</div>
            <div className="mt-1 text-[11px] text-slate-500">{row.original.runItemsLabel}</div>
            <div className="mt-0.5 text-[11px] text-slate-500">{formatDate(row.original.startedAt)}</div>
          </div>
        ),
      }),
      columnHelper.accessor("modelKey", {
        id: "model",
        header: "Model",
        enableHiding: false,
        cell: ({ row }) => (
          <div className="min-w-32 leading-tight">
            <div className="break-all font-semibold text-ink">{row.original.modelKey}</div>
            <div className="mt-0.5 break-all text-[11px] text-slate-500">{row.original.model}</div>
          </div>
        ),
      }),
      columnHelper.accessor("failedCount", {
        id: "failed",
        header: "Failed",
        cell: ({ getValue }) => (
          <span className={`block min-w-16 ${getValue() ? "font-semibold text-coral" : ""}`}>{formatInt(getValue())}</span>
        ),
      }),
      columnHelper.accessor("evaluatorErrorCount", {
        id: "evalErrors",
        header: "Eval errors",
        cell: ({ getValue }) => (
          <span className={`block min-w-16 ${getValue() ? "font-semibold text-coral" : ""}`}>{formatInt(getValue())}</span>
        ),
      }),
      ...scoreMetrics.map((scoreMetric) =>
        columnHelper.accessor((row) => row.scoreAggregates.find((score) => score.id === scoreMetric.id)?.mean ?? null, {
          id: scoreColumnId(scoreMetric.id),
          header: scoreMetric.label,
          cell: ({ row }) => <ScoreCell score={row.original.scoreAggregates.find((item) => item.id === scoreMetric.id)} />,
        }),
      ),
      columnHelper.accessor((row) => row.latency.avg, {
        id: "latencyAvg",
        header: "Avg time",
        cell: ({ getValue }) => <span className="block min-w-16">{formatSeconds(getValue())}</span>,
      }),
      columnHelper.accessor((row) => row.latency.p50, {
        id: "latencyP50",
        header: "P50 time",
        cell: ({ getValue }) => <span className="block min-w-16">{formatSeconds(getValue())}</span>,
      }),
      columnHelper.accessor((row) => row.latency.p90, {
        id: "latencyP90",
        header: "P90 time",
        cell: ({ getValue }) => <span className="block min-w-16">{formatSeconds(getValue())}</span>,
      }),
      columnHelper.accessor((row) => row.usage.input_tokens, {
        id: "inputTokens",
        header: "Input tokens",
        cell: ({ getValue }) => <span className="block min-w-20">{formatInt(getValue())}</span>,
      }),
      columnHelper.accessor((row) => row.usage.cached_tokens, {
        id: "cachedTokens",
        header: "Cached tokens",
        cell: ({ getValue }) => <span className="block min-w-20">{formatInt(getValue())}</span>,
      }),
      columnHelper.accessor((row) => row.usage.output_tokens, {
        id: "outputTokens",
        header: "Output tokens",
        cell: ({ getValue }) => <span className="block min-w-20">{formatInt(getValue())}</span>,
      }),
      columnHelper.accessor((row) => row.usage.reasoning_tokens, {
        id: "reasoningTokens",
        header: "Reasoning tokens",
        cell: ({ getValue }) => <span className="block min-w-24">{formatInt(getValue())}</span>,
      }),
      columnHelper.accessor((row) => row.usage.total_tokens, {
        id: "totalTokens",
        header: "Total tokens",
        cell: ({ getValue }) => <span className="block min-w-20 font-medium text-ink">{formatInt(getValue())}</span>,
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
    [scoreMetrics],
  );

  const columnIds = useMemo(() => columns.map((column) => column.id).filter((id): id is string => Boolean(id)), [columns]);
  const columnIdsKey = columnIds.join("|");
  const defaultColumnVisibility = useMemo<VisibilityState>(() => ({ hashes: false }), []);

  useEffect(() => {
    setColumnOrder((current) => {
      const existing = current.filter((id) => columnIds.includes(id));
      const missing = columnIds.filter((id) => !existing.includes(id));
      return [...existing, ...missing];
    });
    setColumnVisibility((current) => {
      const next: VisibilityState = {};
      for (const id of columnIds) {
        if (Object.prototype.hasOwnProperty.call(current, id)) {
          next[id] = current[id];
        } else if (Object.prototype.hasOwnProperty.call(defaultColumnVisibility, id)) {
          next[id] = defaultColumnVisibility[id];
        }
      }
      return next;
    });
  }, [columnIdsKey, columnIds, defaultColumnVisibility]);

  const table = useReactTable({
    data: filteredRows,
    columns,
    state: { sorting, columnVisibility, columnOrder },
    onSortingChange: setSorting,
    onColumnVisibilityChange: setColumnVisibility,
    onColumnOrderChange: setColumnOrder,
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
      <div className="mb-4 flex flex-wrap items-start gap-3">
        <ModelControls
          hiddenModelKeys={hiddenModelKeys}
          modelOptions={modelOptions}
          setHiddenModelKeys={setHiddenModelKeys}
        />
        <ColumnControls
          columnOrder={columnOrder}
          defaultColumnVisibility={defaultColumnVisibility}
          setColumnOrder={setColumnOrder}
          setColumnVisibility={setColumnVisibility}
          table={table}
        />
        <div className="min-h-10 rounded-md border border-line bg-white px-3 py-2 text-sm text-slate-600 shadow-soft">
          {formatInt(filteredRows.length)} rows
        </div>
      </div>
      <DataTable
        table={table}
        empty="No matching models."
        getCellRowSpan={(cell) =>
          cell.column.id === "run" ? runCellRowSpan(cell.row, table.getRowModel().rows) : 1
        }
        getRowClassName={(row) => runTones[row.original.runGroupIndex % runTones.length]}
      />
    </>
  );
}

function ScoreCell({ score }: { score?: ScoreAggregate }) {
  if (!score) {
    return <span className="block min-w-20">-</span>;
  }
  return (
    <div className="min-w-24 whitespace-nowrap">
      <span className="font-medium text-ink">{formatNumber(score.mean)}</span>
      <span className="ml-1 text-[11px] text-slate-500">
        ({formatInt(score.count)})
        {score.errorCount ? <span className="ml-2 font-semibold text-coral">{formatInt(score.errorCount)} errors</span> : null}
      </span>
    </div>
  );
}

function itemLabelForRun(modelSummaries: ModelRunSummary[]): string {
  const counts = [...new Set(modelSummaries.map((model) => model.itemRunCount))].sort((left, right) => left - right);
  if (!counts.length) {
    return "0 items";
  }
  if (counts.length === 1) {
    return `${formatInt(counts[0])} items`;
  }
  return `${formatInt(counts[0])}-${formatInt(counts[counts.length - 1])} items`;
}

function ModelControls({
  hiddenModelKeys,
  modelOptions,
  setHiddenModelKeys,
}: {
  hiddenModelKeys: string[];
  modelOptions: string[];
  setHiddenModelKeys: Dispatch<SetStateAction<string[]>>;
}) {
  return (
    <details className="rounded-md border border-line bg-white shadow-soft">
      <summary className="min-h-10 cursor-pointer px-3 py-2 text-sm font-semibold text-ink">Models</summary>
      <div className="flex max-w-xl flex-wrap gap-2 border-t border-line p-3">
        <button
          type="button"
          onClick={() => setHiddenModelKeys([])}
          className="min-h-9 rounded-md border border-line px-3 py-1.5 text-sm font-medium text-ink hover:border-leaf"
        >
          Show all
        </button>
        {modelOptions.map((modelKey) => {
          const checked = !hiddenModelKeys.includes(modelKey);
          return (
            <label
              key={modelKey}
              className="inline-flex min-h-9 cursor-pointer items-center gap-2 rounded-md border border-line bg-white px-3 py-1.5 text-sm text-ink"
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={(event) => {
                  setHiddenModelKeys((current) =>
                    event.target.checked ? current.filter((key) => key !== modelKey) : [...current, modelKey],
                  );
                }}
                className="h-4 w-4 accent-leaf"
              />
              <span className="break-all">{modelKey}</span>
            </label>
          );
        })}
      </div>
    </details>
  );
}

function ColumnControls({
  columnOrder,
  defaultColumnVisibility,
  setColumnOrder,
  setColumnVisibility,
  table,
}: {
  columnOrder: ColumnOrderState;
  defaultColumnVisibility: VisibilityState;
  setColumnOrder: Dispatch<SetStateAction<ColumnOrderState>>;
  setColumnVisibility: Dispatch<SetStateAction<VisibilityState>>;
  table: Table<ModelRunRow>;
}) {
  const orderedColumns = orderedLeafColumns(table, columnOrder);

  return (
    <details className="rounded-md border border-line bg-white shadow-soft">
      <summary className="min-h-10 cursor-pointer px-3 py-2 text-sm font-semibold text-ink">Columns</summary>
      <div className="max-h-96 w-[22rem] max-w-[calc(100vw-2rem)] overflow-auto border-t border-line p-2">
        <button
          type="button"
          onClick={() => {
            setColumnOrder(table.getAllLeafColumns().map((column) => column.id));
            setColumnVisibility(defaultColumnVisibility);
          }}
          className="mb-2 min-h-9 rounded-md border border-line px-3 py-1.5 text-sm font-medium text-ink hover:border-leaf"
        >
          Reset
        </button>
        <div className="space-y-1">
          {orderedColumns.map((column, index) => (
            <div key={column.id} className="grid grid-cols-[1fr_auto_auto] items-center gap-2 rounded-md px-2 py-1 hover:bg-mist">
              <label className="flex min-h-9 items-center gap-2 overflow-hidden text-sm text-ink">
                <input
                  type="checkbox"
                  checked={column.getIsVisible()}
                  disabled={!column.getCanHide()}
                  onChange={(event) => column.toggleVisibility(event.target.checked)}
                  className="h-4 w-4 shrink-0 accent-leaf disabled:opacity-30"
                />
                <span className="truncate">{columnLabel(column.columnDef.header, column.id)}</span>
              </label>
              <button
                type="button"
                disabled={index === 0}
                onClick={() => moveColumn(column.id, -1, table, setColumnOrder)}
                className="min-h-8 rounded-md border border-line px-2 text-xs font-medium text-ink disabled:opacity-30"
              >
                Up
              </button>
              <button
                type="button"
                disabled={index === orderedColumns.length - 1}
                onClick={() => moveColumn(column.id, 1, table, setColumnOrder)}
                className="min-h-8 rounded-md border border-line px-2 text-xs font-medium text-ink disabled:opacity-30"
              >
                Down
              </button>
            </div>
          ))}
        </div>
      </div>
    </details>
  );
}

function orderedLeafColumns(table: Table<ModelRunRow>, columnOrder: ColumnOrderState) {
  const allColumns = table.getAllLeafColumns();
  const byId = new Map(allColumns.map((column) => [column.id, column]));
  const ordered = columnOrder.map((id) => byId.get(id)).filter((column): column is (typeof allColumns)[number] => Boolean(column));
  const missing = allColumns.filter((column) => !columnOrder.includes(column.id));
  return [...ordered, ...missing];
}

function runCellRowSpan(row: Row<ModelRunRow>, rows: Row<ModelRunRow>[]): number {
  const index = rows.findIndex((item) => item.id === row.id);
  if (index < 0) {
    return 1;
  }
  if (rows[index - 1]?.original.runKey === row.original.runKey) {
    return 0;
  }
  let span = 1;
  for (let next = index + 1; next < rows.length; next += 1) {
    if (rows[next].original.runKey !== row.original.runKey) {
      break;
    }
    span += 1;
  }
  return span;
}

function moveColumn(
  columnId: string,
  direction: -1 | 1,
  table: Table<ModelRunRow>,
  setColumnOrder: Dispatch<SetStateAction<ColumnOrderState>>,
) {
  setColumnOrder((current) => {
    const ids = current.length ? [...current] : table.getAllLeafColumns().map((column) => column.id);
    const index = ids.indexOf(columnId);
    const target = index + direction;
    if (index < 0 || target < 0 || target >= ids.length) {
      return ids;
    }
    const [moved] = ids.splice(index, 1);
    ids.splice(target, 0, moved);
    return ids;
  });
}

function columnLabel(header: unknown, fallback: string): string {
  return typeof header === "string" ? header : fallback;
}

function scoreColumnId(scoreId: string): string {
  return `score:${scoreId}`;
}
