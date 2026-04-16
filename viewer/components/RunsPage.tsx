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
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { formatNumber } from "../lib/evals";
import type { ModelRunSummary, RunSummary, ScoreAggregate } from "../lib/types";
import { formatDate, formatInt, formatSeconds, formatShortHash } from "./format";
import { DataTable, EmptyState, ErrorState, LoadingState, PageTitle } from "./ui";

type ModelRunRow = ModelRunSummary & { runGroupIndex: number; runItemsLabel: string };
type ChartConfig = { id: string; metricId: string };
type ChartMetric = {
  id: string;
  label: string;
  group: string;
  format: (value: number | null | undefined) => string;
  axis: (value: number) => string;
  value: (row: ModelRunRow) => number | null;
  scaleMax?: number;
};
type ChartGroup = {
  runKey: string;
  experimentName: string;
  startedAt: string;
  values: Map<string, number | null>;
};
type ChartDatum = Record<string, string | number | null> & {
  experimentName: string;
  runKey: string;
  runLabel: string;
};

const columnHelper = createColumnHelper<ModelRunRow>();
const runTones = ["bg-white", "bg-slate-50"];
const modelPalette = ["#334155", "#3b7662", "#4f46e5", "#0f766e", "#9333ea", "#0369a1", "#be185d", "#854d0e"];

export function RunsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sorting, setSorting] = useState<SortingState>([{ id: "started", desc: true }]);
  const [columnVisibility, setColumnVisibility] = useState<VisibilityState>({});
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>([]);
  const [hiddenModelKeys, setHiddenModelKeys] = useState<string[]>([]);
  const [charts, setCharts] = useState<ChartConfig[]>([]);
  const [chartsTouched, setChartsTouched] = useState(false);
  const [chartDraftMetricId, setChartDraftMetricId] = useState("");
  const [expandedChartId, setExpandedChartId] = useState<string | null>(null);
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

  const recentRuns = useMemo(() => newestRuns(runs, 5), [runs]);

  const modelRows = useMemo(
    () =>
      recentRuns.flatMap((run, runGroupIndex) =>
        run.modelSummaries.map((modelSummary) => ({
          ...modelSummary,
          runGroupIndex,
          runItemsLabel: itemLabelForRun(run.modelSummaries),
        })),
      ),
    [recentRuns],
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

  const chartMetrics = useMemo(() => buildChartMetrics(scoreMetrics), [scoreMetrics]);
  const chartMetricsKey = chartMetrics.map((metric) => metric.id).join("|");

  useEffect(() => {
    setChartDraftMetricId((current) =>
      current && chartMetrics.some((metric) => metric.id === current) ? current : (chartMetrics[0]?.id ?? ""),
    );
    setCharts((current) => {
      const valid = current.filter((chart) => chartMetrics.some((metric) => metric.id === chart.metricId));
      if (chartsTouched) {
        return valid;
      }
      return defaultCharts(chartMetrics).map((chart) => valid.find((item) => item.metricId === chart.metricId) ?? chart);
    });
  }, [chartMetricsKey, chartMetrics, chartsTouched]);

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

  const expandedChart = charts.find((chart) => chart.id === expandedChartId) ?? null;

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
        <ChartControls
          chartDraftMetricId={chartDraftMetricId}
          charts={charts}
          metrics={chartMetrics}
          setChartDraftMetricId={setChartDraftMetricId}
          onAdd={() => {
            if (!chartDraftMetricId || charts.some((chart) => chart.metricId === chartDraftMetricId)) {
              return;
            }
            setChartsTouched(true);
            setCharts((current) => [...current, { id: chartDraftMetricId, metricId: chartDraftMetricId }]);
          }}
        />
      </div>
      <RunCharts
        charts={charts}
        metrics={chartMetrics}
        rows={filteredRows}
        onOpen={setExpandedChartId}
        onRemove={(chartId) => {
          setChartsTouched(true);
          setCharts((current) => current.filter((chart) => chart.id !== chartId));
        }}
      />
      <DataTable
        table={table}
        empty="No matching models."
        getCellRowSpan={(cell) =>
          cell.column.id === "run" ? runCellRowSpan(cell.row, table.getRowModel().rows) : 1
        }
        getRowClassName={(row) => runTones[row.original.runGroupIndex % runTones.length]}
      />
      {expandedChart ? (
        <ChartModal
          metric={chartMetrics.find((metric) => metric.id === expandedChart.metricId) ?? null}
          rows={filteredRows}
          onClose={() => setExpandedChartId(null)}
        />
      ) : null}
    </>
  );
}

function ChartControls({
  chartDraftMetricId,
  charts,
  metrics,
  setChartDraftMetricId,
  onAdd,
}: {
  chartDraftMetricId: string;
  charts: ChartConfig[];
  metrics: ChartMetric[];
  setChartDraftMetricId: Dispatch<SetStateAction<string>>;
  onAdd: () => void;
}) {
  const canAdd = Boolean(chartDraftMetricId) && !charts.some((chart) => chart.metricId === chartDraftMetricId);
  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border border-line bg-white p-2 shadow-soft">
      <select
        value={chartDraftMetricId}
        onChange={(event) => setChartDraftMetricId(event.target.value)}
        className="min-h-9 max-w-72 rounded-md border border-line bg-white px-3 py-1.5 text-sm text-ink outline-none focus:border-leaf"
        aria-label="Metric"
      >
        {metricGroups(metrics).map((group) => (
          <optgroup key={group.name} label={group.name}>
            {group.metrics.map((metric) => (
              <option key={metric.id} value={metric.id}>
                {metric.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
      <button
        type="button"
        disabled={!canAdd}
        onClick={onAdd}
        className="min-h-9 rounded-md bg-ink px-3 py-1.5 text-sm font-semibold text-white hover:bg-leaf disabled:cursor-not-allowed disabled:bg-slate-300"
      >
        Add Chart
      </button>
    </div>
  );
}

function RunCharts({
  charts,
  metrics,
  rows,
  onOpen,
  onRemove,
}: {
  charts: ChartConfig[];
  metrics: ChartMetric[];
  rows: ModelRunRow[];
  onOpen: (chartId: string) => void;
  onRemove: (chartId: string) => void;
}) {
  return (
    <section className="mb-5">
      {charts.length ? (
        <div className="overflow-x-auto pb-2">
          <div className="flex min-w-max gap-3">
            {charts.map((chart) => {
              const metric = metrics.find((item) => item.id === chart.metricId);
              return metric ? (
                <ChartCard
                  key={chart.id}
                  chart={chart}
                  metric={metric}
                  rows={rows}
                  onOpen={onOpen}
                  onRemove={onRemove}
                />
              ) : null;
            })}
          </div>
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-line bg-white p-6 text-sm text-slate-600">
          Add a chart to compare runs.
        </div>
      )}
    </section>
  );
}

function ChartCard({
  chart,
  metric,
  rows,
  onOpen,
  onRemove,
}: {
  chart: ChartConfig;
  metric: ChartMetric;
  rows: ModelRunRow[];
  onOpen: (chartId: string) => void;
  onRemove: (chartId: string) => void;
}) {
  return (
    <div className="w-[22rem] shrink-0 rounded-md border border-line bg-white p-2 shadow-soft">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-ink">{metric.label}</h3>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={() => onRemove(chart.id)}
            className="flex h-7 w-7 items-center justify-center rounded-md border border-line text-lg leading-none text-ink hover:border-coral hover:text-coral"
            aria-label={`Remove ${metric.label}`}
          >
            &times;
          </button>
        </div>
      </div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => onOpen(chart.id)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onOpen(chart.id);
          }
        }}
        className="mt-2 block w-full cursor-pointer text-left outline-none focus:ring-2 focus:ring-leaf/40"
        aria-label={`Open ${metric.label}`}
      >
        <ChartView metric={metric} rows={rows} compact />
      </div>
    </div>
  );
}

function ChartModal({
  metric,
  rows,
  onClose,
}: {
  metric: ChartMetric | null;
  rows: ModelRunRow[];
  onClose: () => void;
}) {
  if (!metric) {
    return null;
  }
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-ink/30 p-4" role="dialog" aria-modal="true">
      <div className="flex max-h-[92vh] w-full max-w-6xl flex-col rounded-md bg-white shadow-soft">
        <div className="flex items-start justify-between gap-3 border-b border-line p-4">
          <div>
            <div className="text-xs font-semibold uppercase tracking-normal text-leaf">{metric.group}</div>
            <h2 className="mt-1 text-2xl font-semibold text-ink">{metric.label}</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-line px-3 py-2 text-sm font-semibold text-ink hover:border-ink"
          >
            Close
          </button>
        </div>
        <div className="overflow-auto p-4">
          <ChartView metric={metric} rows={rows} large />
        </div>
      </div>
    </div>
  );
}

function ChartView({
  compact = false,
  large = false,
  metric,
  rows,
}: {
  compact?: boolean;
  large?: boolean;
  metric: ChartMetric;
  rows: ModelRunRow[];
}) {
  const groups = useMemo(() => buildChartGroups(rows, metric), [metric, rows]);
  const modelKeys = useMemo(() => chartModelKeys(rows), [rows]);
  const chartData = useMemo(() => buildChartData(groups, modelKeys), [groups, modelKeys]);
  const values = groups.flatMap((group) =>
    modelKeys.map((modelKey) => group.values.get(modelKey)).filter((value): value is number => isChartNumber(value)),
  );

  if (!groups.length || !modelKeys.length || !values.length) {
    return (
      <div className="flex h-32 items-center justify-center rounded-md border border-line bg-mist text-sm text-slate-500">
        No values yet.
      </div>
    );
  }

  const height = large ? 440 : 160;
  const yMax = Math.max(metric.scaleMax ?? 0, niceMax(Math.max(...values) * 1.08));
  const margin = { top: 10, right: 12, bottom: 4, left: large ? 34 : 8 };

  return (
    <div className="overflow-hidden rounded-md border border-line bg-white">
      {compact ? null : <ChartLegend modelKeys={modelKeys} />}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={margin}>
            <CartesianGrid stroke="#e2e8f0" vertical={false} />
            <XAxis dataKey="runLabel" tick={{ fill: "#64748b", fontSize: large ? 11 : 10 }} tickLine={false} axisLine={{ stroke: "#cbd5e1" }} />
            <YAxis
              width={large ? 62 : 42}
              tick={{ fill: "#64748b", fontSize: large ? 11 : 10 }}
              tickFormatter={metric.axis}
              tickLine={false}
              axisLine={false}
              domain={[0, yMax]}
            />
            <Tooltip content={<ChartTooltip metric={metric} />} />
            {modelKeys.map((modelKey) => (
              <Line
                key={modelKey}
                type="monotone"
                dataKey={modelKey}
                stroke={modelColor(modelKey, modelKeys)}
                strokeWidth={large ? 2.5 : 2}
                dot={{ r: large ? 3 : 2 }}
                activeDot={{ r: large ? 5 : 4 }}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function ChartLegend({ modelKeys }: { modelKeys: string[] }) {
  return (
    <div className="flex max-h-12 flex-wrap gap-x-2 gap-y-1 overflow-hidden border-b border-line px-2 py-1.5">
      {modelKeys.map((modelKey) => (
        <span key={modelKey} className="inline-flex items-center gap-1.5 text-xs text-slate-600">
          <span className="h-2 w-2 rounded-sm" style={{ backgroundColor: modelColor(modelKey, modelKeys) }} />
          <span className="max-w-24 truncate">{modelKey}</span>
        </span>
      ))}
    </div>
  );
}

function ChartTooltip({
  active,
  label,
  metric,
  payload,
}: {
  active?: boolean;
  label?: string;
  metric: ChartMetric;
  payload?: Array<{ color?: string; dataKey?: string | number; value?: number | null }>;
}) {
  if (!active || !payload?.length) {
    return null;
  }
  return (
    <div className="rounded-md border border-line bg-white p-3 text-xs shadow-soft">
      <div className="mb-2 font-semibold text-ink">{label}</div>
      <div className="space-y-1">
        {payload
          .filter((item) => isChartNumber(item.value))
          .map((item) => (
            <div key={String(item.dataKey)} className="flex items-center justify-between gap-4 text-slate-600">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: item.color }} />
                {item.dataKey}
              </span>
              <span className="font-medium text-ink">{metric.format(item.value)}</span>
            </div>
          ))}
      </div>
    </div>
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

function newestRuns(runs: RunSummary[], count: number): RunSummary[] {
  return [...runs]
    .sort((left, right) => timestamp(right.startedAt) - timestamp(left.startedAt))
    .slice(0, count);
}

function timestamp(value?: string | null): number {
  if (!value) {
    return 0;
  }
  const parsed = new Date(value).getTime();
  return Number.isFinite(parsed) ? parsed : 0;
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

function buildChartMetrics(scoreMetrics: ScoreAggregate[]): ChartMetric[] {
  const scoreCharts = scoreMetrics.map<ChartMetric>((scoreMetric) => ({
    id: `score:${scoreMetric.id}`,
    label: scoreMetric.label,
    group: "Scores",
    format: (value) => formatNumber(value),
    axis: axisScore,
    value: (row) => row.scoreAggregates.find((score) => score.id === scoreMetric.id)?.mean ?? null,
    scaleMax: 1,
  }));
  return [
    ...scoreCharts,
    {
      id: "latency:avg",
      label: "Avg time",
      group: "Latency",
      format: formatSeconds,
      axis: axisSeconds,
      value: (row) => row.latency.avg,
    },
    {
      id: "latency:p50",
      label: "P50 time",
      group: "Latency",
      format: formatSeconds,
      axis: axisSeconds,
      value: (row) => row.latency.p50,
    },
    {
      id: "latency:p90",
      label: "P90 time",
      group: "Latency",
      format: formatSeconds,
      axis: axisSeconds,
      value: (row) => row.latency.p90,
    },
    {
      id: "tokens:input",
      label: "Input tokens",
      group: "Tokens",
      format: formatInt,
      axis: compactNumber,
      value: (row) => row.usage.input_tokens,
    },
    {
      id: "tokens:cached",
      label: "Cached tokens",
      group: "Tokens",
      format: formatInt,
      axis: compactNumber,
      value: (row) => row.usage.cached_tokens,
    },
    {
      id: "tokens:output",
      label: "Output tokens",
      group: "Tokens",
      format: formatInt,
      axis: compactNumber,
      value: (row) => row.usage.output_tokens,
    },
    {
      id: "tokens:reasoning",
      label: "Reasoning tokens",
      group: "Tokens",
      format: formatInt,
      axis: compactNumber,
      value: (row) => row.usage.reasoning_tokens,
    },
    {
      id: "tokens:total",
      label: "Total tokens",
      group: "Tokens",
      format: formatInt,
      axis: compactNumber,
      value: (row) => row.usage.total_tokens,
    },
    {
      id: "counts:failed",
      label: "Failed",
      group: "Counts",
      format: formatInt,
      axis: compactNumber,
      value: (row) => row.failedCount,
    },
    {
      id: "counts:evalErrors",
      label: "Eval errors",
      group: "Counts",
      format: formatInt,
      axis: compactNumber,
      value: (row) => row.evaluatorErrorCount,
    },
  ];
}

function defaultCharts(metrics: ChartMetric[]): ChartConfig[] {
  const preferredIds = [metrics.find((metric) => metric.id.startsWith("score:"))?.id, "latency:avg", "tokens:total"].filter(
    (id): id is string => Boolean(id) && metrics.some((metric) => metric.id === id),
  );
  return [...new Set(preferredIds)].map((metricId) => ({ id: metricId, metricId }));
}

function metricGroups(metrics: ChartMetric[]) {
  const groups = new Map<string, ChartMetric[]>();
  for (const metric of metrics) {
    groups.set(metric.group, [...(groups.get(metric.group) ?? []), metric]);
  }
  return [...groups.entries()].map(([name, groupMetrics]) => ({ name, metrics: groupMetrics }));
}

function buildChartGroups(rows: ModelRunRow[], metric: ChartMetric): ChartGroup[] {
  const groups = new Map<string, ChartGroup>();
  for (const row of rows) {
    const group =
      groups.get(row.runKey) ??
      ({
        runKey: row.runKey,
        experimentName: row.experimentName,
        startedAt: row.startedAt,
        values: new Map<string, number | null>(),
      } satisfies ChartGroup);
    group.values.set(row.modelKey, metric.value(row));
    groups.set(row.runKey, group);
  }
  return [...groups.values()].sort((left, right) => timestamp(left.startedAt) - timestamp(right.startedAt));
}

function buildChartData(groups: ChartGroup[], modelKeys: string[]): ChartDatum[] {
  return groups.map((group) => {
    const datum: ChartDatum = {
      experimentName: group.experimentName,
      runKey: group.runKey,
      runLabel: shortRunTick(group.startedAt),
    };
    for (const modelKey of modelKeys) {
      datum[modelKey] = group.values.get(modelKey) ?? null;
    }
    return datum;
  });
}

function chartModelKeys(rows: ModelRunRow[]): string[] {
  return [...new Set(rows.map((row) => row.modelKey))].sort((left, right) => left.localeCompare(right));
}

function modelColor(modelKey: string, modelKeys: string[]): string {
  const index = modelKeys.indexOf(modelKey);
  return modelPalette[(index < 0 ? 0 : index) % modelPalette.length];
}

function isChartNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function niceMax(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 1;
  }
  const magnitude = 10 ** Math.floor(Math.log10(value));
  const scaled = value / magnitude;
  const nice = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return nice * magnitude;
}

function axisScore(value: number): string {
  if (value === 0 || value === 1) {
    return value.toFixed(0);
  }
  return value.toFixed(2);
}

function axisSeconds(value: number): string {
  if (value < 10) {
    return `${value.toFixed(1)}s`;
  }
  return `${value.toFixed(0)}s`;
}

function compactNumber(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, notation: "compact" }).format(value);
}

function shortRunTick(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Run";
  }
  return date.toLocaleString(undefined, { day: "numeric", hour: "numeric", minute: "2-digit", month: "numeric" });
}
