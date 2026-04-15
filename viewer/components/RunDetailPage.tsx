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

import {
  formatNumber,
  outputPreview,
  recordHasEvaluatorError,
  recordScore,
  scoreNumber,
} from "../lib/evals";
import type { EvalResult, ItemRunRecord, RunDetail, ScoreMetric, StepRecord } from "../lib/types";
import { formatDate, formatInt, formatScore, formatSeconds } from "./format";
import {
  DataTable,
  EmptyState,
  ErrorState,
  JsonBlock,
  LoadingState,
  PageTitle,
  SearchInput,
  SelectInput,
  Stat,
  StatGrid,
  StatusBadge,
  Toggle,
  Toolbar,
} from "./ui";

const columnHelper = createColumnHelper<ItemRunRecord>();

type DetailTab = "dataset" | "output" | "scores" | "steps" | "calls" | "data" | "error";

export function RunDetailPage({ runKey }: { runKey: string }) {
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [scoreFilter, setScoreFilter] = useState("");
  const [stepFilter, setStepFilter] = useState("");
  const [threshold, setThreshold] = useState("");
  const [failedOnly, setFailedOnly] = useState(false);
  const [evalErrorsOnly, setEvalErrorsOnly] = useState(false);
  const [selected, setSelected] = useState<ItemRunRecord | null>(null);
  const [tab, setTab] = useState<DetailTab>("output");
  const [sorting, setSorting] = useState<SortingState>([{ id: "item_index", desc: false }]);

  useEffect(() => {
    fetch(`/api/runs/${encodeURIComponent(runKey)}`)
      .then(async (response) => {
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error ?? "Run could not be loaded.");
        }
        setDetail(payload);
      })
      .catch((caught: Error) => setError(caught.message))
      .finally(() => setLoading(false));
  }, [runKey]);

  const filteredRecords = useMemo(() => {
    if (!detail) {
      return [];
    }
    const needle = query.trim().toLowerCase();
    const maxScore = threshold.trim() ? Number(threshold) : null;
    return detail.records.filter((record) => {
      if (failedOnly && record.status !== "failed") {
        return false;
      }
      if (evalErrorsOnly && !recordHasEvaluatorError(record)) {
        return false;
      }
      if (modelFilter && record.model_key !== modelFilter) {
        return false;
      }
      if (scoreFilter) {
        const metric = detail.metrics.find((item) => item.id === scoreFilter);
        if (!metric || recordScore(record, metric) === null) {
          return false;
        }
      }
      if (stepFilter && !(record.steps ?? []).some((step) => step.key === stepFilter)) {
        return false;
      }
      if (maxScore !== null && Number.isFinite(maxScore)) {
        const scores = detail.metrics.map((metric) => scoreNumber(recordScore(record, metric))).filter((score) => score !== null);
        if (!scores.some((score) => score < maxScore)) {
          return false;
        }
      }
      if (!needle) {
        return true;
      }
      return [
        record.item_id,
        record.model_key,
        record.model,
        record.status,
        record.output?.text ?? "",
        record.error?.message ?? "",
        ...Object.values(record.item ?? {}),
      ]
        .join(" ")
        .toLowerCase()
        .includes(needle);
    });
  }, [detail, evalErrorsOnly, failedOnly, modelFilter, query, scoreFilter, stepFilter, threshold]);

  const columns = useMemo(
    () => [
      columnHelper.accessor("item_index", {
        header: "Item",
        cell: ({ row }) => (
          <div className="min-w-32">
            <button
              type="button"
              onClick={() => {
                setSelected(row.original);
                setTab("output");
              }}
              className="font-semibold text-ink hover:text-leaf"
            >
              {row.original.item_id}
            </button>
            <div className="mt-1 text-xs text-slate-500">#{row.original.item_index}</div>
          </div>
        ),
      }),
      columnHelper.accessor("model_key", {
        header: "Model",
        cell: ({ getValue }) => <span className="block min-w-28">{getValue()}</span>,
      }),
      columnHelper.accessor("status", {
        header: "Status",
        cell: ({ getValue }) => <StatusBadge status={getValue()} />,
      }),
      columnHelper.display({
        id: "output",
        header: "Output",
        cell: ({ row }) => <p className="max-h-24 min-w-80 overflow-hidden whitespace-pre-wrap text-sm">{outputPreview(row.original, 360) || "-"}</p>,
      }),
      columnHelper.display({
        id: "scores",
        header: "Scores",
        cell: ({ row }) => (
          <div className="flex min-w-56 flex-wrap gap-1">
            {(detail?.metrics ?? []).slice(0, 4).map((metric) => (
              <span key={metric.id} className="rounded-md border border-line bg-mist px-2 py-1 text-xs text-ink">
                {metric.label}: {formatScore(recordScore(row.original, metric))}
              </span>
            ))}
            {!(detail?.metrics ?? []).length ? <span>-</span> : null}
          </div>
        ),
      }),
      columnHelper.accessor("duration_s", {
        header: "Time",
        cell: ({ getValue }) => formatSeconds(getValue()),
      }),
      columnHelper.accessor((row) => row.usage?.total_tokens ?? 0, {
        id: "tokens",
        header: "Tokens",
        cell: ({ getValue }) => formatInt(getValue()),
      }),
      columnHelper.display({
        id: "open",
        header: "",
        cell: ({ row }) => (
          <button
            type="button"
            onClick={() => {
              setSelected(row.original);
              setTab("output");
            }}
            className="rounded-md border border-line px-3 py-2 text-xs font-semibold text-ink hover:border-leaf hover:text-leaf"
          >
            Open
          </button>
        ),
      }),
    ],
    [detail?.metrics],
  );

  const table = useReactTable({
    data: filteredRecords,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  const scoreMatrix = useMemo(() => (detail ? buildScoreMatrix(detail.records, detail.metrics) : []), [detail]);

  if (loading) {
    return <LoadingState />;
  }
  if (error) {
    return <ErrorState message={error} />;
  }
  if (!detail) {
    return <EmptyState title="Run not found" body="Choose another run from the list." />;
  }

  return (
    <>
      <PageTitle eyebrow="Run detail" title={detail.summary.experimentName}>
        <div className="flex flex-wrap gap-2">
          <Link href="/" className="rounded-md border border-line bg-white px-3 py-2 text-sm font-semibold text-ink hover:border-ink">
            All runs
          </Link>
          <Link href="/compare" className="rounded-md bg-ink px-3 py-2 text-sm font-semibold text-white hover:bg-leaf">
            Compare
          </Link>
        </div>
      </PageTitle>

      <StatGrid>
        <Stat label="Status" value={<StatusBadge status={detail.summary.status} />} detail={formatDate(detail.summary.startedAt)} />
        <Stat label="Items" value={formatInt(detail.summary.itemRunCount)} detail={`${formatInt(detail.summary.failedCount)} failed`} />
        <Stat label="Eval errors" value={formatInt(detail.summary.evaluatorErrorCount)} detail="Recorded by evaluators" />
        <Stat label="Tokens" value={formatInt(detail.summary.totalTokens)} detail={`Avg ${formatSeconds(detail.summary.latency.avg)}`} />
      </StatGrid>

      <section className="mb-6">
        <div className="mb-3 flex items-center justify-between gap-3">
          <h2 className="text-xl font-semibold text-ink">Scores by model</h2>
          <div className="flex flex-wrap gap-2">
            {detail.summary.artifacts.map((artifact) => (
              <a
                key={artifact.name}
                href={artifact.href}
                className="rounded-md border border-line bg-white px-3 py-2 text-xs font-semibold text-ink hover:border-leaf hover:text-leaf"
              >
                {artifactLabel(artifact.name)}
              </a>
            ))}
          </div>
        </div>
        <ScoreMatrix rows={scoreMatrix} modelKeys={detail.summary.modelKeys} />
      </section>

      <Toolbar>
        <SearchInput value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search items" />
        <SelectInput value={modelFilter} onChange={(event) => setModelFilter(event.target.value)} aria-label="Model">
          <option value="">All models</option>
          {detail.summary.modelKeys.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))}
        </SelectInput>
        <SelectInput value={scoreFilter} onChange={(event) => setScoreFilter(event.target.value)} aria-label="Score">
          <option value="">All scores</option>
          {detail.metrics.map((metric) => (
            <option key={metric.id} value={metric.id}>
              {metric.label}
            </option>
          ))}
        </SelectInput>
        <SelectInput value={stepFilter} onChange={(event) => setStepFilter(event.target.value)} aria-label="Step">
          <option value="">All steps</option>
          {detail.stepKeys.map((step) => (
            <option key={step} value={step}>
              {step}
            </option>
          ))}
        </SelectInput>
        <SearchInput
          type="number"
          step="0.01"
          value={threshold}
          onChange={(event) => setThreshold(event.target.value)}
          placeholder="Score below"
          className="min-w-36"
        />
        <Toggle checked={failedOnly} onChange={setFailedOnly} label="Failed" />
        <Toggle checked={evalErrorsOnly} onChange={setEvalErrorsOnly} label="Eval errors" />
      </Toolbar>

      <DataTable table={table} empty="No matching item runs." />
      {selected ? <RecordDrawer record={selected} metrics={detail.metrics} tab={tab} setTab={setTab} onClose={() => setSelected(null)} /> : null}
    </>
  );
}

function ScoreMatrix({
  rows,
  modelKeys,
}: {
  rows: Array<{ metric: ScoreMetric; byModel: Record<string, { mean: number | null; count: number }> }>;
  modelKeys: string[];
}) {
  if (!rows.length) {
    return <EmptyState title="No scores yet" body="This run has no recorded score values." />;
  }
  return (
    <div className="overflow-hidden rounded-md border border-line bg-white shadow-soft">
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-left text-sm">
          <thead className="bg-mist text-xs uppercase tracking-normal text-slate-600">
            <tr>
              <th className="border-b border-line px-3 py-3">Score</th>
              {modelKeys.map((model) => (
                <th key={model} className="border-b border-line px-3 py-3 text-right">
                  {model}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.metric.id} className="border-b border-line last:border-b-0">
                <td className="px-3 py-3 font-medium text-ink">{row.metric.label}</td>
                {modelKeys.map((model) => (
                  <td key={model} className="px-3 py-3 text-right text-slate-700">
                    {formatNumber(row.byModel[model]?.mean)}{" "}
                    <span className="text-xs text-slate-400">({row.byModel[model]?.count ?? 0})</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RecordDrawer({
  record,
  metrics,
  tab,
  setTab,
  onClose,
}: {
  record: ItemRunRecord;
  metrics: ScoreMetric[];
  tab: DetailTab;
  setTab: (tab: DetailTab) => void;
  onClose: () => void;
}) {
  const tabs: Array<{ id: DetailTab; label: string }> = [
    { id: "output", label: "Output" },
    { id: "scores", label: "Scores" },
    { id: "steps", label: "Steps" },
    { id: "dataset", label: "Dataset" },
    { id: "calls", label: "Calls" },
    { id: "data", label: "Data" },
    { id: "error", label: "Error" },
  ];
  return (
    <div className="fixed inset-0 z-50 bg-ink/20">
      <aside className="ml-auto flex h-full w-full max-w-4xl flex-col bg-white shadow-soft">
        <div className="border-b border-line p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-sm text-slate-500">
                {record.model_key} - repetition {record.repetition}
              </div>
              <h2 className="mt-1 text-2xl font-semibold text-ink">{record.item_id}</h2>
            </div>
            <button type="button" onClick={onClose} className="rounded-md border border-line px-3 py-2 text-sm font-semibold text-ink hover:border-ink">
              Close
            </button>
          </div>
          <div className="mt-4 flex flex-wrap gap-2">
            {tabs.map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setTab(item.id)}
                className={`rounded-md border px-3 py-2 text-sm font-semibold ${
                  tab === item.id ? "border-ink bg-ink text-white" : "border-line bg-white text-ink hover:border-leaf"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {tab === "output" ? <TextPanel text={record.output?.text ?? ""} /> : null}
          {tab === "scores" ? <ScoresPanel record={record} metrics={metrics} /> : null}
          {tab === "steps" ? <StepsPanel steps={record.steps ?? []} /> : null}
          {tab === "dataset" ? <JsonBlock value={record.item} /> : null}
          {tab === "calls" ? <JsonBlock value={record.generations ?? []} /> : null}
          {tab === "data" ? <JsonBlock value={record} /> : null}
          {tab === "error" ? <JsonBlock value={record.error ?? "No error recorded."} /> : null}
        </div>
      </aside>
    </div>
  );
}

function TextPanel({ text }: { text: string }) {
  return <pre className="whitespace-pre-wrap rounded-md border border-line bg-mist p-4 text-sm leading-6 text-ink">{text || "No output recorded."}</pre>;
}

function ScoresPanel({ record, metrics }: { record: ItemRunRecord; metrics: ScoreMetric[] }) {
  if (!metrics.length) {
    return <EmptyState title="No scores" body="This item has no recorded scores." />;
  }
  return (
    <div className="space-y-3">
      {metrics.map((metric) => {
        const result = findEval(record, metric);
        return (
          <div key={metric.id} className="rounded-md border border-line p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="font-semibold text-ink">{metric.label}</div>
              <div className="rounded-md border border-line bg-mist px-2 py-1 text-sm text-ink">{formatScore(result?.score ?? null)}</div>
            </div>
            {result?.comment ? <p className="mt-2 text-sm text-slate-700">{result.comment}</p> : null}
            {result?.error ? <p className="mt-2 text-sm font-medium text-coral">{result.error.message}</p> : null}
          </div>
        );
      })}
    </div>
  );
}

function StepsPanel({ steps }: { steps: StepRecord[] }) {
  if (!steps.length) {
    return <EmptyState title="No steps" body="This item has no recorded steps." />;
  }
  return (
    <div className="space-y-3">
      {steps.map((step) => (
        <div key={step.key} className="rounded-md border border-line p-3">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="font-semibold text-ink">{step.key}</div>
            <div className="text-sm text-slate-500">
              {step.status} - {formatSeconds(step.duration_s)}
            </div>
          </div>
          <pre className="max-h-56 overflow-auto whitespace-pre-wrap rounded-md bg-mist p-3 text-sm text-ink">
            {step.output?.text || "No output recorded."}
          </pre>
        </div>
      ))}
    </div>
  );
}

function buildScoreMatrix(records: ItemRunRecord[], metrics: ScoreMetric[]) {
  const modelKeys = [...new Set(records.map((record) => record.model_key))].sort();
  return metrics.map((metric) => {
    const byModel: Record<string, { mean: number | null; count: number }> = {};
    for (const model of modelKeys) {
      const values = records
        .filter((record) => record.model_key === model)
        .map((record) => scoreNumber(recordScore(record, metric)))
        .filter((score) => score !== null);
      byModel[model] = {
        mean: values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null,
        count: values.length,
      };
    }
    return { metric, byModel };
  });
}

function findEval(record: ItemRunRecord, metric: ScoreMetric): EvalResult | undefined {
  if (metric.scope === "item_run") {
    return (record.evals ?? []).find((result) => result.key === metric.scoreKey);
  }
  const step = (record.steps ?? []).find((item) => item.key === metric.stepKey);
  return (step?.evals ?? []).find((result) => result.key === metric.scoreKey);
}

function artifactLabel(name: string): string {
  switch (name) {
    case "results.csv":
      return "Results CSV";
    case "scores.csv":
      return "Scores CSV";
    case "steps.csv":
      return "Steps CSV";
    case "results.jsonl":
      return "Run data";
    case "manifest.json":
      return "Summary";
    default:
      return name;
  }
}
