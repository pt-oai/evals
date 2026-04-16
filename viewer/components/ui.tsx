"use client";

import { flexRender, type Cell, type Row, type Table } from "@tanstack/react-table";
import { ReactNode } from "react";

import { statusLabel } from "../lib/evals";
import type { RunSummary } from "../lib/types";
import { statusTone } from "./format";

export function StatusBadge({ status }: { status: RunSummary["status"] | "success" | "failed" | "skipped" }) {
  return (
    <span className={`inline-flex min-w-24 justify-center rounded-md border px-2 py-1 text-xs font-semibold ${statusTone(status)}`}>
      {statusLabel(status)}
    </span>
  );
}

export function PageTitle({
  eyebrow,
  subtitle,
  title,
  children,
}: {
  eyebrow?: string;
  subtitle?: ReactNode;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-col gap-3 border-b border-line pb-5 md:flex-row md:items-end md:justify-between">
      <div>
        {eyebrow ? <p className="mb-1 text-sm font-semibold uppercase tracking-normal text-leaf">{eyebrow}</p> : null}
        <h1 className="text-3xl font-semibold text-ink md:text-4xl">{title}</h1>
        {subtitle ? <div className="mt-2 text-sm text-slate-500">{subtitle}</div> : null}
      </div>
      {children}
    </div>
  );
}

export function StatGrid({ children }: { children: ReactNode }) {
  return <div className="mb-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{children}</div>;
}

export function Stat({ label, value, detail }: { label: string; value: ReactNode; detail?: ReactNode }) {
  return (
    <div className="rounded-md border border-line bg-white p-4 shadow-soft">
      <div className="text-xs font-semibold uppercase tracking-normal text-slate-500">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-ink">{value}</div>
      {detail ? <div className="mt-1 text-sm text-slate-600">{detail}</div> : null}
    </div>
  );
}

export function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-md border border-line bg-white p-8 text-center shadow-soft">
      <h2 className="text-xl font-semibold text-ink">{title}</h2>
      <p className="mx-auto mt-2 max-w-xl text-sm text-slate-600">{body}</p>
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-coral/30 bg-coral/10 p-4 text-sm font-medium text-coral">
      {message}
    </div>
  );
}

export function LoadingState() {
  return <div className="rounded-md border border-line bg-white p-6 text-sm text-slate-600 shadow-soft">Loading runs...</div>;
}

export function Toolbar({ children }: { children: ReactNode }) {
  return <div className="mb-4 flex flex-wrap items-center gap-3 rounded-md border border-line bg-white p-3 shadow-soft">{children}</div>;
}

export function SearchInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={`min-h-10 min-w-60 rounded-md border border-line bg-white px-3 py-2 text-sm text-ink outline-none focus:border-leaf ${props.className ?? ""}`}
    />
  );
}

export function SelectInput(props: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      {...props}
      className={`min-h-10 rounded-md border border-line bg-white px-3 py-2 text-sm text-ink outline-none focus:border-leaf ${props.className ?? ""}`}
    />
  );
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
}) {
  return (
    <label className="inline-flex min-h-10 cursor-pointer items-center gap-2 rounded-md border border-line bg-white px-3 py-2 text-sm text-ink">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-4 w-4 accent-leaf"
      />
      {label}
    </label>
  );
}

export function DataTable<T>({
  table,
  empty,
  getCellRowSpan,
  getRowClassName,
}: {
  table: Table<T>;
  empty: string;
  getCellRowSpan?: (cell: Cell<T, unknown>) => number;
  getRowClassName?: (row: Row<T>) => string;
}) {
  return (
    <div className="overflow-hidden rounded-md border border-line bg-white shadow-soft">
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse text-left text-xs">
          <thead className="bg-mist text-xs uppercase tracking-normal text-slate-600">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => {
                  const label = header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext());
                  const sorted = header.column.getIsSorted();
                  return (
                    <th key={header.id} className="border-b border-line px-2 py-2 font-semibold">
                      {header.isPlaceholder ? null : header.column.getCanSort() ? (
                        <button
                          type="button"
                          onClick={header.column.getToggleSortingHandler()}
                          className="flex w-full items-center justify-between gap-2 text-left"
                        >
                          <span>{label}</span>
                          <span className="text-slate-400">{sorted === "asc" ? "Asc" : sorted === "desc" ? "Desc" : ""}</span>
                        </button>
                      ) : (
                        <div className="flex w-full items-center justify-between gap-2 text-left">
                          <span>{label}</span>
                        </div>
                      )}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.length ? (
              table.getRowModel().rows.map((row) => (
                <tr
                  key={row.id}
                  className={`border-b border-line last:border-b-0 ${getRowClassName?.(row) ?? ""}`}
                >
                  {row.getVisibleCells().map((cell) => {
                    const rowSpan = getCellRowSpan?.(cell) ?? 1;
                    if (rowSpan === 0) {
                      return null;
                    }
                    return (
                      <td key={cell.id} rowSpan={rowSpan} className="overflow-hidden px-2 py-2 align-top text-slate-700">
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </td>
                    );
                  })}
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={table.getAllLeafColumns().length} className="px-3 py-8 text-center text-sm text-slate-500">
                  {empty}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function JsonBlock({ value }: { value: unknown }) {
  return (
    <pre className="max-h-[50vh] overflow-auto rounded-md border border-line bg-slate-950 p-4 text-xs leading-5 text-slate-100">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}
