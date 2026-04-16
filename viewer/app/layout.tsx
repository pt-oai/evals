import type { Metadata } from "next";
import Link from "next/link";

import { loadViewerInfo } from "../lib/server/viewer";
import "./globals.css";

export const metadata: Metadata = {
  title: "Prism Evals",
  description: "Review local Prism Evals runs.",
};

export const dynamic = "force-dynamic";

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const viewerInfo = loadViewerInfo();

  return (
    <html lang="en">
      <body>
        <header className="border-b border-line bg-white/90">
          <div className="flex w-full flex-col gap-4 px-5 py-5 md:flex-row md:items-center md:justify-between">
            <div>
              <Link href="/" className="text-2xl font-semibold text-ink">
                <span className="inline-flex items-center gap-2">
                  <span aria-hidden="true" className="grid h-7 w-7 grid-cols-3 overflow-hidden rounded-md border border-line bg-white p-1 shadow-soft">
                    <span className="rounded-sm bg-leaf" />
                    <span className="rounded-sm bg-ink" />
                    <span className="rounded-sm bg-coral" />
                  </span>
                  Prism Evals
                </span>
              </Link>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                <span className="font-medium">{viewerInfo.tag}</span>
                {viewerInfo.updateAvailable && viewerInfo.latestTag ? (
                  <span className="rounded-md bg-mist px-1.5 py-0.5 font-medium text-leaf">
                    {viewerInfo.latestTag} available
                  </span>
                ) : null}
              </div>
            </div>
            <nav className="flex flex-wrap items-center gap-2 text-sm">
              <Link
                href="/"
                className="rounded-md border border-line bg-white px-3 py-2 font-medium text-ink hover:border-ink"
              >
                Runs
              </Link>
              <Link
                href="/compare"
                className="rounded-md border border-line bg-white px-3 py-2 font-medium text-ink hover:border-ink"
              >
                Compare
              </Link>
            </nav>
          </div>
        </header>
        <main className="w-full px-5 py-6">{children}</main>
      </body>
    </html>
  );
}
