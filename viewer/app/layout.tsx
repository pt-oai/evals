import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "Eval Runs",
  description: "Review local eval runs.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <header className="border-b border-line bg-white/90">
          <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-5 md:flex-row md:items-center md:justify-between">
            <Link href="/" className="text-2xl font-semibold text-ink">
              Eval Runs
            </Link>
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
        <main className="mx-auto max-w-7xl px-5 py-6">{children}</main>
      </body>
    </html>
  );
}
