import { NextResponse } from "next/server";

import { loadRunDetail } from "../../../../../lib/server/runs";

export const runtime = "nodejs";

interface Context {
  params: Promise<{ runKey: string }>;
}

export async function GET(_request: Request, context: Context) {
  try {
    const { runKey } = await context.params;
    const detail = await loadRunDetail(decodeURIComponent(runKey));
    return NextResponse.json(detail.records);
  } catch (error) {
    return NextResponse.json({ error: errorMessage(error) }, { status: 404 });
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Could not load records.";
}
