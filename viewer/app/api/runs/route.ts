import { NextResponse } from "next/server";

import { loadRunSummaries } from "../../../lib/server/runs";

export const runtime = "nodejs";

export async function GET() {
  try {
    return NextResponse.json(await loadRunSummaries());
  } catch (error) {
    return NextResponse.json({ error: errorMessage(error) }, { status: 500 });
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Could not load runs.";
}
