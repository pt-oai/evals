import { NextResponse } from "next/server";

import { loadCompare, loadLanes } from "../../../lib/server/runs";

export const runtime = "nodejs";

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const baselineRun = url.searchParams.get("baselineRun");
    const baselineModel = url.searchParams.get("baselineModel");
    const candidateRun = url.searchParams.get("candidateRun");
    const candidateModel = url.searchParams.get("candidateModel");
    if (!baselineRun || !baselineModel || !candidateRun || !candidateModel) {
      return NextResponse.json({ lanes: await loadLanes() });
    }
    return NextResponse.json(
      await loadCompare({
        baselineRun,
        baselineModel,
        candidateRun,
        candidateModel,
      }),
    );
  } catch (error) {
    return NextResponse.json({ error: errorMessage(error) }, { status: 400 });
  }
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Could not compare runs.";
}
