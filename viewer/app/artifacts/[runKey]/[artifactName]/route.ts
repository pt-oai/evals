import { promises as fs } from "fs";
import path from "path";

import { resolveArtifact } from "../../../../lib/server/runs";

export const runtime = "nodejs";

interface Context {
  params: Promise<{ runKey: string; artifactName: string }>;
}

export async function GET(_request: Request, context: Context) {
  try {
    const { runKey, artifactName } = await context.params;
    const file = await resolveArtifact(decodeURIComponent(runKey), decodeURIComponent(artifactName));
    const body = await fs.readFile(file);
    const name = path.basename(file);
    return new Response(body, {
      headers: {
        "content-disposition": `attachment; filename="${name}"`,
        "content-type": contentType(name),
      },
    });
  } catch (error) {
    return Response.json({ error: errorMessage(error) }, { status: 404 });
  }
}

function contentType(name: string): string {
  if (name.endsWith(".csv")) {
    return "text/csv; charset=utf-8";
  }
  if (name.endsWith(".jsonl")) {
    return "application/x-ndjson; charset=utf-8";
  }
  if (name.endsWith(".json")) {
    return "application/json; charset=utf-8";
  }
  return "application/octet-stream";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Could not load artifact.";
}
