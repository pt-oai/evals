import { promises as fs } from "fs";
import path from "path";

import { resolveMedia } from "../../../../lib/server/runs";

export const runtime = "nodejs";

interface Context {
  params: Promise<{ runKey: string; mediaPath: string[] }>;
}

export async function GET(_request: Request, context: Context) {
  try {
    const { runKey, mediaPath } = await context.params;
    const relativePath = mediaPath.map((part) => decodeURIComponent(part)).join("/");
    const file = await resolveMedia(decodeURIComponent(runKey), relativePath);
    const body = await fs.readFile(file);
    return new Response(body, {
      headers: {
        "content-type": contentType(path.basename(file)),
      },
    });
  } catch (error) {
    return Response.json({ error: errorMessage(error) }, { status: 404 });
  }
}

function contentType(name: string): string {
  if (name.endsWith(".png")) {
    return "image/png";
  }
  if (name.endsWith(".jpg") || name.endsWith(".jpeg")) {
    return "image/jpeg";
  }
  if (name.endsWith(".webp")) {
    return "image/webp";
  }
  if (name.endsWith(".gif")) {
    return "image/gif";
  }
  return "application/octet-stream";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Could not load media.";
}
