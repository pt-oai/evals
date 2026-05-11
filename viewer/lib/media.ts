export type MediaKind = "image" | "audio" | "file";

export function mediaKind(mimeType: string): MediaKind {
  if (mimeType.startsWith("image/")) {
    return "image";
  }
  if (mimeType.startsWith("audio/")) {
    return "audio";
  }
  return "file";
}
