import { RunDetailPage } from "../../../components/RunDetailPage";

interface PageProps {
  params: Promise<{ runKey: string }>;
}

export default async function Page({ params }: PageProps) {
  const { runKey } = await params;
  return <RunDetailPage runKey={decodeURIComponent(runKey)} />;
}
