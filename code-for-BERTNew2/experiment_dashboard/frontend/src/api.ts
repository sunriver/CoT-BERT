import type {
  CompareResponse,
  PointsFile,
  PointsScatterResponse,
  RunDetail,
  RunListItem,
} from './types'

const base = ''

export type RunsTab = 'all' | 'senteval' | 'alignment' | 'gold'

export async function fetchRuns(
  q?: string,
  tab: RunsTab = 'senteval'
): Promise<RunListItem[]> {
  const u = new URL(`${base}/api/runs`, window.location.origin)
  if (q) u.searchParams.set('q', q)
  u.searchParams.set('tab', tab)
  const r = await fetch(u.toString())
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function fetchRun(id: string): Promise<RunDetail> {
  const r = await fetch(`${base}/api/runs/${encodeURIComponent(id)}`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function fetchCompare(runIds: string[]): Promise<CompareResponse> {
  const u = new URL(`${base}/api/compare`, window.location.origin)
  u.searchParams.set('run_ids', runIds.join(','))
  const r = await fetch(u.toString())
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function refreshIndex(): Promise<void> {
  await fetch(`${base}/api/refresh`, { method: 'POST' })
}

export function artifactUrl(runId: string, filename: string): string {
  return `${base}/api/runs/${encodeURIComponent(runId)}/files/${encodeURIComponent(filename)}`
}

export async function fetchPointsFiles(): Promise<PointsFile[]> {
  const r = await fetch(`${base}/api/points-files`)
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}

export async function fetchPointsScatter(params: {
  files: string[]
  runId?: string
  datasetId: 'sts_test' | 'sick_test'
  limit?: number
}): Promise<PointsScatterResponse> {
  const u = new URL(`${base}/api/points-scatter`, window.location.origin)
  u.searchParams.set('files', params.files.join(','))
  u.searchParams.set('dataset_id', params.datasetId)
  if (params.runId) u.searchParams.set('run_id', params.runId)
  if (params.limit != null) u.searchParams.set('limit', String(params.limit))
  const r = await fetch(u.toString())
  if (!r.ok) throw new Error(await r.text())
  return r.json()
}
