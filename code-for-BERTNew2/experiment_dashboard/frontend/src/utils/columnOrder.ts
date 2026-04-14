/** 与 naive-ui data-table getColKey 一致，用于列顺序持久化 */
export function getColumnStableKey(col: unknown): string | null {
  if (col && typeof col === 'object' && 'type' in col) {
    const t = (col as { type?: string }).type
    if (t === 'selection') return '__n_selection__'
    if (t === 'expand') return '__n_expand__'
  }
  if (col && typeof col === 'object' && 'key' in col) {
    const k = (col as { key?: unknown }).key
    if (k != null && k !== '') return String(k)
  }
  return null
}

export function applyColumnOrder<T>(base: T[], savedKeys: string[] | undefined | null): T[] {
  if (!savedKeys || savedKeys.length === 0) return base
  const byKey = new Map<string, T>()
  const defaultKeys: string[] = []
  for (const col of base) {
    const k = getColumnStableKey(col)
    if (!k) continue
    byKey.set(k, col)
    defaultKeys.push(k)
  }
  const seen = new Set<string>()
  const out: T[] = []
  for (const k of savedKeys) {
    if (byKey.has(k) && !seen.has(k)) {
      out.push(byKey.get(k)!)
      seen.add(k)
    }
  }
  for (const k of defaultKeys) {
    if (!seen.has(k)) {
      out.push(byKey.get(k)!)
      seen.add(k)
    }
  }
  return out
}
