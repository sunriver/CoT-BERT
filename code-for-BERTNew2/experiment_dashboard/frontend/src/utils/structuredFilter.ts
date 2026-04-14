import type { RunListItem } from '../types'

/** 自第二行起与上一有效条件的关系；NOT 表示「且不含本行」（AND NOT） */
export type StructuredJoin = 'AND' | 'OR' | 'NOT'

export type MatchMode = 'exact' | 'contains' | 'fuzzy'

export interface StructuredFilterRow {
  join: StructuredJoin | null
  field: string | null
  value: string
  match: MatchMode
}

export type StructuredFilterTrainParamsSource = 'merged' | 'eval_test'

export interface StructuredFilterOptions {
  /** 「全部」Tab 用 eval_test 参数列筛选用 eval_test */
  trainParamsSource?: StructuredFilterTrainParamsSource
}

function trainParamsFlatForFilter(
  row: RunListItem,
  src: StructuredFilterTrainParamsSource | undefined
): Record<string, unknown> {
  if (src === 'eval_test') {
    const e = row.train_params_flat_eval_test
    return e && typeof e === 'object' ? e : {}
  }
  const m = row.train_params_flat
  return m && typeof m === 'object' ? m : {}
}

function getField(row: RunListItem, field: string, opts?: StructuredFilterOptions): unknown {
  const flat = trainParamsFlatForFilter(row, opts?.trainParamsSource)
  if (field === 'gold_img_sick_test') return row.gold_cosine_png_by_dataset?.sick_test
  if (field === 'gold_img_sts_test') return row.gold_cosine_png_by_dataset?.sts_test
  if (field.startsWith('train_params_flat.')) {
    const k = field.slice('train_params_flat.'.length)
    return flat[k]
  }
  if (field !== 'train_params_flat' && field in row) {
    return (row as unknown as Record<string, unknown>)[field]
  }
  return flat[field]
}

function asNumber(x: unknown): number | null {
  if (typeof x === 'number' && Number.isFinite(x)) return x
  if (typeof x === 'string' && x.trim() !== '') {
    const n = Number(x)
    if (!Number.isNaN(n)) return n
  }
  return null
}

function parseBool(raw: string): boolean | null {
  const t = raw.trim().toLowerCase()
  if (t === 'true') return true
  if (t === 'false') return false
  return null
}

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

/**与 columnFilter 中一致：SQL LIKE 语义转 RegExp */
function sqlLikeToRegex(pattern: string, insensitive: boolean): RegExp {
  let re = ''
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i]
    if (c === '%') re += '.*'
    else if (c === '_') re += '.'
    else re += c.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  }
  return new RegExp(`^${re}$`, insensitive ? 'i' : '')
}

function cellExact(cell: unknown, raw: string): boolean {
  const t = raw.trim()
  if (!t) return true
  if (t.toLowerCase() === 'null') return cell == null
  const b = parseBool(t)
  if (b !== null) return Boolean(cell) === b
  if (/^-?\d/.test(t)) {
    const n = Number(t)
    if (!Number.isNaN(n)) {
      const cn = asNumber(cell)
      return cn !== null && cn === n
    }
  }
  return String(cell ?? '') === t
}

function cellContains(cell: unknown, raw: string): boolean {
  const pat = raw.trim()
  if (!pat) return true
  const s = cell == null ? '' : String(cell)
  return new RegExp(escapeRegExp(pat), 'i').test(s)
}

function cellFuzzy(cell: unknown, raw: string): boolean {
  const t = raw.trim()
  if (!t) return true
  const s = cell == null ? '' : String(cell)
  if (/[%_]/.test(t)) {
    try {
      return sqlLikeToRegex(t, true).test(s)
    } catch {
      return false
    }
  }
  return new RegExp(escapeRegExp(t), 'i').test(s)
}

function evalOne(
  r: StructuredFilterRow,
  dataRow: RunListItem,
  opts?: StructuredFilterOptions
): boolean {
  if (!r.field?.trim() || !r.value.trim()) return true
  const cell = getField(dataRow, r.field, opts)
  switch (r.match) {
    case 'exact':
      return cellExact(cell, r.value)
    case 'contains':
      return cellContains(cell, r.value)
    case 'fuzzy':
      return cellFuzzy(cell, r.value)
    default:
      return true
  }
}

export function structuredFilterHasActiveRows(rows: StructuredFilterRow[]): boolean {
  return rows.some((r) => Boolean(r.field?.trim() && r.value.trim()))
}

export function rowMatchesStructuredFilter(
  rows: StructuredFilterRow[],
  dataRow: RunListItem,
  opts?: StructuredFilterOptions
): boolean {
  const active = rows.filter((r) => r.field?.trim() && r.value.trim())
  if (!active.length) return true

  let acc = evalOne(active[0], dataRow, opts)
  for (let i = 1; i < active.length; i++) {
    const r = active[i]
    const j = r.join ?? 'AND'
    const next = evalOne(r, dataRow, opts)
    if (j === 'AND') acc = acc && next
    else if (j === 'OR') acc = acc || next
    else acc = acc && !next
  }
  return acc
}
