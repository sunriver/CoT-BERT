<script setup lang="ts">
import { computed, h, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import type { DataTableColumns } from 'naive-ui'
import {
  NButton,
  NCard,
  NDataTable,
  NImage,
  NInput,
  NSelect,
  NSpace,
  NTabPane,
  NTabs,
  NTag,
  useMessage,
} from 'naive-ui'
import type { RunListItem } from '../types'
import type { RunsTab } from '../api'
import { artifactUrl, fetchRuns, refreshIndex } from '../api'
import type { MatchMode, StructuredFilterRow, StructuredJoin } from '../utils/structuredFilter'
import {
  rowMatchesStructuredFilter,
  structuredFilterHasActiveRows,
} from '../utils/structuredFilter'
import { applyColumnOrder, getColumnStableKey } from '../utils/columnOrder'
import Sortable from 'sortablejs'

/** 与 eval_test.train_config_full 经后端 flatten 后的键一致（model_* / data_*） */
const DEFAULT_TRAIN_PARAM_COLUMN_KEYS: readonly string[] = [
  'model_temp',
  'model_scd_temp',
  'model_mask_embedding_sentence_template',
  'model_mask_embedding_sentence_different_template',
  'model_mask_embedding_sentence_negative_template',
  'model_dropout_different_prob',
  'model_dropout_negative_prob',
  'model_enable_custom_dropout_for_last_column',
  'model_hidden_dropout_prob_for_last_column',
  'data_train_file',
]

const TRAIN_PARAM_COLS_STORAGE = 'bert-exp-dash-train-param-cols-v2'
const TRAIN_PARAM_COLS_STORAGE_LEGACY = 'bert-exp-dash-train-param-cols-v1'
const COLUMN_ORDER_STORAGE = 'bert-exp-dash-column-order-v1'
const STRUCTURED_FILTER_STORAGE = 'bert-exp-dash-structured-filter-v1'

/** 运行列表 Tab 顺序（含后端 tab=all 汇总） */
const RUN_LIST_TABS: readonly RunsTab[] = ['all', 'senteval', 'alignment', 'gold']

function loadColumnOrder(): Record<RunsTab, string[]> {
  const empty = (): Record<RunsTab, string[]> => ({
    all: [],
    senteval: [],
    alignment: [],
    gold: [],
  })
  try {
    const raw = localStorage.getItem(COLUMN_ORDER_STORAGE)
    if (!raw) return empty()
    const o = JSON.parse(raw) as Record<string, unknown>
    const pick = (k: RunsTab): string[] =>
      Array.isArray(o[k]) ? (o[k] as string[]).filter((x) => typeof x === 'string') : []
    return {
      all: pick('all'),
      senteval: pick('senteval'),
      alignment: pick('alignment'),
      gold: pick('gold'),
    }
  } catch {
    return empty()
  }
}

function persistColumnOrder(v: Record<RunsTab, string[]>) {
  try {
    localStorage.setItem(COLUMN_ORDER_STORAGE, JSON.stringify(v))
  } catch {
    /* ignore */
  }
}

function defaultTrainParamCols(): Record<RunsTab, string[]> {
  const cols = [...DEFAULT_TRAIN_PARAM_COLUMN_KEYS]
  return { all: cols, senteval: cols, alignment: cols, gold: cols }
}

/** 默认键在前（保持上述顺序），再追加其余已选键，去重 */
function mergeTrainParamKeysWithDefaults(existing: string[]): string[] {
  const seen = new Set<string>()
  const out: string[] = []
  for (const k of DEFAULT_TRAIN_PARAM_COLUMN_KEYS) {
    if (!seen.has(k)) {
      seen.add(k)
      out.push(k)
    }
  }
  for (const k of existing) {
    if (typeof k !== 'string' || !k) continue
    if (!seen.has(k)) {
      seen.add(k)
      out.push(k)
    }
  }
  return out
}

function persistTrainParamCols(v: Record<RunsTab, string[]>) {
  try {
    localStorage.setItem(TRAIN_PARAM_COLS_STORAGE, JSON.stringify(v))
  } catch {
    /* ignore */
  }
}

function loadTrainParamCols(): Record<RunsTab, string[]> {
  const pickTab = (o: Record<string, unknown>, k: RunsTab): string[] | null => {
    if (!Object.prototype.hasOwnProperty.call(o, k)) return null
    const v = o[k]
    if (!Array.isArray(v)) return null
    return v.filter((x): x is string => typeof x === 'string')
  }
  try {
    const v2raw = localStorage.getItem(TRAIN_PARAM_COLS_STORAGE)
    if (v2raw) {
      const o = JSON.parse(v2raw) as Record<string, unknown>
      const base = defaultTrainParamCols()
      const out: Record<RunsTab, string[]> = { ...base }
      for (const t of RUN_LIST_TABS) {
        const p = pickTab(o, t)
        if (p) out[t] = p
      }
      return out
    }

    const v1raw = localStorage.getItem(TRAIN_PARAM_COLS_STORAGE_LEGACY)
    if (v1raw) {
      const o = JSON.parse(v1raw) as Record<string, unknown>
      const out: Record<RunsTab, string[]> = {
        all: mergeTrainParamKeysWithDefaults([]),
        senteval: [],
        alignment: [],
        gold: [],
      }
      for (const t of RUN_LIST_TABS) {
        if (t === 'all') continue
        const p = pickTab(o, t) ?? []
        out[t] = mergeTrainParamKeysWithDefaults(p)
      }
      persistTrainParamCols(out)
      return out
    }

    return defaultTrainParamCols()
  } catch {
    return defaultTrainParamCols()
  }
}

const router = useRouter()
const message = useMessage()
const rows = ref<RunListItem[]>([])
const loading = ref(false)
const tableWrapRef = ref<HTMLElement | null>(null)
const tableBodyMaxHeight = ref(420)
let tableResizeObserver: ResizeObserver | null = null

function fallbackTableBodyHeight(): number {
  return Math.max(280, Math.floor(window.innerHeight * 0.5))
}

function measureTableBodyMaxHeight() {
  const el = tableWrapRef.value
  if (!el) {
    tableBodyMaxHeight.value = fallbackTableBodyHeight()
    return
  }
  const h = Math.floor(el.getBoundingClientRect().height)
  tableBodyMaxHeight.value = h >= 120 ? h : fallbackTableBodyHeight()
}
const filter = ref('')
const checked = ref<string[]>([])

/** CSV 导出：顶层列优先顺序（与 RunListItem 一致；不含 train.*） */
const RUN_LIST_CSV_TOP_KEY_ORDER: readonly string[] = [
  'run_id',
  'eval_run_tag',
  'timestamp',
  'source_dir',
  'has_eval_test',
  'has_alignment',
  'has_run_summary',
  'stsb_test_spearman',
  'sick_test_spearman',
  'sts_avg_percent',
  'sts12_wmean_spearman',
  'sts13_wmean_spearman',
  'sts14_wmean_spearman',
  'sts15_wmean_spearman',
  'sts16_wmean_spearman',
  'best_metric',
  'git_commit_short',
  'alignment_cot_align_mean',
  'alignment_cot_unif_mean',
  'gold_pearson_mean_cot_mask',
  'gold_pearson_mean_cot_mask_mlp',
  'gold_pearson_mean_bert_base_cls',
  'gold_cosine_pngs',
  'gold_cosine_png_by_dataset',
]

function downloadTextFile(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function csvEscapeCell(val: string): string {
  if (/[",\n\r]/.test(val)) {
    return `"${val.replace(/"/g, '""')}"`
  }
  return val
}

function flattenRunForCsv(row: RunListItem): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [key, val] of Object.entries(row)) {
    if (key === 'train_params_flat' || key === 'train_params_flat_eval_test') continue
    if (key === 'gold_cosine_png_by_dataset') {
      out[key] = JSON.stringify(val ?? {})
      continue
    }
    if (key === 'gold_cosine_pngs') {
      out[key] = JSON.stringify(val ?? [])
      continue
    }
    if (val === null || val === undefined) {
      out[key] = ''
    } else if (typeof val === 'boolean' || typeof val === 'number') {
      out[key] = String(val)
    } else if (typeof val === 'string') {
      out[key] = val
    } else {
      out[key] = JSON.stringify(val)
    }
  }
  const flat = row.train_params_flat
  if (flat && typeof flat === 'object') {
    for (const [k, v] of Object.entries(flat)) {
      out[`train.${k}`] =
        v === null || v === undefined ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v)
    }
  }
  const fe = row.train_params_flat_eval_test
  if (fe && typeof fe === 'object') {
    for (const [k, v] of Object.entries(fe)) {
      out[`train_eval_test.${k}`] =
        v === null || v === undefined ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v)
    }
  }
  return out
}

function orderedCsvHeaders(keySet: Set<string>): string[] {
  const top = RUN_LIST_CSV_TOP_KEY_ORDER.filter((k) => keySet.has(k))
  const used = new Set(top)
  const train = [...keySet]
    .filter((k) => k.startsWith('train.') && !k.startsWith('train_eval_test.'))
    .sort()
  for (const k of train) used.add(k)
  const trainEval = [...keySet].filter((k) => k.startsWith('train_eval_test.')).sort()
  for (const k of trainEval) used.add(k)
  const rest = [...keySet].filter((k) => !used.has(k)).sort()
  return [...top, ...train, ...trainEval, ...rest]
}

function runsToCsv(selected: RunListItem[]): string {
  const flatRows = selected.map(flattenRunForCsv)
  const keySet = new Set<string>()
  for (const r of flatRows) {
    for (const k of Object.keys(r)) keySet.add(k)
  }
  const headers = orderedCsvHeaders(keySet)
  const headLine = headers.map((h) => csvEscapeCell(h)).join(',')
  const bodyLines = flatRows.map((row) =>
    headers.map((h) => csvEscapeCell(row[h] ?? '')).join(',')
  )
  return [headLine, ...bodyLines].join('\n')
}

function exportSelectedCsv() {
  if (checked.value.length < 1) {
    message.warning('请至少选择一个 run')
    return
  }
  const selected = checked.value
    .map((id) => rows.value.find((r) => r.run_id === id))
    .filter((r): r is RunListItem => !!r)
  if (selected.length === 0) {
    message.warning('未找到选中行的数据')
    return
  }
  const csv = runsToCsv(selected)
  const ts = new Date().toISOString().slice(0, 19).replace(/[-:]/g, '').replace('T', '-')
  const name = `runs_export_${activeTab.value}_${ts}.csv`
  downloadTextFile(name, `\uFEFF${csv}`, 'text/csv;charset=utf-8')
  message.success(`已导出 ${selected.length} 条为 CSV`)
}

const activeTab = ref<RunsTab>('senteval')
const columnOrderByTab = ref<Record<RunsTab, string[]>>(loadColumnOrder())
const trainParamColsByTab = ref<Record<RunsTab, string[]>>(loadTrainParamCols())

const activeTrainParamKeys = computed(() => trainParamColsByTab.value[activeTab.value])

/** 「全部」Tab 训练参数列仅展示 eval_test；其它 Tab 用多源合并后的 train_params_flat */
function trainParamsFlatForTableRow(row: RunListItem): Record<string, unknown> {
  if (activeTab.value === 'all') {
    const e = row.train_params_flat_eval_test
    return e && typeof e === 'object' ? e : {}
  }
  const m = row.train_params_flat
  return m && typeof m === 'object' ? m : {}
}

const STATIC_FILTER_FIELDS: { label: string; value: string }[] = [
  { label: 'Run ID', value: 'run_id' },
  { label: '时间', value: 'timestamp' },
  { label: '来源目录', value: 'source_dir' },
  { label: 'git 提交', value: 'git_commit_short' },
  { label: 'STSB test ρ', value: 'stsb_test_spearman' },
  { label: 'SICK test ρ', value: 'sick_test_spearman' },
  { label: 'STS Avg %', value: 'sts_avg_percent' },
  { label: 'STS12 ρ wmean', value: 'sts12_wmean_spearman' },
  { label: 'STS13 ρ wmean', value: 'sts13_wmean_spearman' },
  { label: 'STS14 ρ wmean', value: 'sts14_wmean_spearman' },
  { label: 'STS15 ρ wmean', value: 'sts15_wmean_spearman' },
  { label: 'STS16 ρ wmean', value: 'sts16_wmean_spearman' },
  { label: 'best_metric', value: 'best_metric' },
  { label: 'cot align（均值）', value: 'alignment_cot_align_mean' },
  { label: 'cot uniform（均值）', value: 'alignment_cot_unif_mean' },
  { label: 'ρ cot_mask', value: 'gold_pearson_mean_cot_mask' },
  { label: 'ρ cot_mask_mlp', value: 'gold_pearson_mean_cot_mask_mlp' },
  { label: 'ρ bert_base_cls', value: 'gold_pearson_mean_bert_base_cls' },
  { label: 'Gold SICK 散点图文件', value: 'gold_img_sick_test' },
  { label: 'Gold STS 散点图文件', value: 'gold_img_sts_test' },
  { label: 'has_eval_test', value: 'has_eval_test' },
  { label: 'has_alignment', value: 'has_alignment' },
  { label: 'has_run_summary', value: 'has_run_summary' },
]

const joinSelectOptions = [
  { label: 'AND', value: 'AND' as StructuredJoin },
  { label: 'OR', value: 'OR' as StructuredJoin },
  { label: 'NOT', value: 'NOT' as StructuredJoin },
]

const matchSelectOptions = [
  { label: '精确', value: 'exact' as MatchMode },
  { label: '包含', value: 'contains' as MatchMode },
  { label: '模糊', value: 'fuzzy' as MatchMode },
]

function newStructuredRow(isFirst: boolean): StructuredFilterRow {
  return {
    join: isFirst ? null : 'AND',
    field: null,
    value: '',
    match: 'exact',
  }
}

function cloneStructuredRows(rows: StructuredFilterRow[]): StructuredFilterRow[] {
  return rows.map((r) => ({
    join: r.join,
    field: r.field,
    value: r.value,
    match: r.match,
  }))
}

function normalizeStructuredRows(raw: unknown): StructuredFilterRow[] {
  const one = [newStructuredRow(true)]
  if (!Array.isArray(raw)) return one
  const out: StructuredFilterRow[] = []
  for (let i = 0; i < raw.length; i++) {
    const item = raw[i]
    if (!item || typeof item !== 'object') continue
    const o = item as Record<string, unknown>
    const join: StructuredJoin | null =
      i === 0 ? null : o.join === 'OR' || o.join === 'NOT' ? o.join : 'AND'
    const fieldRaw = o.field
    const field =
      fieldRaw === null || fieldRaw === undefined || fieldRaw === ''
        ? null
        : String(fieldRaw)
    const value = typeof o.value === 'string' ? o.value : String(o.value ?? '')
    let match: MatchMode = 'exact'
    if (o.match === 'contains' || o.match === 'fuzzy') match = o.match
    out.push({ join, field, value, match })
  }
  return out.length ? out : one
}

function defaultStructuredFilterByTab(): Record<RunsTab, StructuredFilterRow[]> {
  return {
    all: [newStructuredRow(true)],
    senteval: [newStructuredRow(true)],
    alignment: [newStructuredRow(true)],
    gold: [newStructuredRow(true)],
  }
}

function loadStructuredFilterByTab(): Record<RunsTab, StructuredFilterRow[]> {
  const base = defaultStructuredFilterByTab()
  try {
    const raw = localStorage.getItem(STRUCTURED_FILTER_STORAGE)
    if (!raw) return base
    const o = JSON.parse(raw) as unknown
    if (!o || typeof o !== 'object') return base
    const rec = o as Record<string, unknown>
    for (const t of RUN_LIST_TABS) {
      if (Object.prototype.hasOwnProperty.call(rec, t)) {
        base[t] = normalizeStructuredRows(rec[t])
      }
    }
    return base
  } catch {
    return base
  }
}

function persistStructuredFilter(data: Record<RunsTab, StructuredFilterRow[]>) {
  try {
    localStorage.setItem(STRUCTURED_FILTER_STORAGE, JSON.stringify(data))
  } catch {
    /* ignore */
  }
}

const structuredFilterByTab = ref<Record<RunsTab, StructuredFilterRow[]>>(
  loadStructuredFilterByTab()
)
const structuredRows = ref<StructuredFilterRow[]>(
  cloneStructuredRows(structuredFilterByTab.value[activeTab.value])
)

function addStructuredRow() {
  structuredRows.value = [...structuredRows.value, newStructuredRow(false)]
}

function removeStructuredRow(idx: number) {
  if (structuredRows.value.length <= 1) return
  const next = structuredRows.value.slice()
  next.splice(idx, 1)
  next[0].join = null
  structuredRows.value = next
}

/** 表格列 key → 高级检索「字段」下拉中的 value */
function tableColumnKeyToFilterField(colKey: string): string | null {
  if (colKey === '__n_selection__') return null
  if (colKey.startsWith('train_param_')) return colKey.slice('train_param_'.length)
  return colKey
}

function getCellRawValueForFilter(row: RunListItem, colKey: string): unknown {
  if (colKey === 'gold_img_sick_test') return row.gold_cosine_png_by_dataset?.sick_test
  if (colKey === 'gold_img_sts_test') return row.gold_cosine_png_by_dataset?.sts_test
  if (colKey.startsWith('train_param_')) {
    const k = colKey.slice('train_param_'.length)
    return trainParamsFlatForTableRow(row)[k]
  }
  if (colKey !== 'train_params_flat' && colKey in row) {
    return (row as unknown as Record<string, unknown>)[colKey]
  }
  return trainParamsFlatForTableRow(row)[colKey]
}

function formatCellToFilterValue(v: unknown): string {
  if (v === null || v === undefined) return ''
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number' && Number.isFinite(v)) {
    return Number.isInteger(v) ? String(v) : String(v)
  }
  if (typeof v === 'string') return v
  try {
    const s = JSON.stringify(v)
    return s.length > 800 ? `${s.slice(0, 797)}…` : s
  } catch {
    return String(v)
  }
}

function addStructuredConditionFromCell(filterField: string, valueStr: string) {
  const val = valueStr.trim()
  if (!filterField || !val) {
    message.info('该单元格无可用内容')
    return
  }
  const match: MatchMode =
    val.length > 64 || /[%_]/.test(val) ? 'contains' : 'exact'

  const cur = structuredRows.value
  const emptyFirst =
    cur.length === 1 && !cur[0].field && !cur[0].value.trim()
  if (emptyFirst) {
    cur[0].field = filterField
    cur[0].value = val
    cur[0].match = match
    structuredRows.value = [...cur]
    message.success('已填入高级检索（首行）')
    return
  }
  structuredRows.value = [
    ...cur,
    {
      join: 'AND',
      field: filterField,
      value: val,
      match,
    },
  ]
  message.success('已追加高级检索条件（AND）')
}

function onTableCellContextMenu(e: MouseEvent) {
  const td = (e.target as HTMLElement | null)?.closest?.('tbody td') as HTMLTableCellElement | null
  if (!td) return
  const wrap = tableWrapRef.value
  if (!wrap?.contains(td)) return
  e.preventDefault()

  const tr = td.closest('tr')
  const tbody = td.closest('tbody')
  if (!tr || !tbody) return

  const colIndex = td.cellIndex
  const cols = columns.value
  if (colIndex < 0 || colIndex >= cols.length) return

  const col = cols[colIndex] as DataTableColumns<RunListItem>[number]
  const colKey = getColumnStableKey(col)
  if (!colKey) return

  const filterField = tableColumnKeyToFilterField(colKey)
  if (!filterField) return

  const bodyRows = tbody.querySelectorAll('tr')
  let rowIdx = -1
  bodyRows.forEach((r, i) => {
    if (r === tr) rowIdx = i
  })
  if (rowIdx < 0 || rowIdx >= displayRows.value.length) return

  const row = displayRows.value[rowIdx]
  const raw = getCellRawValueForFilter(row, colKey)
  const str = formatCellToFilterValue(raw)
  if (!str) {
    message.info('该单元格为空，未添加条件')
    return
  }
  addStructuredConditionFromCell(filterField, str)
}

const displayRows = computed(() => {
  const data = rows.value
  if (!structuredFilterHasActiveRows(structuredRows.value)) return data
  const filterOpts =
    activeTab.value === 'all' ? { trainParamsSource: 'eval_test' as const } : undefined
  return data.filter((row) => rowMatchesStructuredFilter(structuredRows.value, row, filterOpts))
})

const trainParamKeyOptions = computed(() => {
  const s = new Set<string>(DEFAULT_TRAIN_PARAM_COLUMN_KEYS)
  for (const r of rows.value) {
    const p =
      activeTab.value === 'all' ? r.train_params_flat_eval_test : r.train_params_flat
    if (p && typeof p === 'object') {
      for (const k of Object.keys(p)) s.add(k)
    }
  }
  return Array.from(s)
    .sort()
    .map((k) => ({ label: k, value: k }))
})

const filterFieldOptions = computed(() => {
  const seen = new Set<string>()
  const out: { label: string; value: string }[] = []
  for (const o of STATIC_FILTER_FIELDS) {
    if (!seen.has(o.value)) {
      seen.add(o.value)
      out.push(o)
    }
  }
  for (const o of trainParamKeyOptions.value) {
    if (!seen.has(o.value)) {
      seen.add(o.value)
      out.push(o)
    }
  }
  return out
})

function onTrainParamKeysUpdate(v: string[]) {
  const t = activeTab.value
  trainParamColsByTab.value = { ...trainParamColsByTab.value, [t]: v }
  persistTrainParamCols(trainParamColsByTab.value)
}

function formatTrainParamCell(v: unknown): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number' && Number.isFinite(v)) {
    return Number.isInteger(v) ? String(v) : String(v)
  }
  if (typeof v === 'string') return v || '—'
  try {
    const s = JSON.stringify(v)
    return s.length > 120 ? `${s.slice(0, 117)}…` : s
  } catch {
    return String(v)
  }
}

function cmpTrainParam(a: unknown, b: unknown): number {
  if (a == null && b == null) return 0
  if (a == null || a === undefined) return 1
  if (b == null || b === undefined) return -1
  if (typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b)) {
    return a - b
  }
  return String(a).localeCompare(String(b), undefined, { numeric: true })
}

function customTrainParamColumns(keys: string[]): DataTableColumns<RunListItem> {
  return keys.map((key) => ({
    title: key,
    key: `train_param_${key}`,
    width: 156,
    resizable: true,
    ellipsis: { tooltip: true },
    sorter: (a, b) =>
      cmpTrainParam(trainParamsFlatForTableRow(a)[key], trainParamsFlatForTableRow(b)[key]),
    render: (row) => formatTrainParamCell(trainParamsFlatForTableRow(row)[key]),
  }))
}

function cmpStrNullLast(a: string | null, b: string | null): number {
  const as = a ?? ''
  const bs = b ?? ''
  if (!as && !bs) return 0
  if (!as) return 1
  if (!bs) return -1
  return as.localeCompare(bs)
}

function cmpNumNullLast(a: number | null, b: number | null): number {
  if (a == null && b == null) return 0
  if (a == null) return 1
  if (b == null) return -1
  return a - b
}

function selectionCol(): DataTableColumns<RunListItem>[0] {
  return { type: 'selection', multiple: true, resizable: true }
}

function runIdCol(): DataTableColumns<RunListItem>[0] {
  return {
    title: 'Run ID',
    key: 'run_id',
    resizable: true,
    ellipsis: { tooltip: true },
    sorter: (a, b) => a.run_id.localeCompare(b.run_id),
    render: (row) =>
      h(
        NButton,
        {
          text: true,
          type: 'primary',
          onClick: () => router.push({ name: 'run-detail', params: { id: row.run_id } }),
        },
        { default: () => row.run_id }
      ),
  }
}

function timeCol(): DataTableColumns<RunListItem>[0] {
  return {
    title: '时间',
    key: 'timestamp',
    width: 200,
    resizable: true,
    ellipsis: { tooltip: true },
    sorter: (a, b) => cmpStrNullLast(a.timestamp, b.timestamp),
  }
}

/** SentEval results.STS12–STS16 聚合 Spearman（wmean，缺则用 mean） */
function stsYearMetricColumns(): DataTableColumns<RunListItem> {
  const specs: { key: keyof RunListItem; title: string }[] = [
    { key: 'sts12_wmean_spearman', title: 'STS12 ρ wmean' },
    { key: 'sts13_wmean_spearman', title: 'STS13 ρ wmean' },
    { key: 'sts14_wmean_spearman', title: 'STS14 ρ wmean' },
    { key: 'sts15_wmean_spearman', title: 'STS15 ρ wmean' },
    { key: 'sts16_wmean_spearman', title: 'STS16 ρ wmean' },
  ]
  return specs.map(({ key, title }) => ({
    title,
    key: key as string,
    width: 118,
    resizable: true,
    sorter: (a: RunListItem, b: RunListItem) =>
      cmpNumNullLast(a[key] as number | null, b[key] as number | null),
    render: (row: RunListItem) => {
      const v = row[key] as number | null | undefined
      return v != null ? Number(v).toFixed(4) : '—'
    },
  }))
}

function gitCol(): DataTableColumns<RunListItem>[0] {
  return {
    title: 'git',
    key: 'git_commit_short',
    width: 90,
    resizable: true,
    sorter: (a, b) => cmpStrNullLast(a.git_commit_short, b.git_commit_short),
    render: (row) =>
      row.git_commit_short
        ? h(NTag, { size: 'small' }, { default: () => row.git_commit_short! })
        : '—',
  }
}

const columnsSenteval = computed<DataTableColumns<RunListItem>>(() => [
  selectionCol(),
  runIdCol(),
  timeCol(),
  ...customTrainParamColumns(activeTrainParamKeys.value),
  {
    title: 'STSB test ρ',
    key: 'stsb_test_spearman',
    width: 110,
    resizable: true,
    sorter: (a, b) => cmpNumNullLast(a.stsb_test_spearman, b.stsb_test_spearman),
    render: (row) =>
      row.stsb_test_spearman != null ? row.stsb_test_spearman.toFixed(4) : '—',
  },
  {
    title: 'SICK test ρ',
    key: 'sick_test_spearman',
    width: 110,
    resizable: true,
    sorter: (a, b) => cmpNumNullLast(a.sick_test_spearman, b.sick_test_spearman),
    render: (row) =>
      row.sick_test_spearman != null ? row.sick_test_spearman.toFixed(4) : '—',
  },
  {
    title: 'STS Avg %',
    key: 'sts_avg_percent',
    width: 100,
    resizable: true,
    sorter: (a, b) => cmpNumNullLast(a.sts_avg_percent, b.sts_avg_percent),
    render: (row) =>
      row.sts_avg_percent != null ? row.sts_avg_percent.toFixed(2) : '—',
  },
  ...stsYearMetricColumns(),
  {
    title: 'best_metric',
    key: 'best_metric',
    width: 110,
    resizable: true,
    sorter: (a, b) => cmpNumNullLast(a.best_metric, b.best_metric),
    render: (row) =>
      row.best_metric != null ? row.best_metric.toFixed(4) : '—',
  },
  gitCol(),
])

const columnsAlignment = computed<DataTableColumns<RunListItem>>(() => [
  selectionCol(),
  runIdCol(),
  timeCol(),
  ...customTrainParamColumns(activeTrainParamKeys.value),
  {
    title: 'cot align (均值)',
    key: 'alignment_cot_align_mean',
    width: 130,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(a.alignment_cot_align_mean, b.alignment_cot_align_mean),
    render: (row) =>
      row.alignment_cot_align_mean != null
        ? row.alignment_cot_align_mean.toFixed(4)
        : '—',
  },
  {
    title: 'cot uniform (均值)',
    key: 'alignment_cot_unif_mean',
    width: 140,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(a.alignment_cot_unif_mean, b.alignment_cot_unif_mean),
    render: (row) =>
      row.alignment_cot_unif_mean != null
        ? row.alignment_cot_unif_mean.toFixed(4)
        : '—',
  },
  {
    title: '来源目录',
    key: 'source_dir',
    resizable: true,
    ellipsis: { tooltip: true },
    sorter: (a, b) => cmpStrNullLast(a.source_dir, b.source_dir),
  },
  gitCol(),
])

const columnsGold = computed<DataTableColumns<RunListItem>>(() => [
  selectionCol(),
  runIdCol(),
  timeCol(),
  ...customTrainParamColumns(activeTrainParamKeys.value),
  {
    title: 'ρ cot_mask',
    key: 'gold_pearson_mean_cot_mask',
    width: 120,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(a.gold_pearson_mean_cot_mask, b.gold_pearson_mean_cot_mask),
    render: (row) =>
      row.gold_pearson_mean_cot_mask != null
        ? row.gold_pearson_mean_cot_mask.toFixed(4)
        : '—',
  },
  {
    title: 'ρ cot_mask_mlp',
    key: 'gold_pearson_mean_cot_mask_mlp',
    width: 130,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(
        a.gold_pearson_mean_cot_mask_mlp,
        b.gold_pearson_mean_cot_mask_mlp
      ),
    render: (row) =>
      row.gold_pearson_mean_cot_mask_mlp != null
        ? row.gold_pearson_mean_cot_mask_mlp.toFixed(4)
        : '—',
  },
  {
    title: 'ρ bert_base_cls',
    key: 'gold_pearson_mean_bert_base_cls',
    width: 130,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(
        a.gold_pearson_mean_bert_base_cls,
        b.gold_pearson_mean_bert_base_cls
      ),
    render: (row) =>
      row.gold_pearson_mean_bert_base_cls != null
        ? row.gold_pearson_mean_bert_base_cls.toFixed(4)
        : '—',
  },
  {
    title: 'SICK scatter',
    key: 'gold_img_sick_test',
    width: 236,
    resizable: true,
    render: (row) => {
      const fn = row.gold_cosine_png_by_dataset?.sick_test
      if (!fn) return '—'
      return h(NImage, {
        width: 220,
        height: 160,
        src: artifactUrl(row.run_id, fn),
        alt: fn,
        objectFit: 'contain',
        style: { display: 'block' },
      })
    },
  },
  {
    title: 'STS scatter',
    key: 'gold_img_sts_test',
    width: 236,
    resizable: true,
    render: (row) => {
      const fn = row.gold_cosine_png_by_dataset?.sts_test
      if (!fn) return '—'
      return h(NImage, {
        width: 220,
        height: 160,
        src: artifactUrl(row.run_id, fn),
        alt: fn,
        objectFit: 'contain',
        style: { display: 'block' },
      })
    },
  },
  {
    title: '来源目录',
    key: 'source_dir',
    resizable: true,
    ellipsis: { tooltip: true },
    sorter: (a, b) => cmpStrNullLast(a.source_dir, b.source_dir),
  },
  gitCol(),
])

/** 汇总：每行一个 run_id，合并 SentEval + Alignment + Gold 列（单一 source_dir / git） */
const columnsUnified = computed<DataTableColumns<RunListItem>>(() => [
  selectionCol(),
  runIdCol(),
  timeCol(),
  ...customTrainParamColumns(activeTrainParamKeys.value),
  {
    title: 'STSB test ρ',
    key: 'stsb_test_spearman',
    width: 110,
    resizable: true,
    sorter: (a, b) => cmpNumNullLast(a.stsb_test_spearman, b.stsb_test_spearman),
    render: (row) =>
      row.stsb_test_spearman != null ? row.stsb_test_spearman.toFixed(4) : '—',
  },
  {
    title: 'SICK test ρ',
    key: 'sick_test_spearman',
    width: 110,
    resizable: true,
    sorter: (a, b) => cmpNumNullLast(a.sick_test_spearman, b.sick_test_spearman),
    render: (row) =>
      row.sick_test_spearman != null ? row.sick_test_spearman.toFixed(4) : '—',
  },
  {
    title: 'STS Avg %',
    key: 'sts_avg_percent',
    width: 100,
    resizable: true,
    sorter: (a, b) => cmpNumNullLast(a.sts_avg_percent, b.sts_avg_percent),
    render: (row) =>
      row.sts_avg_percent != null ? row.sts_avg_percent.toFixed(2) : '—',
  },
  ...stsYearMetricColumns(),
  {
    title: 'best_metric',
    key: 'best_metric',
    width: 110,
    resizable: true,
    sorter: (a, b) => cmpNumNullLast(a.best_metric, b.best_metric),
    render: (row) =>
      row.best_metric != null ? row.best_metric.toFixed(4) : '—',
  },
  {
    title: 'cot align (均值)',
    key: 'alignment_cot_align_mean',
    width: 130,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(a.alignment_cot_align_mean, b.alignment_cot_align_mean),
    render: (row) =>
      row.alignment_cot_align_mean != null
        ? row.alignment_cot_align_mean.toFixed(4)
        : '—',
  },
  {
    title: 'cot uniform (均值)',
    key: 'alignment_cot_unif_mean',
    width: 140,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(a.alignment_cot_unif_mean, b.alignment_cot_unif_mean),
    render: (row) =>
      row.alignment_cot_unif_mean != null
        ? row.alignment_cot_unif_mean.toFixed(4)
        : '—',
  },
  {
    title: 'ρ cot_mask',
    key: 'gold_pearson_mean_cot_mask',
    width: 120,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(a.gold_pearson_mean_cot_mask, b.gold_pearson_mean_cot_mask),
    render: (row) =>
      row.gold_pearson_mean_cot_mask != null
        ? row.gold_pearson_mean_cot_mask.toFixed(4)
        : '—',
  },
  {
    title: 'ρ cot_mask_mlp',
    key: 'gold_pearson_mean_cot_mask_mlp',
    width: 130,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(
        a.gold_pearson_mean_cot_mask_mlp,
        b.gold_pearson_mean_cot_mask_mlp
      ),
    render: (row) =>
      row.gold_pearson_mean_cot_mask_mlp != null
        ? row.gold_pearson_mean_cot_mask_mlp.toFixed(4)
        : '—',
  },
  {
    title: 'ρ bert_base_cls',
    key: 'gold_pearson_mean_bert_base_cls',
    width: 130,
    resizable: true,
    sorter: (a, b) =>
      cmpNumNullLast(
        a.gold_pearson_mean_bert_base_cls,
        b.gold_pearson_mean_bert_base_cls
      ),
    render: (row) =>
      row.gold_pearson_mean_bert_base_cls != null
        ? row.gold_pearson_mean_bert_base_cls.toFixed(4)
        : '—',
  },
  {
    title: 'SICK scatter',
    key: 'gold_img_sick_test',
    width: 236,
    resizable: true,
    render: (row) => {
      const fn = row.gold_cosine_png_by_dataset?.sick_test
      if (!fn) return '—'
      return h(NImage, {
        width: 220,
        height: 160,
        src: artifactUrl(row.run_id, fn),
        alt: fn,
        objectFit: 'contain',
        style: { display: 'block' },
      })
    },
  },
  {
    title: 'STS scatter',
    key: 'gold_img_sts_test',
    width: 236,
    resizable: true,
    render: (row) => {
      const fn = row.gold_cosine_png_by_dataset?.sts_test
      if (!fn) return '—'
      return h(NImage, {
        width: 220,
        height: 160,
        src: artifactUrl(row.run_id, fn),
        alt: fn,
        objectFit: 'contain',
        style: { display: 'block' },
      })
    },
  },
  {
    title: '来源目录',
    key: 'source_dir',
    resizable: true,
    ellipsis: { tooltip: true },
    sorter: (a, b) => cmpStrNullLast(a.source_dir, b.source_dir),
  },
  gitCol(),
])

const columnsBase = computed<DataTableColumns<RunListItem>>(() => {
  if (activeTab.value === 'all') return columnsUnified.value
  if (activeTab.value === 'alignment') return columnsAlignment.value
  if (activeTab.value === 'gold') return columnsGold.value
  return columnsSenteval.value
})

const columns = computed(() => {
  const base = columnsBase.value
  const saved = columnOrderByTab.value[activeTab.value]
  return applyColumnOrder(base, saved?.length ? saved : null)
})

let sortableInst: Sortable | null = null

function destroyColumnSortable() {
  sortableInst?.destroy()
  sortableInst = null
}

function reorderColumnOrder(oldIndex: number, newIndex: number) {
  const tab = activeTab.value
  const base = columnsBase.value
  const defaultKeys = base.map(getColumnStableKey).filter(Boolean) as string[]
  if (!defaultKeys.length) return
  let order =
    columnOrderByTab.value[tab] && columnOrderByTab.value[tab]!.length > 0
      ? [...columnOrderByTab.value[tab]!]
      : [...defaultKeys]
  const setEq =
    order.length === defaultKeys.length && defaultKeys.every((k) => order.includes(k))
  if (!setEq) order = [...defaultKeys]
  if (oldIndex < 0 || newIndex < 0 || oldIndex >= order.length || newIndex >= order.length) {
    return
  }
  const [moved] = order.splice(oldIndex, 1)
  order.splice(newIndex, 0, moved)
  columnOrderByTab.value = { ...columnOrderByTab.value, [tab]: order }
  persistColumnOrder(columnOrderByTab.value)
}

function setupColumnSortable() {
  destroyColumnSortable()
  const wrap = tableWrapRef.value
  if (!wrap) return
  const row = wrap.querySelector('thead tr')
  if (!row) return
  sortableInst = Sortable.create(row as HTMLElement, {
    animation: 180,
    draggable: 'th',
    onEnd: (evt) => {
      const oi = evt.oldIndex
      const ni = evt.newIndex
      if (oi === undefined || ni === undefined || oi === ni) return
      reorderColumnOrder(oi, ni)
    },
  })
}

const columnStructureSignature = computed(() =>
  columnsBase.value.map(getColumnStableKey).join('\u0001')
)

const scrollX = computed(() => {
  const extra = activeTrainParamKeys.value.length * 160
  if (activeTab.value === 'all') return 3780 + extra
  if (activeTab.value === 'alignment') return 1280 + extra
  if (activeTab.value === 'gold') return 2100 + extra
  return 1680 + extra
})

const tabHint = computed(() => {
  const adv =
    '「高级检索」可多行组合：自第二行起可选 AND / OR / NOT（NOT 表示且不含该行）；匹配方式：精确 / 包含 / 模糊（模糊可在关键词中使用 % 与 _ 通配）。检索条件与表头列顺序均按 Tab 分别保存在本机浏览器，下次打开自动恢复。在表格数据区右键单元格，可将该列与单元格内容填入检索条件。'
  if (activeTab.value === 'all') {
    return `全部 run：每行唯一 run_id，合并三类评估结果列；「自定义表头」中的训练参数仅来自 eval_test_*.json 的 train_config_full（无 SentEval 文件时参数列为「—」）。数据来自 /api/runs?tab=all。${adv}`
  }
  if (activeTab.value === 'alignment') {
    return `run_alignment_uniformity_benchmark.py → alignment_uniformity_benchmark*.json（表中为 cot_bert_local 在多数据集上的 alignment / uniformity 均值）。${adv}`
  }
  if (activeTab.value === 'gold') {
    return `gold_cosine_scatter.py：Pearson 均值 +每行内嵌 SICK / STS 散点图（按数据集分列）。${adv}`
  }
  return `cot_bert_evaluation.py → eval_test_*.json；可用「自定义表头」从 train_config_full 中选字段。${adv}`
})

async function load() {
  loading.value = true
  try {
    rows.value = await fetchRuns(filter.value.trim() || undefined, activeTab.value)
  } catch (e) {
    message.error(String(e))
  } finally {
    loading.value = false
  }
}

async function onRefreshIndex() {
  try {
    await refreshIndex()
    message.success('索引已刷新')
    await load()
  } catch (e) {
    message.error(String(e))
  }
}

function goCompare() {
  if (checked.value.length < 1) {
    message.warning('请至少选择一个 run')
    return
  }
  router.push({
    name: 'compare',
    query: { run_ids: checked.value.join(',') },
  })
}

let filterDebounce = 0
watch(filter, () => {
  window.clearTimeout(filterDebounce)
  filterDebounce = window.setTimeout(load, 320)
})
watch(activeTab, (newTab, oldTab) => {
  if (oldTab !== undefined) {
    structuredFilterByTab.value[oldTab] = cloneStructuredRows(structuredRows.value)
    persistStructuredFilter(structuredFilterByTab.value)
  }
  structuredRows.value = cloneStructuredRows(structuredFilterByTab.value[newTab])
  checked.value = []
  load()
})

watch(
  structuredRows,
  () => {
    structuredFilterByTab.value[activeTab.value] = cloneStructuredRows(structuredRows.value)
    persistStructuredFilter(structuredFilterByTab.value)
  },
  { deep: true }
)

watch([rows, loading, activeTab, columnStructureSignature], () => {
  nextTick(() => {
    measureTableBodyMaxHeight()
    setupColumnSortable()
  })
})

onMounted(() => {
  tableResizeObserver = new ResizeObserver(() => measureTableBodyMaxHeight())
  nextTick(() => {
    if (tableWrapRef.value) tableResizeObserver?.observe(tableWrapRef.value)
    measureTableBodyMaxHeight()
    setupColumnSortable()
  })
  window.addEventListener('resize', measureTableBodyMaxHeight)
  load()
})
onUnmounted(() => {
  structuredFilterByTab.value[activeTab.value] = cloneStructuredRows(structuredRows.value)
  persistStructuredFilter(structuredFilterByTab.value)
  window.clearTimeout(filterDebounce)
  window.removeEventListener('resize', measureTableBodyMaxHeight)
  destroyColumnSortable()
  tableResizeObserver?.disconnect()
  tableResizeObserver = null
})
</script>

<template>
  <div class="run-list-page">
    <n-card class="run-list-card" title="实验运行列表" :bordered="true">
      <div class="run-list-body">
        <n-space vertical :size="10" style="width: 100%">
          <n-space align="center" style="flex-wrap: wrap">
            <n-input
              v-model:value="filter"
              placeholder="按 run_id / tag 过滤（服务端）"
              clearable
              style="width: 280px; max-width: 100%"
            />
            <n-button :loading="loading" @click="load">刷新列表</n-button>
            <n-button secondary @click="onRefreshIndex">重新扫描目录</n-button>
            <n-button type="primary" :disabled="checked.length < 1" @click="goCompare">
              对比选中 ({{ checked.length }})
            </n-button>
            <n-button secondary :disabled="checked.length < 1" @click="exportSelectedCsv">
              导出 CSV ({{ checked.length }})
            </n-button>
            <span v-if="rows.length" class="filter-row-count">
              列筛选：{{ displayRows.length }} / {{ rows.length }}
            </span>
          </n-space>
          <div class="advanced-filter">
            <div class="advanced-filter-head">
              <span class="advanced-filter-title">高级检索</span>
              <span class="advanced-filter-sub">客户端列筛选 · 多行条件</span>
            </div>
            <div
              v-for="(r, idx) in structuredRows"
              :key="idx"
              class="advanced-filter-row"
            >
              <div class="adv-cell adv-join">
                <n-select
                  v-if="idx > 0"
                  :value="r.join ?? 'AND'"
                  :options="joinSelectOptions"
                  :consistent-menu-width="false"
                  @update:value="(v) => (r.join = v as StructuredJoin)"
                />
              </div>
              <div class="adv-cell adv-field">
                <n-select
                  v-model:value="r.field"
                  filterable
                  clearable
                  placeholder="字段"
                  :options="filterFieldOptions"
                  :consistent-menu-width="false"
                />
              </div>
              <div class="adv-cell adv-value">
                <n-input v-model:value="r.value" clearable placeholder="关键词" />
              </div>
              <div class="adv-cell adv-match">
                <n-select
                  v-model:value="r.match"
                  :options="matchSelectOptions"
                  :consistent-menu-width="false"
                />
              </div>
              <div class="adv-cell adv-actions">
                <n-button
                  quaternary
                  circle
                  size="small"
                  :disabled="structuredRows.length <= 1"
                  title="删除本行"
                  @click="removeStructuredRow(idx)"
                >
                  −
                </n-button>
                <n-button
                  v-if="idx === structuredRows.length - 1"
                  quaternary
                  circle
                  size="small"
                  title="添加一行"
                  @click="addStructuredRow"
                >
                  +
                </n-button>
              </div>
            </div>
          </div>
        </n-space>
        <n-tabs v-model:value="activeTab" type="line" animated class="run-list-tabs">
          <n-tab-pane name="all" tab="全部 (run_id)" />
          <n-tab-pane name="senteval" tab="SentEval" />
          <n-tab-pane name="alignment" tab="Alignment / Uniformity" />
          <n-tab-pane name="gold" tab="Gold–Cosine" />
        </n-tabs>
        <div class="column-settings">
          <span class="column-settings-label">自定义表头</span>
          <n-select
            :value="activeTrainParamKeys"
            multiple
            filterable
            clearable
            :options="trainParamKeyOptions"
            :loading="loading"
            placeholder="选择 train_config_full 中的字段，追加为表格列（可多选、可搜索）"
            :consistent-menu-width="false"
            max-tag-count="responsive"
            class="column-settings-select"
            @update:value="onTrainParamKeysUpdate"
          />
          <span v-if="!loading && trainParamKeyOptions.length === 0" class="column-settings-empty">
            <template v-if="activeTab === 'all'">
              当前列表中无 eval_test 或均无 train_config_full，本 Tab 训练参数列无可用键
            </template>
            <template v-else>当前列表无 eval_test 或缺少 train_config_full，无可用参数键</template>
          </span>
        </div>
        <p class="tab-hint">{{ tabHint }}</p>
        <div
          ref="tableWrapRef"
          class="run-list-table-wrap"
          @contextmenu="onTableCellContextMenu"
        >
          <n-data-table
            v-model:checked-row-keys="checked"
            :max-height="tableBodyMaxHeight"
            :columns="columns"
            :data="displayRows"
            :loading="loading"
            :row-key="(r: RunListItem) => r.run_id"
            :scroll-x="scrollX"
            :min-row-height="activeTab === 'gold' || activeTab === 'all' ? 180 : undefined"
            striped
            class="run-list-table"
          />
        </div>
        <p v-if="!loading && rows.length === 0" class="run-list-empty">
          当前 Tab 下没有可用的 run（请确认已启动后端 <code>http://127.0.0.1:8000</code>、Vite 代理
          <code>/api</code> 正常，并点击「重新扫描目录」）
        </p>
        <p
          v-else-if="
            !loading &&
            rows.length > 0 &&
            displayRows.length === 0 &&
            structuredFilterHasActiveRows(structuredRows)
          "
          class="run-list-empty"
        >
          没有符合「高级检索」条件的运行，请修改条件或清空关键词。
        </p>
      </div>
    </n-card>
  </div>
</template>

<style scoped>
.run-list-page {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
}
.run-list-card {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.run-list-card :deep(.n-card-header) {
  flex-shrink: 0;
}
.run-list-card :deep(.n-card__content) {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
}
.run-list-body {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow: hidden;
}
.run-list-tabs {
  flex-shrink: 0;
}
.run-list-tabs :deep(.n-tabs-nav) {
  flex-wrap: wrap;
}
.run-list-table-wrap {
  flex: 1;
  min-height: 0;
  overflow: hidden;
  width: 100%;
  max-width: 100%;
}
.run-list-table-wrap :deep(thead th) {
  cursor: grab;
}
.run-list-table-wrap :deep(thead th.sortable-chosen) {
  cursor: grabbing;
}
.run-list-empty {
  margin: 0;
  font-size: 13px;
  color: var(--n-text-color-3);
  flex-shrink: 0;
}
.run-list-empty code {
  font-size: 12px;
}
.tab-hint {
  margin: 0;
  font-size: 13px;
  color: var(--n-text-color-3);
  flex-shrink: 0;
}
.tab-hint code {
  font-size: 12px;
}
.column-settings {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px 12px;
  padding: 10px 12px;
  margin-bottom: 4px;
  border-radius: var(--n-border-radius);
  border: 1px solid var(--n-border-color);
  background: var(--n-action-color);
}
.column-settings-label {
  flex: 0 0 auto;
  font-size: 14px;
  font-weight: 600;
  color: var(--n-text-color);
}
.column-settings-select {
  flex: 1 1 280px;
  min-width: min(100%, 320px);
  max-width: 100%;
}
.column-settings-empty {
  flex: 1 1 100%;
  font-size: 12px;
  color: var(--n-text-color-3);
}
.filter-row-count {
  font-size: 12px;
  color: var(--n-text-color-3);
  white-space: nowrap;
}
.advanced-filter {
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  border: 1px solid var(--n-border-color);
  border-radius: var(--n-border-radius);
  padding: 12px 14px;
  background: var(--n-color);
}
.advanced-filter-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 10px;
}
.advanced-filter-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--n-text-color);
}
.advanced-filter-sub {
  font-size: 12px;
  color: var(--n-text-color-3);
}
.advanced-filter-row {
  display: grid;
  grid-template-columns: minmax(72px, 92px) minmax(140px, 1fr) minmax(160px, 2fr) minmax(88px, 100px) auto;
  gap: 8px 10px;
  align-items: center;
  margin-bottom: 8px;
}
.advanced-filter-row:last-child {
  margin-bottom: 0;
}
.adv-cell :deep(.n-base-selection),
.adv-cell :deep(.n-input) {
  width: 100%;
}
.adv-actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
  min-width: 72px;
}
@media (max-width: 720px) {
  .advanced-filter-row {
    grid-template-columns: 1fr;
 }
  .adv-actions {
    justify-content: flex-start;
  }
}
</style>
