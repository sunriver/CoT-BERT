<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { DataTableColumns } from 'naive-ui'
import {
  NButton,
  NCard,
  NDataTable,
  NSpace,
  useMessage,
} from 'naive-ui'
import * as echarts from 'echarts'
import type { CompareResponse } from '../types'
import { fetchCompare } from '../api'

const route = useRoute()
const router = useRouter()
const message = useMessage()
const data = ref<CompareResponse | null>(null)
const loading = ref(false)
const chartRef = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

const runIds = computed(() => {
  const raw = route.query.run_ids
  if (!raw || typeof raw !== 'string') return []
  return raw.split(',').map((s) => s.trim()).filter(Boolean)
})

const metricTableColumns = computed<DataTableColumns<Record<string, unknown>>>(() => {
  const d = data.value
  if (!d) return []
  const cols: DataTableColumns<Record<string, unknown>> = [
    { title: '指标', key: 'metric', fixed: 'left', width: 160, resizable: true },
  ]
  for (const rid of d.run_ids) {
    cols.push({
      title: rid.length > 24 ? rid.slice(0, 22) + '…' : rid,
      key: rid,
      resizable: true,
      render: (row) => {
        const v = row[rid] as number | null | undefined
        if (v == null) return '—'
        return typeof v === 'number' ? v.toFixed(4) : String(v)
      },
    })
  }
  return cols
})

const metricTableRows = computed(() => {
  const d = data.value
  if (!d) return []
  return d.metric_keys.map((mk) => {
    const row: Record<string, unknown> = { metric: mk }
    for (const rid of d.run_ids) {
      row[rid] = d.metric_matrix[rid]?.[mk] ?? null
    }
    return row
  })
})

const diffRows = computed(() => {
  const d = data.value
  if (!d) return []
  return Object.entries(d.param_diff_only).map(([k, vals]) => ({
    param: k,
    ...vals,
  }))
})

const diffColumns = computed<DataTableColumns<Record<string, unknown>>>(() => {
  const d = data.value
  if (!d) return [{ title: '参数', key: 'param' }]
  const cols: DataTableColumns<Record<string, unknown>> = [
    {
      title: '参数（仅差异）',
      key: 'param',
      width: 280,
      resizable: true,
      ellipsis: { tooltip: true },
    },
  ]
  for (const rid of d.run_ids) {
    cols.push({
      title: rid.length > 20 ? rid.slice(0, 18) + '…' : rid,
      key: rid,
      resizable: true,
      ellipsis: { tooltip: true },
      render: (row) => {
        const v = row[rid]
        if (v === undefined) return '—'
        if (typeof v === 'object') return JSON.stringify(v)
        return String(v)
      },
    })
  }
  return cols
})

function renderChart() {
  const d = data.value
  const el = chartRef.value
  if (!d || !el) return
  if (!chart) chart = echarts.init(el)
  const metrics = d.metric_keys
  const series = d.run_ids.map((rid) => ({
    name: rid,
    type: 'bar',
    data: metrics.map((m) => d.metric_matrix[rid]?.[m] ??0),
  }))
  chart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { type: 'scroll', bottom: 0 },
    grid: { left: 60, right: 24, bottom: 80, top: 24 },
    xAxis: { type: 'category', data: metrics, axisLabel: { rotate: 30 } },
    yAxis: { type: 'value', scale: true },
    series,
  })
}

async function loadCompare() {
  if (runIds.value.length < 1) {
    data.value = null
    message.warning('未指定 run_ids')
    return
  }
  loading.value = true
  try {
    data.value = await fetchCompare(runIds.value)
    setTimeout(renderChart, 0)
  } catch (e) {
    message.error(String(e))
    data.value = null
  } finally {
    loading.value = false
  }
}

onMounted(loadCompare)
watch(() => route.query.run_ids, loadCompare)

watch(
  () => data.value,
  () => {
    setTimeout(renderChart, 0)
  }
)
</script>

<template>
  <div class="view-scroll">
  <n-space vertical size="large">
    <n-space>
      <n-button @click="router.push({ name: 'runs' })">返回列表</n-button>
    </n-space>
    <n-card v-if="loading" title="加载对比…" />
    <template v-else-if="data">
      <n-card title="指标柱状对比">
        <div ref="chartRef" style="width: 100%; height: 380px" />
      </n-card>
      <n-card title="指标表">
        <n-data-table
          :columns="metricTableColumns"
          :data="metricTableRows"
          :scroll-x="200 + data.run_ids.length * 140"
          size="small"
        />
      </n-card>
      <n-card title="超参差异（仅显示不同键）">
        <n-data-table
          :columns="diffColumns"
          :data="diffRows"
          :scroll-x="400 + data.run_ids.length * 200"
          :max-height="480"
          virtual-scroll
          :row-key="(r) => String(r.param)"
        />
      </n-card>
    </template>
  </n-space>
  </div>
</template>

<style scoped>
.view-scroll {
  flex: 1;
  min-height: 0;
  overflow: auto;
  width: 100%;
  max-width: 100%;
  box-sizing: border-box;
  -webkit-overflow-scrolling: touch;
}
</style>
