<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import type { DataTableColumns } from 'naive-ui'
import {
  NButton,
  NCard,
  NDataTable,
  NImage,
  NSpace,
  NTabPane,
  NTabs,
  NTag,
  useMessage,
} from 'naive-ui'
import type { RunDetail as RunDetailT } from '../types'
import { artifactUrl, fetchRun } from '../api'

const route = useRoute()
const router = useRouter()
const message = useMessage()
const run = ref<RunDetailT | null>(null)
const loading = ref(true)

const id = computed(() => route.params.id as string)

const paramRows = computed(() => {
  if (!run.value) return []
  return Object.entries(run.value.train_params_flat).map(([key, value]) => ({
    key,
    value:
      typeof value === 'object' && value !== null
        ? JSON.stringify(value)
        : String(value),
  }))
})

const paramColumns: DataTableColumns<{ key: string; value: string }> = [
  { title: '参数', key: 'key', width: 320, resizable: true, ellipsis: { tooltip: true } },
  { title: '值', key: 'value', resizable: true, ellipsis: { tooltip: true } },
]

const summaryRows = computed(() => {
  if (!run.value?.summary_table?.tasks || !run.value.summary_table.scores)
    return []
  const tasks = run.value.summary_table.tasks
  const scores = run.value.summary_table.scores
  return tasks.map((t, i) => ({ task: t, score: scores[i] }))
})

const summaryColumns: DataTableColumns<{ task: string; score: number }> = [
  { title: '任务', key: 'task', resizable: true },
  {
    title: '分数 (×100)',
    key: 'score',
    resizable: true,
    render: (row) => row.score?.toFixed?.(2) ?? row.score,
  },
]

onMounted(async () => {
  loading.value = true
  try {
    run.value = await fetchRun(id.value)
  } catch (e) {
    message.error(String(e))
  } finally {
    loading.value = false
  }
})

const imageArtifacts = computed(() =>
  (run.value?.artifacts ?? []).filter((a) =>
    /\.(png|jpe?g|webp)$/i.test(a.name)
  )
)
const otherArtifacts = computed(() =>
  (run.value?.artifacts ?? []).filter(
    (a) => !/\.(png|jpe?g|webp)$/i.test(a.name)
  )
)
</script>

<template>
  <div class="view-scroll">
  <n-space vertical size="large">
    <n-space>
      <n-button @click="router.push({ name: 'runs' })">返回列表</n-button>
      <n-tag v-if="run">{{ run.run_id }}</n-tag>
    </n-space>
    <n-card v-if="loading" title="加载中…" />
    <n-card v-else-if="run" :title="`Run: ${run.run_id}`">
      <n-tabs type="line" animated>
        <n-tab-pane name="overview" tab="概览">
          <n-space vertical>
            <p>
              <strong>eval_run_tag:</strong> {{ run.eval_run_tag }}
              <span v-if="run.eval_run_tag_source"> ({{ run.eval_run_tag_source }})</span>
            </p>
            <p><strong>timestamp:</strong> {{ run.timestamp }}</p>
            <p v-if="run.git">
              <strong>git:</strong> {{ run.git.branch }} @ {{ run.git.commit_short }}
            </p>
            <p v-if="run.trainer_state_summary">
              <strong>trainer:</strong> step
              {{ run.trainer_state_summary.global_step }}, best_metric
              {{ run.trainer_state_summary.best_metric }}
            </p>
            <h4>STS 汇总表</h4>
            <n-data-table
              v-if="summaryRows.length"
              :columns="summaryColumns"
              :data="summaryRows"
              size="small"
            />
            <p v-else>无 summary_table</p>
            <h4>源文件</h4>
            <ul>
              <li v-for="(p, k) in run.source_files" :key="k">{{ k }}: {{ p }}</li>
            </ul>
          </n-space>
        </n-tab-pane>
        <n-tab-pane name="params" tab="训练超参">
          <n-data-table
            :columns="paramColumns"
            :data="paramRows"
            :max-height="480"
            virtual-scroll
            :row-key="(r) => r.key"
          />
        </n-tab-pane>
        <n-tab-pane name="extra" tab="金标 / Alignment">
          <n-space vertical>
            <h4>run_summary（节选 JSON）</h4>
            <pre class="json-block">{{
              JSON.stringify(run.run_summary, null, 2)?.slice(0, 12000)
            }}</pre>
            <h4>alignment_uniformity（节选 JSON）</h4>
            <pre class="json-block">{{
              JSON.stringify(run.alignment_uniformity, null, 2)?.slice(0, 12000)
            }}</pre>
          </n-space>
        </n-tab-pane>
        <n-tab-pane name="senteval" tab="SentEval 原始 results">
          <pre class="json-block">{{
            JSON.stringify(run.results_raw, null, 2)?.slice(0, 24000)
          }}</pre>
        </n-tab-pane>
        <n-tab-pane name="artifacts" tab="图表与文件">
          <n-space vertical>
            <div v-for="a in imageArtifacts" :key="a.name" class="fig">
              <p>{{ a.name }}</p>
              <n-image
                width="600"
                :src="artifactUrl(run.run_id, a.name)"
                :alt="a.name"
              />
            </div>
            <ul>
              <li v-for="a in otherArtifacts" :key="a.name">
                <a
                  :href="artifactUrl(run.run_id, a.name)"
                  target="_blank"
                  rel="noopener"
                >{{ a.name }}</a>
              </li>
            </ul>
          </n-space>
        </n-tab-pane>
      </n-tabs>
    </n-card>
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
.json-block {
  font-size: 11px;
  overflow: auto;
  max-height: 400px;
  background: var(--n-code-color);
  padding: 8px;
  border-radius: 4px;
}
.fig {
  margin-bottom: 16px;
}
</style>
