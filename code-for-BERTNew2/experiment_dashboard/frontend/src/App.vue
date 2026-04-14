<script setup lang="ts">
import {
  NConfigProvider,
  NLayout,
  NLayoutContent,
  NLayoutHeader,
  NMenu,
  NMessageProvider,
} from 'naive-ui'
import { computed, h, ref } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'

const route = useRoute()
const activeKey = computed(() => {
  if (route.path.startsWith('/compare')) return 'compare'
  if (route.path.startsWith('/runs')) return 'runs'
  return 'runs'
})

const menuOptions = ref([
  {
    label: () => h(RouterLink, { to: { name: 'runs' } }, { default: () => '运行列表' }),
    key: 'runs',
  },
])
</script>

<template>
  <n-config-provider :theme="null">
    <n-message-provider>
      <n-layout
        class="app-shell"
        style="height: 100%; min-height: 100%; display: flex; flex-direction: column"
      >
        <n-layout-header
          bordered
          style="
            flex-shrink: 0;
            padding: 0 clamp(12px, 2vw, 24px);
            height: 56px;
            display: flex;
            align-items: center;
          "
        >
          <strong style="margin-right: clamp(12px, 2vw, 24px); white-space: nowrap">CoT-BERT 实验看板</strong>
          <n-menu mode="horizontal" :value="activeKey" :options="menuOptions" style="min-width: 0; flex: 1" />
        </n-layout-header>
        <n-layout-content
          :native-scrollbar="false"
          class="app-main"
          content-style="
            padding: clamp(12px, 2vw, 20px);
            box-sizing: border-box;
            flex: 1;
            min-height: 0;
            width: 100%;
            max-width: 100%;
            overflow: hidden;
            display: flex;
            flex-direction: column;
          "
        >
          <router-view v-slot="{ Component }">
            <div class="router-view-root">
              <component :is="Component" />
            </div>
          </router-view>
        </n-layout-content>
      </n-layout>
    </n-message-provider>
  </n-config-provider>
</template>

<style scoped>
.app-shell {
  box-sizing: border-box;
}
.router-view-root {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  width: 100%;
  max-width: 100%;
}
</style>
