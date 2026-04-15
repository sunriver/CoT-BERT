import { createRouter, createWebHistory } from 'vue-router'
import RunList from '../views/RunList.vue'
import RunDetail from '../views/RunDetail.vue'
import CompareView from '../views/CompareView.vue'
import GoldCosinePointsView from '../views/GoldCosinePointsView.vue'

export default createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/runs' },
    { path: '/runs', name: 'runs', component: RunList },
    { path: '/runs/:id', name: 'run-detail', component: RunDetail, props: true },
    { path: '/compare', name: 'compare', component: CompareView },
    {
      path: '/gold-cosine-points',
      name: 'gold-cosine-points',
      component: GoldCosinePointsView,
    },
  ],
})
