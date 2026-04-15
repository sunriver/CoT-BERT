export interface RunListItem {
  run_id: string
  eval_run_tag: string | null
  timestamp: string | null
  source_dir: string | null
  has_eval_test: boolean
  has_alignment: boolean
  has_run_summary: boolean
  stsb_test_spearman: number | null
  sick_test_spearman: number | null
  sts_avg_percent: number | null
  sts12_wmean_spearman: number | null
  sts13_wmean_spearman: number | null
  sts14_wmean_spearman: number | null
  sts15_wmean_spearman: number | null
  sts16_wmean_spearman: number | null
  best_metric: number | null
  git_commit_short: string | null
  alignment_cot_align_mean: number | null
  alignment_cot_unif_mean: number | null
  gold_pearson_mean_cot_mask: number | null
  gold_pearson_mean_cot_mask_mlp: number | null
  gold_pearson_mean_bert_base_cls: number | null
  gold_cosine_pngs: string[]
  gold_cosine_png_by_dataset: Record<string, string>
  train_params_flat: Record<string, unknown>
  /** 仅 eval_test.train_config_full 展开；旧版 API 可能缺省 */
  train_params_flat_eval_test?: Record<string, unknown>
}

export interface ArtifactRef {
  name: string
  path: string
}

export interface RunDetail {
  run_id: string
  eval_run_tag: string | null
  eval_run_tag_source: string | null
  timestamp: string | null
  source_files: Record<string, string>
  summary_metrics: Record<string, unknown>
  summary_table: { mode?: string; tasks?: string[]; scores?: number[] } | null
  train_params_flat: Record<string, unknown>
  trainer_state_summary: Record<string, unknown> | null
  git: Record<string, unknown> | null
  results_raw: Record<string, unknown> | null
  run_summary: Record<string, unknown> | null
  alignment_uniformity: Record<string, unknown> | null
  artifacts: ArtifactRef[]
}

export interface CompareResponse {
  run_ids: string[]
  metric_keys: string[]
  metric_matrix: Record<string, Record<string, number | null>>
  param_keys: string[]
  param_matrix: Record<string, Record<string, unknown>>
  param_diff_only: Record<string, Record<string, unknown>>
}

export interface PointsFile {
  name: string
  mtime: number | null
}

export interface PointsScatterPoint {
  gold: number
  cos: number
}

/** 与对比表列一致，便于每个 points CSV 子图旁展示（可由后端直接下发，或由前端根据 run 索引匹配填充） */
export interface PointsSeriesRunMeta {
  model_temp: string
  model_scd_temp: string
  model_enable: string
  model_hidden_dropout: string
  model_hidden_dropout_prob_for_last_column: string
  model_dropout_negative_prob: string
  model_dropout_different_prob: string
  sick_test_rho: string
  sts_avg_pct: string
  stsb_test_rho: string
}

export interface PointsSeries {
  file: string
  label: string
  data: PointsScatterPoint[]
  meta?: PointsSeriesRunMeta | null
}

export interface PointsScatterResponse {
  series: PointsSeries[]
  spearman?: Record<string, number> | null
}
