from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RunListItem(BaseModel):
    run_id: str
    eval_run_tag: Optional[str] = None
    timestamp: Optional[str] = None
    source_dir: Optional[str] = None
    has_eval_test: bool = False
    has_alignment: bool = False
    has_run_summary: bool = False
    stsb_test_spearman: Optional[float] = None
    sick_test_spearman: Optional[float] = None
    sts_avg_percent: Optional[float] = None
    sts12_wmean_spearman: Optional[float] = None
    sts13_wmean_spearman: Optional[float] = None
    sts14_wmean_spearman: Optional[float] = None
    sts15_wmean_spearman: Optional[float] = None
    sts16_wmean_spearman: Optional[float] = None
    best_metric: Optional[float] = None
    git_commit_short: Optional[str] = None
    alignment_cot_align_mean: Optional[float] = None
    alignment_cot_unif_mean: Optional[float] = None
    gold_pearson_mean_cot_mask: Optional[float] = None
    gold_pearson_mean_cot_mask_mlp: Optional[float] = None
    gold_pearson_mean_bert_base_cls: Optional[float] = None
    gold_cosine_pngs: List[str] = Field(default_factory=list)
    gold_cosine_png_by_dataset: Dict[str, str] = Field(default_factory=dict)
    train_params_flat: Dict[str, Any] = Field(default_factory=dict)
    train_params_flat_eval_test: Dict[str, Any] = Field(
        default_factory=dict,
        description="仅来自 eval_test.train_config_full；无 eval 时为空",
    )


class ArtifactRef(BaseModel):
    name: str
    path: str


class RunDetail(BaseModel):
    run_id: str
    eval_run_tag: Optional[str] = None
    eval_run_tag_source: Optional[str] = None
    timestamp: Optional[str] = None
    source_files: Dict[str, str] = Field(default_factory=dict)
    summary_metrics: Dict[str, Any] = Field(default_factory=dict)
    summary_table: Optional[Dict[str, Any]] = None
    train_params_flat: Dict[str, Any] = Field(default_factory=dict)
    trainer_state_summary: Optional[Dict[str, Any]] = None
    git: Optional[Dict[str, Any]] = None
    results_raw: Optional[Dict[str, Any]] = None
    run_summary: Optional[Dict[str, Any]] = None
    alignment_uniformity: Optional[Dict[str, Any]] = None
    artifacts: List[ArtifactRef] = Field(default_factory=list)


class CompareResponse(BaseModel):
    run_ids: List[str]
    metric_keys: List[str]
    metric_matrix: Dict[str, Dict[str, Optional[float]]]
    param_keys: List[str]
    param_matrix: Dict[str, Dict[str, Any]]
    param_diff_only: Dict[str, Dict[str, Any]]


class PointsFile(BaseModel):
    name: str
    mtime: Optional[float] = None


class PointsSeries(BaseModel):
    file: str
    label: str
    data: List[Dict[str, float]]  # [{"gold": float, "cos": float}, ...]


class PointsScatterResponse(BaseModel):
    series: List[PointsSeries]
    spearman: Optional[Dict[str, float]] = None
