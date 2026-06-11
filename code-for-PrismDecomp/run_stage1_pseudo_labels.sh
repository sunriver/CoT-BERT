#!/usr/bin/env bash
# Stage1: 从 wiki1m 生成 7 维模板伪标签
set -e
cd "$(dirname "$0")"
python template_pseudo_label.py \
  --input ../data/wiki1m_for_simcse.txt \
  --output preprocessed/wiki_pseudo_labels.jsonl \
  --stats_output preprocessed/pseudo_label_stats.json \
  --model_path "${MODEL_PATH:-/Users/lmf/Documents/local/code/pretrain_models/bert-base-uncased}" \
  --templates configs/aspect_templates.yaml \
  --batch_size 32 \
  "$@"
