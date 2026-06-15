#!/usr/bin/env bash
# Stage1: 训练共享压缩器 + 全库提取 7×d 主题缓存
set -e
cd "$(dirname "$0")"
MODEL_PATH="${MODEL_PATH:-/Users/lmf/Documents/local/code/pretrain_models/bert-base-uncased}"
INPUT="${INPUT:-../data/wiki1m_for_simcse.txt}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

EXTRA=()
if [[ -n "$MAX_SAMPLES" ]]; then
  EXTRA+=(--max_samples "$MAX_SAMPLES")
fi

python3 train_theme_compressor.py \
  --input "$INPUT" \
  --compressor_out preprocessed/theme_compressor.pt \
  --model_path "$MODEL_PATH" \
  --templates configs/aspect_templates.yaml \
  --batch_size 32 \
  "${EXTRA[@]}" \
  "$@"

python3 template_pseudo_label.py \
  --input "$INPUT" \
  --output preprocessed/wiki_theme_cache.jsonl \
  --stats_output preprocessed/theme_cache_stats.json \
  --model_path "$MODEL_PATH" \
  --compressor_ckpt preprocessed/theme_compressor.pt \
  --templates configs/aspect_templates.yaml \
  --batch_size 32 \
  --resume \
  "${EXTRA[@]}"
