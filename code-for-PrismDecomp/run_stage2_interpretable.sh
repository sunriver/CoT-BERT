#!/usr/bin/env bash
# Stage2: 使用伪标签训练 PrismDecomp（可解释分解）
set -e
cd "$(dirname "$0")"
bash train.sh configs/train_interpretable.yaml "$@"
