#!/bin/bash

# PrismDecomp训练脚本
# 基于棱镜分解的多语义句子表示学习

echo "开始PrismDecomp训练..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练（第一个参数为配置文件，默认 train_default.yaml）
CONFIG="${1:-configs/train_default.yaml}"
shift || true
python prism_decomp_train.py "$CONFIG" "$@"

echo "PrismDecomp训练完成！"
