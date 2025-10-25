#!/bin/bash

# PrismDecomp训练脚本
# 基于棱镜分解的多语义句子表示学习

echo "开始PrismDecomp训练..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python prism_decomp_train.py configs/train_default.yaml

echo "PrismDecomp训练完成！"
