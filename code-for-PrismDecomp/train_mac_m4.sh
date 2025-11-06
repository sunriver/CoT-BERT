#!/bin/bash

# PrismDecomp Mac M4训练脚本
# 基于棱镜分解的多语义句子表示学习

echo "开始PrismDecomp Mac M4训练..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python prism_decomp_train.py configs/train_mac_m4.yaml

echo "PrismDecomp Mac M4训练完成！"
