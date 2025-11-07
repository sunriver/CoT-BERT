#!/bin/bash

# PrismDecomp Mac M4评估脚本
# 基于棱镜分解的多语义句子表示学习

echo "开始PrismDecomp Mac M4评估..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python prism_decomp_evaluation.py configs/evaluation_mac_m4.yaml

echo "PrismDecomp Mac M4评估完成！"

