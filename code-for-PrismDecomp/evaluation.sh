#!/bin/bash

# PrismDecomp评估脚本
# 基于棱镜分解的多语义句子表示学习

echo "开始PrismDecomp评估..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python prism_decomp_evaluation.py configs/evaluation_default.yaml

echo "PrismDecomp评估完成！"
