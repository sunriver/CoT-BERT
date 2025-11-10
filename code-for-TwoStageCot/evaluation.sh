#!/bin/bash

# TwoStageCoT评估脚本
# 基于两阶段思维链的句子表示学习

echo "开始TwoStageCoT评估..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python two_stage_cot_evaluation.py configs/evaluation_default.yaml

echo "TwoStageCoT评估完成！"

