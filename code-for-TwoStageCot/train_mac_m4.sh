#!/bin/bash

# TwoStageCoT Mac M4训练脚本
# 基于两阶段思维链的句子表示学习

echo "开始TwoStageCoT Mac M4训练..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python two_stage_cot_train.py configs/train_mac_m4.yaml

echo "TwoStageCoT Mac M4训练完成！"

