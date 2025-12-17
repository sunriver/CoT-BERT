#!/bin/bash

# ProcessSupervisionCoT Mac M4 评估脚本

echo "开始 ProcessSupervisionCoT Mac M4 评估..."

source ../.venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python process_supervision_cot_evaluation.py configs/evaluation_mac_m4.yaml

echo "ProcessSupervisionCoT Mac M4 评估完成！"


