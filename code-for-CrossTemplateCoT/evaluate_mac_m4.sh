#!/bin/bash

# CrossTemplateCoT Mac M4 评估脚本

echo "开始 CrossTemplateCoT Mac M4 评估..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python cross_template_cot_evaluation.py configs/evaluation_mac_m4.yaml

echo "CrossTemplateCoT Mac M4 评估完成！"


