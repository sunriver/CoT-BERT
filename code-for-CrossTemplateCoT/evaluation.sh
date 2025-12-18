#!/bin/bash

# CrossTemplateCoT 评估脚本（默认配置）

echo "开始 CrossTemplateCoT 默认评估..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行评估
python cross_template_cot_evaluation.py configs/evaluation_default.yaml

echo "CrossTemplateCoT 默认评估完成！"


