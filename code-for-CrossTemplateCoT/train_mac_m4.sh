#!/bin/bash

# CrossTemplateCoT Mac M4 训练脚本

echo "开始 CrossTemplateCoT Mac M4 训练..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python cross_template_cot_train.py configs/train_mac_m4.yaml

echo "CrossTemplateCoT Mac M4 训练完成！"


