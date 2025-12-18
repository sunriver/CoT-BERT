#!/bin/bash

# CrossTemplateCoT 训练脚本（默认配置）

echo "开始 CrossTemplateCoT 默认训练..."

# 激活虚拟环境
source ../.venv/bin/activate

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练
python cross_template_cot_train.py configs/train_default.yaml

echo "CrossTemplateCoT 默认训练完成！"


