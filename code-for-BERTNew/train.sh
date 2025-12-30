#!/bin/bash

# CoT-BERT训练脚本
# 基于Chain-of-Thought的句子表示学习

echo "开始CoT-BERT训练..."

# 激活虚拟环境
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/../../.venv"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "警告: 虚拟环境未找到: $VENV_PATH"
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 运行训练（使用默认配置文件）
python cot_bert_train.py configs/train_default.yaml

echo "CoT-BERT训练完成！"

