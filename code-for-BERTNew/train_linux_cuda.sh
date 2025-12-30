#!/bin/bash

# CoT-BERT Linux CUDA训练脚本
# 基于Chain-of-Thought的句子表示学习

echo "开始CoT-BERT Linux CUDA训练..."

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

# 运行训练
python cot_bert_train.py configs/train_linux_cuda.yaml

echo "CoT-BERT Linux CUDA训练完成！"

