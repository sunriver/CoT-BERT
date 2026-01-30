#!/bin/bash
# TexLeJEPA 综合训练脚本：根据平台自动选 YAML，激活 CoT-BERT/.venv

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/../.venv"

if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
else
    echo "警告: 虚拟环境未找到: $VENV_PATH"
fi

cd "$SCRIPT_DIR"
export PYTHONPATH="${PYTHONPATH}:$SCRIPT_DIR"

echo "TexLeJEPA 训练（按当前平台选择 config）..."
python tex_lejepa_train.py

echo "TexLeJEPA 训练完成。"
