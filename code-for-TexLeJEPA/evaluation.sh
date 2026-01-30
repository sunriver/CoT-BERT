#!/bin/bash
# TexLeJEPA 综合评估脚本：根据平台自动选 YAML，激活 CoT-BERT/.venv，集成 SentEval

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

echo "TexLeJEPA 评估（按当前平台选择 config）..."
python tex_lejepa_evaluation.py

echo "TexLeJEPA 评估完成。"
