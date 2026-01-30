#!/bin/bash
# TexLeJEPA Mac M4 专用训练脚本

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"
cd "$SCRIPT_DIR"
export PYTHONPATH="${PYTHONPATH}:$SCRIPT_DIR"

echo "TexLeJEPA Mac M4 训练..."
python tex_lejepa_train.py configs/train_mac_m4.yaml

echo "TexLeJEPA Mac M4 训练完成。"
