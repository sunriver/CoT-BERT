#!/bin/bash
# TexLeJEPA Mac M4 专用评估脚本

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"
cd "$SCRIPT_DIR"
export PYTHONPATH="${PYTHONPATH}:$SCRIPT_DIR"

echo "TexLeJEPA Mac M4 评估..."
python tex_lejepa_evaluation.py configs/evaluation_mac_m4.yaml

echo "TexLeJEPA Mac M4 评估完成。"
