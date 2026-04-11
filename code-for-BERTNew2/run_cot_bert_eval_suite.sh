#!/usr/bin/env bash
# Linux：在 code-for-BERTNew2 下依次运行 SentEval 评估、金标散点图、Alignment/Uniformity benchmark。
#
# 推荐先激活虚拟环境（与「默认已开启 venv」一致）：
#   source ../.venv/bin/activate
# 未激活时：若存在 CoT-BERT/.venv/bin/python 则自动使用该解释器；否则回退 python3。
#
# 用法（在 code-for-BERTNew2 目录）:
#   chmod +x run_cot_bert_eval_suite.sh
#   ./run_cot_bert_eval_suite.sh
#
# cot_bert_evaluation 第一个参数为覆盖 yaml，会与平台默认 configs/evaluation_*.yaml 深度合并。
# train_linux_cuda.yaml 含训练专用键，若 argparse 报错，请改用仅含评估字段的覆盖配置。
set -euo pipefail

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$CODE_DIR"

COT_ROOT="$(cd "$CODE_DIR/.." && pwd)"
VENV_PY="$COT_ROOT/.venv/bin/python"

if command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
elif [[ -x "$VENV_PY" ]]; then
  PYTHON="$VENV_PY"
else
  PYTHON="python3"
  echo "[run_cot_bert_eval_suite] 未在 PATH 中找到 python，且 $VENV_PY 不可执行，已回退为 python3。建议: source $COT_ROOT/.venv/bin/activate" >&2
fi

echo "[run_cot_bert_eval_suite] Using: $PYTHON"

"$PYTHON" cot_bert_evaluation.py configs/evaluation_linux_cuda.yaml
"$PYTHON" gold_cosine_scatter.py
"$PYTHON" run_alignment_uniformity_benchmark.py

echo "[run_cot_bert_eval_suite] 全部完成。"
