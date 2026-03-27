#!/usr/bin/env python3
"""
从 Hugging Face Trainer 保存的 trainer_state.json 读取 log_history，绘制训练曲线。

用法:
  python scripts/visualize_trainer_state.py ../result/CoT-Bert/trainer_state.json
  python scripts/visualize_trainer_state.py ../result/CoT-Bert/trainer_state.json -o ../result/CoT-Bert/plots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict


def load_history(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        state = json.load(f)
    return state.get("log_history", [])


def series_from_history(history: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """按 step 聚合标量序列；同一 step 多条记录时保留最后一次。"""
    loss_lr: dict[int, tuple[float, float | None]] = {}
    eval_metrics: dict[str, dict[int, float]] = defaultdict(dict)

    for row in history:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row:
            lr = row.get("learning_rate")
            loss_lr[step] = (float(row["loss"]), float(lr) if lr is not None else None)
        for k, v in row.items():
            if not k.startswith("eval_") or not isinstance(v, (int, float)):
                continue
            eval_metrics[k][step] = float(v)

    steps_sorted = sorted(loss_lr.keys())
    loss_series = [(s, loss_lr[s][0]) for s in steps_sorted]
    lr_series = [(s, loss_lr[s][1]) for s in steps_sorted if loss_lr[s][1] is not None]

    out: dict[str, list[tuple[int, float]]] = {
        "loss": loss_series,
        "learning_rate": lr_series,
    }
    for name, step_val in eval_metrics.items():
        out[name] = sorted(step_val.items(), key=lambda x: x[0])
    return out


def plot_all(series: dict[str, list[tuple[int, float]]], out_dir: str, title_prefix: str = "") -> None:
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    if series.get("loss"):
        xs, ys = zip(*series["loss"])
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o", markersize=3)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title(f"{title_prefix}Training loss".strip())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trainer_loss.png"), dpi=150)
        plt.close()

    if series.get("learning_rate"):
        xs, ys = zip(*series["learning_rate"])
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o", markersize=3)
        plt.xlabel("step")
        plt.ylabel("learning rate")
        plt.title(f"{title_prefix}Learning rate".strip())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trainer_learning_rate.png"), dpi=150)
        plt.close()

    eval_keys = [k for k in series if k.startswith("eval_") and series[k]]
    if eval_keys:
        plt.figure(figsize=(9, 5))
        for k in sorted(eval_keys):
            xs, ys = zip(*series[k])
            plt.plot(xs, ys, marker="o", markersize=3, label=k)
        plt.xlabel("step")
        plt.ylabel("metric")
        plt.title(f"{title_prefix}Validation metrics".strip())
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "trainer_eval_metrics.png"), dpi=150)
        plt.close()

    # 每个 eval 指标单独一张（论文里好挑一张用）
    for k in sorted(eval_keys):
        xs, ys = zip(*series[k])
        plt.figure(figsize=(8, 4))
        plt.plot(xs, ys, marker="o", markersize=3, color="C0")
        plt.xlabel("step")
        plt.ylabel(k)
        plt.title(f"{title_prefix}{k}".strip())
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        safe = k.replace("/", "_")
        plt.savefig(os.path.join(out_dir, f"trainer_{safe}.png"), dpi=150)
        plt.close()

    print(f"Saved figures to: {os.path.abspath(out_dir)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="可视化 trainer_state.json 中的 log_history")
    parser.add_argument(
        "trainer_state_json",
        type=str,
        help="trainer_state.json 路径",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        help="图片输出目录（默认：与 json 同目录下的 trainer_plots/）",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="",
        help="图标题前缀（可选）",
    )
    args = parser.parse_args()

    path = os.path.abspath(args.trainer_state_json)
    if not os.path.isfile(path):
        print(f"文件不存在: {path}", file=sys.stderr)
        return 1

    out_dir = args.output_dir
    if not out_dir:
        out_dir = os.path.join(os.path.dirname(path), "trainer_plots")

    history = load_history(path)
    if not history:
        print("log_history 为空", file=sys.stderr)
        return 1

    series = series_from_history(history)
    try:
        plot_all(series, out_dir, title_prefix=args.title_prefix)
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
