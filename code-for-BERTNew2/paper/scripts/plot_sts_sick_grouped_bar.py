#!/usr/bin/env python3
"""Grouped bar chart: STS-B vs SICK-R (Spearman rho x 100) from Table 1 subset."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Order: PromptBERT -> ConPVP -> SimCSE -> DiffCSE -> CoT-BERT -> MV-RaCL -> RankEnc.
ROWS: list[tuple[str, float, float]] = [
    ("PromptBERT", 81.60, 69.87),
    ("ConPVP", 80.82, 73.38),
    ("SimCSE", 76.85, 72.23),
    ("DiffCSE", 80.59, 71.23),
    ("CoT-BERT", 82.40, 71.41),
    ("MV-RaCL", 81.75, 74.65),
    ("RankEnc.", 81.55, 75.78),
]

LABELS = [r[0] for r in ROWS]
STS_B = np.array([r[1] for r in ROWS])
SICK_R = np.array([r[2] for r in ROWS])


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "figures" / "sts_stickr_grouped.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "black",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    n = len(LABELS)
    x = np.arange(n)
    width = 0.36

    fig_w, fig_h = 7.0, 3.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    bars1 = ax.bar(x - width / 2, STS_B, width, label="STS-B", color="#2c5282", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, SICK_R, width, label="SICK-R", color="#9b2c2c", edgecolor="black", linewidth=0.5)

    ax.set_ylabel(r"Spearman $\rho \times 100$")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=28, ha="right")
    ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="0.4")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#e0e0e0", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55, n - 0.45)

    ymin = min(STS_B.min(), SICK_R.min()) - 2.0
    ymax = max(STS_B.max(), SICK_R.max()) + 2.5
    ax.set_ylim(ymin, ymax)

    for b in (*bars1, *bars2):
        b.set_zorder(2)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
