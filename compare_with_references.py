"""
compare_with_references.py
--------------------------
Comparative analysis of the proposed bone health ensemble model against
benchmark results reported in the literature.

Reference table (IEEE style)
----------------------------
[1] Y. Wu, J. Chao, M. Bao, and N. Zhang, \"Predictive value of machine learning
    on fracture risk in osteoporosis: A systematic review and meta-analysis," BMJ Open,
    vol. 13, no. 12, e071430, 2023.
[2] F. Liu, H. Jang, R. Kijowski, T. Bradshaw, and A. B. McMillan,
    \"Deep learning MR imaging–based attenuation correction for PET/MR imaging," Radiology,
    vol. 286, no. 2, pp. 676–684, 2018.
[3] R. Jennane, W. J. Ohley, S. Majumdar, and G. Lemineur, \"Fractal analysis of bone
    X-ray tomographic microscopy projections," IEEE Trans. Med. Imaging, vol. 20, no. 5,
    pp. 443–449, 2001.
[4] G. Litjens, T. Kooi, B. E. Bejnordi, et al., \"A survey on deep learning in medical
    image analysis," Med. Image Anal., vol. 42, pp. 60–88, 2017.

Usage
-----
    python compare_with_references.py

The script prints the comparison table to stdout and saves two benchmark
figures to:
- output/range_plot.png
- output/bar_plot.png

If output/our_metrics.json exists, it will be used to add the \"Ours\" entry.
Expected format:
{
  \"metric\": \"AUC\",
  \"value\": 0.92
}
or
{
  \"metric\": \"AUC\",
  \"min\": 0.90,
  \"max\": 0.94
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class StudyMetric:
    label: str
    year: int
    metric: str
    min_value: float
    max_value: float
    note: str
    ref_id: str

    @property
    def midpoint(self) -> float:
        return (self.min_value + self.max_value) / 2.0


REFERENCE_STUDIES: List[StudyMetric] = [
    StudyMetric("Wu et al.", 2023, "AUC", 0.80, 0.90, "Fracture risk ML", "[1]"),
    StudyMetric("Liu et al.", 2018, "SSIM", 0.90, 0.90, "PET/MR DL", "[2]"),
    StudyMetric("Jennane et al.", 2001, "Correlation", 0.70, 0.85, "Texture vs BMD", "[3]"),
    StudyMetric("Litjens et al.", 2017, "AUC", 0.85, 0.95, "DL survey", "[4]"),
]


DEFAULT_OURS_METRIC = "AUC"

def load_ours_metrics(path: str = "output/our_metrics.json") -> Optional[StudyMetric]:
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if "value" in data:
        min_val = max_val = float(data["value"])
    else:
        min_val = float(data.get("min", np.nan))
        max_val = float(data.get("max", np.nan))

    if np.isnan(min_val):
        return None

    return StudyMetric(
        "Ours (CBHS Model)",
        2026,
        data.get("metric", DEFAULT_OURS_METRIC),
        min_val,
        max_val,
        "Proposed Ensemble",
        "*",
    )

def print_table(studies: List[StudyMetric]) -> None:
    print("\n📊 Comparison With Prior Work\n")
    print("-" * 75)
    print(f"{'Study':<25}{'Year':<6}{'Metric':<12}{'Range':<15}{'Ref'}")
    print("-" * 75)

    for s in studies:
        rng = f"{s.min_value:.2f}-{s.max_value:.2f}"
        print(f"{s.label:<25}{s.year:<6}{s.metric:<12}{rng:<15}{s.ref_id}")

    print("-" * 75)

def plot_range(studies: List[StudyMetric], path: str = "output/range_plot.png") -> None:
    os.makedirs("output", exist_ok=True)

    labels = [s.label for s in studies]
    y = np.arange(len(studies))

    plt.figure(figsize=(10, 5))

    for i, s in enumerate(studies):
        if "Ours" in s.label:
            color = "black"
            lw = 4
        else:
            color = "gray"
            lw = 2

        plt.hlines(i, s.min_value, s.max_value, color=color, linewidth=lw)
        plt.scatter(s.midpoint, i, color=color, s=80)

    plt.yticks(y, labels)
    plt.xlabel("Metric Value")
    plt.title("Performance Comparison Across Studies")

    plt.grid(axis="x", linestyle="--", alpha=0.5)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✅ Range plot saved: {path}")

def plot_bar(studies: List[StudyMetric], path: str = "output/bar_plot.png") -> None:
    os.makedirs("output", exist_ok=True)

    labels = [s.label for s in studies]
    mids = [s.midpoint for s in studies]

    plt.figure(figsize=(10, 5))

    for i, s in enumerate(studies):
        if "Ours" in s.label:
            plt.bar(i, mids[i], color="black")
        else:
            plt.bar(i, mids[i], color="lightgray")

        plt.text(i, mids[i] + 0.02, f"{mids[i]:.2f}", ha="center")

    plt.xticks(range(len(labels)), labels, rotation=15)
    plt.ylabel("Metric Value")
    plt.title("Midpoint Performance Comparison")

    plt.ylim(0, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"✅ Bar plot saved: {path}")


if __name__ == "__main__":
    studies = REFERENCE_STUDIES.copy()

    ours = load_ours_metrics()
    if ours:
        studies.append(ours)
    else:
        print("ℹ No 'our_metrics.json' found → plotting only references")

    print_table(studies)
    plot_range(studies)
    plot_bar(studies)
