"""
compare_with_references.py
--------------------------
Comparative analysis of the proposed bone health ensemble model against
benchmark results reported in the literature.

Reference table (IEEE style)
----------------------------
[1] Y. Wu, J. Chao, M. Bao, and N. Zhang, "Predictive value of machine learning
    on fracture risk in osteoporosis: A systematic review and meta-analysis," BMJ Open,
    vol. 13, no. 12, e071430, 2023.
[2] F. Liu, H. Jang, R. Kijowski, T. Bradshaw, and A. B. McMillan,
    "Deep learning MR imaging–based attenuation correction for PET/MR imaging," Radiology,
    vol. 286, no. 2, pp. 676–684, 2018.
[3] R. Jennane, W. J. Ohley, S. Majumdar, and G. Lemineur, "Fractal analysis of bone
    X-ray tomographic microscopy projections," IEEE Trans. Med. Imaging, vol. 20, no. 5,
    pp. 443–449, 2001.
[4] G. Litjens, T. Kooi, B. E. Bejnordi, et al., "A survey on deep learning in medical
    image analysis," Med. Image Anal., vol. 42, pp. 60–88, 2017.

Usage
-----
    python compare_with_references.py

The script prints the comparison table to stdout and saves two benchmark
figures to:
- output/comparison_metric_ranges.png
- output/comparison_metric_midpoints.pnggit add compare_with_references.py generate_bone_health_report.py

If output/our_metrics.json exists, it will be used to add the "Ours" entry.
Expected format:
{
  "metric": "AUC",
  "value": 0.92
}
or
{
  "metric": "AUC",
  "min": 0.90,
  "max": 0.94
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Benchmark data (from provided references/summary)
# ---------------------------------------------------------------------------

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
    StudyMetric(
        label="Wu et al.",
        year=2023,
        metric="AUC",
        min_value=0.80,
        max_value=0.90,
        note="Meta-analysis fracture risk",
        ref_id="[1]",
    ),
    StudyMetric(
        label="Liu et al.",
        year=2018,
        metric="SSIM",
        min_value=0.90,
        max_value=0.90,
        note="PET/MR attenuation correction",
        ref_id="[2]",
    ),
    StudyMetric(
        label="Jennane et al.",
        year=2001,
        metric="r (corr)",
        min_value=0.70,
        max_value=0.85,
        note="Fractal texture vs BMD",
        ref_id="[3]",
    ),
    StudyMetric(
        label="Litjens et al.",
        year=2017,
        metric="AUC",
        min_value=0.85,
        max_value=0.95,
        note="Surveyed AUC ranges",
        ref_id="[4]",
    ),
]


# ---------------------------------------------------------------------------
# Optional: include our model metrics if provided
# ---------------------------------------------------------------------------

DEFAULT_OURS_METRIC = "AUC"

def load_ours_metrics(path: str = "output/our_metrics.json") -> Optional[StudyMetric]:
    """Load optional metrics for the proposed model from a JSON file."""
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"⚠ Could not read {path}: {exc}")
        return None

    metric = str(data.get("metric", DEFAULT_OURS_METRIC))
    if "value" in data:
        min_value = max_value = float(data["value"])
    else:
        min_value = float(data.get("min", np.nan))
        max_value = float(data.get("max", np.nan))

    if np.isnan(min_value) or np.isnan(max_value):
        print("⚠ our_metrics.json missing 'value' or ('min','max') fields.")
        return None

    return StudyMetric(
        label="Ours (Proposed)",
        year=2026,
        metric=metric,
        min_value=min_value,
        max_value=max_value,
        note="Proposed ensemble model",
        ref_id="*",
    )


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(studies: List[StudyMetric]) -> None:
    """Print a publication-ready comparison table to stdout."""
    col_w = [26, 6, 12, 16, 10]
    header = (
        f"{'Study':<{col_w[0]}}"
        f"{'Year':^{col_w[1]}}"
        f"{'Metric':^{col_w[2]}}"
        f"{'Range':^{col_w[3]}}"
        f"{'Ref':^{col_w[4]}}"
    )
    sep = "-" * sum(col_w)
    print("\nComparison With Prior Work (Reported Metrics)")
    print(sep)
    print(header)
    print(sep)
    for s in studies:
        rng = f"{s.min_value:.2f}–{s.max_value:.2f}"
        print(
            f"{s.label:<{col_w[0]}}"
            f"{s.year:^{col_w[1]}}"
            f"{s.metric:^{col_w[2]}}"
            f"{rng:^{col_w[3]}}"
            f"{s.ref_id:^{col_w[4]}}"
        )
    print(sep)
    print("* Ours loaded from output/our_metrics.json (if present).\n")


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

METRIC_COLORS = {
    "AUC": "#4C78A8",
    "SSIM": "#F58518",
    "r (corr)": "#54A24B",
}

def _metric_color(metric: str) -> str:
    return METRIC_COLORS.get(metric, "#B279A2")

def generate_metric_range_plot(
    studies: List[StudyMetric],
    output_path: str = "output/comparison_metric_ranges.png",
) -> None:
    """Plot min-max ranges per study with a midpoint marker."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    labels = [f"{s.label} ({s.metric})" for s in studies]
    y = np.arange(len(studies))

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for i, s in enumerate(studies):
        color = _metric_color(s.metric)
        ax.plot([s.min_value, s.max_value], [i, i], color=color, linewidth=3)
        ax.scatter([s.midpoint], [i], color=color, s=60, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Reported Metric Range", fontsize=12)
    ax.set_title("Reported Metric Ranges by Study", fontsize=14)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], color=c, lw=3, label=m)
        for m, c in METRIC_COLORS.items()
    ]
    ax.legend(handles=handles, title="Metric", loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to '{output_path}'.")

def generate_metric_midpoint_plot(
    studies: List[StudyMetric],
    output_path: str = "output/comparison_metric_midpoints.png",
) -> None:
    """Plot metric midpoints with color coding by metric type."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    labels = [s.label for s in studies]
    mids = [s.midpoint for s in studies]
    colors = [_metric_color(s.metric) for s in studies]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, mids, color=colors, alpha=0.85)

    # Annotate bars with metric type
    for bar, s in zip(bars, studies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{s.metric}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Midpoint Value", fontsize=12)
    ax.set_title("Midpoint Comparison of Reported Metrics", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to '{output_path}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    studies = REFERENCE_STUDIES.copy()
    ours = load_ours_metrics()
    if ours is not None:
        studies.append(ours)
    else:
        print("ℹ No output/our_metrics.json found. Skipping 'Ours' in plots.")

    print_comparison_table(studies)
    generate_metric_range_plot(studies, "output/comparison_metric_ranges.png")
    generate_metric_midpoint_plot(studies, "output/comparison_metric_midpoints.png")