"""
visualizer.py - Generate doctor-friendly reports and population charts.

Produces:
- Individual patient health score cards (PNG)
- Population health score distribution chart (PNG)
- Feature deviation heatmap (PNG)
- Age-wise health trend chart (PNG)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import OUTPUT_DIR
from feature_extractor import ALL_FEATURE_NAMES
from health_scale_calculator import HealthResult

logger = logging.getLogger(__name__)

# Color palette for score bands
SCORE_COLORS = {
    range(9, 11): "#2ecc71",   # green  – healthy
    range(7, 9):  "#f1c40f",   # yellow – mild
    range(5, 7):  "#e67e22",   # orange – moderate
    range(3, 5):  "#e74c3c",   # red    – severe
    range(1, 3):  "#8e44ad",   # purple – critical
}


def _score_color(score: int) -> str:
    for band, color in SCORE_COLORS.items():
        if score in band:
            return color
    return "#95a5a6"


# ──────────────────────────────────────────────────────────────────────────────
# Individual patient report
# ──────────────────────────────────────────────────────────────────────────────


def plot_patient_report(result: HealthResult, save_dir: Optional[str] = None) -> str:
    """Generate a single-page health report for one patient.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or os.path.join(OUTPUT_DIR, "patient_reports")
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Bone Health Report – Patient {result.patient_id}",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    _draw_score_card(axes[0], result)
    _draw_feature_deviation_chart(axes[1], result)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{result.patient_id}_report.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_score_card(ax: plt.Axes, result: HealthResult) -> None:
    """Draw a large health score gauge on *ax*."""
    color = _score_color(result.health_score)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background circle
    circle = plt.Circle((0.5, 0.55), 0.35, color=color, alpha=0.15)
    ax.add_patch(circle)
    circle2 = plt.Circle((0.5, 0.55), 0.35, fill=False, edgecolor=color, linewidth=3)
    ax.add_patch(circle2)

    # Score number
    ax.text(
        0.5, 0.60,
        str(result.health_score),
        ha="center", va="center",
        fontsize=60, fontweight="bold", color=color,
    )
    ax.text(
        0.5, 0.42,
        "/10",
        ha="center", va="center",
        fontsize=18, color="#555555",
    )

    # Severity label
    ax.text(
        0.5, 0.28,
        result.severity_label(),
        ha="center", va="center",
        fontsize=12, color="#333333",
        style="italic",
    )

    # Patient info
    info_lines = [
        f"Patient ID : {result.patient_id}",
        f"Age        : {result.age if result.age is not None else 'N/A'}",
        f"Gender     : {result.gender or 'N/A'}",
        f"Baseline   : {result.stratum_used}  (n={result.stratum_n})",
        f"Composite Δ: {result.composite_deviation:.2f} std",
    ]
    for i, line in enumerate(info_lines):
        ax.text(0.05, 0.18 - i * 0.05, line, ha="left", va="top", fontsize=9, color="#444444")

    # Top deviations
    ax.text(0.05, -0.08, "Top Deviating Features:", ha="left", va="top",
            fontsize=9, fontweight="bold", color="#333333")
    for rank, (feat, zscore) in enumerate(result.top_deviations, start=1):
        ax.text(0.05, -0.08 - rank * 0.05,
                f"  {rank}. {feat}: {zscore:+.2f} std",
                ha="left", va="top", fontsize=8.5, color="#555555")


def _draw_feature_deviation_chart(ax: plt.Axes, result: HealthResult) -> None:
    """Horizontal bar chart of per-feature z-scores."""
    feats = [f for f in ALL_FEATURE_NAMES if result.feature_zscores.get(f) is not None]
    zscores = [result.feature_zscores[f] for f in feats]

    if not feats:
        ax.text(0.5, 0.5, "No features available", ha="center", va="center")
        ax.axis("off")
        return

    colors = ["#e74c3c" if z < 0 else "#2ecc71" for z in zscores]
    y_pos = range(len(feats))

    ax.barh(list(y_pos), zscores, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="#333333", linewidth=1.2)
    ax.axvline(-2, color="#f39c12", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(2, color="#f39c12", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Z-score (deviation from age-gender baseline)", fontsize=9)
    ax.set_title("Feature Deviations from Baseline", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    red_patch = mpatches.Patch(color="#e74c3c", label="Below baseline")
    green_patch = mpatches.Patch(color="#2ecc71", label="Above baseline")
    ax.legend(handles=[green_patch, red_patch], fontsize=8, loc="lower right")


# ──────────────────────────────────────────────────────────────────────────────
# Population charts
# ──────────────────────────────────────────────────────────────────────────────


def plot_score_distribution(
    results: List[HealthResult],
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> str:
    """Bar chart of health score distribution, optionally split by diagnosis."""
    save_dir = save_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    scores = [r.health_score for r in results]
    diagnoses = df["diagnosis"].tolist() if "diagnosis" in df.columns else ["unknown"] * len(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall distribution
    ax = axes[0]
    unique_scores = list(range(1, 11))
    counts = [scores.count(s) for s in unique_scores]
    bar_colors = [_score_color(s) for s in unique_scores]
    ax.bar(unique_scores, counts, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Health Score (1–10)", fontsize=10)
    ax.set_ylabel("Number of Patients", fontsize=10)
    ax.set_title("Health Score Distribution (All Patients)", fontsize=11)
    ax.set_xticks(unique_scores)
    ax.grid(axis="y", alpha=0.3)

    # By diagnosis
    ax = axes[1]
    diag_scores: Dict[str, List[int]] = {}
    for score, diag in zip(scores, diagnoses):
        diag_scores.setdefault(str(diag), []).append(score)

    diag_list = sorted(diag_scores.keys())
    palette = sns.color_palette("Set2", len(diag_list))
    for i, diag in enumerate(diag_list):
        ds = diag_scores[diag]
        cnt = [ds.count(s) for s in unique_scores]
        ax.bar(
            [s + 0.25 * (i - len(diag_list) / 2) for s in unique_scores],
            cnt,
            width=0.25,
            color=palette[i],
            label=diag,
            edgecolor="white",
        )
    ax.set_xlabel("Health Score (1–10)", fontsize=10)
    ax.set_ylabel("Number of Patients", fontsize=10)
    ax.set_title("Health Score Distribution by Diagnosis", fontsize=11)
    ax.set_xticks(unique_scores)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "score_distribution.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Score distribution chart saved to %s", path)
    return path


def plot_age_trend(
    results: List[HealthResult],
    save_dir: Optional[str] = None,
) -> str:
    """Scatter plot of health score vs age with regression line."""
    save_dir = save_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    ages = [r.age for r in results if r.age is not None]
    scores = [r.health_score for r in results if r.age is not None]

    if not ages:
        logger.warning("No age data available for age trend chart")
        return ""

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [_score_color(s) for s in scores]
    ax.scatter(ages, scores, c=colors, alpha=0.6, s=40, edgecolors="white")

    # Regression line
    z = np.polyfit(ages, scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(ages), max(ages), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=1.5, alpha=0.7, label="Trend")

    ax.set_xlabel("Age (years)", fontsize=10)
    ax.set_ylabel("Health Score (1–10)", fontsize=10)
    ax.set_title("Health Score vs Age", fontsize=11)
    ax.set_ylim(0, 11)
    ax.set_yticks(range(1, 11))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "age_trend.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Age trend chart saved to %s", path)
    return path


def plot_feature_heatmap(
    results: List[HealthResult],
    max_patients: int = 50,
    save_dir: Optional[str] = None,
) -> str:
    """Heatmap of feature z-scores across (a sample of) patients."""
    save_dir = save_dir or OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)

    sample = results[:max_patients]
    feat_names = [
        f for f in ALL_FEATURE_NAMES
        if any(r.feature_zscores.get(f) is not None for r in sample)
    ]

    if not feat_names or not sample:
        logger.warning("Not enough data for feature heatmap")
        return ""

    matrix = []
    patient_labels = []
    for r in sample:
        row_vals = [
            r.feature_zscores.get(f) if r.feature_zscores.get(f) is not None else 0.0
            for f in feat_names
        ]
        matrix.append(row_vals)
        patient_labels.append(r.patient_id)

    matrix_arr = np.array(matrix)
    # Clip extreme values for better color scale
    matrix_arr = np.clip(matrix_arr, -5, 5)

    fig, ax = plt.subplots(figsize=(max(10, len(feat_names) * 0.7), max(8, len(sample) * 0.25)))
    sns.heatmap(
        matrix_arr,
        ax=ax,
        xticklabels=feat_names,
        yticklabels=patient_labels,
        cmap="RdYlGn_r",
        center=0,
        vmin=-5,
        vmax=5,
        linewidths=0.3,
        cbar_kws={"label": "Z-score (deviation from baseline)"},
    )
    ax.set_xlabel("Feature", fontsize=10)
    ax.set_ylabel("Patient ID", fontsize=10)
    ax.set_title(f"Feature Deviation Heatmap (first {len(sample)} patients)", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_heatmap.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature heatmap saved to %s", path)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: generate all population charts
# ──────────────────────────────────────────────────────────────────────────────


def generate_all_charts(
    results: List[HealthResult],
    df: pd.DataFrame,
    save_dir: Optional[str] = None,
) -> List[str]:
    """Generate all population-level visualizations and return saved paths."""
    paths = []
    paths.append(plot_score_distribution(results, df, save_dir))
    paths.append(plot_age_trend(results, save_dir))
    paths.append(plot_feature_heatmap(results, save_dir=save_dir))
    return [p for p in paths if p]
