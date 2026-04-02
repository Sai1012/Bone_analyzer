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
import textwrap
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch

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
# Individual patient report – clinical dashboard layout
# ──────────────────────────────────────────────────────────────────────────────

# Soft medical colour palette used by the clinical dashboard
_REPORT_COLORS = {
    "bg":     "#F7F9FB",
    "card":   "#FFFFFF",
    "border": "#E0E0E0",
    "green":  "#4CAF50",
    "red":    "#E57373",
    "yellow": "#FBC02D",
    "text":   "#333333",
}


def add_card(ax: plt.Axes) -> None:
    """Paint *ax* with a white card background and a rounded border."""
    ax.set_facecolor(_REPORT_COLORS["card"])
    card = FancyBboxPatch(
        (0, 0), 1, 1,
        boxstyle="round,pad=0.02",
        transform=ax.transAxes,
        fc=_REPORT_COLORS["card"],
        ec=_REPORT_COLORS["border"],
        lw=1,
        zorder=0,
    )
    ax.add_patch(card)


def _report_score_color(score: float) -> str:
    """Return a colour from the medical palette for *score* (1–10)."""
    if score >= 8:
        return _REPORT_COLORS["green"]
    if score >= 5:
        return _REPORT_COLORS["yellow"]
    return _REPORT_COLORS["red"]


def draw_gauge(ax: plt.Axes, score: float) -> None:
    """Draw a semicircular gauge for *score* (1–10) on *ax* (no card)."""
    ax.axis("equal")
    ax.axis("off")
    ax.set_facecolor("none")
    # Expanded limits prevent arc clipping
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.2)

    # Background arc
    angles = np.linspace(180, 0, 200)
    ax.plot(
        np.cos(np.deg2rad(angles)),
        np.sin(np.deg2rad(angles)),
        lw=18, color="#E1E4E8", solid_capstyle="round", clip_on=False,
    )

    # Coloured value arc
    value_angle = 180 - (score / 10) * 180
    arc = np.linspace(180, value_angle, 200)
    ax.plot(
        np.cos(np.deg2rad(arc)),
        np.sin(np.deg2rad(arc)),
        lw=18, color=_report_score_color(score), solid_capstyle="round", clip_on=False,
    )

    # Needle
    needle_rad = np.deg2rad(value_angle)
    ax.plot(
        [0, np.cos(needle_rad) * 0.85],
        [0, np.sin(needle_rad) * 0.85],
        lw=3.5, color=_REPORT_COLORS["text"], clip_on=False,
    )
    ax.add_patch(Circle((0, 0), 0.03, color=_REPORT_COLORS["text"], clip_on=False))

    ax.text(0, -0.12, f"{score:.1f}/10",
            ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, 0.25, "Risk Level",
            ha="center", fontsize=10, color=_REPORT_COLORS["text"])


def draw_feature_chart(ax: plt.Axes, features: Dict[str, Optional[float]]) -> None:
    """Horizontal bar chart of feature z-scores inside a card."""
    add_card(ax)

    valid = {k: v for k, v in features.items() if v is not None}
    if not valid:
        ax.text(0.5, 0.5, "No feature data available", ha="center", va="center")
        ax.axis("off")
        return

    items = sorted(valid.items(), key=lambda x: abs(x[1]), reverse=True)
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [_REPORT_COLORS["green"] if v > 0 else _REPORT_COLORS["red"] for v in values]

    ax.barh(labels, values, color=colors, edgecolor="black", alpha=0.8)
    ax.axvline(0,  color="#999999", lw=1)
    ax.axvline( 1, color="#cccccc", lw=0.8, ls="--")
    ax.axvline(-1, color="#cccccc", lw=0.8, ls="--")
    ax.axvline( 2, color="#dddddd", lw=0.8, ls=":")
    ax.axvline(-2, color="#dddddd", lw=0.8, ls=":")
    ax.set_title("Feature Deviation (Z-scores)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Deviation from Baseline", fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.2, color="#EEEEEE")
    ax.scatter([], [], color=_REPORT_COLORS["green"], label="Above baseline")
    ax.scatter([], [], color=_REPORT_COLORS["red"],   label="Below baseline")
    ax.legend(loc="lower right", fontsize=8, frameon=False)


def draw_risk_table(
    ax: plt.Axes,
    risk_factors: List[Tuple[str, str, str]],
) -> None:
    """Draw a risk-factor table inside a card on *ax*.

    *risk_factors* is a list of (factor_name, value, level) where
    level is one of "High", "Medium", "Low".
    """
    add_card(ax)
    ax.axis("off")
    ax.text(0.05, 0.88, "Risk Factor Analysis", fontsize=12, fontweight="bold",
            transform=ax.transAxes)
    y = 0.65
    for factor, value, level in risk_factors:
        color = (
            _REPORT_COLORS["red"]    if level == "High"   else
            _REPORT_COLORS["yellow"] if level == "Medium" else
            _REPORT_COLORS["green"]
        )
        ax.text(0.05, y, factor, fontsize=10, transform=ax.transAxes)
        ax.text(0.45, y, value,  fontsize=10, transform=ax.transAxes)
        ax.text(0.70, y, level,  fontsize=9,  transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="none", alpha=0.3))
        y -= 0.22


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers – derive display content from HealthResult
# ──────────────────────────────────────────────────────────────────────────────


def _risk_factors_from_result(result: HealthResult) -> List[Tuple[str, str, str]]:
    """Build a risk-factor list from the top deviating features."""
    top = sorted(
        [(k, v) for k, v in result.feature_zscores.items() if v is not None],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:3]
    rows: List[Tuple[str, str, str]] = []
    for feat, zscore in top:
        abs_z = abs(zscore)
        level = "High" if abs_z >= 2.0 else ("Medium" if abs_z >= 1.0 else "Low")
        rows.append((feat, f"{zscore:+.2f}σ", level))
    return rows


def _interpretation_from_result(result: HealthResult) -> str:
    """Return a short clinical interpretation string."""
    label = result.severity_label()
    top = result.top_deviations
    if top:
        feat, abs_z = top[0]
        raw_z = result.feature_zscores.get(feat, 0.0) or 0.0
        direction = "above" if raw_z > 0 else "below"
        return (
            f"Score indicates {label.lower()}. Most notable deviation: "
            f"{feat} ({raw_z:+.2f}σ {direction} baseline)."
        )
    return f"Score indicates {label.lower()}."


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def plot_patient_report(result: HealthResult, save_dir: Optional[str] = None) -> str:
    """Generate a single-page clinical dashboard report for one patient.

    Layout: GridSpec 3 rows × 2 columns
      Row 0: patient info card | gauge (standalone, no card)
      Row 1: feature deviation chart (spans both columns)
      Row 2: interpretation card | risk-factor table card

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or os.path.join(OUTPUT_DIR, "patient_reports")
    os.makedirs(save_dir, exist_ok=True)

    sns.set_style("white")
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(_REPORT_COLORS["bg"])
    gs = GridSpec(3, 2, height_ratios=[1.2, 1.5, 1.2], hspace=0.5, wspace=0.4)
    fig.suptitle(
        "Bone Health Analysis Report",
        fontsize=18, fontweight="bold", y=0.98,
    )

    # ── Row 0, Col 0: patient info card ──────────────────────────────────────
    ax_info = fig.add_subplot(gs[0, 0])
    add_card(ax_info)
    ax_info.axis("off")
    age_str    = str(int(result.age)) if result.age is not None else "N/A"
    gender_str = result.gender or "N/A"
    ax_info.text(0.05, 0.80, f"Patient ID:     {result.patient_id}", fontsize=11,
                 transform=ax_info.transAxes)
    ax_info.text(0.05, 0.62, f"Age:            {age_str}",           fontsize=11,
                 transform=ax_info.transAxes)
    ax_info.text(0.05, 0.44, f"Gender:         {gender_str}",        fontsize=11,
                 transform=ax_info.transAxes)
    ax_info.text(0.05, 0.26, f"Baseline Group: {result.stratum_used}", fontsize=11,
                 transform=ax_info.transAxes)

    # ── Row 0, Col 1: gauge (no card) ────────────────────────────────────────
    ax_gauge = fig.add_subplot(gs[0, 1])
    draw_gauge(ax_gauge, result.health_score)

    # ── Row 1, full width: feature deviation chart ───────────────────────────
    ax_feat = fig.add_subplot(gs[1, :])
    draw_feature_chart(ax_feat, result.feature_zscores)

    # ── Row 2, Col 0: interpretation card ────────────────────────────────────
    ax_interp = fig.add_subplot(gs[2, 0])
    add_card(ax_interp)
    ax_interp.axis("off")
    interp_text = _interpretation_from_result(result)
    ax_interp.text(0.05, 0.85, "Interpretation", fontsize=12, fontweight="bold",
                   transform=ax_interp.transAxes)
    ax_interp.text(
        0.05, 0.55,
        "\n".join(textwrap.wrap(interp_text, 52)),
        fontsize=10, transform=ax_interp.transAxes,
    )

    # ── Row 2, Col 1: risk factor table card ─────────────────────────────────
    ax_risk = fig.add_subplot(gs[2, 1])
    draw_risk_table(ax_risk, _risk_factors_from_result(result))

    fig.text(
        0.5, 0.02,
        "This report is AI-assisted and for clinical support only.",
        ha="center", fontsize=9, color="#888888",
    )

    path = os.path.join(save_dir, f"{result.patient_id}_report.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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
