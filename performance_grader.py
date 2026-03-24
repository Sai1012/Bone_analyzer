"""
performance_grader.py - Evaluate system performance for Bone Health Analysis.

Reads health scores from output/health_scores.xlsx (or CSV fallback) and
computes four categories of performance metrics:

  1. Group Separation Analysis  – mean/std of health score per diagnosis group
  2. Classification Metrics     – confusion matrix, accuracy, precision,
                                   recall, F1-score, balanced accuracy
  3. Correlation Analysis        – Pearson & Spearman correlation between
                                   health score and clinical T-score / Z-score
  4. System Reliability          – success rate, fallback baseline usage,
                                   runtime statistics

Outputs
-------
  output/confusion_matrix.png
  output/score_distribution_by_diagnosis.png
  output/score_vs_tscore.png
  output/score_vs_zscore.png
  output/correlation_heatmap.png
  output/classification_report.txt
  output/performance_metrics_summary.json

Usage
-----
  python performance_grader.py
  python performance_grader.py --output-dir /path/to/output
  python performance_grader.py --score-thresholds 8,6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# Default score → class mapping (configurable via CLI)
# score >= NORMAL_THRESHOLD  → Normal
# score >= OSTEOPENIA_THRESHOLD → Osteopenia
# otherwise → Osteoporosis
DEFAULT_NORMAL_THRESHOLD = 8.0
DEFAULT_OSTEOPENIA_THRESHOLD = 6.0

CLASS_ORDER = ["normal", "osteopenia", "osteoporosis"]

# Palette used across all plots
PALETTE = {
    "normal": "#2ecc71",
    "osteopenia": "#f39c12",
    "osteoporosis": "#e74c3c",
}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_health_scores(output_dir: str = OUTPUT_DIR) -> pd.DataFrame:
    """Load health scores from Excel (preferred) or CSV fallback.

    Raises
    ------
    FileNotFoundError
        If neither health_scores.xlsx nor health_scores.csv is found.
    """
    xlsx_path = os.path.join(output_dir, "health_scores.xlsx")
    csv_path = os.path.join(output_dir, "health_scores.csv")

    if os.path.exists(xlsx_path):
        print(f"Loading data from {xlsx_path}...")
        df = pd.read_excel(xlsx_path)
    elif os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No health scores file found in '{output_dir}'.\n"
            "Run main.py first to generate outputs."
        )

    # Normalise string columns to lower-case for consistent comparisons
    for col in ("actual_diagnosis", "gender", "severity"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    print(f"✓ Loaded {len(df)} patients")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Score → class mapping
# ──────────────────────────────────────────────────────────────────────────────


def score_to_class(
    score: float,
    normal_threshold: float = DEFAULT_NORMAL_THRESHOLD,
    osteopenia_threshold: float = DEFAULT_OSTEOPENIA_THRESHOLD,
) -> str:
    """Map a numeric health score to a diagnosis class string."""
    if score >= normal_threshold:
        return "normal"
    elif score >= osteopenia_threshold:
        return "osteopenia"
    else:
        return "osteoporosis"


def assign_predicted_classes(
    df: pd.DataFrame,
    normal_threshold: float = DEFAULT_NORMAL_THRESHOLD,
    osteopenia_threshold: float = DEFAULT_OSTEOPENIA_THRESHOLD,
) -> pd.DataFrame:
    """Add a *predicted_diagnosis* column based on health_score thresholds."""
    df = df.copy()
    df["predicted_diagnosis"] = df["health_score"].apply(
        lambda s: score_to_class(s, normal_threshold, osteopenia_threshold)
    )
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 1. Group Separation Analysis
# ──────────────────────────────────────────────────────────────────────────────


def group_separation_analysis(df: pd.DataFrame) -> Dict:
    """Compute mean and std of health score per diagnosis group.

    Returns a dict with keys 'groups' (per-class stats) and 'separation'
    (pairwise mean differences).
    """
    print("\n" + "=" * 60)
    print("1. GROUP SEPARATION ANALYSIS")
    print("=" * 60)

    if "actual_diagnosis" not in df.columns:
        print("  ⚠ 'actual_diagnosis' column not found – skipping group analysis.")
        return {}

    result: Dict = {"groups": {}, "separation": {}}

    for cls in CLASS_ORDER:
        group = df[df["actual_diagnosis"] == cls]["health_score"]
        if group.empty:
            continue
        stats_dict = {
            "n": int(len(group)),
            "mean": float(round(group.mean(), 3)),
            "std": float(round(group.std(), 3)),
            "min": float(round(group.min(), 3)),
            "max": float(round(group.max(), 3)),
            "median": float(round(group.median(), 3)),
        }
        result["groups"][cls] = stats_dict
        label = cls.capitalize()
        print(
            f"  {label:<15s}  n={stats_dict['n']:3d}  "
            f"Mean={stats_dict['mean']:.2f} ± {stats_dict['std']:.2f}  "
            f"[{stats_dict['min']:.1f} – {stats_dict['max']:.1f}]"
        )

    # Pairwise mean differences
    group_means = {k: v["mean"] for k, v in result["groups"].items()}
    for i, cls_a in enumerate(CLASS_ORDER):
        for cls_b in CLASS_ORDER[i + 1 :]:
            if cls_a in group_means and cls_b in group_means:
                diff = round(abs(group_means[cls_a] - group_means[cls_b]), 3)
                key = f"{cls_a}_vs_{cls_b}"
                result["separation"][key] = diff
                print(f"  Separation {cls_a} vs {cls_b}: Δmean = {diff:.3f}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2. Classification Metrics
# ──────────────────────────────────────────────────────────────────────────────


def classification_metrics(
    df: pd.DataFrame,
    output_dir: str,
) -> Dict:
    """Compute and save classification performance metrics."""
    print("\n" + "=" * 60)
    print("2. CLASSIFICATION METRICS")
    print("=" * 60)

    if "actual_diagnosis" not in df.columns or "predicted_diagnosis" not in df.columns:
        print("  ⚠ Required columns missing – skipping classification metrics.")
        return {}

    y_true = df["actual_diagnosis"].values
    y_pred = df["predicted_diagnosis"].values

    # Filter to valid labels only
    valid_mask = np.isin(y_true, CLASS_ORDER) & np.isin(y_pred, CLASS_ORDER)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]

    if len(y_true) == 0:
        print("  ⚠ No valid labelled samples found.")
        return {}

    present_labels = [c for c in CLASS_ORDER if c in y_true]

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=present_labels, average="weighted", zero_division=0)

    result = {
        "accuracy": float(round(acc, 4)),
        "balanced_accuracy": float(round(bal_acc, 4)),
        "precision_weighted": float(round(prec, 4)),
        "recall_weighted": float(round(rec, 4)),
        "f1_weighted": float(round(f1, 4)),
        "n_samples": int(len(y_true)),
    }

    print(f"  Accuracy:          {acc * 100:.2f}%")
    print(f"  Balanced accuracy: {bal_acc * 100:.2f}%")
    print(f"  Precision (wtd):   {prec:.4f}")
    print(f"  Recall (wtd):      {rec:.4f}")
    print(f"  F1-score (wtd):    {f1:.4f}")

    # ── Per-class metrics ─────────────────────────────────────────────────────
    report_str = classification_report(
        y_true, y_pred, labels=present_labels, zero_division=0
    )
    print("\n  Per-class classification report:")
    for line in report_str.splitlines():
        print(f"    {line}")

    txt_path = os.path.join(output_dir, "classification_report.txt")
    with open(txt_path, "w") as fh:
        fh.write("Bone Health Analysis – Classification Report\n")
        fh.write("=" * 50 + "\n\n")
        fh.write(f"Accuracy:          {acc * 100:.2f}%\n")
        fh.write(f"Balanced accuracy: {bal_acc * 100:.2f}%\n")
        fh.write(f"Precision (wtd):   {prec:.4f}\n")
        fh.write(f"Recall (wtd):      {rec:.4f}\n")
        fh.write(f"F1-score (wtd):    {f1:.4f}\n")
        fh.write(f"Samples:           {len(y_true)}\n\n")
        fh.write("Per-class Report\n")
        fh.write("-" * 50 + "\n")
        fh.write(report_str)
    print(f"\n✓ Saved: {txt_path}")

    # ── Confusion matrix plot ─────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    result["confusion_matrix"] = cm.tolist()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_labels,
        yticklabels=present_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix – Bone Health Classification", fontsize=13)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {cm_path}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 3. Correlation Analysis
# ──────────────────────────────────────────────────────────────────────────────


def _scatter_with_regression(
    x: pd.Series,
    y: pd.Series,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: str,
    hue: Optional[pd.Series] = None,
) -> Dict:
    """Create a scatter plot with linear regression line and return r/p values."""
    mask = x.notna() & y.notna()
    xv = x[mask].values.astype(float)
    yv = y[mask].values.astype(float)

    if len(xv) < 3:
        logger.warning("Not enough data points for regression in %s", title)
        return {}

    pearson_r, pearson_p = stats.pearsonr(xv, yv)
    spearman_r, spearman_p = stats.spearmanr(xv, yv)

    fig, ax = plt.subplots(figsize=(7, 5))

    if hue is not None:
        hue_valid = hue[mask]
        for cls in CLASS_ORDER:
            idx = hue_valid == cls
            if idx.any():
                ax.scatter(
                    xv[idx.values],
                    yv[idx.values],
                    label=cls.capitalize(),
                    color=PALETTE.get(cls, "grey"),
                    alpha=0.65,
                    s=40,
                )
        ax.legend(title="Diagnosis", fontsize=9)
    else:
        ax.scatter(xv, yv, alpha=0.55, s=40, color="#3498db")

    # Regression line
    m, b = np.polyfit(xv, yv, 1)
    x_line = np.linspace(xv.min(), xv.max(), 200)
    ax.plot(x_line, m * x_line + b, "r--", linewidth=1.5, label="Regression")

    annotation = (
        f"Pearson r = {pearson_r:.3f}  (p={pearson_p:.3e})\n"
        f"Spearman ρ = {spearman_r:.3f}  (p={spearman_p:.3e})"
    )
    ax.text(
        0.03, 0.97, annotation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {save_path}")

    return {
        "n": int(len(xv)),
        "pearson_r": float(round(pearson_r, 4)),
        "pearson_p": float(round(pearson_p, 6)),
        "spearman_r": float(round(spearman_r, 4)),
        "spearman_p": float(round(spearman_p, 6)),
    }


def correlation_analysis(df: pd.DataFrame, output_dir: str) -> Dict:
    """Compute correlations and generate scatter + heatmap plots."""
    print("\n" + "=" * 60)
    print("3. CORRELATION ANALYSIS")
    print("=" * 60)

    result: Dict = {}
    hue = df.get("actual_diagnosis") if "actual_diagnosis" in df.columns else None

    # ── Health score vs T-score ───────────────────────────────────────────────
    if "t_score" in df.columns:
        t_corr = _scatter_with_regression(
            x=df["t_score"],
            y=df["health_score"],
            xlabel="T-score",
            ylabel="Health Score (1–10)",
            title="Health Score vs Clinical T-score",
            save_path=os.path.join(output_dir, "score_vs_tscore.png"),
            hue=hue,
        )
        if t_corr:
            result["health_vs_tscore"] = t_corr
            print(
                f"  T-score correlation:  Pearson r={t_corr['pearson_r']:.3f}  "
                f"Spearman ρ={t_corr['spearman_r']:.3f}"
            )
    else:
        print("  ⚠ 't_score' column not found – skipping T-score correlation.")

    # ── Health score vs Z-score ───────────────────────────────────────────────
    if "z_score" in df.columns:
        z_corr = _scatter_with_regression(
            x=df["z_score"],
            y=df["health_score"],
            xlabel="Z-score",
            ylabel="Health Score (1–10)",
            title="Health Score vs Clinical Z-score",
            save_path=os.path.join(output_dir, "score_vs_zscore.png"),
            hue=hue,
        )
        if z_corr:
            result["health_vs_zscore"] = z_corr
            print(
                f"  Z-score correlation:  Pearson r={z_corr['pearson_r']:.3f}  "
                f"Spearman ρ={z_corr['spearman_r']:.3f}"
            )
    else:
        print("  ⚠ 'z_score' column not found – skipping Z-score correlation.")

    # ── Feature correlation heatmap ───────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep only meaningful columns for the heatmap (avoid dozens of z-score columns)
    preferred = [
        "health_score", "composite_deviation_std", "age",
        "t_score", "z_score",
        "zscore_t_score", "zscore_z_score",
        "zscore_bone_density_mean", "zscore_joint_space_width",
        "zscore_cortical_thickness", "zscore_bmi",
    ]
    heatmap_cols = [c for c in preferred if c in numeric_cols]
    if len(heatmap_cols) >= 3:
        corr_matrix = df[heatmap_cols].corr()
        fig, ax = plt.subplots(figsize=(max(8, len(heatmap_cols)), max(6, len(heatmap_cols) - 1)))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title("Feature Correlation Matrix", fontsize=13)
        plt.tight_layout()
        hm_path = os.path.join(output_dir, "correlation_heatmap.png")
        fig.savefig(hm_path, dpi=150)
        plt.close(fig)
        print(f"✓ Saved: {hm_path}")

        result["heatmap_columns"] = heatmap_cols

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. System Reliability
# ──────────────────────────────────────────────────────────────────────────────


def system_reliability(df: pd.DataFrame) -> Dict:
    """Compute system reliability statistics."""
    print("\n" + "=" * 60)
    print("4. SYSTEM RELIABILITY")
    print("=" * 60)

    total = len(df)
    result: Dict = {"total_patients": total}

    # Success rate: patients with a valid health score (1–10)
    valid_scores = df["health_score"].between(1, 10, inclusive="both")
    success_count = int(valid_scores.sum())
    result["success_rate"] = float(round(success_count / total, 4)) if total > 0 else 0.0
    result["successful_patients"] = success_count
    print(f"  Total patients:    {total}")
    print(f"  Valid scores:      {success_count}  ({result['success_rate'] * 100:.1f}%)")

    # Fallback baseline usage
    if "baseline_stratum" in df.columns:
        fallback_mask = df["baseline_stratum"].astype(str).str.contains(
            r"all_genders|all_ages|global", na=False
        )
        fallback_count = int(fallback_mask.sum())
        result["fallback_baseline_count"] = fallback_count
        result["fallback_baseline_rate"] = float(
            round(fallback_count / total, 4) if total > 0 else 0.0
        )
        print(
            f"  Fallback baseline: {fallback_count}  "
            f"({result['fallback_baseline_rate'] * 100:.1f}%)"
        )

    # Score distribution summary
    result["score_stats"] = {
        "mean": float(round(df["health_score"].mean(), 3)),
        "std": float(round(df["health_score"].std(), 3)),
        "min": int(df["health_score"].min()),
        "max": int(df["health_score"].max()),
        "median": float(df["health_score"].median()),
    }
    s = result["score_stats"]
    print(
        f"  Score stats:       mean={s['mean']:.2f} ± {s['std']:.2f}  "
        f"[{s['min']}–{s['max']}]"
    )

    # Missing data
    missing = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}
    result["missing_values"] = missing
    if missing:
        print(f"  Missing values:    {sum(missing.values())} cells across {len(missing)} columns")
    else:
        print("  Missing values:    None")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Visualisations
# ──────────────────────────────────────────────────────────────────────────────


def plot_score_distribution_by_diagnosis(df: pd.DataFrame, output_dir: str) -> None:
    """Save a boxplot of health score distribution per diagnosis group."""
    if "actual_diagnosis" not in df.columns:
        print("  ⚠ 'actual_diagnosis' missing – skipping distribution plot.")
        return

    present = [c for c in CLASS_ORDER if c in df["actual_diagnosis"].values]
    if not present:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df[df["actual_diagnosis"].isin(present)].copy()
    # Keep CLASS_ORDER ordering
    plot_df["actual_diagnosis"] = pd.Categorical(
        plot_df["actual_diagnosis"], categories=present, ordered=True
    )
    plot_df = plot_df.sort_values("actual_diagnosis")

    sns.boxplot(
        data=plot_df,
        x="actual_diagnosis",
        y="health_score",
        hue="actual_diagnosis",
        palette=PALETTE,
        order=present,
        ax=ax,
        width=0.5,
        flierprops={"marker": "o", "alpha": 0.5, "markersize": 4},
        legend=False,
    )
    sns.stripplot(
        data=plot_df,
        x="actual_diagnosis",
        y="health_score",
        hue="actual_diagnosis",
        palette=PALETTE,
        order=present,
        ax=ax,
        size=3,
        alpha=0.35,
        jitter=True,
        legend=False,
    )
    ax.set_xlabel("Diagnosis", fontsize=12)
    ax.set_ylabel("Health Score (1–10)", fontsize=12)
    ax.set_title("Health Score Distribution by Diagnosis", fontsize=13)
    ax.set_ylim(0, 11)
    ax.axhline(DEFAULT_NORMAL_THRESHOLD, color="green", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(DEFAULT_OSTEOPENIA_THRESHOLD, color="orange", linestyle="--", linewidth=1, alpha=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, "score_distribution_by_diagnosis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"✓ Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────────────────────


def save_metrics_json(metrics: Dict, output_dir: str) -> None:
    """Persist all numeric results to a JSON file."""
    path = os.path.join(output_dir, "performance_metrics_summary.json")

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=4, default=_convert)
    print(f"✓ Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="performance_grader.py",
        description="Evaluate performance metrics for the Bone Health Analysis system",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory containing health_scores.xlsx/.csv and where plots will be saved",
    )
    parser.add_argument(
        "--score-thresholds",
        default=f"{DEFAULT_NORMAL_THRESHOLD},{DEFAULT_OSTEOPENIA_THRESHOLD}",
        help=(
            "Comma-separated thresholds for Normal,Osteopenia boundaries "
            f"(default: {DEFAULT_NORMAL_THRESHOLD},{DEFAULT_OSTEOPENIA_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    # Parse thresholds
    try:
        normal_thr, osteopenia_thr = [float(x) for x in args.score_thresholds.split(",")]
    except ValueError:
        print(
            f"❌ Invalid --score-thresholds '{args.score_thresholds}'.  "
            "Expected two comma-separated numbers, e.g. '8,6'",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    t_start = time.time()

    print("\n" + "=" * 60)
    print("   BONE HEALTH ANALYSIS – PERFORMANCE GRADER")
    print("=" * 60)
    print(f"  Score thresholds:  Normal≥{normal_thr}  Osteopenia≥{osteopenia_thr}")

    # ── Load data ─────────────────────────────────────────────────────────────
    try:
        df = load_health_scores(args.output_dir)
    except FileNotFoundError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        sys.exit(1)

    df = assign_predicted_classes(df, normal_thr, osteopenia_thr)

    # ── Run all four analyses ─────────────────────────────────────────────────
    all_metrics: Dict = {
        "score_thresholds": {
            "normal": normal_thr,
            "osteopenia": osteopenia_thr,
            "osteoporosis_below": osteopenia_thr,
        }
    }

    all_metrics["group_separation"] = group_separation_analysis(df)
    all_metrics["classification"] = classification_metrics(df, args.output_dir)
    all_metrics["correlation"] = correlation_analysis(df, args.output_dir)
    all_metrics["reliability"] = system_reliability(df)

    # ── Extra visualisation: boxplot ──────────────────────────────────────────
    print("\nGenerating plots...")
    plot_score_distribution_by_diagnosis(df, args.output_dir)

    # ── Persist results ───────────────────────────────────────────────────────
    all_metrics["runtime_seconds"] = round(time.time() - t_start, 2)
    save_metrics_json(all_metrics, args.output_dir)

    elapsed = time.time() - t_start
    print(f"\n✅ Performance grading complete in {elapsed:.1f}s")
    print(f"   All outputs saved to: {args.output_dir}\n")


if __name__ == "__main__":
    main()
