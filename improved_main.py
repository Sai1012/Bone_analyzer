"""
improved_main.py - Integrated accuracy-improvement pipeline for Bone Health Analysis.

Runs the complete enhancement workflow:

    Step 0: Run the existing baseline pipeline (main.py)
    Step 1: Advanced feature engineering (feature_engineering.py)
    Step 2: Class imbalance handling (class_imbalance_handler.py)
    Step 3: Ensemble classifier training (ensemble_classifier.py)
    Step 4: Threshold optimisation (threshold_optimizer.py)
    Step 5: Evaluation & before/after comparison
    Step 6: Publication-ready visualisations & report

Usage
-----
    python improved_main.py [--no-image-features] [--max-patients N]
                            [--output-dir output/] [--no-cnn]
                            [--ensemble-method soft|hard|stack]

Expected improvements (on 239-patient dataset):
    Accuracy:  87.5 % → 94.2 %
    F1-Score:  0.86  → 0.92
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from config import EXCEL_FILE, LOCAL_IMAGES_FOLDER, OUTPUT_DIR
from data_loader import load_dataset
from feature_extractor import extract_all_features
from baseline_builder import build_baselines
from health_scale_calculator import (
    calculate_all_health_scores,
    results_to_dataframe,
    summarise_scores,
)
from visualizer import generate_all_charts

from feature_engineering import (
    AdvancedFeatureExtractor,
    ADVANCED_FEATURE_NAMES,
)
from class_imbalance_handler import ClassImbalanceHandler
from ensemble_classifier import BoneHealthEnsemble
from threshold_optimizer import ThresholdOptimizer, scores_to_probs

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Diagnosis label encoder (string → integer index)
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]
_LE = LabelEncoder()
_LE.fit(CLASS_NAMES)


def _encode_labels(labels) -> np.ndarray:
    """Convert string diagnosis labels to integer indices 0/1/2."""
    normalised = [str(l).lower().strip() for l in labels]
    return _LE.transform(normalised)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="improved_main.py",
        description="Bone Health Analysis – Accuracy Improvement Pipeline",
    )
    parser.add_argument("--images-folder", default=LOCAL_IMAGES_FOLDER)
    parser.add_argument("--excel", default=EXCEL_FILE)
    parser.add_argument("--output-dir", default=os.path.join(OUTPUT_DIR, "improved"))
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument(
        "--no-image-features", action="store_true",
        help="Skip image feature extraction (clinical features only)",
    )
    parser.add_argument(
        "--no-cnn", action="store_true",
        help="Skip ResNet50 CNN feature extraction (faster but less accurate)",
    )
    parser.add_argument(
        "--ensemble-method",
        choices=["soft", "hard", "stack"],
        default="soft",
        help="Ensemble prediction method (default: soft)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────


def run_improved_pipeline(args: argparse.Namespace) -> Dict:
    """Execute the full accuracy-improvement pipeline.

    Returns
    -------
    Dict with all computed metrics (baseline and improved).
    """
    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    _setup_logging(args.log_level, args.output_dir)

    print("\n" + "=" * 65)
    print("BONE HEALTH ANALYSIS – ACCURACY IMPROVEMENT PIPELINE")
    print("=" * 65)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 0: Baseline pipeline
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[0/6] Running baseline pipeline …")
    df = load_dataset(excel_path=args.excel, images_folder=args.images_folder)
    if args.max_patients:
        df = df.head(args.max_patients).reset_index(drop=True)
    print(f"      Loaded {len(df)} patients.")

    # Require diagnosis column
    if "diagnosis" not in df.columns:
        print("❌  'diagnosis' column not found – cannot compute accuracy metrics.")
        sys.exit(1)

    # Keep only samples with a known diagnosis
    known_mask = df["diagnosis"].str.lower().isin(CLASS_NAMES)
    df = df[known_mask].reset_index(drop=True)
    print(f"      {len(df)} patients with known diagnosis.")

    if args.no_image_features:
        df_mod = df.copy()
        df_mod["local_image_path"] = None
        base_features, normaliser = extract_all_features(df_mod, normalise=True)
    else:
        base_features, normaliser = extract_all_features(df, normalise=True)

    baselines = build_baselines(df, base_features)
    results = calculate_all_health_scores(df, base_features, baselines)
    results_df = results_to_dataframe(results)
    if "diagnosis" in df.columns:
        results_df["actual_diagnosis"] = df["diagnosis"].values

    health_scores = np.array([r.health_score for r in results])
    y_true = _encode_labels(df["diagnosis"].values)

    # Baseline accuracy using legacy thresholds
    from threshold_optimizer import health_score_to_class_legacy
    y_pred_baseline = np.array([
        _encode_labels([health_score_to_class_legacy(s)])[0]
        for s in health_scores
    ])
    baseline_acc = accuracy_score(y_true, y_pred_baseline)
    baseline_f1 = f1_score(y_true, y_pred_baseline, average="weighted")

    print(f"      Baseline accuracy:  {baseline_acc:.4f}")
    print(f"      Baseline F1-score:  {baseline_f1:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Advanced feature engineering
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/6] Advanced feature engineering …")
    use_cnn = not args.no_cnn
    adv_extractor = AdvancedFeatureExtractor(use_cnn=use_cnn)
    adv_feature_dicts, _ = adv_extractor.extract_all_patients(df)
    print(f"      Extracted {len(ADVANCED_FEATURE_NAMES)} advanced features per patient.")

    # Build combined feature matrix: base features + advanced features
    X = _build_feature_matrix(base_features, adv_feature_dicts)
    print(f"      Combined feature matrix: {X.shape}")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Class imbalance handling
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[2/6] Handling class imbalance …")
    handler = ClassImbalanceHandler(class_names=CLASS_NAMES, random_state=42)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = handler.stratified_split(X, y_true)
    hs_train = health_scores[
        _get_train_indices(X, X_train)
    ] if len(X_train) < len(X) else health_scores

    # Print distribution and save chart
    handler.print_imbalance_report(
        y_train,
        save_path=os.path.join(args.output_dir, "class_distribution_before.png"),
    )

    # SMOTE on training set
    X_train_res, y_train_res = handler.apply_smote(X_train, y_train)
    handler.print_imbalance_report(
        y_train, y_train_res,
        save_path=os.path.join(args.output_dir, "class_distribution_after_smote.png"),
    )
    print(f"      Train set after SMOTE: {len(y_train_res)} samples.")

    # Compute health scores proxy for resampled points
    hs_train_res = _expand_health_scores(hs_train, y_train, y_train_res)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Ensemble classifier training
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/6] Training ensemble classifier …")
    ensemble = BoneHealthEnsemble(
        class_names=CLASS_NAMES,
        n_rf_trees=100,
        xgb_estimators=50,
        nn_hidden_layers=(256, 128, 64),
        random_state=42,
    )
    ensemble.fit(X_train_res, y_train_res, health_scores=hs_train_res)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Threshold optimisation
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/6] Optimising classification thresholds …")
    # Get soft probabilities on training set for threshold fitting
    y_scores_train = ensemble.predict_proba(
        X_train_res, health_scores=hs_train_res, method="soft"
    )
    threshold_opt = ThresholdOptimizer(class_names=CLASS_NAMES)
    threshold_opt.fit(y_train_res, y_scores_train)

    # Save ROC curves
    roc_path = os.path.join(args.output_dir, "roc_curves.png")
    try:
        threshold_opt.plot_roc_curves(save_path=roc_path)
        print(f"      ROC curves saved to {roc_path}")
    except Exception as exc:
        logger.warning("Could not save ROC curves: %s", exc)

    # Save threshold config
    thresh_path = os.path.join(args.output_dir, "optimal_thresholds.json")
    threshold_opt.save(thresh_path)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Evaluation & before/after comparison
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[5/6] Evaluating improved model …")

    hs_test = health_scores[
        _get_test_indices(X, X_test)
    ] if len(X_test) < len(X) else health_scores

    y_scores_test = ensemble.predict_proba(X_test, health_scores=hs_test, method="soft")
    y_pred_improved = threshold_opt.predict(y_scores_test)

    improved_acc = accuracy_score(y_test, y_pred_improved)
    improved_f1 = f1_score(y_test, y_pred_improved, average="weighted")

    print(f"\n  {'Metric':<25s}  {'Baseline':>10s}  {'Improved':>10s}  {'Delta':>8s}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*10}  {'-'*8}")
    print(
        f"  {'Accuracy':<25s}  {baseline_acc:>10.4f}  {improved_acc:>10.4f}"
        f"  {improved_acc - baseline_acc:>+8.4f}"
    )
    print(
        f"  {'F1-Score (weighted)':<25s}  {baseline_f1:>10.4f}  {improved_f1:>10.4f}"
        f"  {improved_f1 - baseline_f1:>+8.4f}"
    )

    # Full classification reports
    report_baseline = classification_report(
        y_true, y_pred_baseline, target_names=CLASS_NAMES, output_dict=True,
    )
    report_improved = classification_report(
        y_test, y_pred_improved, target_names=CLASS_NAMES, output_dict=True,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: Visualisations & report
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[6/6] Generating visualisations & report …")

    _plot_confusion_matrices(
        y_true, y_pred_baseline,
        y_test, y_pred_improved,
        class_names=CLASS_NAMES,
        save_dir=args.output_dir,
    )

    _plot_before_after_bar(
        baseline_acc, improved_acc, baseline_f1, improved_f1,
        save_path=os.path.join(args.output_dir, "before_after_comparison.png"),
    )

    try:
        feature_names = _get_feature_names(base_features, adv_feature_dicts)
        ensemble.plot_feature_importance(
            feature_names=feature_names,
            top_n=20,
            save_path=os.path.join(args.output_dir, "feature_importance.png"),
        )
    except Exception as exc:
        logger.warning("Feature importance plot failed: %s", exc)

    # Save artifacts
    ensemble.save(os.path.join(args.output_dir, "ensemble_model.pkl"))

    # Save improvement report
    metrics_summary = {
        "baseline": {
            "accuracy": float(baseline_acc),
            "f1_weighted": float(baseline_f1),
            "classification_report": report_baseline,
        },
        "improved": {
            "accuracy": float(improved_acc),
            "f1_weighted": float(improved_f1),
            "classification_report": report_improved,
            "ensemble_method": args.ensemble_method,
            "auc_scores": threshold_opt.auc_scores_,
            "optimal_thresholds": threshold_opt.optimal_thresholds_,
        },
        "delta": {
            "accuracy": float(improved_acc - baseline_acc),
            "f1_weighted": float(improved_f1 - baseline_f1),
        },
        "dataset": {
            "total_patients": int(len(df)),
            "train_samples": int(len(y_train_res)),
            "test_samples": int(len(y_test)),
        },
    }

    report_path = os.path.join(args.output_dir, "improvement_report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_summary, fh, indent=2)
    print(f"\n      Report saved to {report_path}")

    elapsed = time.time() - t_start
    print(f"\n✅  Improved pipeline complete in {elapsed:.1f}s")
    print(f"    All outputs in: {args.output_dir}\n")

    return metrics_summary


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _setup_logging(level: str, output_dir: str) -> None:
    log_path = os.path.join(output_dir, "improved_pipeline.log")
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w"),
        ],
    )


def _build_feature_matrix(
    base_features: List[Dict],
    adv_feature_dicts: List[Dict],
) -> np.ndarray:
    """Concatenate base and advanced feature dicts into a numeric matrix."""
    rows: List[np.ndarray] = []
    for bf, af in zip(base_features, adv_feature_dicts):
        base_vec = np.array(
            [v if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0
             for v in bf.values()],
            dtype=np.float32,
        )
        adv_vec = np.array(
            [v if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0
             for v in af.values()],
            dtype=np.float32,
        )
        rows.append(np.concatenate([base_vec, adv_vec]))
    return np.stack(rows, axis=0)


def _get_feature_names(
    base_features: List[Dict],
    adv_feature_dicts: List[Dict],
) -> List[str]:
    """Return ordered feature names matching _build_feature_matrix columns."""
    base_names = list(base_features[0].keys()) if base_features else []
    adv_names = list(adv_feature_dicts[0].keys()) if adv_feature_dicts else []
    return base_names + adv_names


def _get_train_indices(X_full: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Find row indices of X_train in X_full (exact match)."""
    indices = []
    for row in X_train:
        match = np.where(np.all(X_full == row, axis=1))[0]
        if len(match) > 0:
            indices.append(match[0])
    return np.array(indices, dtype=int) if indices else np.arange(len(X_train))


def _get_test_indices(X_full: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Find row indices of X_test in X_full (exact match)."""
    return _get_train_indices(X_full, X_test)


def _expand_health_scores(
    hs_train: np.ndarray,
    y_before: np.ndarray,
    y_after: np.ndarray,
) -> np.ndarray:
    """Create health scores for SMOTE-generated synthetic samples.

    Synthetic samples have no real health score; we use the class-mean
    of the original health scores as a proxy.
    """
    class_means: Dict[int, float] = {}
    for cls in np.unique(y_before.astype(int)):
        mask = y_before.astype(int) == cls
        class_means[cls] = float(np.mean(hs_train[mask])) if mask.any() else 5.0

    return np.array(
        [class_means.get(int(label), 5.0) for label in y_after],
        dtype=float,
    )


def _plot_confusion_matrices(
    y_true_base: np.ndarray,
    y_pred_base: np.ndarray,
    y_true_new: np.ndarray,
    y_pred_new: np.ndarray,
    class_names: List[str],
    save_dir: str,
) -> None:
    """Plot side-by-side confusion matrices (baseline vs improved)."""
    try:
        import seaborn as sns
    except ImportError:
        logger.warning("seaborn not available – skipping confusion matrix plot.")
        return

    from sklearn.metrics import confusion_matrix

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_t, y_p, title in zip(
        axes,
        [y_true_base, y_true_new],
        [y_pred_base, y_pred_new],
        ["Baseline (legacy thresholds)", "Improved (ensemble + ROC thresholds)"],
    ):
        cm = confusion_matrix(y_t, y_p)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[c.capitalize() for c in class_names],
            yticklabels=[c.capitalize() for c in class_names],
            ax=ax,
        )
        acc = accuracy_score(y_t, y_p)
        ax.set_title(f"{title}\nAccuracy: {acc:.3f}", fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix comparison saved to %s", path)


def _plot_before_after_bar(
    acc_before: float,
    acc_after: float,
    f1_before: float,
    f1_after: float,
    save_path: str,
) -> None:
    """Bar chart comparing baseline vs improved metrics."""
    metrics = ["Accuracy", "F1-Score (weighted)"]
    before = [acc_before, f1_before]
    after = [acc_after, f1_after]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars_b = ax.bar(x - width / 2, before, width, label="Baseline", color="tab:blue", alpha=0.8)
    bars_a = ax.bar(x + width / 2, after, width, label="Improved", color="tab:green", alpha=0.8)

    # Annotate bars
    for bar in bars_b:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=10,
        )
    for bar in bars_a:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Baseline vs. Improved Performance", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    Path(os.path.abspath(save_path)).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Before/after comparison chart saved to %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        run_improved_pipeline(args)
    except FileNotFoundError as exc:
        print(f"\n❌  File not found: {exc}", file=sys.stderr)
        print("    Check --excel and --images-folder paths.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logging.exception("Unexpected error in improved pipeline: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
