"""
threshold_optimizer.py - ROC-AUC based threshold optimisation for bone health classification.

Finds optimal decision thresholds for each class (Normal / Osteopenia /
Osteoporosis) using Youden's J statistic on predicted probability scores.

Algorithm
---------
1. Compute one-vs-rest ROC curves for each class.
2. Find the threshold that maximises Youden's J = Sensitivity + Specificity - 1.
3. Compare classification accuracy with old vs new thresholds.
4. Visualise ROC curves and optimal threshold points.

Usage
-----
    from threshold_optimizer import ThresholdOptimizer
    opt = ThresholdOptimizer(class_names=["normal", "osteopenia", "osteoporosis"])
    opt.fit(y_true, y_scores)         # y_scores: (n_samples, 3) probability matrix
    y_pred = opt.predict(y_scores)    # classify using optimal thresholds
    opt.plot_roc_curves(save_path="output/roc_curves.png")
"""

from __future__ import annotations

import json
import logging
import os
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
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Default class names and legacy thresholds
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CLASSES = ["normal", "osteopenia", "osteoporosis"]

# Health scores → class mapping (existing hardcoded thresholds from config.py)
DEFAULT_SCORE_THRESHOLDS = {
    "normal": 8.0,      # score >= 8.0
    "osteopenia": 6.0,  # 6.0 <= score < 8.0
    # osteoporosis: score < 6.0
}


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────


def scores_to_probs(health_scores: np.ndarray) -> np.ndarray:
    """Convert 1-10 health scores into soft class probabilities.

    Maps each integer health score to a 3-dimensional probability vector
    [P(normal), P(osteopenia), P(osteoporosis)] using a soft Gaussian
    assignment so that the probabilities sum to 1 and vary smoothly.

    Parameters
    ----------
    health_scores:
        1-D array of health scores in range [1, 10].

    Returns
    -------
    ndarray of shape (n_samples, 3).
    """
    # Gaussian centres for each class (on the 1-10 scale)
    centres = np.array([8.5, 6.5, 4.0])  # normal, osteopenia, osteoporosis
    sigma = 1.5

    scores = np.asarray(health_scores, dtype=float).reshape(-1, 1)
    # Unnormalised Gaussian responses
    responses = np.exp(-0.5 * ((scores - centres) / sigma) ** 2)
    # Normalise to sum to 1 per row
    probs = responses / responses.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def health_score_to_class_legacy(score: float) -> str:
    """Classify a health score using the legacy hardcoded thresholds."""
    if score >= DEFAULT_SCORE_THRESHOLDS["normal"]:
        return "normal"
    if score >= DEFAULT_SCORE_THRESHOLDS["osteopenia"]:
        return "osteopenia"
    return "osteoporosis"


# ──────────────────────────────────────────────────────────────────────────────
# ThresholdOptimizer
# ──────────────────────────────────────────────────────────────────────────────


class ThresholdOptimizer:
    """Find optimal classification thresholds via ROC-AUC analysis.

    Parameters
    ----------
    class_names:
        Ordered list of class labels.  Must match column order of probability
        matrices passed to ``fit`` / ``predict``.
    """

    def __init__(self, class_names: List[str] = DEFAULT_CLASSES) -> None:
        self.class_names = class_names
        self.n_classes = len(class_names)
        # Filled by fit()
        self.fpr_: Dict[str, np.ndarray] = {}
        self.tpr_: Dict[str, np.ndarray] = {}
        self.thresholds_roc_: Dict[str, np.ndarray] = {}
        self.optimal_thresholds_: Dict[str, float] = {}
        self.auc_scores_: Dict[str, float] = {}
        self._fitted = False

    # ── fit / predict ─────────────────────────────────────────────────────────

    def fit(self, y_true: np.ndarray, y_scores: np.ndarray) -> "ThresholdOptimizer":
        """Compute ROC curves and optimal thresholds via Youden's J.

        Parameters
        ----------
        y_true:
            1-D array of integer class indices (0 = normal, 1 = osteopenia,
            2 = osteoporosis) or string labels matching ``class_names``.
        y_scores:
            2-D array of shape (n_samples, n_classes) with predicted
            probabilities.

        Returns
        -------
        self (fitted).
        """
        y_true_idx = self._to_indices(y_true)
        y_bin = label_binarize(y_true_idx, classes=list(range(self.n_classes)))

        for i, cls in enumerate(self.class_names):
            fpr, tpr, thresholds = roc_curve(y_bin[:, i], y_scores[:, i])
            auc = roc_auc_score(y_bin[:, i], y_scores[:, i])

            # Youden's J statistic: maximise sensitivity + specificity
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            optimal_thresh = float(thresholds[best_idx])

            self.fpr_[cls] = fpr
            self.tpr_[cls] = tpr
            self.thresholds_roc_[cls] = thresholds
            self.optimal_thresholds_[cls] = optimal_thresh
            self.auc_scores_[cls] = float(auc)

            logger.info(
                "Class %-15s  AUC=%.3f  optimal_threshold=%.3f  "
                "sensitivity=%.3f  specificity=%.3f",
                cls,
                auc,
                optimal_thresh,
                tpr[best_idx],
                1.0 - fpr[best_idx],
            )

        self._fitted = True
        return self

    def predict(self, y_scores: np.ndarray) -> np.ndarray:
        """Classify samples using optimal per-class thresholds.

        Uses a two-step approach:
        1. Find classes whose threshold is exceeded.
        2. Among those, pick the one with the highest probability.
        If no threshold is exceeded, pick the argmax class.

        Parameters
        ----------
        y_scores:
            2-D array of shape (n_samples, n_classes).

        Returns
        -------
        1-D integer array of predicted class indices.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        thresholds = np.array(
            [self.optimal_thresholds_.get(cls, 0.5) for cls in self.class_names]
        )
        # Mask: which classes exceed their threshold
        exceeds = y_scores >= thresholds[np.newaxis, :]  # (n, n_classes)
        # For rows where at least one class exceeds → argmax within those
        # For rows where none exceeds → plain argmax
        preds = np.argmax(y_scores, axis=1)
        for i in range(len(y_scores)):
            active = np.where(exceeds[i])[0]
            if len(active) > 0:
                preds[i] = active[np.argmax(y_scores[i, active])]
        return preds

    def predict_labels(self, y_scores: np.ndarray) -> List[str]:
        """Like ``predict`` but returns string class names."""
        idx = self.predict(y_scores)
        return [self.class_names[i] for i in idx]

    # ── Validation helpers ────────────────────────────────────────────────────

    def compare_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compare classification accuracy with old vs new thresholds.

        Parameters
        ----------
        y_true:
            Ground-truth class labels (indices or strings).
        y_scores:
            (n_samples, n_classes) probability matrix.
        health_scores:
            Optional 1-D array of health scores for legacy classification.

        Returns
        -------
        Dict with 'old' and 'new' accuracy and per-class metrics.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before compare_thresholds().")

        y_true_idx = self._to_indices(y_true)
        y_pred_new = self.predict(y_scores)

        report_new = classification_report(
            y_true_idx, y_pred_new,
            target_names=self.class_names,
            output_dict=True,
        )

        result: Dict = {
            "new": {
                "accuracy": accuracy_score(y_true_idx, y_pred_new),
                "classification_report": report_new,
                "optimal_thresholds": self.optimal_thresholds_,
                "auc_scores": self.auc_scores_,
            }
        }

        # Legacy (old) thresholds using health scores if provided
        if health_scores is not None:
            y_pred_old = np.array(
                [self._label_to_idx(health_score_to_class_legacy(s))
                 for s in health_scores]
            )
            report_old = classification_report(
                y_true_idx, y_pred_old,
                target_names=self.class_names,
                output_dict=True,
            )
            result["old"] = {
                "accuracy": accuracy_score(y_true_idx, y_pred_old),
                "classification_report": report_old,
                "score_thresholds": DEFAULT_SCORE_THRESHOLDS,
            }

        # Fallback "old" using argmax (no threshold optimisation)
        if "old" not in result:
            y_pred_base = np.argmax(y_scores, axis=1)
            report_base = classification_report(
                y_true_idx, y_pred_base,
                target_names=self.class_names,
                output_dict=True,
            )
            result["old"] = {
                "accuracy": accuracy_score(y_true_idx, y_pred_base),
                "classification_report": report_base,
                "note": "Baseline argmax (no threshold optimisation)",
            }

        acc_old = result["old"]["accuracy"]
        acc_new = result["new"]["accuracy"]
        logger.info(
            "Threshold comparison: old_accuracy=%.3f  new_accuracy=%.3f  delta=%.3f",
            acc_old, acc_new, acc_new - acc_old,
        )
        return result

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_roc_curves(self, save_path: Optional[str] = None) -> str:
        """Plot one-vs-rest ROC curves with optimal threshold markers.

        Parameters
        ----------
        save_path:
            File path for saving the figure (PNG).  If None, a default path
            in the current directory is used.

        Returns
        -------
        Absolute path to the saved figure.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before plot_roc_curves().")

        if save_path is None:
            save_path = "roc_curves.png"

        fig, axes = plt.subplots(1, self.n_classes, figsize=(5 * self.n_classes, 5))
        if self.n_classes == 1:
            axes = [axes]

        colours = ["tab:blue", "tab:orange", "tab:green"]

        for i, (cls, ax) in enumerate(zip(self.class_names, axes)):
            colour = colours[i % len(colours)]
            fpr = self.fpr_[cls]
            tpr = self.tpr_[cls]
            auc = self.auc_scores_[cls]
            opt_thresh = self.optimal_thresholds_[cls]

            ax.plot(fpr, tpr, colour, lw=2, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

            # Mark optimal threshold point
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))
            ax.scatter(
                fpr[best_idx], tpr[best_idx],
                marker="*", s=200, color=colour,
                zorder=5, label=f"Optimal threshold = {opt_thresh:.3f}",
            )

            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=11)
            ax.set_ylabel("True Positive Rate", fontsize=11)
            ax.set_title(f"ROC – {cls.capitalize()}", fontsize=13)
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        Path(os.path.abspath(save_path)).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("ROC curves saved to %s", save_path)
        return os.path.abspath(save_path)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save optimal thresholds and AUC scores to a JSON file."""
        data = {
            "class_names": self.class_names,
            "optimal_thresholds": self.optimal_thresholds_,
            "auc_scores": self.auc_scores_,
        }
        Path(os.path.abspath(path)).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("ThresholdOptimizer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ThresholdOptimizer":
        """Load a previously saved ThresholdOptimizer from JSON."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        opt = cls(class_names=data["class_names"])
        opt.optimal_thresholds_ = data["optimal_thresholds"]
        opt.auc_scores_ = data["auc_scores"]
        opt._fitted = True
        return opt

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_indices(self, y: np.ndarray) -> np.ndarray:
        """Convert string labels or integer indices to a consistent int array."""
        y = np.asarray(y)
        if y.dtype.kind in ("U", "S", "O"):  # string-like
            mapping = {cls: i for i, cls in enumerate(self.class_names)}
            return np.array([mapping.get(str(label).lower(), 0) for label in y])
        return y.astype(int)

    def _label_to_idx(self, label: str) -> int:
        mapping = {cls: i for i, cls in enumerate(self.class_names)}
        return mapping.get(label.lower(), 0)
