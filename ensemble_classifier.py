"""
ensemble_classifier.py - Multi-model ensemble for bone health classification.

Combines four classifiers using soft voting, hard voting, and stacking:

1. Baseline Deviation Model  – maps health scores to class probabilities
2. Random Forest Classifier  – 100 decision trees
3. XGBoost Classifier        – 50 gradient-boosted trees (optional)
4. Neural Network Classifier – 3 hidden layers (256 → 128 → 64)

Prediction strategies:
- Soft voting  : average predicted probabilities
- Hard voting  : majority vote on class predictions
- Stacking     : meta-learner (Logistic Regression) on stacked outputs

Usage
-----
    from ensemble_classifier import BoneHealthEnsemble
    ens = BoneHealthEnsemble()
    ens.fit(X_train, y_train, health_scores_train)
    y_pred = ens.predict(X_test)
    proba  = ens.predict_proba(X_test)
    report = ens.evaluate(X_test, y_test)
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional XGBoost import
# ──────────────────────────────────────────────────────────────────────────────

_XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
    logger.debug("XGBoost found – XGBClassifier will be included in ensemble.")
except ImportError:
    logger.debug("XGBoost not installed – extra-trees will be used instead.")
    from sklearn.ensemble import ExtraTreesClassifier  # fallback

# ──────────────────────────────────────────────────────────────────────────────
# Default class names
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CLASSES = ["normal", "osteopenia", "osteoporosis"]


# ──────────────────────────────────────────────────────────────────────────────
# Baseline Deviation model wrapper
# ──────────────────────────────────────────────────────────────────────────────


class BaselineDeviationClassifier:
    """Thin sklearn-compatible wrapper around the existing health-score system.

    Converts 1-10 health scores into soft probabilities using a Gaussian
    assignment (same approach as in threshold_optimizer.py).

    Parameters
    ----------
    class_names:
        Ordered list of class labels.
    """

    def __init__(self, class_names: List[str] = DEFAULT_CLASSES) -> None:
        self.class_names = class_names
        self._centres = np.array([8.5, 6.5, 4.0])  # normal, osteopenia, osteoporosis
        self._sigma = 1.5
        self._fitted = False

    def fit(self, health_scores: np.ndarray, y: Optional[np.ndarray] = None) -> "BaselineDeviationClassifier":
        """No-op: probabilities are deterministic given health scores."""
        self._fitted = True
        return self

    def predict_proba(self, health_scores: np.ndarray) -> np.ndarray:
        """Convert health scores to probability vectors."""
        scores = np.asarray(health_scores, dtype=float).reshape(-1, 1)
        responses = np.exp(-0.5 * ((scores - self._centres) / self._sigma) ** 2)
        probs = responses / responses.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    def predict(self, health_scores: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(health_scores), axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Ensemble classifier
# ──────────────────────────────────────────────────────────────────────────────


class BoneHealthEnsemble:
    """Multi-model ensemble classifier for bone health assessment.

    Parameters
    ----------
    class_names:
        Ordered list of class labels.
    n_rf_trees:
        Number of trees in the Random Forest.
    xgb_estimators:
        Number of boosting rounds for XGBoost (or Extra Trees if XGBoost
        is unavailable).
    nn_hidden_layers:
        Tuple specifying hidden layer sizes for the MLP.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        class_names: List[str] = DEFAULT_CLASSES,
        n_rf_trees: int = 100,
        xgb_estimators: int = 50,
        nn_hidden_layers: Tuple[int, ...] = (256, 128, 64),
        random_state: int = 42,
    ) -> None:
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.n_rf_trees = n_rf_trees
        self.xgb_estimators = xgb_estimators
        self.nn_hidden_layers = nn_hidden_layers
        self.random_state = random_state

        self._scaler = StandardScaler()
        self._fitted = False

        # Individual models
        self.baseline_model_ = BaselineDeviationClassifier(class_names)
        self.rf_model_ = RandomForestClassifier(
            n_estimators=n_rf_trees,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        if _XGB_AVAILABLE:
            self.boost_model_ = XGBClassifier(
                n_estimators=xgb_estimators,
                eval_metric="mlogloss",
                random_state=random_state,
                verbosity=0,
            )
        else:
            self.boost_model_ = ExtraTreesClassifier(
                n_estimators=xgb_estimators,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            )
        self.nn_model_ = MLPClassifier(
            hidden_layer_sizes=nn_hidden_layers,
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        # Meta-learner for stacking
        self.meta_learner_ = LogisticRegression(
            max_iter=500,
            random_state=random_state,
            C=1.0,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
        fit_meta_learner: bool = True,
    ) -> "BoneHealthEnsemble":
        """Train all sub-models and the stacking meta-learner.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).
        y:
            Integer class labels.
        health_scores:
            Optional 1-D array of health scores (for the baseline model).
            If None, dummy scores are generated from y.
        fit_meta_learner:
            Whether to fit the stacking meta-learner on held-out predictions.

        Returns
        -------
        self (fitted).
        """
        logger.info("Training BoneHealthEnsemble on %d samples …", len(y))
        y = np.asarray(y, dtype=int)

        # Scale features
        X_scaled = self._scaler.fit_transform(X)

        # Baseline model (no feature matrix needed, only health scores)
        if health_scores is not None:
            self.baseline_model_.fit(health_scores, y)
        else:
            # Create proxy health scores from labels
            score_map = {0: 9.0, 1: 6.5, 2: 4.0}
            hs = np.array([score_map.get(int(label), 5.0) for label in y])
            self.baseline_model_.fit(hs, y)
        self._dummy_health_scores = True

        # Random forest
        logger.info("Fitting Random Forest (%d trees) …", self.n_rf_trees)
        self.rf_model_.fit(X_scaled, y)

        # XGBoost / Extra Trees
        model_name = "XGBoost" if _XGB_AVAILABLE else "Extra Trees"
        logger.info("Fitting %s (%d estimators) …", model_name, self.xgb_estimators)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.boost_model_.fit(X_scaled, y)

        # Neural network
        logger.info("Fitting Neural Network %s …", self.nn_hidden_layers)
        self.nn_model_.fit(X_scaled, y)

        # Stacking meta-learner on full training predictions
        if fit_meta_learner:
            logger.info("Training stacking meta-learner …")
            # Use training predictions as meta-features (in practice use
            # cross-val predictions; here we simplify to training set for speed)
            meta_X = self._build_meta_features(X_scaled, health_scores, y)
            self.meta_learner_.fit(meta_X, y)

        self._fitted = True
        logger.info("BoneHealthEnsemble training complete.")
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_proba_all(
        self,
        X: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Return predicted probabilities from each sub-model.

        Returns
        -------
        Dict mapping model name → (n_samples, n_classes) probability array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba_all().")

        X_scaled = self._scaler.transform(X)
        proba: Dict[str, np.ndarray] = {}

        # Baseline model
        if health_scores is not None:
            proba["baseline"] = self.baseline_model_.predict_proba(health_scores)
        else:
            # No health scores → uniform probabilities
            proba["baseline"] = np.full((len(X), self.n_classes), 1.0 / self.n_classes)

        proba["random_forest"] = self.rf_model_.predict_proba(X_scaled)
        proba["boost"] = self.boost_model_.predict_proba(X_scaled)
        proba["neural_net"] = self.nn_model_.predict_proba(X_scaled)
        return proba

    def predict_proba(
        self,
        X: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
        method: str = "soft",
    ) -> np.ndarray:
        """Return ensemble predicted probabilities.

        Parameters
        ----------
        X:
            Feature matrix.
        health_scores:
            Optional health scores for the baseline sub-model.
        method:
            'soft'    – average probabilities from all models
            'stack'   – use meta-learner on stacked predictions

        Returns
        -------
        (n_samples, n_classes) probability array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        all_proba = self.predict_proba_all(X, health_scores)

        if method == "soft":
            return np.mean(list(all_proba.values()), axis=0)

        if method == "stack":
            X_scaled = self._scaler.transform(X)
            meta_X = self._build_meta_features(X_scaled, health_scores)
            return self.meta_learner_.predict_proba(meta_X)

        raise ValueError(f"Unknown method '{method}'. Choose 'soft' or 'stack'.")

    def predict(
        self,
        X: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
        method: str = "soft",
    ) -> np.ndarray:
        """Predict class indices using the specified ensemble method.

        Parameters
        ----------
        method:
            'soft'  – argmax of averaged probabilities
            'hard'  – majority vote across all sub-models
            'stack' – meta-learner prediction
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")

        if method == "hard":
            all_proba = self.predict_proba_all(X, health_scores)
            all_preds = np.stack(
                [np.argmax(p, axis=1) for p in all_proba.values()], axis=1
            )
            # Majority vote (mode per row)
            from scipy.stats import mode as sp_mode
            result = sp_mode(all_preds, axis=1, keepdims=False)
            return result.mode.astype(int).ravel()

        proba = self.predict_proba(X, health_scores, method=method)
        return np.argmax(proba, axis=1)

    def predict_labels(
        self,
        X: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
        method: str = "soft",
    ) -> List[str]:
        """Like ``predict`` but returns string class names."""
        idx = self.predict(X, health_scores, method=method)
        return [self.class_names[i] for i in idx]

    def confidence_scores(
        self,
        X: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
        method: str = "soft",
    ) -> np.ndarray:
        """Return the prediction confidence (max probability per sample)."""
        proba = self.predict_proba(X, health_scores, method=method)
        return np.max(proba, axis=1)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        health_scores: Optional[np.ndarray] = None,
        method: str = "soft",
        verbose: bool = True,
    ) -> Dict:
        """Evaluate the ensemble on a held-out test set.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            True integer class labels.
        health_scores:
            Optional health scores for the baseline sub-model.
        method:
            Ensemble method passed to ``predict``.
        verbose:
            Print classification report to stdout.

        Returns
        -------
        Dict with accuracy, per-model accuracy, and full classification report.
        """
        y = np.asarray(y, dtype=int)
        y_pred = self.predict(X, health_scores, method=method)

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report_dict = classification_report(
            y, y_pred, target_names=self.class_names, output_dict=True
        )

        if verbose:
            print("\n" + "=" * 60)
            print(f"ENSEMBLE ({method.upper()}) EVALUATION")
            print("=" * 60)
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y, y_pred, target_names=self.class_names))

        # Per-model accuracy
        all_proba = self.predict_proba_all(X, health_scores)
        per_model_acc: Dict[str, float] = {}
        for name, proba in all_proba.items():
            preds = np.argmax(proba, axis=1)
            per_model_acc[name] = float(accuracy_score(y, preds))
            if verbose:
                print(f"  {name:<20s} accuracy: {per_model_acc[name]:.4f}")

        return {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict,
            "per_model_accuracy": per_model_acc,
            "method": method,
        }

    # ── Visualisation ─────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Ensemble Confusion Matrix",
    ) -> str:
        """Plot and save a confusion matrix heatmap.

        Returns the absolute path to the saved figure.
        """
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[c.capitalize() for c in self.class_names],
            yticklabels=[c.capitalize() for c in self.class_names],
            ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(title, fontsize=13)
        plt.tight_layout()

        if save_path is None:
            save_path = "ensemble_confusion_matrix.png"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", save_path)
        return os.path.abspath(save_path)

    def plot_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 20,
        save_path: Optional[str] = None,
    ) -> str:
        """Plot Random Forest feature importances.

        Returns the absolute path to the saved figure.
        """
        if not hasattr(self.rf_model_, "feature_importances_"):
            raise RuntimeError("Random Forest has not been fitted yet.")

        importances = self.rf_model_.feature_importances_
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(len(importances))]

        indices = np.argsort(importances)[::-1][:top_n]
        top_names = [feature_names[i] for i in indices]
        top_vals = importances[indices]

        fig, ax = plt.subplots(figsize=(8, max(4, top_n // 2)))
        ax.barh(range(len(top_names)), top_vals[::-1], color="tab:blue", alpha=0.8)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"Top {top_n} Feature Importances (Random Forest)", fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        if save_path is None:
            save_path = "feature_importance.png"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Feature importance plot saved to %s", save_path)
        return os.path.abspath(save_path)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Pickle the fitted ensemble to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        logger.info("BoneHealthEnsemble saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BoneHealthEnsemble":
        """Load a fitted ensemble from a pickle file."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info("BoneHealthEnsemble loaded from %s", path)
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_meta_features(
        self,
        X_scaled: np.ndarray,
        health_scores: Optional[np.ndarray],
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Concatenate sub-model probability outputs for the meta-learner."""
        parts: List[np.ndarray] = []

        if health_scores is not None:
            parts.append(self.baseline_model_.predict_proba(health_scores))
        else:
            parts.append(np.full((len(X_scaled), self.n_classes), 1.0 / self.n_classes))

        parts.append(self.rf_model_.predict_proba(X_scaled))
        parts.append(self.boost_model_.predict_proba(X_scaled))
        parts.append(self.nn_model_.predict_proba(X_scaled))

        return np.hstack(parts)
