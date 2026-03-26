"""
class_imbalance_handler.py - Handle class imbalance in bone health classification.

Addresses the dataset skew (Normal: 15 %, Osteopenia: 64 %, Osteoporosis: 21 %)
via four complementary strategies:

1. SMOTE            – Synthetic Minority Oversampling Technique
2. Class Weights    – Inverse-frequency weights for loss / scoring
3. Stratified Split – Train/test splits that preserve class proportions
4. Data Augmentation– Geometric + photometric transforms on X-ray images

Usage
-----
    from class_imbalance_handler import ClassImbalanceHandler
    handler = ClassImbalanceHandler()
    X_resampled, y_resampled = handler.apply_smote(X_train, y_train)
    class_weights = handler.compute_class_weights(y_train)
    X_tr, X_te, y_tr, y_te = handler.stratified_split(X, y)
    aug_img = handler.augment_image(img_array)
    handler.print_imbalance_report(y_before, y_after)
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Optional SMOTE import
# ──────────────────────────────────────────────────────────────────────────────

_SMOTE_AVAILABLE = False
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    _SMOTE_AVAILABLE = True
    logger.debug("imbalanced-learn found – SMOTE available.")
except ImportError:
    logger.debug("imbalanced-learn not installed – SMOTE will be skipped.")


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def _flip_image(img: np.ndarray, flip_code: int) -> np.ndarray:
    """Flip: 0 = vertical, 1 = horizontal, -1 = both."""
    return cv2.flip(img, flip_code)


def _adjust_contrast(img: np.ndarray, alpha: float, beta: float = 0) -> np.ndarray:
    """Apply contrast (alpha) and brightness (beta) adjustment."""
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


def _add_gaussian_noise(img: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Add zero-mean Gaussian noise."""
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _apply_clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE (contrast-limited adaptive histogram equalisation)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


# ──────────────────────────────────────────────────────────────────────────────
# ClassImbalanceHandler
# ──────────────────────────────────────────────────────────────────────────────


class ClassImbalanceHandler:
    """Utility class that bundles all class-imbalance mitigation strategies.

    Parameters
    ----------
    class_names:
        Ordered class labels matching integer indices 0, 1, 2, …
    random_state:
        Random seed for reproducibility.
    smote_k_neighbors:
        Number of nearest neighbours used in SMOTE.
    test_size:
        Fraction of data to hold out as test set.
    """

    def __init__(
        self,
        class_names: List[str] = ("normal", "osteopenia", "osteoporosis"),
        random_state: int = 42,
        smote_k_neighbors: int = 5,
        test_size: float = 0.2,
    ) -> None:
        self.class_names = list(class_names)
        self.random_state = random_state
        self.smote_k_neighbors = smote_k_neighbors
        self.test_size = test_size

    # ── 1. SMOTE ──────────────────────────────────────────────────────────────

    def apply_smote(
        self,
        X: np.ndarray,
        y: np.ndarray,
        strategy: str = "auto",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to oversample minority classes.

        Falls back to simple random oversampling if imbalanced-learn is not
        installed.

        Parameters
        ----------
        X:
            Feature matrix (n_samples, n_features).
        y:
            Class label array (integer indices).
        strategy:
            SMOTE sampling strategy (see imbalanced-learn docs).
            'auto' = resample all classes to match the majority.

        Returns
        -------
        X_resampled, y_resampled
        """
        if _SMOTE_AVAILABLE:
            # Use BorderlineSMOTE if any class has very few samples (< 6),
            # otherwise standard SMOTE
            counts = np.bincount(y.astype(int))
            min_count = counts.min()
            k = min(self.smote_k_neighbors, min_count - 1) if min_count > 1 else 1
            if k < 1:
                logger.warning(
                    "Not enough minority samples for SMOTE (min_count=%d). "
                    "Falling back to random oversampling.", min_count
                )
                return self._random_oversample(X, y)

            try:
                smote = SMOTE(
                    sampling_strategy=strategy,
                    k_neighbors=k,
                    random_state=self.random_state,
                )
                X_res, y_res = smote.fit_resample(X, y)
                logger.info(
                    "SMOTE applied: %d → %d samples.", len(y), len(y_res)
                )
                return X_res, y_res
            except Exception as exc:
                logger.warning("SMOTE failed (%s). Falling back to random oversampling.", exc)
                return self._random_oversample(X, y)
        else:
            logger.info(
                "imbalanced-learn not available. Using random oversampling."
            )
            return self._random_oversample(X, y)

    def _random_oversample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple random oversampling of minority classes."""
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        max_count = counts.max()

        X_parts = [X]
        y_parts = [y]
        for cls, cnt in zip(classes, counts):
            n_extra = max_count - cnt
            if n_extra > 0:
                idx = np.where(y == cls)[0]
                sampled = rng.choice(idx, size=n_extra, replace=True)
                X_parts.append(X[sampled])
                y_parts.append(y[sampled])

        X_res = np.vstack(X_parts)
        y_res = np.concatenate(y_parts)
        # Shuffle
        perm = rng.permutation(len(y_res))
        return X_res[perm], y_res[perm]

    # ── 2. Class Weights ──────────────────────────────────────────────────────

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights.

        Uses sklearn's ``compute_class_weight('balanced', ...)`` which sets
        weight_i = n_samples / (n_classes × count_i).

        Parameters
        ----------
        y:
            Integer class label array.

        Returns
        -------
        Dict mapping class index → weight.
        """
        classes = np.unique(y.astype(int))
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y.astype(int),
        )
        weight_dict = {int(cls): float(w) for cls, w in zip(classes, weights)}
        for cls, w in weight_dict.items():
            name = self.class_names[cls] if cls < len(self.class_names) else str(cls)
            logger.info("Class weight %-15s (idx=%d): %.4f", name, cls, w)
        return weight_dict

    def sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Convert class weights to per-sample weights for use in sklearn.

        Parameters
        ----------
        y:
            Integer class label array of length n.

        Returns
        -------
        1-D float array of per-sample weights (length n).
        """
        class_weights = self.compute_class_weights(y)
        return np.array([class_weights[int(label)] for label in y])

    # ── 3. Stratified Split ───────────────────────────────────────────────────

    def stratified_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train/test split preserving class proportions.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Integer label array.
        n_splits:
            Number of splits (only the first split is returned).

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        sss = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_idx, test_idx = next(sss.split(X, y))
        logger.info(
            "Stratified split: train=%d, test=%d (test_size=%.0f%%)",
            len(train_idx), len(test_idx), self.test_size * 100,
        )
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def stratified_split_df(
        self,
        df: pd.DataFrame,
        label_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Stratified split for a DataFrame.

        Parameters
        ----------
        df:
            Patient DataFrame.
        label_col:
            Column containing class labels (string or integer).

        Returns
        -------
        df_train, df_test
        """
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        y = df[label_col].values
        train_idx, test_idx = next(sss.split(np.zeros(len(df)), y))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)

    # ── 4. Data Augmentation ──────────────────────────────────────────────────

    def augment_image(
        self,
        img: np.ndarray,
        rotations: Tuple[float, ...] = (-15, 15),
        flip: bool = True,
        contrast_range: Tuple[float, float] = (0.8, 1.3),
        add_noise: bool = True,
        apply_clahe: bool = True,
    ) -> List[np.ndarray]:
        """Generate augmented variants of a single X-ray image.

        Parameters
        ----------
        img:
            Grayscale uint8 image (H × W).
        rotations:
            Tuple of rotation angles (degrees) to apply.  Each angle produces
            one augmented variant.
        flip:
            If True, include a horizontally flipped variant.
        contrast_range:
            (min_alpha, max_alpha) for random contrast scaling.
        add_noise:
            If True, add Gaussian noise to one variant.
        apply_clahe:
            If True, apply CLAHE to one variant.

        Returns
        -------
        List of augmented images (same dtype as input).
        """
        rng = np.random.RandomState(self.random_state)
        augmented: List[np.ndarray] = []

        # Rotations
        for angle in rotations:
            augmented.append(_rotate_image(img, angle))

        # Horizontal flip
        if flip:
            augmented.append(_flip_image(img, 1))

        # Contrast adjustment
        alpha = float(rng.uniform(*contrast_range))
        augmented.append(_adjust_contrast(img, alpha=alpha))

        # Gaussian noise
        if add_noise:
            sigma = float(rng.uniform(2.0, 8.0))
            augmented.append(_add_gaussian_noise(img, sigma=sigma))

        # CLAHE
        if apply_clahe:
            augmented.append(_apply_clahe(img))

        return augmented

    def augment_minority_images(
        self,
        image_paths: List[Optional[str]],
        labels: List[str],
        minority_classes: Optional[List[str]] = None,
        save_dir: Optional[str] = None,
    ) -> Tuple[List[Optional[str]], List[str]]:
        """Augment images for minority-class patients and (optionally) save them.

        Parameters
        ----------
        image_paths:
            List of image paths (one per patient).
        labels:
            Corresponding string diagnosis labels.
        minority_classes:
            Classes to augment.  Defaults to ``["normal"]`` (15 % of data).
        save_dir:
            If provided, augmented images are saved here.

        Returns
        -------
        Extended lists (original + augmented) of paths and labels.
        """
        if minority_classes is None:
            minority_classes = ["normal"]

        aug_paths: List[Optional[str]] = list(image_paths)
        aug_labels: List[str] = list(labels)

        for img_path, label in zip(image_paths, labels):
            if label.lower() not in [c.lower() for c in minority_classes]:
                continue
            if img_path is None or not Path(str(img_path)).exists():
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            variants = self.augment_image(img)
            for vi, variant in enumerate(variants):
                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    stem = Path(str(img_path)).stem
                    out_name = f"{stem}_aug{vi}.png"
                    out_path = os.path.join(save_dir, out_name)
                    cv2.imwrite(out_path, variant)
                    aug_paths.append(out_path)
                else:
                    aug_paths.append(None)
                aug_labels.append(label)

        logger.info(
            "Augmentation: %d original → %d total (%d added)",
            len(image_paths), len(aug_paths), len(aug_paths) - len(image_paths),
        )
        return aug_paths, aug_labels

    # ── 5. Imbalance Report ───────────────────────────────────────────────────

    def print_imbalance_report(
        self,
        y_before: np.ndarray,
        y_after: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
    ) -> str:
        """Print (and optionally plot) class distribution before/after resampling.

        Parameters
        ----------
        y_before:
            Class labels before resampling.
        y_after:
            Class labels after resampling (optional).
        save_path:
            If provided, save a bar chart to this path.

        Returns
        -------
        Report text string.
        """
        def _count(y: np.ndarray) -> Dict[str, int]:
            counts: Dict[str, int] = {}
            for i, name in enumerate(self.class_names):
                counts[name] = int(np.sum(y.astype(int) == i))
            return counts

        counts_before = _count(np.asarray(y_before))
        lines = [
            "=" * 55,
            "CLASS IMBALANCE REPORT",
            "=" * 55,
            "",
            "Before resampling:",
        ]
        total_before = sum(counts_before.values())
        for name, cnt in counts_before.items():
            pct = 100 * cnt / total_before if total_before > 0 else 0
            lines.append(f"  {name:<15s}: {cnt:4d}  ({pct:.1f} %)")

        if y_after is not None:
            counts_after = _count(np.asarray(y_after))
            total_after = sum(counts_after.values())
            lines += ["", "After resampling:"]
            for name, cnt in counts_after.items():
                pct = 100 * cnt / total_after if total_after > 0 else 0
                lines.append(f"  {name:<15s}: {cnt:4d}  ({pct:.1f} %)")

        lines += ["", "=" * 55]
        report = "\n".join(lines)
        print(report)

        if save_path is not None:
            self._plot_distribution(
                counts_before,
                counts_after=(counts_after if y_after is not None else None),
                save_path=save_path,
            )

        return report

    def _plot_distribution(
        self,
        counts_before: Dict[str, int],
        counts_after: Optional[Dict[str, int]],
        save_path: str,
    ) -> None:
        """Save a grouped bar chart comparing before/after class distributions."""
        x = np.arange(len(self.class_names))
        width = 0.35 if counts_after else 0.6

        fig, ax = plt.subplots(figsize=(8, 5))
        before_vals = [counts_before.get(n, 0) for n in self.class_names]

        if counts_after:
            after_vals = [counts_after.get(n, 0) for n in self.class_names]
            ax.bar(x - width / 2, before_vals, width, label="Before", color="tab:blue", alpha=0.8)
            ax.bar(x + width / 2, after_vals, width, label="After", color="tab:green", alpha=0.8)
            ax.legend()
        else:
            ax.bar(x, before_vals, width, color="tab:blue", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([n.capitalize() for n in self.class_names])
        ax.set_ylabel("Sample Count")
        ax.set_title("Class Distribution Before vs. After Resampling")
        ax.grid(axis="y", alpha=0.3)

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Class distribution chart saved to %s", save_path)
