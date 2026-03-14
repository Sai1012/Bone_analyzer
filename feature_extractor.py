"""
feature_extractor.py - Extract image-based and clinical features from each patient.

Image features are extracted with OpenCV; clinical features come from the
DataFrame produced by data_loader.py.  All features are normalised to [0, 1]
using per-feature statistics collected from the full dataset (fitted once,
applied to each patient).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Feature names (keep a canonical order)
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_FEATURE_NAMES: List[str] = [
    "bone_density_mean",
    "bone_density_std",
    "joint_space_width",
    "cortical_thickness",
    "texture_contrast",
    "texture_homogeneity",
    "texture_energy",
    "texture_correlation",
    "edge_density",
    "geometric_area",
    "geometric_perimeter",
]

CLINICAL_FEATURE_NAMES: List[str] = [
    "bmi",
    "t_score",
    "z_score",
    "walking_distance",
    "smoker",
    "alcoholic",
    "diabetic",
    "hypothyroidism",
    "estrogen_use",
    "history_of_fracture",
    "family_history",
]

ALL_FEATURE_NAMES: List[str] = IMAGE_FEATURE_NAMES + CLINICAL_FEATURE_NAMES


# ──────────────────────────────────────────────────────────────────────────────
# Image feature extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_image_features(image_path: Optional[str]) -> Dict[str, Optional[float]]:
    """Extract bone-health-relevant features from a single X-ray image.

    Returns a dict of feature_name → float (or None if extraction fails).
    """
    features: Dict[str, Optional[float]] = {k: None for k in IMAGE_FEATURE_NAMES}

    if image_path is None or not Path(image_path).exists():
        logger.debug("Image not available: %s", image_path)
        return features

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning("Could not read image: %s", image_path)
        return features

    try:
        # ── Bone density ───────────────────────────────────────────────────────
        features["bone_density_mean"] = float(np.mean(img))
        features["bone_density_std"] = float(np.std(img))

        # ── Joint space width ─────────────────────────────────────────────────
        features["joint_space_width"] = _estimate_joint_space(img)

        # ── Cortical thickness ────────────────────────────────────────────────
        features["cortical_thickness"] = _estimate_cortical_thickness(img)

        # ── Texture features (GLCM) ───────────────────────────────────────────
        glcm_feats = _compute_glcm_features(img)
        features.update(glcm_feats)

        # ── Edge density ──────────────────────────────────────────────────────
        features["edge_density"] = _compute_edge_density(img)

        # ── Geometric features ────────────────────────────────────────────────
        geo = _compute_geometric_features(img)
        features.update(geo)

    except Exception as exc:  # pragma: no cover
        logger.error("Feature extraction failed for %s: %s", image_path, exc)

    return features


# ── Private image helpers ──────────────────────────────────────────────────────


def _estimate_joint_space(img: np.ndarray) -> float:
    """Estimate joint space width using horizontal edge detection.

    The joint space in a knee X-ray appears as a bright (low-density)
    horizontal band between the femur and tibia.  We locate this band via
    horizontal Sobel gradients and measure its extent in the central column.
    """
    h, w = img.shape
    # Work on the central third of the image
    col_start = w // 3
    col_end = 2 * w // 3
    central = img[:, col_start:col_end]

    # Horizontal projection (mean intensity per row)
    row_means = np.mean(central, axis=1).astype(np.float32)

    # Smooth
    kernel_size = max(3, h // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(
        row_means.reshape(-1, 1), (1, kernel_size), 0
    ).flatten()

    # The joint space is the bright band in the middle third of the image
    mid_start = h // 3
    mid_end = 2 * h // 3
    mid_region = smoothed[mid_start:mid_end]

    # Find the peak brightness and measure width above 80 % of the peak
    peak = float(np.max(mid_region))
    threshold = 0.80 * peak
    above = (mid_region >= threshold).astype(np.uint8)
    if above.sum() == 0:
        return 0.0

    # Width = number of consecutive rows above threshold around the peak
    peak_idx = int(np.argmax(mid_region))
    left = peak_idx
    while left > 0 and mid_region[left - 1] >= threshold:
        left -= 1
    right = peak_idx
    while right < len(mid_region) - 1 and mid_region[right + 1] >= threshold:
        right += 1

    width_pixels = right - left + 1
    # Normalise by image height so the value is scale-independent
    return float(width_pixels) / h


def _estimate_cortical_thickness(img: np.ndarray) -> float:
    """Estimate cortical bone thickness using morphological operations.

    Cortical bone is the dense outer shell.  We threshold to find bright
    regions, apply erosion, and measure how much erodes away.
    """
    # Threshold to bright bone regions
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Successive erosions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = binary.copy()
    max_iter = 10
    thickness_sum = 0.0
    for i in range(max_iter):
        new_eroded = cv2.erode(eroded, kernel, iterations=1)
        shell = cv2.bitwise_and(eroded, cv2.bitwise_not(new_eroded))
        thickness_sum += float(shell.sum()) / (255.0 * img.size)
        eroded = new_eroded
        if eroded.sum() == 0:
            break

    return thickness_sum


def _compute_glcm_features(img: np.ndarray) -> Dict[str, float]:
    """Compute GLCM-based texture features."""
    # Reduce bit depth to speed up GLCM computation
    img_8 = (img // 16).astype(np.uint8)  # 256 → 16 levels

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        glcm = graycomatrix(
            img_8,
            distances=[1, 3],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=16,
            symmetric=True,
            normed=True,
        )

    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    correlation = float(np.mean(graycoprops(glcm, "correlation")))

    return {
        "texture_contrast": contrast,
        "texture_homogeneity": homogeneity,
        "texture_energy": energy,
        "texture_correlation": correlation,
    }


def _compute_edge_density(img: np.ndarray) -> float:
    """Fraction of pixels classified as edges by Canny."""
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    return float(edges.sum()) / (255.0 * img.size)


def _compute_geometric_features(img: np.ndarray) -> Dict[str, float]:
    """Compute geometric properties of the segmented bone region."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"geometric_area": 0.0, "geometric_perimeter": 0.0}

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, closed=True)

    total_pixels = float(img.size)
    return {
        "geometric_area": area / total_pixels,
        "geometric_perimeter": perimeter / (img.shape[0] + img.shape[1]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Clinical feature extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_clinical_features(row: pd.Series) -> Dict[str, Optional[float]]:
    """Extract clinical features from a single DataFrame row."""
    features: Dict[str, Optional[float]] = {}
    for col in CLINICAL_FEATURE_NAMES:
        val = row.get(col)
        if pd.isna(val) if val is not None else True:
            features[col] = None
        else:
            try:
                features[col] = float(val)
            except (TypeError, ValueError):
                features[col] = None
    return features


# ──────────────────────────────────────────────────────────────────────────────
# Feature normalisation
# ──────────────────────────────────────────────────────────────────────────────


class FeatureNormaliser:
    """Min-max normaliser fitted on the full dataset feature matrix."""

    def __init__(self) -> None:
        self._min: Dict[str, float] = {}
        self._max: Dict[str, float] = {}
        self._fitted = False

    def fit(self, feature_matrix: List[Dict[str, Optional[float]]]) -> "FeatureNormaliser":
        """Compute min/max statistics from a list of raw feature dicts."""
        from collections import defaultdict

        values: Dict[str, List[float]] = defaultdict(list)
        for feat_dict in feature_matrix:
            for k, v in feat_dict.items():
                if v is not None and not np.isnan(v):
                    values[k].append(v)

        for k, vals in values.items():
            self._min[k] = float(np.min(vals))
            self._max[k] = float(np.max(vals))

        self._fitted = True
        return self

    def transform(
        self, feat_dict: Dict[str, Optional[float]]
    ) -> Dict[str, Optional[float]]:
        """Normalise a single feature dict to [0, 1]."""
        if not self._fitted:
            raise RuntimeError("FeatureNormaliser must be fitted before transform.")
        out: Dict[str, Optional[float]] = {}
        for k, v in feat_dict.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                out[k] = None
                continue
            lo = self._min.get(k)
            hi = self._max.get(k)
            if lo is None or hi is None or hi == lo:
                out[k] = 0.5  # no range information
            else:
                out[k] = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
        return out

    def fit_transform(
        self, feature_matrix: List[Dict[str, Optional[float]]]
    ) -> List[Dict[str, Optional[float]]]:
        self.fit(feature_matrix)
        return [self.transform(fd) for fd in feature_matrix]


# ──────────────────────────────────────────────────────────────────────────────
# High-level pipeline function
# ──────────────────────────────────────────────────────────────────────────────


def extract_all_features(
    df: pd.DataFrame,
    normalise: bool = True,
) -> Tuple[List[Dict[str, Optional[float]]], Optional[FeatureNormaliser]]:
    """Extract and optionally normalise features for every patient in *df*.

    Returns
    -------
    feature_list:
        One dict per patient (same order as df rows).
    normaliser:
        The fitted FeatureNormaliser (None if normalise=False).
    """
    raw_features: List[Dict[str, Optional[float]]] = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 50 == 0:
            logger.info("Extracting features: %d / %d", idx, total)

        img_feats = extract_image_features(row.get("local_image_path"))
        clin_feats = extract_clinical_features(row)
        combined = {**img_feats, **clin_feats}
        raw_features.append(combined)

    if not normalise:
        return raw_features, None

    normaliser = FeatureNormaliser()
    normalised = normaliser.fit_transform(raw_features)
    return normalised, normaliser
