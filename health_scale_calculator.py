"""
health_scale_calculator.py - Compute 1-10 bone health scores from feature deviations.

Algorithm:
1. For each feature, compute z-score = (patient_value - baseline_mean) / baseline_std
2. Aggregate weighted absolute z-scores into a composite deviation
3. Map composite deviation to 1-10 scale via HEALTH_SCALE_THRESHOLDS
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from baseline_builder import BaselineDict, find_baseline
from config import FEATURE_WEIGHTS, HEALTH_SCALE_THRESHOLDS
from feature_extractor import ALL_FEATURE_NAMES

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes / named results
# ──────────────────────────────────────────────────────────────────────────────


class HealthResult:
    """Stores the health score and deviation details for one patient."""

    def __init__(
        self,
        patient_id: str,
        health_score: int,
        composite_deviation: float,
        feature_zscores: Dict[str, Optional[float]],
        stratum_used: str,
        stratum_n: int,
        age: Optional[float],
        gender: Optional[str],
    ) -> None:
        self.patient_id = patient_id
        self.health_score = health_score
        self.composite_deviation = composite_deviation
        self.feature_zscores = feature_zscores
        self.stratum_used = stratum_used
        self.stratum_n = stratum_n
        self.age = age
        self.gender = gender

    # Top-3 features most deviating from baseline
    @property
    def top_deviations(self) -> List[Tuple[str, float]]:
        scored = [
            (k, abs(v)) for k, v in self.feature_zscores.items() if v is not None
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:3]

    def severity_label(self) -> str:
        labels = {
            10: "Perfect",
            9: "Minor deviation",
            8: "Slight deviation",
            7: "Mild deviation",
            6: "Moderate deviation",
            5: "Noticeable degradation",
            4: "Significant degradation",
            3: "Severe degradation",
            2: "Critical",
            1: "Critical",
        }
        return labels.get(self.health_score, "Unknown")

    def to_dict(self) -> Dict:
        d = {
            "patient_id": self.patient_id,
            "age": self.age,
            "gender": self.gender,
            "health_score": self.health_score,
            "severity": self.severity_label(),
            "composite_deviation_std": round(self.composite_deviation, 3),
            "baseline_stratum": self.stratum_used,
            "baseline_n": self.stratum_n,
        }
        # Top deviations
        for rank, (feat, zscore) in enumerate(self.top_deviations, start=1):
            d[f"top_deviation_{rank}_feature"] = feat
            d[f"top_deviation_{rank}_zscore"] = round(zscore, 3)
        # All feature z-scores
        for feat in ALL_FEATURE_NAMES:
            val = self.feature_zscores.get(feat)
            d[f"zscore_{feat}"] = round(val, 3) if val is not None else None
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Core calculation
# ──────────────────────────────────────────────────────────────────────────────


def calculate_health_score(
    patient_features: Dict[str, Optional[float]],
    age: Optional[float],
    gender: Optional[str],
    patient_id: str,
    baselines: BaselineDict,
) -> HealthResult:
    """Compute the 1-10 bone health score for a single patient.

    Parameters
    ----------
    patient_features:
        Normalised feature dict from feature_extractor.
    age, gender:
        Patient demographics used to select the baseline stratum.
    patient_id:
        String identifier for logging / output.
    baselines:
        The baseline dict produced by baseline_builder.

    Returns
    -------
    HealthResult
    """
    stratum_key, stratum_stats = find_baseline(age, gender, baselines)

    if not stratum_stats:
        logger.warning("No baseline found for patient %s (age=%s, gender=%s)", patient_id, age, gender)
        return HealthResult(
            patient_id=patient_id,
            health_score=5,
            composite_deviation=0.0,
            feature_zscores={f: None for f in ALL_FEATURE_NAMES},
            stratum_used="none",
            stratum_n=0,
            age=age,
            gender=gender,
        )

    feat_stats: Dict[str, Dict] = stratum_stats.get("features", {})
    feature_zscores: Dict[str, Optional[float]] = {}
    weighted_devs: List[float] = []
    total_weight = 0.0

    for feat in ALL_FEATURE_NAMES:
        patient_val = patient_features.get(feat)
        if patient_val is None:
            feature_zscores[feat] = None
            continue

        feat_info = feat_stats.get(feat, {})
        baseline_mean = feat_info.get("mean")
        baseline_std = feat_info.get("std")

        if baseline_mean is None:
            feature_zscores[feat] = None
            continue

        if baseline_std is None or baseline_std == 0:
            # Use a small epsilon to avoid division by zero while still
            # capturing deviations
            baseline_std = 1e-6

        zscore = (float(patient_val) - float(baseline_mean)) / float(baseline_std)
        feature_zscores[feat] = zscore

        weight = FEATURE_WEIGHTS.get(feat, 1.0)
        weighted_devs.append(abs(zscore) * weight)
        total_weight += weight

    # Composite deviation: weighted mean of absolute z-scores
    if weighted_devs and total_weight > 0:
        composite = sum(weighted_devs) / total_weight
    else:
        composite = 0.0

    health_score = _deviation_to_score(composite)

    return HealthResult(
        patient_id=patient_id,
        health_score=health_score,
        composite_deviation=composite,
        feature_zscores=feature_zscores,
        stratum_used=stratum_key,
        stratum_n=stratum_stats.get("n", 0),
        age=age,
        gender=gender,
    )


def calculate_all_health_scores(
    df: pd.DataFrame,
    features: List[Dict[str, Optional[float]]],
    baselines: BaselineDict,
) -> List[HealthResult]:
    """Compute health scores for every patient in *df*."""
    results: List[HealthResult] = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 50 == 0:
            logger.info("Calculating health scores: %d / %d", idx, total)

        patient_id = str(row.get("patient_id", f"patient_{idx}"))
        age = row.get("age")
        gender = row.get("gender")

        result = calculate_health_score(
            patient_features=features[idx],
            age=float(age) if age is not None and not (isinstance(age, float) and np.isnan(age)) else None,
            gender=str(gender) if gender not in (None, "none", "nan") else None,
            patient_id=patient_id,
            baselines=baselines,
        )
        results.append(result)

    return results


def results_to_dataframe(results: List[HealthResult]) -> pd.DataFrame:
    """Convert a list of HealthResult objects to a tidy DataFrame."""
    return pd.DataFrame([r.to_dict() for r in results])


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _deviation_to_score(deviation: float) -> int:
    """Map a composite deviation value to an integer 1-10 health score."""
    for max_dev, score in HEALTH_SCALE_THRESHOLDS:
        if deviation <= max_dev:
            return score
    return 1  # extreme deviation


# ──────────────────────────────────────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────────────────────────────────────


def summarise_scores(results: List[HealthResult], df: pd.DataFrame) -> None:
    """Print a human-readable summary of the health score distribution."""
    scores = [r.health_score for r in results]
    print("\n" + "=" * 60)
    print("BONE HEALTH ANALYSIS - SUMMARY")
    print("=" * 60)
    print(f"Total patients analysed: {len(scores)}")
    print(f"Mean health score:       {np.mean(scores):.2f} / 10")
    print(f"Median health score:     {np.median(scores):.2f} / 10")
    print(f"Std dev:                 {np.std(scores):.2f}")
    print()
    print("Score Distribution:")
    from collections import Counter
    dist = Counter(scores)
    for score in range(10, 0, -1):
        count = dist.get(score, 0)
        bar = "█" * count
        print(f"  {score:2d}/10  {bar}  ({count})")

    if "diagnosis" in df.columns:
        print("\nMean Score by Diagnosis:")
        score_series = pd.Series(scores, index=df.index)
        for diag, group in df.groupby("diagnosis"):
            grp_scores = [scores[i] for i in range(len(df)) if df.iloc[i]["diagnosis"] == diag]
            if grp_scores:
                print(f"  {diag:<15s}: {np.mean(grp_scores):.2f}")

    print("=" * 60 + "\n")
