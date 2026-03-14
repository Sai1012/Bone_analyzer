"""
baseline_builder.py - Build age-gender stratified baselines from normal cases.

Baselines are computed from the 36 "normal" diagnosis patients and saved as
JSON so they can be reused across runs without re-processing images.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    AGE_GROUP_LABELS,
    AGE_GROUPS,
    BASELINE_FALLBACK,
    MIN_STRATUM_SAMPLES,
    OUTPUT_DIR,
)
from feature_extractor import ALL_FEATURE_NAMES

logger = logging.getLogger(__name__)

# Saved baseline JSON path
BASELINE_FILE = os.path.join(OUTPUT_DIR, "baselines.json")

# Type alias: stratum key → {"mean": {feat: val}, "std": {feat: val}, ...}
BaselineDict = Dict[str, Dict]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def build_baselines(
    df: pd.DataFrame,
    features: List[Dict[str, Optional[float]]],
) -> BaselineDict:
    """Compute age-gender baselines from normal cases.

    Parameters
    ----------
    df:
        Full patient DataFrame (from data_loader).
    features:
        Feature dicts (same order as df rows), from feature_extractor.

    Returns
    -------
    BaselineDict
        Keyed by stratum string, e.g. ``"40_50_female"``.  Each value
        contains ``mean``, ``std``, ``min``, ``max``, ``n``.
    """
    normal_mask = df["diagnosis"].str.lower() == "normal"
    normal_indices = [i for i, flag in enumerate(normal_mask) if flag]
    logger.info("Found %d normal cases for baseline construction", len(normal_indices))

    # Group normal indices by stratum
    stratum_map: Dict[str, List[int]] = {}
    for idx in normal_indices:
        row = df.iloc[idx]
        key = _get_stratum_key(row)
        stratum_map.setdefault(key, []).append(idx)

    baselines: BaselineDict = {}
    for stratum, indices in stratum_map.items():
        feat_matrix = [features[i] for i in indices]
        baselines[stratum] = _compute_stratum_stats(feat_matrix, stratum, indices)

    # Also store fallback strata (gender-only, age-only, global)
    baselines.update(_build_fallback_strata(normal_indices, df, features))

    _log_baseline_summary(baselines)
    _save_baselines(baselines)
    return baselines


def find_baseline(
    age: Optional[float],
    gender: Optional[str],
    baselines: BaselineDict,
) -> Tuple[str, Dict]:
    """Find the most specific reliable baseline for a given age and gender.

    Returns ``(stratum_key, stratum_stats)``.
    """
    if not BASELINE_FALLBACK:
        key = _make_stratum_key(age, gender)
        return key, baselines.get(key, {})

    # Try exact stratum → gender-only → age-only → global
    candidates = _candidate_keys(age, gender)
    for key in candidates:
        stats = baselines.get(key)
        if stats and stats.get("n", 0) >= MIN_STRATUM_SAMPLES:
            return key, stats

    # Last resort: any available stratum
    if baselines:
        key = next(iter(baselines))
        return key, baselines[key]

    return "none", {}


def load_baselines(path: str = BASELINE_FILE) -> BaselineDict:
    """Load previously saved baselines from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _get_stratum_key(row: pd.Series) -> str:
    age = row.get("age")
    gender = row.get("gender")
    return _make_stratum_key(age, gender)


def _make_stratum_key(age: Optional[float], gender: Optional[str]) -> str:
    age_label = _age_label(age)
    gender_norm = _normalise_gender(gender)
    return f"{age_label}_{gender_norm}"


def _age_label(age: Optional[float]) -> str:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return "unknown_age"
    for lo, hi in AGE_GROUPS:
        if lo <= age < hi:
            return AGE_GROUP_LABELS[(lo, hi)]
    return "unknown_age"


def _normalise_gender(gender: Optional[str]) -> str:
    if gender is None:
        return "unknown_gender"
    g = str(gender).strip().lower()
    if g in ("m", "male"):
        return "male"
    if g in ("f", "female"):
        return "female"
    return "unknown_gender"


def _candidate_keys(
    age: Optional[float], gender: Optional[str]
) -> List[str]:
    """Return fallback stratum keys from most to least specific."""
    age_label = _age_label(age)
    gender_norm = _normalise_gender(gender)
    return [
        f"{age_label}_{gender_norm}",   # exact
        f"all_ages_{gender_norm}",       # gender-only
        f"{age_label}_all_genders",      # age-only
        "global",                        # global
    ]


def _compute_stratum_stats(
    feat_matrix: List[Dict[str, Optional[float]]],
    stratum: str,
    indices: List[int],
) -> Dict:
    """Compute mean/std/min/max for every feature in a stratum."""
    stats: Dict[str, Dict[str, Optional[float]]] = {}
    for feat in ALL_FEATURE_NAMES:
        vals = [
            fd[feat]
            for fd in feat_matrix
            if fd.get(feat) is not None and not np.isnan(fd[feat])
        ]
        if not vals:
            stats[feat] = {"mean": None, "std": None, "min": None, "max": None}
        else:
            arr = np.array(vals, dtype=float)
            stats[feat] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)) if len(arr) > 1 else 0.0,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
    return {
        "stratum": stratum,
        "n": len(feat_matrix),
        "indices": indices,
        "features": stats,
    }


def _build_fallback_strata(
    normal_indices: List[int],
    df: pd.DataFrame,
    features: List[Dict[str, Optional[float]]],
) -> BaselineDict:
    """Create gender-only, age-only, and global fallback strata."""
    fallbacks: BaselineDict = {}

    # Gender-only strata
    for gender in ("male", "female"):
        idx_list = [
            i for i in normal_indices
            if _normalise_gender(df.iloc[i].get("gender")) == gender
        ]
        if idx_list:
            key = f"all_ages_{gender}"
            fallbacks[key] = _compute_stratum_stats(
                [features[i] for i in idx_list], key, idx_list
            )

    # Age-only strata
    for (lo, hi), label in AGE_GROUP_LABELS.items():
        idx_list = [
            i for i in normal_indices
            if _in_age_group(df.iloc[i].get("age"), lo, hi)
        ]
        if idx_list:
            key = f"{label}_all_genders"
            fallbacks[key] = _compute_stratum_stats(
                [features[i] for i in idx_list], key, idx_list
            )

    # Global
    if normal_indices:
        key = "global"
        fallbacks[key] = _compute_stratum_stats(
            [features[i] for i in normal_indices], key, normal_indices
        )

    return fallbacks


def _in_age_group(age: Optional[float], lo: int, hi: int) -> bool:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return False
    return lo <= age < hi


def _log_baseline_summary(baselines: BaselineDict) -> None:
    logger.info("Baseline strata created:")
    for key, stats in baselines.items():
        if "_all_genders" not in key and "all_ages_" not in key and key != "global":
            logger.info("  %-30s  n=%d", key, stats.get("n", 0))


def _save_baselines(baselines: BaselineDict, path: str = BASELINE_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert numpy types to native Python for JSON serialization
    serializable = json.loads(json.dumps(baselines, default=_json_default))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, indent=2)
    logger.info("Baselines saved to %s", path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
