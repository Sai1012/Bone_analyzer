"""
data_loader.py - Load the osteoporosis knee dataset and link to local images.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    COLUMN_MAP,
    EXCEL_FILE,
    IMAGE_EXTENSION,
    IMAGE_PREFIXES,
    LOCAL_IMAGES_FOLDER,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def load_dataset(
    excel_path: str = EXCEL_FILE,
    images_folder: str = LOCAL_IMAGES_FOLDER,
) -> pd.DataFrame:
    """Load the Excel metadata file and return a cleaned DataFrame.

    Parameters
    ----------
    excel_path:
        Path to ``osteoporosis_knee_dataset.xlsx``.
    images_folder:
        Local directory that contains the 239 JPEG X-ray files.

    Returns
    -------
    pd.DataFrame
        One row per patient with cleaned, renamed columns and an additional
        ``local_image_path`` column pointing to the actual file on disk
        (``None`` when the file cannot be found).
    """
    logger.info("Loading dataset from %s", excel_path)

    df = pd.read_excel(excel_path, engine="openpyxl")
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    df = _rename_columns(df)
    df = _clean_values(df)
    df = _link_images(df, images_folder)
    df = _validate(df)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Excel headers to clean internal keys using COLUMN_MAP."""
    rename = {}
    for col in df.columns:
        stripped = col  # try exact match first
        if stripped in COLUMN_MAP:
            rename[col] = COLUMN_MAP[stripped]
        else:
            # Try stripping leading/trailing whitespace from the stored key
            for excel_key, internal_key in COLUMN_MAP.items():
                if col.strip() == excel_key.strip():
                    rename[col] = internal_key
                    break
    df = df.rename(columns=rename)
    return df


def _clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise string values, handle missing data."""

    # Lowercase string columns for consistency
    str_cols = [
        "patient_id", "gender", "joint_pain", "smoker", "alcoholic",
        "diabetic", "hypothyroidism", "seizure_disorder", "estrogen_use",
        "history_of_fracture", "dialysis", "family_history", "eating_habits",
        "medical_history", "site", "obesity", "diagnosis",
        "image_file", "occupation",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace("nan", None)

    # Binary yes/no columns → 1/0
    binary_cols = [
        "joint_pain", "smoker", "alcoholic", "diabetic", "hypothyroidism",
        "seizure_disorder", "estrogen_use", "history_of_fracture", "dialysis",
        "family_history",
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda v: _yesno_to_int(v))

    # Numeric columns – coerce to float, leave NaN for missing
    numeric_cols = [
        "age", "menopause_age", "height", "weight", "num_pregnancies",
        "walking_distance", "t_score", "z_score", "bmi",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _yesno_to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("yes", "1", "true"):
        return 1
    if v in ("no", "0", "false"):
        return 0
    return None


def _link_images(df: pd.DataFrame, images_folder: str) -> pd.DataFrame:
    """Add a ``local_image_path`` column resolving each image to disk."""
    folder = Path(images_folder)
    paths: list[Optional[str]] = []

    for _, row in df.iterrows():
        image_file = _resolve_image_filename(row)
        if image_file is None:
            paths.append(None)
            continue

        candidate = folder / image_file
        if candidate.exists():
            paths.append(str(candidate))
        else:
            # Try case-insensitive search
            found = _find_file_case_insensitive(folder, image_file)
            paths.append(found)

    df["local_image_path"] = paths
    found_count = sum(1 for p in paths if p is not None)
    logger.info(
        "Linked %d / %d images in %s", found_count, len(df), images_folder
    )
    if found_count < len(df):
        missing = df[df["local_image_path"].isna()]["patient_id"].tolist()
        logger.warning("Missing images for patients: %s", missing[:10])
    return df


def _resolve_image_filename(row: pd.Series) -> Optional[str]:
    """Return the expected JPEG filename for a row."""
    # Prefer the explicit image_file column when present
    if "image_file" in row.index and row["image_file"] not in (None, "none", "nan", ""):
        fname = str(row["image_file"]).strip()
        if not fname.upper().endswith(IMAGE_EXTENSION.upper()):
            fname += IMAGE_EXTENSION
        return fname

    # Fall back to constructing from patient_id
    if "patient_id" in row.index and row["patient_id"] not in (None, "none", "nan", ""):
        pid = str(row["patient_id"]).strip().upper()
        return pid + IMAGE_EXTENSION

    return None


def _find_file_case_insensitive(folder: Path, filename: str) -> Optional[str]:
    """Search *folder* for *filename* ignoring case; return path or None."""
    if not folder.exists():
        return None
    target = filename.lower()
    for f in folder.iterdir():
        if f.name.lower() == target:
            return str(f)
    return None


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Log data quality summary."""
    total = len(df)
    if "age" in df.columns:
        missing_age = df["age"].isna().sum()
        logger.info("Missing age values: %d / %d", missing_age, total)
    if "gender" in df.columns:
        logger.info("Gender distribution:\n%s", df["gender"].value_counts())
    if "diagnosis" in df.columns:
        logger.info("Diagnosis distribution:\n%s", df["diagnosis"].value_counts())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = load_dataset()
    print(data.head())
    print(data.dtypes)
    print("Shape:", data.shape)
