"""
app.py - Run the improved CBHS model on a single new patient (no retraining).

Usage example
-------------

python app.py \
    --image /path/to/new_knee.jpeg \
    --patient-id NEW001 \
    --age 65 \
    --gender female \
    --bmi 23.5 \
    --t-score -1.8 \
    --z-score -1.2 \
    --walking-distance 2.0 \
    --smoker 0 \
    --alcoholic 0 \
    --diabetic 0 \
    --hypothyroidism 0 \
    --estrogen-use 0 \
    --history-of-fracture 0 \
    --family-history 1

This script:
  - Loads pre-trained artifacts from output/ and output/improved/
  - Extracts features for the new patient
  - Computes z-score-based health score
  - Computes ensemble probabilities
  - Blends the two into a final 1–10 health score
  - Updates HealthResult.health_score to the blended value
  - Saves a PNG report to output/improved/patient_reports/<patient_id>_report.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from baseline_builder import load_baselines
from config import OUTPUT_DIR
from ensemble_classifier import BoneHealthEnsemble
from feature_engineering import AdvancedFeatureExtractor, ADVANCED_FEATURE_NAMES
from feature_extractor import (
    ALL_FEATURE_NAMES,
    extract_clinical_features,
    extract_image_features,
    load_normaliser,
)
from health_scale_calculator import calculate_health_score
from threshold_optimizer import ThresholdOptimizer
from visualizer import plot_patient_report

CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]


def _prob_to_score(proba: np.ndarray) -> float:
    """Map class probabilities to a continuous score (1–10 scale)."""
    centres = np.array([9.0, 6.5, 4.0], dtype=float)  # normal, osteopenia, osteoporosis
    return float(np.dot(proba, centres))


def _blend_score(
    ensemble_score: float,
    zscore_score: float,
    w_ensemble: float = 0.7,
    w_zscore: float = 0.3,
) -> float:
    """Blend ensemble probability score with z-score health score."""
    blended = (w_ensemble * ensemble_score) + (w_zscore * zscore_score)
    return float(np.clip(blended, 1.0, 10.0))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.py",
        description="Run improved CBHS model on a single new patient (no retraining).",
    )
    p.add_argument("--image", required=True, help="Path to knee X-ray (JPEG/PNG).")
    p.add_argument("--patient-id", required=True, help="Patient identifier.")
    p.add_argument("--age", type=float, required=True, help="Age in years.")
    p.add_argument(
        "--gender",
        required=True,
        choices=["male", "female", "unknown"],
        help="Patient gender.",
    )

    # Clinical numeric
    p.add_argument("--bmi", type=float, required=True)
    p.add_argument("--t-score", type=float, required=True)
    p.add_argument("--z-score", type=float, required=True)
    p.add_argument("--walking-distance", type=float, required=True)

    # Binary flags (0/1)
    p.add_argument("--smoker", type=int, choices=[0, 1], required=True)
    p.add_argument("--alcoholic", type=int, choices=[0, 1], required=True)
    p.add_argument("--diabetic", type=int, choices=[0, 1], required=True)
    p.add_argument("--hypothyroidism", type=int, choices=[0, 1], required=True)
    p.add_argument("--estrogen-use", type=int, choices=[0, 1], required=True)
    p.add_argument("--history-of-fracture", type=int, choices=[0, 1], required=True)
    p.add_argument("--family-history", type=int, choices=[0, 1], required=True)

    # Paths to artifacts (defaults assume you already ran improved_main.py)
    p.add_argument(
        "--baselines",
        default=os.path.join(OUTPUT_DIR, "baselines.json"),
        help="Path to baselines.json (default: output/baselines.json).",
    )
    p.add_argument(
        "--normaliser",
        default=os.path.join(OUTPUT_DIR, "feature_normaliser.json"),
        help="Path to feature_normaliser.json (default: output/feature_normaliser.json).",
    )
    p.add_argument(
        "--model",
        default=os.path.join(OUTPUT_DIR, "improved", "ensemble_model.pkl"),
        help="Path to saved ensemble model (default: output/improved/ensemble_model.pkl).",
    )
    p.add_argument(
        "--thresholds",
        default=os.path.join(OUTPUT_DIR, "improved", "optimal_thresholds.json"),
        help="Path to saved optimal thresholds (default: output/improved/optimal_thresholds.json).",
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(OUTPUT_DIR, "improved"),
        help="Directory for saving the patient report PNG.",
    )
    p.add_argument(
        "--ensemble-method",
        choices=["soft", "stack"],
        default="soft",
        help="Ensemble probability method (default: soft).",
    )
    p.add_argument(
        "--no-cnn",
        action="store_true",
        help="Skip CNN-based advanced features (if you only trained with non-CNN).",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # ------------------------------------------------------------------
    # 1. Load trained artifacts (NO retraining)
    # ------------------------------------------------------------------
    if not os.path.exists(args.baselines):
        raise FileNotFoundError(f"Baselines JSON not found: {args.baselines}")
    if not os.path.exists(args.normaliser):
        raise FileNotFoundError(f"Feature normaliser JSON not found: {args.normaliser}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Ensemble model not found: {args.model}")
    if not os.path.exists(args.thresholds):
        raise FileNotFoundError(f"Thresholds JSON not found: {args.thresholds}")

    baselines = load_baselines(args.baselines)
    normaliser = load_normaliser(args.normaliser)
    ensemble = BoneHealthEnsemble.load(args.model)
    threshold_opt = ThresholdOptimizer.load(args.thresholds)

    # ------------------------------------------------------------------
    # 2. Build a clinical Series for this new patient
    # ------------------------------------------------------------------
    clin_data: Dict[str, float] = {
        "bmi": args.bmi,
        "t_score": args.t_score,
        "z_score": args.z_score,
        "walking_distance": args.walking_distance,
        "smoker": args.smoker,
        "alcoholic": args.alcoholic,
        "diabetic": args.diabetic,
        "hypothyroidism": args.hypothyroidism,
        "estrogen_use": args.estrogen_use,
        "history_of_fracture": args.history_of_fracture,
        "family_history": args.family_history,
        "age": args.age,
        "gender": args.gender,
        "patient_id": args.patient_id,
    }
    row = pd.Series(clin_data)

    # ------------------------------------------------------------------
    # 3. Extract base (image + clinical) features and normalise
    # ------------------------------------------------------------------
    img_feats = extract_image_features(image_path)
    clin_feats = extract_clinical_features(row)
    combined = {**img_feats, **clin_feats}
    norm_feats = normaliser.transform(combined)

    # ------------------------------------------------------------------
    # 4. Z-score based health score via existing calculator
    # ------------------------------------------------------------------
    result = calculate_health_score(
        patient_features=norm_feats,
        age=args.age,
        gender=args.gender,
        patient_id=args.patient_id,
        baselines=baselines,
    )
    zscore_health = float(result.health_score)

    # ------------------------------------------------------------------
    # 5. Advanced features for ensemble, then ensemble probabilities
    # ------------------------------------------------------------------
    use_cnn = not args.no_cnn
    adv_extractor = AdvancedFeatureExtractor(use_cnn=use_cnn)
    adv_feats = adv_extractor.extract_image_features(image_path)

    base_vec = np.array(
        [norm_feats.get(k) or 0.0 for k in ALL_FEATURE_NAMES],
        dtype=np.float32,
    )
    adv_vec = np.array(
        [adv_feats.get(k) or 0.0 for k in ADVANCED_FEATURE_NAMES],
        dtype=np.float32,
    )
    X = np.concatenate([base_vec, adv_vec]).reshape(1, -1)

    proba = ensemble.predict_proba(
        X,
        health_scores=np.array([zscore_health]),
        method=args.ensemble_method,
    )[0]

    # Threshold optimizer -> predicted class index
    pred_idx = int(threshold_opt.predict(proba.reshape(1, -1))[0])
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(np.max(proba))

    # ------------------------------------------------------------------
    # 6. Blend z-score health score + ensemble probability-based score
    # ------------------------------------------------------------------
    ensemble_score = _prob_to_score(proba)
    blended_score = _blend_score(
        ensemble_score,
        zscore_health,
        w_ensemble=0.7,
        w_zscore=0.3,
    )

    # Overwrite result.health_score to use blended (CBHS) score
    result.health_score = int(round(blended_score))

    # ------------------------------------------------------------------
    # 7. Save only the PNG report, and print summary to stdout
    # ------------------------------------------------------------------
    report_dir = os.path.join(args.output_dir, "patient_reports")
    os.makedirs(report_dir, exist_ok=True)

    report_path = plot_patient_report(result, save_dir=report_dir)

    print("\n=== CBHS INFERENCE RESULT ===")
    print(f"Patient ID:        {args.patient_id}")
    print(f"Predicted class:   {pred_label} (confidence {confidence:.2f})")
    print(f"Z-score health:    {zscore_health:.2f} / 10")
    print(f"Ensemble score:    {ensemble_score:.2f} / 10")
    print(f"Blended CBHS:      {blended_score:.2f} / 10")
    print(f"Final (rounded):   {result.health_score} / 10")
    print(f"Report PNG saved:  {report_path}\n")


if __name__ == "__main__":
    main()
