"""
main.py - Orchestrate the complete Bone Health Analysis pipeline.

Usage
-----
# Run full pipeline (all patients, all charts)
python main.py

# Run on a specific subset
python main.py --max-patients 10

# Skip individual patient reports (much faster for large runs)
python main.py --no-patient-reports

# Use a custom images folder
python main.py --images-folder /path/to/images

# Use a custom Excel file
python main.py --excel /path/to/dataset.xlsx

# Skip image feature extraction (clinical features only)
python main.py --no-image-features
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure the project root is on the path when running as a script
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
from visualizer import generate_all_charts, plot_patient_report

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Bone Health Analysis System – 1-10 health scale from X-rays + clinical data",
    )
    parser.add_argument(
        "--images-folder",
        default=LOCAL_IMAGES_FOLDER,
        help="Path to directory containing the JPEG X-ray images (default: from config.py)",
    )
    parser.add_argument(
        "--excel",
        default=EXCEL_FILE,
        help="Path to osteoporosis_knee_dataset.xlsx (default: from config.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for saving all outputs (default: output/)",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit processing to the first N patients (useful for quick tests)",
    )
    parser.add_argument(
        "--no-patient-reports",
        action="store_true",
        help="Skip generating individual patient report PNGs (faster for large runs)",
    )
    parser.add_argument(
        "--no-image-features",
        action="store_true",
        help="Skip image-based feature extraction (only clinical features are used)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    """Execute the full analysis pipeline.

    Returns a DataFrame with one row per patient containing health scores and
    feature deviations.
    """
    t_start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    df = load_dataset(
        excel_path=args.excel,
        images_folder=args.images_folder,
    )

    if args.max_patients:
        logger.info("Limiting to first %d patients (--max-patients)", args.max_patients)
        df = df.head(args.max_patients).reset_index(drop=True)

    print(f"      Loaded {len(df)} patients.")

    # ── Step 2: Extract features ─────────────────────────────────────────────
    print("\n[2/5] Extracting features...")

    if args.no_image_features:
        logger.info("Image feature extraction disabled (--no-image-features)")
        # Temporarily null out image paths so extractor returns None image features
        df_mod = df.copy()
        df_mod["local_image_path"] = None
        features, normaliser = extract_all_features(df_mod, normalise=True)
    else:
        features, normaliser = extract_all_features(df, normalise=True)

    print(f"      Extracted features for {len(features)} patients.")

    # ── Step 3: Build baselines from normal cases ────────────────────────────
    print("\n[3/5] Building age-gender baselines from normal cases...")
    baselines = build_baselines(df, features)
    normal_count = sum(1 for k, v in baselines.items()
                       if "_all_genders" not in k and "all_ages_" not in k and k != "global")
    print(f"      Created {normal_count} primary strata.")

    # ── Step 4: Calculate health scores ─────────────────────────────────────
    print("\n[4/5] Calculating health scores (1-10)...")
    results = calculate_all_health_scores(df, features, baselines)
    print(f"      Scored {len(results)} patients.")

    summarise_scores(results, df)

    # Save results CSV and Excel
    results_df = results_to_dataframe(results)
    # Merge original diagnosis and raw clinical scores for reference
    if "diagnosis" in df.columns:
        results_df["actual_diagnosis"] = df["diagnosis"].values
    if "t_score" in df.columns:
        results_df["t_score"] = df["t_score"].values
    if "z_score" in df.columns:
        results_df["z_score"] = df["z_score"].values

    csv_path = os.path.join(args.output_dir, "health_scores.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"      Results saved to {csv_path}")

    xlsx_path = os.path.join(args.output_dir, "health_scores.xlsx")
    results_df.to_excel(xlsx_path, index=False)
    print(f"      Results saved to {xlsx_path}")

    # ── Step 5: Visualisations ───────────────────────────────────────────────
    print("\n[5/5] Generating visualisations...")

    chart_paths = generate_all_charts(results, df, save_dir=args.output_dir)
    for p in chart_paths:
        print(f"      Saved: {p}")

    if not args.no_patient_reports:
        patient_report_dir = os.path.join(args.output_dir, "patient_reports")
        os.makedirs(patient_report_dir, exist_ok=True)
        for result in results:
            try:
                plot_patient_report(result, save_dir=patient_report_dir)
            except Exception as exc:
                logger.warning("Failed to generate report for %s: %s", result.patient_id, exc)
        print(f"      Patient reports saved to {patient_report_dir}/")

    elapsed = time.time() - t_start
    print(f"\n✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   All outputs in: {args.output_dir}\n")

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_pipeline(args)
    except FileNotFoundError as exc:
        print(f"\n❌ File not found: {exc}", file=sys.stderr)
        print("   Check --excel and --images-folder paths.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error in pipeline: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
