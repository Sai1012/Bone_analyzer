import os
import tempfile
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from baseline_builder import load_baselines
from feature_engineering import AdvancedFeatureExtractor, ADVANCED_FEATURE_NAMES
from feature_extractor import (
    ALL_FEATURE_NAMES,
    CLINICAL_FEATURE_NAMES,
    extract_clinical_features,
    extract_image_features,
    load_normaliser,
)
from health_scale_calculator import calculate_health_score, results_to_dataframe
from ensemble_classifier import BoneHealthEnsemble
from threshold_optimizer import ThresholdOptimizer
from visualizer import plot_patient_report


CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]


def _prob_to_score(proba: np.ndarray) -> float:
    centres = np.array([9.0, 6.5, 4.0], dtype=float)  # normal, osteopenia, osteoporosis
    return float(np.dot(proba, centres))


def _blend_score(
    ensemble_score: float,
    zscore_score: float,
    w_ensemble: float = 0.7,
    w_zscore: float = 0.3,
) -> float:
    blended = (w_ensemble * ensemble_score) + (w_zscore * zscore_score)
    return float(np.clip(blended, 1.0, 10.0))


st.set_page_config(page_title="CBHS Single-Patient Test", layout="wide")

st.title("CBHS – Single Patient Test (Improved Pipeline)")
st.caption(
    "Upload one knee X-ray + clinical inputs and get a blended ensemble + z-score health score."
)

with st.sidebar:
    st.header("Model & Artifact Paths")
    baselines_path = st.text_input("Baselines JSON", "output/baselines.json")
    normaliser_path = st.text_input(
        "Feature normaliser JSON", "output/improved/feature_normaliser.json"
    )
    model_path = st.text_input(
        "Ensemble model PKL", "output/improved/ensemble_model.pkl"
    )
    thresholds_path = st.text_input(
        "Optimal thresholds JSON", "output/improved/optimal_thresholds.json"
    )
    output_dir = st.text_input("Output directory", "output/improved")
    ensemble_method = st.selectbox("Ensemble method", ["soft", "stack"], index=0)
    use_cnn = st.checkbox("Use CNN features (requires PyTorch)", value=False)

st.subheader("Patient Details")
col1, col2, col3 = st.columns(3)
with col1:
    patient_id = st.text_input("Patient ID", "uploaded_1")
    age = st.number_input("Age", min_value=1, max_value=120, value=60)
with col2:
    gender = st.selectbox("Gender", ["male", "female", "unknown"], index=0)
with col3:
    image_file = st.file_uploader(
        "Upload Knee X-ray (JPEG/PNG)", type=["jpg", "jpeg", "png"]
    )

st.subheader("Clinical Inputs")

binary_fields = {
    "smoker": "Smoker",
    "alcoholic": "Alcoholic",
    "diabetic": "Diabetic",
    "hypothyroidism": "Hypothyroidism",
    "estrogen_use": "Estrogen use",
    "history_of_fracture": "History of fracture",
    "family_history": "Family history",
}

numeric_fields = {
    "bmi": (0.0, 60.0, 22.0),
    "t_score": (-5.0, 5.0, -1.0),
    "z_score": (-5.0, 5.0, -1.0),
    "walking_distance": (0.0, 20.0, 2.0),
}

clin_values: Dict[str, Optional[float]] = {}

c1, c2, c3 = st.columns(3)
with c1:
    for key in ["bmi", "t_score", "z_score"]:
        lo, hi, default = numeric_fields[key]
        clin_values[key] = st.number_input(
            key, min_value=lo, max_value=hi, value=default
        )
with c2:
    lo, hi, default = numeric_fields["walking_distance"]
    clin_values["walking_distance"] = st.number_input(
        "walking_distance", min_value=lo, max_value=hi, value=default
    )
with c3:
    for key, label in binary_fields.items():
        clin_values[key] = 1 if st.selectbox(label, ["No", "Yes"], index=0) == "Yes" else 0

if st.button("Analyze"):
    if image_file is None:
        st.error("Please upload a knee X-ray image.")
        st.stop()

    missing = [
        p for p in [baselines_path, normaliser_path, model_path] if not os.path.exists(p)
    ]
    if missing:
        st.error("Missing required files: " + ", ".join(missing))
        st.stop()

    # Save uploaded image to temp
    suffix = os.path.splitext(image_file.name)[1].lower() or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image_file.read())
        image_path = tmp.name

    # Build clinical series
    row_data = {**clin_values, "age": age, "gender": gender, "patient_id": patient_id}
    row = pd.Series(row_data)

    # Load normaliser & baselines
    normaliser = load_normaliser(normaliser_path)
    baselines = load_baselines(baselines_path)

    # Extract base features
    img_feats = extract_image_features(image_path)
    clin_feats = extract_clinical_features(row)
    combined = {**img_feats, **clin_feats}
    norm_feats = normaliser.transform(combined)

    # Z-score based health score
    result = calculate_health_score(
        patient_features=norm_feats,
        age=age,
        gender=gender,
        patient_id=patient_id,
        baselines=baselines,
    )
    zscore_health = float(result.health_score)

    # Advanced image features for ensemble
    adv_extractor = AdvancedFeatureExtractor(use_cnn=use_cnn)
    adv_feats = adv_extractor.extract_image_features(image_path)

    base_vec = np.array(
        [norm_feats.get(k) or 0.0 for k in ALL_FEATURE_NAMES], dtype=np.float32
    )
    adv_vec = np.array(
        [adv_feats.get(k) or 0.0 for k in ADVANCED_FEATURE_NAMES], dtype=np.float32
    )
    X = np.concatenate([base_vec, adv_vec]).reshape(1, -1)

    ensemble = BoneHealthEnsemble.load(model_path)
    proba = ensemble.predict_proba(
        X, health_scores=np.array([zscore_health]), method=ensemble_method
    )[0]

    # Threshold-based class label if thresholds exist
    if os.path.exists(thresholds_path):
        threshold_opt = ThresholdOptimizer.load(thresholds_path)
        pred_idx = int(threshold_opt.predict(proba.reshape(1, -1))[0])
    else:
        pred_idx = int(np.argmax(proba))

    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(np.max(proba))

    ensemble_score = _prob_to_score(proba)
    blended_score = _blend_score(
        ensemble_score, zscore_health, w_ensemble=0.7, w_zscore=0.3
    )

    # Replace health_score with blended (rounded int)
    result.health_score = int(round(blended_score))

    st.success("Analysis complete")

    colA, colB, colC = st.columns(3)
    colA.metric("Blended Health Score", f"{blended_score:.2f} / 10")
    colB.metric("Final Score (rounded)", f"{result.health_score} / 10")
    colC.metric("Predicted Class", f"{pred_label} ({confidence:.2f})")

    st.write("**Z-score Health Score (baseline):**", f"{zscore_health:.2f}")
    st.write("**Ensemble Probability Score:**", f"{ensemble_score:.2f}")

    # Generate report PNG
    os.makedirs(output_dir, exist_ok=True)
    report_path = plot_patient_report(
        result, save_dir=os.path.join(output_dir, "patient_reports")
    )
    st.image(report_path, caption="Patient Report")

    # Tabular output
    df_out = results_to_dataframe([result])
    st.dataframe(df_out, use_container_width=True)

    st.download_button(
        label="Download CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name=f"{patient_id}_health_score.csv",
        mime="text/csv",
    )
