"""
config.py - Central configuration for Bone Health Analysis System

Edit LOCAL_IMAGES_FOLDER to point to the directory containing your 239 JPEG files.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# Absolute path to the folder that holds the 239 X-ray JPEGs.
# Change this to match your machine's file system.
LOCAL_IMAGES_FOLDER = os.environ.get(
    "BONE_IMAGES_FOLDER",
    r"C:\Users\admin\Downloads\Bone\Osteoporosis Knee X-ray\images",
)

# Path to the Excel metadata file (relative to this file's directory)
EXCEL_FILE = os.path.join(os.path.dirname(__file__), "osteoporosis_knee_dataset.xlsx")

# Directory for saving all outputs (plots, CSVs, JSON baselines)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ──────────────────────────────────────────────────────────────────────────────
# Image naming patterns (prefix → diagnosis)
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_PREFIXES = {
    "N": "normal",
    "OP": "osteopenia",
    "OS": "osteoporosis",
}

IMAGE_EXTENSION = ".JPEG"

# ──────────────────────────────────────────────────────────────────────────────
# Age stratification groups  (inclusive lower bound, exclusive upper bound)
# ──────────────────────────────────────────────────────────────────────────────

AGE_GROUPS = [
    (0, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 200),
]

AGE_GROUP_LABELS = {
    (0, 30): "under_30",
    (30, 40): "30_40",
    (40, 50): "40_50",
    (50, 60): "50_60",
    (60, 70): "60_70",
    (70, 200): "70_plus",
}

# ──────────────────────────────────────────────────────────────────────────────
# Feature weights used when aggregating per-feature z-scores into a single
# composite deviation.  Weights are normalised internally so they don't need
# to sum to 1.
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_WEIGHTS = {
    # Image-based
    "bone_density_mean": 2.0,
    "bone_density_std": 1.0,
    "joint_space_width": 2.0,
    "cortical_thickness": 2.0,
    "texture_contrast": 1.0,
    "texture_homogeneity": 1.0,
    "texture_energy": 1.0,
    "texture_correlation": 1.0,
    "edge_density": 1.5,
    "geometric_area": 1.0,
    "geometric_perimeter": 1.0,
    # Clinical
    "bmi": 1.5,
    "t_score": 2.5,
    "z_score": 2.0,
    "walking_distance": 1.0,
    "smoker": 1.0,
    "alcoholic": 0.5,
    "diabetic": 0.5,
    "hypothyroidism": 0.5,
    "estrogen_use": 0.5,
    "history_of_fracture": 1.0,
    "family_history": 0.5,
}

# ──────────────────────────────────────────────────────────────────────────────
# Health scale mapping  (composite_deviation_std → health_score)
# Lower composite deviation → higher health score.
# Each tuple: (max_deviation, score)  — first matching threshold wins.
# ──────────────────────────────────────────────────────────────────────────────

HEALTH_SCALE_THRESHOLDS = [
    (0.5, 10),
    (1.0, 9),
    (2.0, 8),
    (3.0, 7),
    (4.0, 6),
    (5.0, 5),
    (6.0, 4),
    (8.0, 3),
    (10.0, 2),
]
# Anything exceeding 10.0 std → score = 1

# ──────────────────────────────────────────────────────────────────────────────
# Baseline builder settings
# ──────────────────────────────────────────────────────────────────────────────

# Minimum number of samples in a stratum before it is considered reliable
MIN_STRATUM_SAMPLES = 3

# Fallback: if a patient's exact age-gender stratum has fewer than
# MIN_STRATUM_SAMPLES, widen to the nearest larger group (gender-only, then
# age-only, then global).
BASELINE_FALLBACK = True

# ──────────────────────────────────────────────────────────────────────────────
# Accuracy improvement settings  (used by improved_main.py and sub-modules)
# ──────────────────────────────────────────────────────────────────────────────

# --- Advanced Feature Engineering (feature_engineering.py) ---
# Whether to attempt CNN feature extraction using ResNet50 (requires PyTorch)
FEATURE_ENGINEERING_USE_CNN = False  # Set True if PyTorch is installed

# Image target size for advanced feature extraction (H, W)
FEATURE_ENGINEERING_TARGET_SIZE = (256, 256)

# --- Class Imbalance (class_imbalance_handler.py) ---
# Random state for reproducibility across SMOTE / splits
IMBALANCE_RANDOM_STATE = 42

# k-nearest neighbours for SMOTE (reduced if minority class is very small)
SMOTE_K_NEIGHBORS = 5

# Fraction of samples to hold out for testing
TEST_SIZE = 0.2

# --- Ensemble Classifier (ensemble_classifier.py) ---
# Number of trees in the Random Forest
RF_N_ESTIMATORS = 100

# Number of boosting rounds (XGBoost) or trees (Extra Trees fallback)
BOOST_N_ESTIMATORS = 50

# Hidden layer sizes for the MLP neural network
NN_HIDDEN_LAYERS = (256, 128, 64)

# Ensemble voting method: "soft" | "hard" | "stack"
ENSEMBLE_METHOD = "soft"

# --- Threshold Optimisation (threshold_optimizer.py) ---
# Class names in the canonical order used throughout improved modules
CLASS_NAMES = ["normal", "osteopenia", "osteoporosis"]

# ──────────────────────────────────────────────────────────────────────────────
# Column name mappings  (Excel header → internal key)
# Trailing whitespace and special characters in the Excel headers are handled
# by data_loader.py; the mapping below uses stripped header names.
# ──────────────────────────────────────────────────────────────────────────────

COLUMN_MAP = {
    "S.No": "sno",
    "Patient Id": "patient_id",
    "Joint Pain:": "joint_pain",
    "Gender": "gender",
    "Age": "age",
    "Menopause Age": "menopause_age",
    "height  (meter)": "height",
    "Weight (KG) ": "weight",
    "Smoker": "smoker",
    "Alcoholic": "alcoholic",
    "Diabetic": "diabetic",
    "Hypothyroidism": "hypothyroidism",
    "Number of Pregnancies": "num_pregnancies",
    "Seizer Disorder": "seizure_disorder",
    "Estrogen Use": "estrogen_use",
    "Occupation ": "occupation",
    "History of Fracture": "history_of_fracture",
    "Dialysis:": "dialysis",
    "Family History of Osteoporosis": "family_history",
    "Maximum Walking distance (km)": "walking_distance",
    "Daily Eating habits": "eating_habits",
    "Medical History": "medical_history",
    "T-score Value": "t_score",
    "Z-Score Value": "z_score",
    "BMI: ": "bmi",
    "Site": "site",
    "Obesity": "obesity",
    "Diagnosis": "diagnosis",
    "image_file": "image_file",
    "image_path": "image_path",
}
