"""
train_model.py
──────────────
Trains a RandomForest activity classifier and a Markov sequence model,
then saves them to the models/ directory.

Usage
─────
  python train_model.py                        # uses synthetic data
  python train_model.py --csv data/train.csv   # uses your real CSV

CSV format expected
───────────────────
  99 feature columns (flattened pose keypoints)  +  1 label column named "activity"

If your CSV has different column names, edit FEATURE_COLS / LABEL_COL below.
"""

import os
import argparse
import pickle
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTIVITIES = ["Walking", "Running", "Sitting", "Standing", "Jumping", "Lying Down"]
FEATURE_DIM = 99
LABEL_COL = "activity"
MODEL_DIR = "models"


# ─── Synthetic data generator ────────────────────────────────────────────────

def _generate_synthetic_data(n_per_class: int = 300) -> pd.DataFrame:
    """
    Creates rough pose signatures for each activity.
    Replace with real sensor / video data for production use.
    """
    rows = []
    rng = np.random.default_rng(42)

    def _noisy(base, noise=0.05, n=n_per_class):
        return base + rng.normal(0, noise, (n, FEATURE_DIM))

    # Walking  – moderate ankle spread, slight knee bend
    walk_base = np.zeros(FEATURE_DIM)
    walk_base[27*3]   =  0.15   # left ankle x spread
    walk_base[28*3]   = -0.15   # right ankle x spread
    walk_base[25*3+1] =  0.60   # left knee y
    walk_base[26*3+1] =  0.60   # right knee y

    # Running – wide ankle spread, more pronounced arm movement
    run_base = walk_base.copy()
    run_base[27*3]   =  0.28
    run_base[28*3]   = -0.28
    run_base[15*3+1] = -0.40   # wrists raised
    run_base[16*3+1] = -0.40

    # Sitting – knees bent sharply (knee_y ≈ hip_y), low hip
    sit_base = np.zeros(FEATURE_DIM)
    sit_base[23*3+1] =  0.35   # hip low
    sit_base[24*3+1] =  0.35
    sit_base[25*3+1] =  0.35   # knee near hip height
    sit_base[26*3+1] =  0.35
    sit_base[27*3+1] =  0.70   # ankle down
    sit_base[28*3+1] =  0.70

    # Standing – straight legs, hips centred
    stand_base = np.zeros(FEATURE_DIM)
    stand_base[0*3+1] = -0.90  # nose high
    stand_base[27*3+1] =  1.10  # ankles low
    stand_base[28*3+1] =  1.10

    # Jumping – ankles above hips (negative y offset)
    jump_base = stand_base.copy()
    jump_base[27*3+1] = -0.30
    jump_base[28*3+1] = -0.30
    jump_base[15*3+1] = -0.80   # arms up
    jump_base[16*3+1] = -0.80

    # Lying Down – nose ≈ hip height, all y-values similar
    lie_base = np.zeros(FEATURE_DIM)
    lie_base[0*3+1]  =  0.05
    lie_base[11*3+1] =  0.05
    lie_base[12*3+1] =  0.05
    lie_base[23*3+1] =  0.05
    lie_base[24*3+1] =  0.05

    for base, label in [
        (walk_base,  "Walking"),
        (run_base,   "Running"),
        (sit_base,   "Sitting"),
        (stand_base, "Standing"),
        (jump_base,  "Jumping"),
        (lie_base,   "Lying Down"),
    ]:
        X = _noisy(base)
        for row in X:
            rows.append({"activity": label, **{f"f{i}": v for i, v in enumerate(row)}})

    df = pd.DataFrame(rows)
    logger.info(f"Generated {len(df)} synthetic samples.")
    return df


# ─── Training ─────────────────────────────────────────────────────────────────

def train(csv_path: str = None):
    os.makedirs(MODEL_DIR, exist_ok=True)

    if csv_path and os.path.exists(csv_path):
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != LABEL_COL]
        X = df[feature_cols].values.astype(np.float32)
        y = df[LABEL_COL].values
    else:
        logger.info("No CSV provided — using synthetic data.")
        df = _generate_synthetic_data()
        feature_cols = [c for c in df.columns if c != LABEL_COL]
        X = df[feature_cols].values.astype(np.float32)
        y = df[LABEL_COL].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    logger.info("Training RandomForestClassifier …")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

    cv_scores = cross_val_score(pipe, X, y_enc, cv=5, scoring="accuracy")
    logger.info(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    model_path = os.path.join(MODEL_DIR, "activity_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": pipe, "label_encoder": le}, f)
    logger.info(f"✅ Saved activity model → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to training CSV")
    args = parser.parse_args()
    train(args.csv)
