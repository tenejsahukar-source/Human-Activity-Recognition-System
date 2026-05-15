"""
ActivityClassifier
──────────────────
Two-mode classifier:

  Mode A – Trained sklearn model (preferred)
    Load a pre-trained RandomForest / SVM from  models/activity_model.pkl
    if it exists.

  Mode B – Rule-based heuristic (zero-dependency fallback)
    Uses joint angles and relative positions derived from the 99-dim
    feature vector to classify: Walking, Running, Sitting, Standing,
    Jumping, Lying Down.

The public interface is identical for both modes:
  predict(features) → (activity_label, confidence, all_probs_dict)
"""

import os
import pickle
import logging
import numpy as np
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ── Label set ────────────────────────────────────────────────────────────────
ACTIVITIES = ["Walking", "Running", "Sitting", "Standing", "Jumping", "Lying Down"]

# ── Landmark indices in the flattened feature vector ─────────────────────────
# After flattening (33, 3) we index as  i*3, i*3+1, i*3+2  for x, y, z
_IDX = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_wrist":     15,
    "right_wrist":    16,
}


def _xy(features: np.ndarray, name: str) -> Tuple[float, float]:
    """Return (x, y) for a named landmark from the flat feature vector."""
    base = _IDX[name] * 3
    return features[base], features[base + 1]


class ActivityClassifier:
    def __init__(self, model_path: str = "models/activity_model.pkl"):
        self.model = None
        self.label_encoder = None
        self._load_model(model_path)

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_model(self, path: str):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                # Support both plain model and {"model": ..., "le": ...} dicts
                if isinstance(data, dict):
                    self.model = data.get("model")
                    self.label_encoder = data.get("label_encoder")
                else:
                    self.model = data
                logger.info(f"✅ Loaded trained model from {path}")
            except Exception as e:
                logger.warning(f"Could not load model ({e}). Using rule-based fallback.")
        else:
            logger.info("No trained model found — using rule-based heuristic classifier.")

    # ── Public API ───────────────────────────────────────────────────────────

    def predict(
        self, features: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Returns
        -------
        activity   : predicted label string
        confidence : float in [0, 1]
        all_probs  : dict {label: probability}
        """
        if self.model is not None:
            return self._sklearn_predict(features)
        return self._rule_based_predict(features)

    # ── Sklearn path ─────────────────────────────────────────────────────────

    def _sklearn_predict(
        self, features: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        X = features.reshape(1, -1)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            classes = (
                self.label_encoder.classes_
                if self.label_encoder
                else self.model.classes_
            )
            idx = int(np.argmax(probs))
            label = str(classes[idx])
            conf = float(probs[idx])
            all_probs = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
        else:
            pred = self.model.predict(X)[0]
            label = str(pred)
            conf = 1.0
            all_probs = {label: 1.0}
        return label, conf, all_probs

    # ── Rule-based heuristic ─────────────────────────────────────────────────

    def _rule_based_predict(
        self, features: np.ndarray
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Heuristic classification based on joint geometry.

        Key signals:
          • hip_y          – vertical position of hips (how low they are)
          • knee_bend      – mean knee angle
          • body_vertical  – how upright the body is
          • ankle_spread   – horizontal distance between ankles (step width)
          • hip_velocity   – approximation from nose–hip distance
        """
        # Raw coordinates (hip-centred, torso-normalised)
        nose_x,           nose_y           = _xy(features, "nose")
        l_hip_x,          l_hip_y          = _xy(features, "left_hip")
        r_hip_x,          r_hip_y          = _xy(features, "right_hip")
        l_knee_x,         l_knee_y         = _xy(features, "left_knee")
        r_knee_x,         r_knee_y         = _xy(features, "right_knee")
        l_ankle_x,        l_ankle_y        = _xy(features, "left_ankle")
        r_ankle_x,        r_ankle_y        = _xy(features, "right_ankle")
        l_shoulder_x,     l_shoulder_y     = _xy(features, "left_shoulder")
        r_shoulder_x,     r_shoulder_y     = _xy(features, "right_shoulder")

        hip_y      = (l_hip_y + r_hip_y) / 2.0
        knee_y     = (l_knee_y + r_knee_y) / 2.0
        ankle_y    = (l_ankle_y + r_ankle_y) / 2.0
        shoulder_y = (l_shoulder_y + r_shoulder_y) / 2.0

        ankle_spread = abs(l_ankle_x - r_ankle_x)
        knee_spread  = abs(l_knee_x  - r_knee_x)

        # Knee bend: angle between hip–knee–ankle vectors
        l_knee_bend = _angle(
            np.array([l_hip_x,   l_hip_y]),
            np.array([l_knee_x,  l_knee_y]),
            np.array([l_ankle_x, l_ankle_y]),
        )
        r_knee_bend = _angle(
            np.array([r_hip_x,   r_hip_y]),
            np.array([r_knee_x,  r_knee_y]),
            np.array([r_ankle_x, r_ankle_y]),
        )
        mean_knee_bend = (l_knee_bend + r_knee_bend) / 2.0

        # Body verticality: shoulder–hip vertical alignment
        body_tilt = abs(shoulder_y - hip_y)   # large = upright in norm. coords

        # ── Decision tree ────────────────────────────────────────────────────

        scores: Dict[str, float] = {a: 0.0 for a in ACTIVITIES}

        # Lying Down: nose is near the same y-level as hips
        if abs(nose_y - hip_y) < 0.3 and body_tilt < 0.4:
            scores["Lying Down"] += 0.9

        # Sitting: hips are LOW and knees are highly bent
        elif hip_y > 0.2 and mean_knee_bend < 110:
            scores["Sitting"] += 0.8 + max(0, (110 - mean_knee_bend) / 200)

        # Jumping: ankles are ABOVE hips in normalised space (negative y offset)
        elif ankle_y < hip_y - 0.3:
            scores["Jumping"] += 0.85

        else:
            # Standing vs Walking vs Running — differentiate by ankle spread + knee bend
            if ankle_spread < 0.15 and mean_knee_bend > 160:
                scores["Standing"] += 0.85
            elif ankle_spread < 0.35 and mean_knee_bend > 140:
                scores["Walking"] += 0.75 + ankle_spread * 0.3
            else:
                scores["Running"] += 0.70 + ankle_spread * 0.2

        # Normalise to sum = 1
        total = sum(scores.values()) or 1.0
        probs = {k: round(v / total, 4) for k, v in scores.items()}

        best = max(probs, key=probs.get)
        return best, probs[best], probs


# ── Geometry helper ───────────────────────────────────────────────────────────

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle (degrees) at point B formed by vectors BA and BC.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
