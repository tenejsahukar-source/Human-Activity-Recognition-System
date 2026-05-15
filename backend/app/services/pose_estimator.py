"""
PoseEstimator
─────────────
Wraps MediaPipe Pose to:
  • Detect 33 body landmarks per frame
  • Return raw landmark objects
  • Flatten landmarks to a normalised feature vector for the classifier
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List


# Number of keypoints × coordinates (x, y, z)
FEATURE_DIM = 33 * 3   # = 99


class PoseEstimator:
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(
        self, frame: np.ndarray
    ) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """
        Run pose estimation on a BGR frame.

        Returns
        -------
        landmarks : list of landmark objects, or None if no person found
        annotated  : frame with skeleton drawn (or original if none found)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_landmarks:
            return None, frame

        # Draw skeleton on a copy
        annotated = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        return results.pose_landmarks.landmark, annotated

    # ── Feature engineering ──────────────────────────────────────────────────

    def landmarks_to_features(self, landmarks) -> np.ndarray:
        """
        Flatten 33 landmarks (x, y, z) → float32 vector of length 99.

        Normalisation strategy
        ─────────────────────
        • Use the hip midpoint as the origin so the vector is
          translation-invariant.
        • Divide by the torso height (shoulder–hip distance) so the
          vector is scale-invariant.
        """
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32
        )  # shape (33, 3)

        # MediaPipe landmark indices
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12

        hip_mid = (coords[LEFT_HIP] + coords[RIGHT_HIP]) / 2.0
        shoulder_mid = (coords[LEFT_SHOULDER] + coords[RIGHT_SHOULDER]) / 2.0

        torso_height = np.linalg.norm(shoulder_mid - hip_mid)
        if torso_height < 1e-6:
            torso_height = 1.0   # safety guard

        # Centre + scale
        coords -= hip_mid
        coords /= torso_height

        return coords.flatten()   # (99,)

    def get_feature_dim(self) -> int:
        return FEATURE_DIM
