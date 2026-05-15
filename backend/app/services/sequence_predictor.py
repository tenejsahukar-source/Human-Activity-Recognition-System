"""
SequencePredictor
─────────────────
Maintains a sliding window of recent activity predictions and uses a
Markov-style transition matrix (learned from common human-motion patterns)
to forecast the next most-likely action.

If a trained LSTM / sequence model exists at  models/sequence_model.pkl,
it is loaded and used instead of the Markov approach.
"""

import os
import pickle
import logging
import numpy as np
from collections import deque, Counter
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

ACTIVITIES = ["Walking", "Running", "Sitting", "Standing", "Jumping", "Lying Down"]

# ── Empirical transition matrix (rows = current, cols = next) ────────────────
# P[i][j] = probability of transitioning from ACTIVITIES[i] to ACTIVITIES[j]
#
#             Walk  Run  Sit  Stand Jump  Lie
_TRANSITION = np.array([
    [0.30,  0.20, 0.10, 0.30, 0.05, 0.05],  # Walking  →
    [0.35,  0.25, 0.05, 0.25, 0.08, 0.02],  # Running  →
    [0.10,  0.05, 0.35, 0.40, 0.02, 0.08],  # Sitting  →
    [0.25,  0.10, 0.20, 0.35, 0.05, 0.05],  # Standing →
    [0.20,  0.30, 0.05, 0.35, 0.08, 0.02],  # Jumping  →
    [0.05,  0.02, 0.25, 0.50, 0.01, 0.17],  # Lying    →
], dtype=np.float32)

_ACT_IDX: Dict[str, int] = {a: i for i, a in enumerate(ACTIVITIES)}

HISTORY_LEN = 10   # sliding window size


class SequencePredictor:
    def __init__(
        self,
        history_len: int = HISTORY_LEN,
        model_path: str = "models/sequence_model.pkl",
    ):
        self.history: deque = deque(maxlen=history_len)
        self.history_len = history_len
        self.seq_model = None
        self._load_model(model_path)

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load_model(self, path: str):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    self.seq_model = pickle.load(f)
                logger.info(f"✅ Loaded sequence model from {path}")
            except Exception as e:
                logger.warning(f"Could not load sequence model ({e}). Using Markov fallback.")

    # ── Public API ───────────────────────────────────────────────────────────

    def update_and_predict(self, activity: str) -> str:
        """
        Push the latest activity label into the history window, then
        return the predicted NEXT action.
        """
        self.history.append(activity)

        if self.seq_model is not None:
            return self._model_predict()
        return self._markov_predict(activity)

    def get_history(self) -> List[str]:
        return list(self.history)

    # ── Markov prediction ─────────────────────────────────────────────────────

    def _markov_predict(self, current: str) -> str:
        """
        Blend two signals:
          1. First-order Markov transition from the current activity.
          2. Frequency of activities in the recent history window.

        This makes the predictor more stable when the same activity
        has been observed for several consecutive frames.
        """
        idx = _ACT_IDX.get(current)
        if idx is None:
            return "Unknown"

        # Signal 1: transition row
        trans_probs = _TRANSITION[idx].copy()

        # Signal 2: history frequency
        hist_list = list(self.history)
        freq_probs = np.zeros(len(ACTIVITIES), dtype=np.float32)
        if hist_list:
            cnt = Counter(hist_list)
            for act, c in cnt.items():
                i = _ACT_IDX.get(act)
                if i is not None:
                    freq_probs[i] = c / len(hist_list)

        # Weighted blend  (60% transition, 40% history)
        blended = 0.6 * trans_probs + 0.4 * freq_probs
        blended /= blended.sum()

        # Pick the top activity that is NOT the current one
        # (i.e. predict a *change* in activity)
        ranked = np.argsort(blended)[::-1]
        for r in ranked:
            candidate = ACTIVITIES[r]
            if candidate != current or blended[r] > 0.7:
                # If confidence is very high (>70%) we can predict same activity
                return candidate

        return ACTIVITIES[ranked[0]]

    # ── Trained model prediction ──────────────────────────────────────────────

    def _model_predict(self) -> str:
        """
        Use a trained sequence model (e.g. LSTM exported as sklearn pipeline).
        The model expects a fixed-length sequence of activity indices.
        """
        try:
            hist = list(self.history)
            # Pad / truncate to history_len
            indices = [_ACT_IDX.get(a, 0) for a in hist]
            while len(indices) < self.history_len:
                indices.insert(0, 0)
            indices = indices[-self.history_len:]

            X = np.array(indices, dtype=np.float32).reshape(1, -1)
            pred = self.seq_model.predict(X)[0]
            return str(pred)
        except Exception as e:
            logger.warning(f"Sequence model predict failed ({e}), falling back.")
            current = list(self.history)[-1] if self.history else "Standing"
            return self._markov_predict(current)
