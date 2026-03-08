"""
app/services/quality_classifier.py
------------------------------------
Loads the trained model once and exposes classify_pose_results(),
which takes pose_results directly from pose_service and returns a
quality assessment dict ready to be injected into the Gemini prompt.

Requires:
    models/climb_quality_classifier.pkl
    models/climb_quality_scaler.pkl

Run once to generate those files:
    python train/generate_data.py
    python train/train_quality_model.py
"""

import os
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import joblib

from app.services.quality_features import extract_quality_features, FEATURE_COLUMNS

logger = logging.getLogger(__name__)

MODEL_PATH  = os.path.join("models", "climb_quality_classifier.pkl")
SCALER_PATH = os.path.join("models", "climb_quality_scaler.pkl")


class ClimbQualityClassifier:
    """Load once at startup, call classify_pose_results() per job."""

    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run: python train/generate_data.py && python train/train_quality_model.py"
            )
        self._clf    = joblib.load(model_path)
        self._scaler = joblib.load(scaler_path)
        logger.info("ClimbQualityClassifier loaded from %s", model_path)

    def classify_pose_results(self, pose_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main entry point. Takes pose_results from pose_service.process_frames().

        Returns:
            {
                "verdict":       "good" | "bad",
                "score":         0–100   (technique score, float),
                "confidence":    0–100   (model confidence, float),
                "feature_flags": {
                    "arm_efficiency":      "good" | "needs_work",
                    "hip_positioning":     "good" | "needs_work",
                    "movement_smoothness": "good" | "needs_work",
                    "hesitation":          "good" | "needs_work",
                    "body_tension":        "good" | "needs_work",
                    "range_of_motion":     "good" | "needs_work",
                },
                "raw_features": { ... }   # full 25-feature dict for Gemini
            }

        Returns None if pose detection rate was too low to classify.
        """
        features = extract_quality_features(pose_results)
        if features is None:
            logger.warning("Too few detected pose frames to classify quality.")
            return None

        x        = np.array([[features[c] for c in FEATURE_COLUMNS]], dtype=np.float32)
        x_scaled = self._scaler.transform(x)
        proba    = self._clf.predict_proba(x_scaled)[0]   # [p_bad, p_good]

        score      = round(float(proba[1]) * 100, 1)
        confidence = round(float(max(proba)) * 100, 1)
        verdict    = "good" if score >= 50 else "bad"

        return {
            "verdict":       verdict,
            "score":         score,
            "confidence":    confidence,
            "feature_flags": self._compute_flags(features),
            "raw_features":  features,
        }

    @staticmethod
    def _compute_flags(f: Dict[str, float]) -> Dict[str, str]:
        def flag(ok: bool) -> str:
            return "good" if ok else "needs_work"
        return {
            "arm_efficiency":      flag(f["mean_elbow_angle"] > 145 and f["straight_arm_pct"] > 0.40),
            "hip_positioning":     flag(f["hip_wall_dist_mean"] < 0.22 and f["hip_drop_count"] < 3),
            "movement_smoothness": flag(f["mean_jerk"] < 1.5 and f["velocity_cv"] < 1.0),
            "hesitation":          flag(f["hesitation_pct"] < 0.25 and f["pause_count"] < 5),
            "body_tension":        flag(f["body_tension_score"] > 0.45),
            "range_of_motion":     flag(f["rom_l_elbow"] > 50 and f["rom_l_knee"] > 45),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_singleton: Optional[ClimbQualityClassifier] = None

def get_classifier() -> ClimbQualityClassifier:
    global _singleton
    if _singleton is None:
        _singleton = ClimbQualityClassifier()
    return _singleton
