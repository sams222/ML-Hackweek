"""
app/services/quality_features.py
---------------------------------
Extracts biomechanical quality features from pose_results — the list of dicts
already produced by app/services/pose_service.process_frames().

Each pose_result dict is expected to have:
    {
        "detected":        bool,
        "landmarks":       list of 33 NormalizedLandmark  (or None if not detected),
        "angles":          dict[str, float],   # joint angles in degrees
        "annotated_frame": np.ndarray,
        "timestamp_ms":    float,
    }

No new MediaPipe calls are made — we reuse whatever pose_service already computed.

Output: a flat dict of 25 floats, ready for ClimbQualityClassifier.
"""

import numpy as np
from typing import List, Dict, Any

# ── MediaPipe landmark indices ────────────────────────────────────────────────
_IDX = {
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
}

# Must match train/generate_data.py FEATURE_COLUMNS exactly
FEATURE_COLUMNS = [
    "hesitation_pct", "pause_count", "max_pause_duration",
    "mean_elbow_angle", "straight_arm_pct", "elbow_angle_variance",
    "hip_wall_dist_mean", "hip_wall_dist_std", "hip_drop_count",
    "mean_jerk", "max_jerk", "jerk_std",
    "velocity_cv",
    "rom_l_elbow", "rom_r_elbow", "rom_l_shoulder", "rom_r_shoulder",
    "rom_l_knee", "rom_r_knee", "rom_l_hip",
    "elbow_lr_correlation", "knee_lr_correlation", "body_tension_score",
    "mean_shoulder_angle", "shoulder_angle_variance",
]


def _lm(landmarks: list, name: str) -> np.ndarray:
    lm = landmarks[_IDX[name]]
    return np.array([lm.x, lm.y, lm.z])


def _longest_run(bool_arr: np.ndarray) -> int:
    max_run = cur_run = 0
    for v in bool_arr:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def extract_quality_features(pose_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Args:
        pose_results: output of pose_service.process_frames() — list of per-frame dicts.

    Returns:
        Dict mapping feature name → float, with keys matching FEATURE_COLUMNS.
        Returns None if fewer than 5 frames had detected poses.
    """
    detected = [r for r in pose_results if r.get("detected") and r.get("landmarks")]

    if len(detected) < 5:
        return None

    landmarks_seq = [r["landmarks"] for r in detected]
    T = len(landmarks_seq)

    # ── 1. Joint angles from landmarks ───────────────────────────────────────
    # pose_service may already compute some angles in r["angles"],
    # but we recompute from landmarks to get the full set we need.
    angle_keys = ["l_elbow", "r_elbow", "l_shoulder", "r_shoulder",
                  "l_knee", "r_knee", "l_hip"]

    def _joint_angle(lms, a_name, b_name, c_name):
        a, b, c = _lm(lms, a_name), _lm(lms, b_name), _lm(lms, c_name)
        ba, bc = a - b, c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    angle_triples = {
        "l_elbow":    ("l_shoulder", "l_elbow",    "l_wrist"),
        "r_elbow":    ("r_shoulder", "r_elbow",    "r_wrist"),
        "l_shoulder": ("l_elbow",    "l_shoulder", "l_hip"),
        "r_shoulder": ("r_elbow",    "r_shoulder", "r_hip"),
        "l_knee":     ("l_hip",      "l_knee",     "l_ankle"),
        "r_knee":     ("r_hip",      "r_knee",     "r_ankle"),
        "l_hip":      ("l_shoulder", "l_hip",      "l_knee"),
    }

    angles: Dict[str, np.ndarray] = {}
    for key, (a, b, c) in angle_triples.items():
        angles[key] = np.array([_joint_angle(lms, a, b, c) for lms in landmarks_seq])

    # ── 2. Hip wall-distance proxy (z-coordinate of hip midpoint) ────────────
    hip_z = np.array([
        (_lm(lms, "l_hip")[2] + _lm(lms, "r_hip")[2]) / 2.0
        for lms in landmarks_seq
    ])

    # ── 3. Full-body velocity via wrist midpoint ──────────────────────────────
    wrist_pos = np.array([
        (_lm(lms, "l_wrist") + _lm(lms, "r_wrist")) / 2.0
        for lms in landmarks_seq
    ])
    body_vel = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)  # (T-1,)

    # ── 4. Hesitation ─────────────────────────────────────────────────────────
    still_thresh = 0.005
    is_still = body_vel < still_thresh
    hesitation_pct     = float(np.mean(is_still))
    pause_count        = int(np.sum(np.diff(is_still.astype(int)) == 1))
    max_pause_duration = float(_longest_run(is_still))

    # ── 5. Arm efficiency ─────────────────────────────────────────────────────
    mean_elbow_angle     = float((angles["l_elbow"].mean() + angles["r_elbow"].mean()) / 2)
    straight_arm_pct     = float(np.mean((angles["l_elbow"] > 155) | (angles["r_elbow"] > 155)))
    elbow_angle_variance = float((angles["l_elbow"].var() + angles["r_elbow"].var()) / 2)

    # ── 6. Hip positioning ────────────────────────────────────────────────────
    hip_wall_dist_mean = float(hip_z.mean())
    hip_wall_dist_std  = float(hip_z.std())
    hip_drop_count     = int(np.sum(np.diff(hip_z) > 0.03))

    # ── 7. Movement smoothness (jerk = 3rd derivative of position) ───────────
    if len(wrist_pos) >= 4:
        jerk_vec       = np.diff(wrist_pos, n=3, axis=0)
        jerk_mag       = np.linalg.norm(jerk_vec, axis=1)
        mean_jerk      = float(jerk_mag.mean())
        max_jerk       = float(jerk_mag.max())
        jerk_std       = float(jerk_mag.std())
    else:
        mean_jerk = max_jerk = jerk_std = 0.0

    # ── 8. Velocity consistency ───────────────────────────────────────────────
    velocity_cv = float(body_vel.std() / (body_vel.mean() + 1e-8))

    # ── 9. Range of motion ────────────────────────────────────────────────────
    rom = {k: float(v.max() - v.min()) for k, v in angles.items()}

    # ── 10. Body tension (left/right coordination) ────────────────────────────
    elbow_lr_corr  = float(np.corrcoef(angles["l_elbow"],    angles["r_elbow"])[0, 1])
    knee_lr_corr   = float(np.corrcoef(angles["l_knee"],     angles["r_knee"])[0, 1])
    body_tension   = float((elbow_lr_corr + knee_lr_corr) / 2)

    # ── 11. Shoulder ──────────────────────────────────────────────────────────
    mean_shoulder_angle     = float((angles["l_shoulder"].mean() + angles["r_shoulder"].mean()) / 2)
    shoulder_angle_variance = float((angles["l_shoulder"].var()  + angles["r_shoulder"].var())  / 2)

    return {
        "hesitation_pct":          hesitation_pct,
        "pause_count":             float(pause_count),
        "max_pause_duration":      max_pause_duration,
        "mean_elbow_angle":        mean_elbow_angle,
        "straight_arm_pct":        straight_arm_pct,
        "elbow_angle_variance":    elbow_angle_variance,
        "hip_wall_dist_mean":      hip_wall_dist_mean,
        "hip_wall_dist_std":       hip_wall_dist_std,
        "hip_drop_count":          float(hip_drop_count),
        "mean_jerk":               mean_jerk,
        "max_jerk":                max_jerk,
        "jerk_std":                jerk_std,
        "velocity_cv":             velocity_cv,
        "rom_l_elbow":             rom["l_elbow"],
        "rom_r_elbow":             rom["r_elbow"],
        "rom_l_shoulder":          rom["l_shoulder"],
        "rom_r_shoulder":          rom["r_shoulder"],
        "rom_l_knee":              rom["l_knee"],
        "rom_r_knee":              rom["r_knee"],
        "rom_l_hip":               rom["l_hip"],
        "elbow_lr_correlation":    elbow_lr_corr,
        "knee_lr_correlation":     knee_lr_corr,
        "body_tension_score":      body_tension,
        "mean_shoulder_angle":     mean_shoulder_angle,
        "shoulder_angle_variance": shoulder_angle_variance,
    }
