"""
MediaPipe VIDEO mode pose estimation, adapted from MediaPipe-PoseLandmarks/main.py.
Replaces LIVE_STREAM + detect_async() with VIDEO mode + detect_for_video().
"""
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from app.config import settings

# Landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle at vertex b in degrees."""
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _lm_to_xy(lm) -> np.ndarray:
    return np.array([lm.x, lm.y])


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """Verbatim from MediaPipe-PoseLandmarks/main.py, adapted for VIDEO mode result."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style,
        )

    return annotated_image


def _build_detector() -> vision.PoseLandmarker:
    model_path = os.path.abspath(settings.mediapipe_model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _is_visible(lm, idx, threshold=0.3) -> bool:
    """Check if a landmark has sufficient visibility."""
    return lm[idx].visibility > threshold if hasattr(lm[idx], 'visibility') else True


def compute_joint_angles(landmarks: list) -> dict[str, float]:
    """Compute key joint angles from a list of NormalizedLandmark."""
    lm = landmarks

    angles = {}

    def safe_angle(name, a_idx, b_idx, c_idx):
        try:
            if not (_is_visible(lm, a_idx) and _is_visible(lm, b_idx) and _is_visible(lm, c_idx)):
                angles[name] = None
                return
            angles[name] = _angle_between(
                _lm_to_xy(lm[a_idx]),
                _lm_to_xy(lm[b_idx]),
                _lm_to_xy(lm[c_idx]),
            )
        except Exception:
            angles[name] = None

    safe_angle("left_elbow", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    safe_angle("right_elbow", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)
    safe_angle("left_knee", LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    safe_angle("right_knee", RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    safe_angle("left_hip", LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE)
    safe_angle("right_hip", RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE)

    # Trunk lean: angle between shoulder midpoint → hip midpoint vector and vertical
    try:
        shoulder_mid = (_lm_to_xy(lm[LEFT_SHOULDER]) + _lm_to_xy(lm[RIGHT_SHOULDER])) / 2
        hip_mid = (_lm_to_xy(lm[LEFT_HIP]) + _lm_to_xy(lm[RIGHT_HIP])) / 2
        trunk_vec = hip_mid - shoulder_mid
        vertical = np.array([0, 1])
        cos_lean = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)
        angles["trunk_lean"] = float(np.degrees(np.arccos(np.clip(cos_lean, -1.0, 1.0))))
    except Exception:
        angles["trunk_lean"] = None

    return angles


def process_frames(frames: list[tuple[np.ndarray, int]]) -> list[dict]:
    """
    Run MediaPipe VIDEO mode on a list of (rgb_frame, timestamp_ms) pairs.

    Returns a list of dicts:
        {
            "frame_idx": int,
            "timestamp_ms": int,
            "annotated_frame": np.ndarray (RGB),
            "landmarks": list | None,
            "angles": dict | None,
            "detected": bool,
        }
    """
    detector = _build_detector()
    results = []

    for idx, (rgb_frame, timestamp_ms) in enumerate(frames):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        detected = bool(result.pose_landmarks)
        annotated = draw_landmarks_on_image(rgb_frame, result) if detected else rgb_frame

        landmarks = result.pose_landmarks[0] if detected else None
        angles = compute_joint_angles(landmarks) if landmarks else None

        results.append(
            {
                "frame_idx": idx,
                "timestamp_ms": timestamp_ms,
                "annotated_frame": annotated,
                "landmarks": landmarks,
                "angles": angles,
                "detected": detected,
            }
        )

    detector.close()
    return results


def select_key_frames(pose_results: list[dict], n: int = 6) -> list[dict]:
    """
    Pick n most informative frames. Strategy:
    1. Only consider frames where pose was detected.
    2. Score by variance of joint angles (high variance = interesting pose).
    3. Ensure temporal spread by dividing into n buckets.
    """
    detected = [r for r in pose_results if r["detected"]]
    if not detected:
        # Fall back to evenly spaced frames from all
        step = max(1, len(pose_results) // n)
        return pose_results[::step][:n]

    if len(detected) <= n:
        return detected

    # Divide into n temporal buckets, pick highest-angle-variance frame per bucket
    bucket_size = len(detected) // n
    key_frames = []
    for i in range(n):
        bucket = detected[i * bucket_size : (i + 1) * bucket_size]
        if not bucket:
            continue
        best = max(
            bucket,
            key=lambda r: np.var([v for v in r["angles"].values() if v is not None])
            if r["angles"]
            else 0,
        )
        key_frames.append(best)

    return key_frames


def compute_angle_stats(pose_results: list[dict]) -> dict:
    """Aggregate angle stats across all detected frames."""
    detected = [r for r in pose_results if r["detected"] and r["angles"]]
    if not detected:
        return {}

    all_angles: dict[str, list[float]] = {}
    for r in detected:
        for k, v in r["angles"].items():
            if v is not None:
                all_angles.setdefault(k, []).append(v)

    stats = {}
    for k, vals in all_angles.items():
        arr = np.array(vals)
        stats[k] = {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    detection_rate = len(detected) / len(pose_results) if pose_results else 0
    stats["detection_rate"] = detection_rate

    return stats
