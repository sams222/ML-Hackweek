"""
train/generate_data.py
-----------------------
Generates a synthetic training dataset of climbing quality features.

Each row = one climbing clip with 25 biomechanical features + label (1=good, 0=bad).
Distributions are grounded in climbing biomechanics research.

Run:
    python train/generate_data.py
    python train/generate_data.py --n 10000 --out data/synthetic_large.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.quality_features import FEATURE_COLUMNS


def _clip(val, lo, hi):
    return float(np.clip(val, lo, hi))


def generate_sample(is_good: bool, rng: np.random.Generator) -> dict:
    if is_good:
        return {
            # Hesitation: good climbers move deliberately, brief planned pauses
            "hesitation_pct":          _clip(rng.beta(2, 9),           0.0,  1.0),
            "pause_count":             _clip(rng.poisson(2),            0,    30),
            "max_pause_duration":      _clip(rng.gamma(2, 3),           0,    60),
            # Arm efficiency: straight arms (>150°), lots of elbow movement
            "mean_elbow_angle":        _clip(rng.normal(158, 8),        90,   180),
            "straight_arm_pct":        _clip(rng.beta(7, 2),            0.0,  1.0),
            "elbow_angle_variance":    _clip(rng.normal(420, 60),       50,   800),
            # Hip positioning: hips close to wall (low z), stable
            "hip_wall_dist_mean":      _clip(rng.normal(0.10, 0.03),   -0.3,  0.5),
            "hip_wall_dist_std":       _clip(rng.normal(0.03, 0.01),    0.0,  0.2),
            "hip_drop_count":          _clip(rng.poisson(1),            0,    20),
            # Movement smoothness: low jerk = fluid movement
            "mean_jerk":               _clip(rng.normal(0.7, 0.2),      0.0,  5.0),
            "max_jerk":                _clip(rng.normal(2.5, 0.6),      0.0,  15.0),
            "jerk_std":                _clip(rng.normal(0.4, 0.1),      0.0,  3.0),
            "velocity_cv":             _clip(rng.normal(0.5, 0.15),     0.0,  3.0),
            # Range of motion: good climbers use full ROM
            "rom_l_elbow":             _clip(rng.normal(80, 12),        10,   130),
            "rom_r_elbow":             _clip(rng.normal(80, 12),        10,   130),
            "rom_l_shoulder":          _clip(rng.normal(95, 15),        10,   150),
            "rom_r_shoulder":          _clip(rng.normal(95, 15),        10,   150),
            "rom_l_knee":              _clip(rng.normal(75, 12),        10,   130),
            "rom_r_knee":              _clip(rng.normal(75, 12),        10,   130),
            "rom_l_hip":               _clip(rng.normal(60, 10),        10,   110),
            # Body tension: high L/R correlation = coordinated movement
            "elbow_lr_correlation":    _clip(rng.normal(0.70, 0.12),   -1.0,  1.0),
            "knee_lr_correlation":     _clip(rng.normal(0.65, 0.12),   -1.0,  1.0),
            "body_tension_score":      _clip(rng.normal(0.72, 0.10),   -1.0,  1.0),
            "mean_shoulder_angle":     _clip(rng.normal(100, 12),       30,   160),
            "shoulder_angle_variance": _clip(rng.normal(350, 60),       20,   800),
        }
    else:
        return {
            # Hesitation: bad climbers freeze, second-guess, hang on holds
            "hesitation_pct":          _clip(rng.beta(6, 3),            0.0,  1.0),
            "pause_count":             _clip(rng.poisson(8),            0,    30),
            "max_pause_duration":      _clip(rng.gamma(5, 6),           0,    60),
            # Arm efficiency: bent arms (over-gripping), stiff
            "mean_elbow_angle":        _clip(rng.normal(118, 15),       70,   180),
            "straight_arm_pct":        _clip(rng.beta(2, 7),            0.0,  1.0),
            "elbow_angle_variance":    _clip(rng.normal(140, 45),       10,   500),
            # Hip positioning: hips away from wall
            "hip_wall_dist_mean":      _clip(rng.normal(0.38, 0.07),   -0.3,  0.5),
            "hip_wall_dist_std":       _clip(rng.normal(0.09, 0.03),    0.0,  0.2),
            "hip_drop_count":          _clip(rng.poisson(5),            0,    20),
            # Movement smoothness: high jerk = lurch, stab, slap
            "mean_jerk":               _clip(rng.normal(2.8, 0.6),      0.0,  5.0),
            "max_jerk":                _clip(rng.normal(7.5, 1.5),      0.0,  15.0),
            "jerk_std":                _clip(rng.normal(1.4, 0.3),      0.0,  3.0),
            "velocity_cv":             _clip(rng.normal(1.5, 0.4),      0.0,  3.0),
            # Range of motion: stiff joints, limited ROM
            "rom_l_elbow":             _clip(rng.normal(32, 10),        10,   130),
            "rom_r_elbow":             _clip(rng.normal(32, 10),        10,   130),
            "rom_l_shoulder":          _clip(rng.normal(40, 12),        10,   150),
            "rom_r_shoulder":          _clip(rng.normal(40, 12),        10,   150),
            "rom_l_knee":              _clip(rng.normal(28, 10),        10,   130),
            "rom_r_knee":              _clip(rng.normal(28, 10),        10,   130),
            "rom_l_hip":               _clip(rng.normal(22, 8),         10,   110),
            # Body tension: uncoordinated, flailing
            "elbow_lr_correlation":    _clip(rng.normal(0.15, 0.20),   -1.0,  1.0),
            "knee_lr_correlation":     _clip(rng.normal(0.10, 0.20),   -1.0,  1.0),
            "body_tension_score":      _clip(rng.normal(0.12, 0.15),   -1.0,  1.0),
            "mean_shoulder_angle":     _clip(rng.normal(75, 15),        30,   160),
            "shoulder_angle_variance": _clip(rng.normal(120, 40),       10,   500),
        }


def generate_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        is_good = bool(rng.random() > 0.5)
        sample  = generate_sample(is_good, rng)
        row     = {col: sample[col] for col in FEATURE_COLUMNS}
        row["label"] = int(is_good)
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default="data/synthetic_climb_data.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = generate_dataset(args.n, args.seed)
    df.to_csv(args.out, index=False)

    good = int(df["label"].sum())
    bad  = len(df) - good
    print(f"✓ Generated {len(df)} samples → {args.out}")
    print(f"  Good: {good} ({100*good/len(df):.1f}%)   Bad: {bad} ({100*bad/len(df):.1f}%)")
    print(f"  Features: {len(FEATURE_COLUMNS)}")
