"""
train/train_quality_model.py
------------------------------
Trains a GradientBoostingClassifier on climbing quality data,
evaluates it, and saves the model + scaler to models/.

Run:
    python train/generate_data.py          # generate data first
    python train/train_quality_model.py

    # Use real labeled data instead / as well:
    python train/train_quality_model.py --data data/real_labeled.csv
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.quality_features import FEATURE_COLUMNS

MODEL_PATH  = "models/climb_quality_classifier.pkl"
SCALER_PATH = "models/climb_quality_scaler.pkl"


def train(
    data_path: str,
    n_estimators: int   = 200,
    max_depth: int      = 4,
    learning_rate: float = 0.05,
    test_size: float    = 0.20,
    seed: int           = 42,
) -> None:
    print(f"Loading data from {data_path} …")
    df = pd.read_csv(data_path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df["label"].values.astype(int)
    print(f"  {len(X)} samples | Good: {y.sum()} ({100*y.mean():.1f}%)  Bad: {(1-y).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    scaler     = StandardScaler()
    X_train_s  = scaler.fit_transform(X_train)
    X_test_s   = scaler.transform(X_test)

    print(f"\nTraining GradientBoostingClassifier "
          f"(n_estimators={n_estimators}, max_depth={max_depth}) …")
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=seed,
    )
    clf.fit(X_train_s, y_train)

    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5, scoring="roc_auc")
    print(f"  5-fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    y_pred = clf.predict(X_test_s)
    auc    = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
    print(f"\n── Test Set Results ──────────────────────────────────────────")
    print(f"  ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["bad", "good"]))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n── Top 10 Feature Importances ────────────────────────────────")
    for rank, i in enumerate(np.argsort(clf.feature_importances_)[::-1][:10], 1):
        print(f"  {rank:2d}. {FEATURE_COLUMNS[i]:<35s} {clf.feature_importances_[i]:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf,    MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n✓ Model  → {MODEL_PATH}")
    print(f"✓ Scaler → {SCALER_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",          type=str,   default="data/synthetic_climb_data.csv")
    parser.add_argument("--n-estimators",  type=int,   default=200)
    parser.add_argument("--max-depth",     type=int,   default=4)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--test-size",     type=float, default=0.20)
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    train(
        data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        test_size=args.test_size,
        seed=args.seed,
    )
