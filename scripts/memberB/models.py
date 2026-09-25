"""
scripts/memberB/models.py

Member B — Black-box (KernelSHAP) track — Stage 2: Models.

Implements exactly two models, per the Review 2 Master Plan:

    1. SVM with RBF kernel
    2. Logistic Regression

Each is wrapped in an sklearn Pipeline (StandardScaler -> model) so that
scaling is always fit on training data only and applied consistently to
whatever is passed through the pipeline later (test data, KernelSHAP
background samples, explained instances) — no separate scaling step for
callers to remember or get wrong.

No hyperparameter tuning: no GridSearchCV, RandomizedSearchCV, Optuna, or
any search. Every hyperparameter here is a fixed, stated default.

This module does NOT implement KernelSHAP, explain_chunk, any parallel
strategy, CodeCarbon, the benchmark grid, CSV output, or analysis/plots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Fixed, deterministic settings (no tuning).
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42

# SVC: as of scikit-learn 1.9, SVC(probability=True) is deprecated in favor
# of explicit CalibratedClassifierCV(SVC(), ...). CalibratedClassifierCV with
# method="sigmoid" is Platt scaling — the exact same calibration method
# SVC(probability=True) used internally — so this is a like-for-like
# replacement, not a new modeling choice. See the note sent alongside this
# file for why ensemble=False / cv=5 were picked.
SVM_KERNEL: str = "rbf"
CALIBRATION_METHOD: str = "sigmoid"  # Platt scaling — same as the old probability=True
CALIBRATION_CV: int = 5              # same fold count SVC(probability=True) used internally
CALIBRATION_ENSEMBLE: bool = False   # single model refit on all data, not a CV ensemble

# LogisticRegression: max_iter raised only so the default lbfgs solver
# reliably converges on scaled features; not a tuned parameter.
LOGREG_MAX_ITER: int = 1000


def build_svm_rbf_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Build an unfit StandardScaler + calibrated RBF-SVM pipeline.

    Structure: StandardScaler -> SVC(RBF) -> sigmoid probability calibration
    -> predict_proba(). All hyperparameters are sklearn defaults except:
      - kernel="rbf"                (per the Review 2 Master Plan)
      - CalibratedClassifierCV(...) (replaces deprecated probability=True,
                                      see note; provides predict_proba)
      - random_state                (determinism)
    """
    base_svm = SVC(kernel=SVM_KERNEL, random_state=random_state)
    calibrated_svm = CalibratedClassifierCV(
        estimator=base_svm,
        method=CALIBRATION_METHOD,
        cv=CALIBRATION_CV,
        ensemble=CALIBRATION_ENSEMBLE,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", calibrated_svm),
        ]
    )


def build_logistic_regression_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Build an unfit StandardScaler + LogisticRegression pipeline.

    All hyperparameters are sklearn defaults except:
      - max_iter      (raised only for convergence, not tuning)
      - random_state  (determinism)
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=LOGREG_MAX_ITER,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_svm_rbf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Fit a fresh StandardScaler + RBF-SVM pipeline on X_train/y_train.

    The scaler is fit inside this call, on X_train only. Works for either
    the 30-feature or 15-feature DataFrame from data_loading.py — the
    pipeline has no hardcoded feature count.
    """
    pipeline = build_svm_rbf_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Fit a fresh StandardScaler + LogisticRegression pipeline on X_train/y_train.

    The scaler is fit inside this call, on X_train only. Works for either
    the 30-feature or 15-feature DataFrame from data_loading.py — the
    pipeline has no hardcoded feature count.
    """
    pipeline = build_logistic_regression_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    return pipeline


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loading import load_memberB_data

    print("=" * 70)
    print("MEMBER B — STAGE 2 MODELS SMOKE TEST")
    print("=" * 70)

    data = load_memberB_data()

    feature_conditions = {
        30: (data.X_train_30, data.X_test_30),
        15: (data.X_train_15, data.X_test_15),
    }

    trainers = {
        "SVM-RBF": train_svm_rbf,
        "LogisticRegression": train_logistic_regression,
    }

    results = []
    all_ok = True

    for n_features, (X_train, X_test) in feature_conditions.items():
        for model_name, train_fn in trainers.items():
            pipeline = train_fn(X_train, data.y_train)
            preds = pipeline.predict(X_test)
            proba = pipeline.predict_proba(X_test)

            accuracy = float(np.mean(preds == data.y_test.values))

            pred_shape_ok = preds.shape == (X_test.shape[0],)
            proba_shape_ok = proba.shape == (X_test.shape[0], 2)
            all_ok &= pred_shape_ok and proba_shape_ok

            print(f"\nModel: {model_name}")
            print(f"  Feature count      : {n_features}")
            print(f"  X_test shape       : {X_test.shape}")
            print(f"  Prediction shape   : {preds.shape}  (expected: ({X_test.shape[0]},))")
            print(f"  predict_proba shape: {proba.shape}  (expected: ({X_test.shape[0]}, 2))")
            print(f"  Accuracy           : {accuracy:.4f}")

            results.append(
                {
                    "model": model_name,
                    "n_features": n_features,
                    "pred_shape_ok": pred_shape_ok,
                    "proba_shape_ok": proba_shape_ok,
                    "accuracy": accuracy,
                }
            )

    print("\n" + "-" * 70)
    print(f"Combinations run: {len(results)} (expected: 4 -> 2 models x 2 feature sets)")
    combos_ok = len(results) == 4
    print(f"[check] all 4 model x feature-count combinations ran: {combos_ok}")
    print(f"[check] all prediction/proba shapes correct: {all_ok}")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if (all_ok and combos_ok) else "SOME CHECKS FAILED")
    print("=" * 70)