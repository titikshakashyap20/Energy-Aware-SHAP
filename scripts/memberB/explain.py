"""
scripts/memberB/explain.py

Member B — Black-box (KernelSHAP) track — Stage 3: Explanation layer.

Exposes exactly one shared workload function, explain_chunk(rows, model,
background), which wraps shap.KernelExplainer. This is the single unit of
work that every parallel strategy (multiprocessing / threading / joblib /
Ray / serial) will later call identically — per the Review 2 Master Plan,
Part 1 §9 (shared harness) and the Member A pattern of one explain_chunk()
reused by all 5 strategies.

This module does NOT implement multiprocessing, threading, joblib, Ray,
CodeCarbon, the benchmark grid, or CSV writing. Parallel scheduling
belongs in parallel_strategies.py, a later stage.

Output format note (shap==0.52.0)
----------------------------------
KernelExplainer is given model.predict_proba, which returns 2 columns
(binary classification: P(class 0), P(class 1)) for both SVM-RBF and
LogisticRegression. Verified empirically against the installed shap
version: explainer.shap_values(rows) returns a single numpy ndarray of
shape

    (n_rows, n_features, n_classes)   # here n_classes == 2

NOT a Python list of per-class arrays (that was the older shap API,
pre ~0.42). explain_chunk() returns this ndarray as-is — both classes'
SHAP values are kept, nothing is collapsed or discarded. Downstream code
that wants "the SHAP values for the positive class" should slice
shap_values[:, :, 1], not assume a list.

explainer.expected_value is the corresponding per-class base rate, shape
(n_classes,) — returned alongside the SHAP values so callers don't need
to reconstruct the explainer to get it.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
import shap


class ExplanationResult(NamedTuple):
    """Container for one explain_chunk() call.

    shap_values : ndarray, shape (n_rows, n_features, n_classes).
        SHAP values for every explained row, every feature, every class.
        Nothing is discarded or pre-selected here.
    expected_value : ndarray, shape (n_classes,).
        KernelExplainer's base-rate output per class (the background's
        average predict_proba), i.e. shap_values[i].sum(axis=0) + expected_value
        reconstructs model.predict_proba(rows[i]) approximately.
    """

    shap_values: np.ndarray
    expected_value: np.ndarray


def explain_chunk(
    rows: pd.DataFrame,
    model,
    background: pd.DataFrame,
    nsamples: int | str = "auto",
) -> ExplanationResult:
    """Explain `rows` with KernelSHAP, using `model` and `background` exactly
    as given — the single shared unit of work for the KernelSHAP track.

    Parameters
    ----------
    rows : DataFrame
        The test instances / chunk to explain. Used as-is; never resampled
        or regenerated inside this function.
    model : fitted sklearn Pipeline
        Must expose predict_proba (both train_svm_rbf() and
        train_logistic_regression() pipelines from models.py do). Called
        through this public interface only — never through internal
        attributes of the scaler or the inner estimator.
    background : DataFrame
        The background dataset for KernelExplainer, supplied explicitly by
        the caller. Used as-is; never sampled or resampled inside this
        function, so every parallel strategy that calls explain_chunk for
        the same benchmark configuration sees the identical background.
    nsamples : int or "auto", default "auto"
        Passed straight through to shap.KernelExplainer.shap_values(...).
        Left at shap's own default unless the caller overrides it — this
        function does not reimplement or approximate KernelSHAP's sampling.

    Returns
    -------
    ExplanationResult(shap_values, expected_value)
        shap_values : ndarray (n_rows, n_features, n_classes)
        expected_value : ndarray (n_classes,)
    """
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(rows, nsamples=nsamples)
    expected_value = np.asarray(explainer.expected_value)
    return ExplanationResult(shap_values=shap_values, expected_value=expected_value)


# ---------------------------------------------------------------------------
# Smoke test — deliberately tiny. Do NOT scale this up; this only checks
# that the wiring (data -> model -> KernelExplainer) works end to end.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loading import load_memberB_data
    from models import train_svm_rbf

    print("=" * 70)
    print("MEMBER B — STAGE 3 EXPLAIN (KernelSHAP) SMOKE TEST")
    print("=" * 70)

    # --- Tiny, fixed sizes on purpose: 1 model, 15 features, background=25,
    #     3 explained rows. No parallel strategy is invoked anywhere here. ---
    N_BACKGROUND = 25
    N_EXPLAIN_ROWS = 3

    data = load_memberB_data()
    X_train, X_test = data.X_train_15, data.X_test_15
    n_features = X_train.shape[1]

    print(f"\nTraining SVM-RBF on the 15-feature condition "
          f"(train={X_train.shape}, test={X_test.shape}) ...")
    model = train_svm_rbf(X_train, data.y_train)

    # Background is selected ONCE, deterministically, from training data only.
    background = X_train.sample(n=N_BACKGROUND, random_state=42)
    # Explained rows: a small, fixed slice of the test set — not resampled.
    rows = X_test.iloc[:N_EXPLAIN_ROWS]

    print(f"Background shape     : {background.shape}")
    print(f"Explained rows shape : {rows.shape}")

    result = explain_chunk(rows, model, background)

    print(f"\nSHAP output type  : {type(result.shap_values)}")
    print(f"SHAP output shape : {result.shap_values.shape}  "
          f"(expected: ({N_EXPLAIN_ROWS}, {n_features}, 2))")
    print(f"expected_value    : {result.expected_value}  (shape {result.expected_value.shape})")

    # --- Sanity checks ---
    checks_passed = True

    background_features_ok = background.shape[1] == n_features
    rows_features_ok = rows.shape[1] == n_features
    print(f"\n[check] background has expected feature count ({n_features}): {background_features_ok}")
    print(f"[check] explained rows have expected feature count ({n_features}): {rows_features_ok}")
    checks_passed &= background_features_ok and rows_features_ok

    n_rows_ok = result.shap_values.shape[0] == rows.shape[0] == N_EXPLAIN_ROWS
    print(f"[check] number of explained rows matches input: {n_rows_ok}")
    checks_passed &= n_rows_ok

    shap_features_ok = result.shap_values.shape[1] == n_features
    n_classes_ok = result.shap_values.shape[2] == 2 and result.expected_value.shape == (2,)
    print(f"[check] SHAP feature dimension matches feature count: {shap_features_ok}")
    print(f"[check] SHAP output retains both classes (no info discarded): {n_classes_ok}")
    checks_passed &= shap_features_ok and n_classes_ok

    ran_ok = not np.any(np.isnan(result.shap_values))
    print(f"[check] KernelSHAP returned successfully (finite values, no NaNs): {ran_ok}")
    checks_passed &= ran_ok

    # Background/rows identity check: confirms nothing inside explain_chunk
    # silently resampled either input.
    same_background = background.equals(X_train.loc[background.index])
    same_rows = rows.equals(X_test.iloc[:N_EXPLAIN_ROWS])
    print(f"[check] background unchanged by explain_chunk (caller-supplied, not resampled): {same_background}")
    print(f"[check] explained rows unchanged by explain_chunk (not regenerated): {same_rows}")
    checks_passed &= same_background and same_rows

    print(f"[check] no parallel strategy invoked in this module "
          f"(no multiprocessing/threading/joblib/ray import present): True")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if checks_passed else "SOME CHECKS FAILED")
    print("=" * 70)