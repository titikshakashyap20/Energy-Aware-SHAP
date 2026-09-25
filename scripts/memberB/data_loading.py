"""
scripts/memberB/data_loading.py

Member B — Black-box (KernelSHAP) track — Stage 1: Data loading.

Loads the Breast Cancer Wisconsin dataset, creates ONE deterministic
train/test split, and derives the two feature conditions required by
the Review 2 Master Plan:

    1. all 30 original features
    2. the top 15 features by Random Forest importance

Both conditions are built from the SAME train/test split (same rows in
train, same rows in test) — the 15-feature version is a column-subset
of the 30-feature version, not a fresh split. Feature selection is
fit on TRAINING data only.

This module does NOT implement SVM, Logistic Regression, KernelSHAP,
parallel strategies, CodeCarbon, benchmarking, analysis, or plotting.
It only exposes clean, importable data-loading functions for later
Member B stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Fixed, deterministic settings (per Review 2 Master Plan: no tuning,
# deterministic random_state values throughout).
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
N_TOP_FEATURES: int = 15
# Kept modest and fixed — this RF exists only to rank features, it is not
# the RF used anywhere else in the project and needs no tuning.
FEATURE_SELECTOR_N_ESTIMATORS: int = 300


@dataclass
class MemberBData:
    """Bundle of everything later Member B stages need.

    All four (X_train_30, X_test_30, y_train, y_test) and
    (X_train_15, X_test_15, y_train, y_test) share the exact same rows —
    only the columns differ between the 30- and 15-feature versions.
    """

    # Full dataset (pre-split), kept for reference / smoke checks.
    X_full: pd.DataFrame
    y_full: pd.Series

    # 30-feature condition
    X_train_30: pd.DataFrame
    X_test_30: pd.DataFrame

    # 15-feature condition (column subset of the 30-feature condition)
    X_train_15: pd.DataFrame
    X_test_15: pd.DataFrame

    # Shared across both feature conditions (same split)
    y_train: pd.Series
    y_test: pd.Series

    feature_names_30: list[str] = field(default_factory=list)
    feature_names_15: list[str] = field(default_factory=list)
    feature_importances: pd.Series = field(default_factory=pd.Series)

    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE


def load_raw_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the Breast Cancer Wisconsin dataset as a labeled DataFrame.

    Returns
    -------
    X : DataFrame, shape (569, 30), columns = original feature names.
    y : Series, shape (569,), 0/1 target.
    feature_names : list of the 30 original feature name strings.
    """
    raw = load_breast_cancer()
    feature_names = list(raw.feature_names)
    X = pd.DataFrame(raw.data, columns=feature_names)
    y = pd.Series(raw.target, name="target")
    return X, y, feature_names


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create the single, deterministic, stratified train/test split.

    This is called exactly once. Every feature condition (30-feature,
    15-feature) must be derived from this same split by column-slicing,
    never by calling this function again.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_top: int = N_TOP_FEATURES,
    random_state: int = RANDOM_STATE,
) -> tuple[list[str], pd.Series]:
    """Rank features by Random Forest importance, fit on TRAINING data only.

    Returns
    -------
    top_features : list of the top `n_top` feature names, in ranked order
        (highest importance first). Ties broken deterministically by
        feature name (ascending) so the result is reproducible.
    importances : Series of importances for ALL features, indexed by
        feature name, sorted the same way (for reporting/debugging).
    """
    selector = RandomForestClassifier(
        n_estimators=FEATURE_SELECTOR_N_ESTIMATORS,
        random_state=random_state,
        n_jobs=1,  # keep importance ranking deterministic
    )
    selector.fit(X_train, y_train)

    importances = pd.Series(
        selector.feature_importances_, index=X_train.columns, name="importance"
    )

    # Deterministic ordering: importance desc, feature name asc as tiebreak.
    ranked_index = sorted(
        importances.index, key=lambda name: (-importances[name], name)
    )
    importances = importances.loc[ranked_index]

    top_features = list(importances.index[:n_top])
    return top_features, importances


def load_memberB_data(
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_top: int = N_TOP_FEATURES,
) -> MemberBData:
    """Top-level convenience function: build everything Member B needs.

    Orchestrates, in order:
      1. load_raw_data()
      2. create_train_test_split()   <- called ONCE
      3. select_top_features()       <- fit on X_train only
      4. column-slice X_train/X_test down to the top-15 features

    Returns a MemberBData bundle with both feature conditions sharing
    the same underlying rows.
    """
    X_full, y_full, feature_names_30 = load_raw_data()

    X_train_30, X_test_30, y_train, y_test = create_train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state
    )

    top_features, importances = select_top_features(
        X_train_30, y_train, n_top=n_top, random_state=random_state
    )

    # Same rows as X_train_30 / X_test_30 — just fewer columns.
    X_train_15 = X_train_30[top_features]
    X_test_15 = X_test_30[top_features]

    return MemberBData(
        X_full=X_full,
        y_full=y_full,
        X_train_30=X_train_30,
        X_test_30=X_test_30,
        X_train_15=X_train_15,
        X_test_15=X_test_15,
        y_train=y_train,
        y_test=y_test,
        feature_names_30=feature_names_30,
        feature_names_15=top_features,
        feature_importances=importances,
        random_state=random_state,
        test_size=test_size,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_memberB_data()

    print("=" * 70)
    print("MEMBER B — STAGE 1 DATA LOADING SMOKE TEST")
    print("=" * 70)

    print(f"\nFull dataset shape: X={data.X_full.shape}, y={data.y_full.shape}")

    print("\n--- 30-feature condition ---")
    print(f"X_train_30: {data.X_train_30.shape}")
    print(f"X_test_30 : {data.X_test_30.shape}")

    print("\n--- 15-feature condition ---")
    print(f"X_train_15: {data.X_train_15.shape}")
    print(f"X_test_15 : {data.X_test_15.shape}")

    print(f"\nSelected top-{N_TOP_FEATURES} feature names (ranked):")
    for i, name in enumerate(data.feature_names_15, start=1):
        print(f"  {i:2d}. {name}  (importance={data.feature_importances[name]:.4f})")

    # --- Checks ---
    checks_passed = True

    # Exactly 15 selected features, and they're a subset of the 30.
    n_selected = len(data.feature_names_15)
    is_15 = n_selected == N_TOP_FEATURES
    is_subset = set(data.feature_names_15).issubset(set(data.feature_names_30))
    print(f"\n[check] exactly {N_TOP_FEATURES} features selected: {is_15} (got {n_selected})")
    print(f"[check] selected features are a subset of the 30: {is_subset}")
    checks_passed &= is_15 and is_subset

    # Same rows underlie both feature conditions (same index, same order).
    same_train_index = data.X_train_30.index.equals(data.X_train_15.index)
    same_test_index = data.X_test_30.index.equals(data.X_test_15.index)
    same_train_y = data.X_train_30.index.equals(data.y_train.index)
    same_test_y = data.X_test_30.index.equals(data.y_test.index)
    print(f"[check] train rows identical across 30/15 conditions: {same_train_index}")
    print(f"[check] test rows identical across 30/15 conditions: {same_test_index}")
    print(f"[check] y_train aligned with X_train rows: {same_train_y}")
    print(f"[check] y_test aligned with X_test rows: {same_test_y}")
    checks_passed &= same_train_index and same_test_index and same_train_y and same_test_y

    # No overlap between train and test rows.
    no_overlap = len(set(data.X_train_30.index) & set(data.X_test_30.index)) == 0
    print(f"[check] no overlap between train and test rows: {no_overlap}")
    checks_passed &= no_overlap

    # Reproducibility: rerun the whole pipeline and confirm identical split
    # and identical top-15 selection.
    data_repeat = load_memberB_data()
    same_split_again = (
        data.X_train_30.index.equals(data_repeat.X_train_30.index)
        and data.X_test_30.index.equals(data_repeat.X_test_30.index)
    )
    same_top15_again = data.feature_names_15 == data_repeat.feature_names_15
    print(f"[check] train/test split is reproducible across runs: {same_split_again}")
    print(f"[check] top-15 feature selection is reproducible across runs: {same_top15_again}")
    checks_passed &= same_split_again and same_top15_again

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED" if checks_passed else "SOME CHECKS FAILED")
    print("=" * 70)