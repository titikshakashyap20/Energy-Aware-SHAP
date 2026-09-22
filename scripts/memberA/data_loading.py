"""
Step 1: Load and preprocess the Adult Income dataset for the TreeSHAP track.
"""

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_adult_income():
    """Fetch Adult Income dataset from OpenML (version 2)."""
    bunch = fetch_openml('adult', version=2, as_frame=True, parser='auto')
    return bunch.data.copy(), bunch.target.copy()


def clean_target(y):
    """Encode target to binary: 0 = <=50K, 1 = >50K."""
    y = y.astype(str).str.strip().str.rstrip('.')
    return (y == '>50K').astype(int)


def handle_missing_values(X_train, X_test):
    """Impute using train-only statistics; apply same values to test."""
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in X_train.columns:
        if X_train[col].isna().any() or X_test[col].isna().any():
            if pd.api.types.is_numeric_dtype(X_train[col]):
                fill_value = X_train[col].median()
            else:
                fill_value = X_train[col].mode().iloc[0]
            X_train[col] = X_train[col].fillna(fill_value)
            X_test[col] = X_test[col].fillna(fill_value)
    return X_train, X_test


def encode_categoricals(X_train, X_test):
    """One-hot encode using train's categories; align test to train's columns."""
    cat_cols = X_train.select_dtypes(include=['category', 'object']).columns.tolist()
    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=False)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    return X_train_enc, X_test_enc


def prepare_data(test_size=0.2, random_state=RANDOM_STATE):
    """Full pipeline: load -> split (raw) -> impute (train-fit) -> encode (train-fit)."""
    X, y = load_adult_income()
    y = clean_target(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, X_test = handle_missing_values(X_train, X_test)
    X_train, X_test = encode_categoricals(X_train, X_test)

    return X_train, X_test, y_train, y_test


def get_explanation_pool():
    """
    Benchmark-only row pool for TreeSHAP explanation (Step 5).

    Recombines X_train and X_test (features only) into one pool of ~48,842
    rows for drawing 1,000 / 10,000 / full-size benchmark chunks. This does
    NOT affect model training or the train/test accuracy evaluation — those
    keep using prepare_data()'s original split as-is.
    """
    X_train, X_test, _, _ = prepare_data()
    return pd.concat([X_train, X_test], ignore_index=True)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"Target balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Feature columns after encoding: {X_train.shape[1]}")

    pool = get_explanation_pool()
    print(f"Explanation pool shape: {pool.shape}")