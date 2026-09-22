"""
Step 2: Train Random Forest and XGBoost models for the TreeSHAP track.
Fixed model configurations — no hyperparameter tuning.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from data_loading import prepare_data

RANDOM_STATE = 42


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()

    rf = train_random_forest(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"Random Forest  — val accuracy: {rf_acc:.4f}")

    xgb = train_xgboost(X_train, y_train)
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"XGBoost        — val accuracy: {xgb_acc:.4f}")