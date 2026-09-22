"""
Step 3: Single unit of work for the TreeSHAP track.
explain_chunk() is what every parallelism strategy (Step 4) will wrap around.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import shap
from data_loading import prepare_data
from models import train_random_forest, train_xgboost


def explain_chunk(rows, model):
    """
    Compute SHAP values for a chunk of rows against a trained tree model.

    Parameters
    ----------
    rows : pandas.DataFrame
        A chunk of the feature matrix to explain (same columns as training data).
    model : trained RandomForestClassifier or XGBClassifier

    Returns
    -------
    numpy.ndarray or list of numpy.ndarray
        Raw output of shap_values() — shape/type depends on model (see note below).
    """
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(rows)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()

    rf = train_random_forest(X_train, y_train)
    chunk = X_test.iloc[:10]

    shap_values = explain_chunk(chunk, rf)

    print(f"Model: RandomForest")
    print(f"Return type: {type(shap_values)}")
    if isinstance(shap_values, list):
        print(f"List length (per-class): {len(shap_values)}")
        print(f"Each element shape: {shap_values[0].shape}")
    else:
        print(f"Array shape: {shap_values.shape}")