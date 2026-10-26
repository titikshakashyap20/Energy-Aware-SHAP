"""
scripts/memberC/train_scheduler.py

Member C — Infrastructure — Stage C3: train and compare the adaptive scheduler.

Reads, read-only:
    results/memberC/edp_dataset.csv
    results/memberC/scheduler_evaluation_protocol.csv
    results/memberC/scheduler_splits.csv

Writes:
    results/memberC/scheduler_model_comparison.csv
    results/memberC/scheduler_final_model.joblib      (reproducible final-model bundle)
    results/memberC/scheduler_model_metadata.json     (versions, features, target, hyperparameters)

Scheduler
---------
pick_strategy(workload characteristics, n_cores) -> strategy.
For every available candidate strategy a regressor predicts
log10(median EDP); the candidate with the lowest prediction is chosen.

Two regressors are compared:  DecisionTreeRegressor, RandomForestRegressor.

Validation (no leakage)
-----------------------
Exhaustive Leave-One-Workload-Out over the 18 folds in scheduler_splits.csv.
In each fold the model is fit ONLY on rows of the other 17 workloads (all
strategies and core counts of the held-out workload are excluded). The
held-out workload's candidate rows are featurised from workload
characteristics + n_cores + candidate strategy ONLY. Measured EDP / time /
energy / oracle / Pareto columns are touched only AFTER the strategy has
been selected, to score the decision.

Only candidates that were actually measured at the held-out
(workload, n_cores) are considered (baseline_available == True). Nothing
is fabricated for KernelSHAP serial at 2/4/8 cores.

Model selection
---------------
Held-out decision performance only (never training fit). Primary criterion:
mean EDP regret over the 72 decision points,
    regret = scheduler_actual_EDP / oracle_EDP - 1.
The simpler Decision Tree is kept unless the Random Forest lowers mean regret
by more than SIMPLICITY_MARGIN (absolute regret, i.e. 0.02 = 2 percentage
points). Hyperparameters are fixed a priori (conservative, not tuned on the
held-out folds), so the only selection step touching held-out results is this
single two-model comparison.

Caveat: 18 workloads, 72 decision points that are correlated within a
workload. The valid claim is generalisation to held-out workloads inside the
evaluated benchmark space, not to arbitrary datasets/models/hardware.

Run:  python scripts/memberC/train_scheduler.py
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

ALL_STRATEGIES = ["serial", "multiprocessing", "threading", "joblib", "ray"]
CONTEXT_COLS = ["track", "model", "n_rows", "n_features", "feature_count", "background_size"]
RAW_INPUT_COLUMNS = CONTEXT_COLS + ["n_cores", "strategy"]  # the ONLY columns a prediction may see
TARGET_DESCRIPTION = "log10(median_edp_joule_seconds)"
MEANINGFUL_MARGIN_PCT = 10.0  # same stated threshold used in C2-B

RANDOM_STATE = 42
SIMPLICITY_MARGIN = 0.02  # RF must reduce mean regret by > this (absolute) to beat the Decision Tree

MODEL_SPECS = {
    "DecisionTree": {
        "class": DecisionTreeRegressor,
        "params": dict(max_depth=6, min_samples_split=10, min_samples_leaf=5, random_state=RANDOM_STATE),
    },
    "RandomForest": {
        "class": RandomForestRegressor,
        "params": dict(n_estimators=300, max_depth=8, min_samples_leaf=3, max_features=0.6,
                       bootstrap=True, random_state=RANDOM_STATE, n_jobs=1),
    },
}

FORBIDDEN_INPUT_SUBSTRINGS = ("measured", "oracle", "edp", "time", "energy", "pareto", "regret")


# --------------------------------------------------------------------------- io
def find_file(relative_path: str) -> Path:
    for root in REPO_ROOT_CANDIDATES:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required input file not found: '{relative_path}'. "
        f"Looked under: {[str(r) for r in REPO_ROOT_CANDIDATES]}."
    )


def output_dir() -> Path:
    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out_dir.exists():
        out_dir = Path.cwd() / "results" / "memberC"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    edp = pd.read_csv(find_file("results/memberC/edp_dataset.csv"))
    protocol = pd.read_csv(find_file("results/memberC/scheduler_evaluation_protocol.csv"))
    splits = pd.read_csv(find_file("results/memberC/scheduler_splits.csv"))
    for name, df, cols in [
        ("edp_dataset.csv", edp, CONTEXT_COLS + ["workload_id", "strategy", "n_cores", "median_edp_joule_seconds",
                                                 "median_wall_time_sec", "median_energy_joules"]),
        ("scheduler_evaluation_protocol.csv", protocol,
         ["split_id", "held_out_workload_id", "n_cores", "candidate_strategy", "measured_median_edp",
          "measured_median_time", "measured_median_energy", "log10_measured_median_edp", "oracle_strategy",
          "oracle_edp", "oracle_time", "oracle_energy", "is_oracle", "baseline_available"] + CONTEXT_COLS),
        ("scheduler_splits.csv", splits,
         ["split_id", "held_out_workload_id", "train_workload_ids", "test_workload_ids"]),
    ]:
        if df.empty:
            raise ValueError(f"{name} loaded with zero rows.")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing required column(s): {missing}")
    if (edp["median_edp_joule_seconds"] <= 0).any():
        raise ValueError("edp_dataset.csv has non-positive median EDP.")
    protocol["baseline_available"] = protocol["baseline_available"].astype(bool)
    protocol["is_oracle"] = protocol["is_oracle"].astype(bool)

    # Cross-file consistency: same workloads everywhere; every fold holds out exactly one complete workload.
    wl_edp = set(edp["workload_id"])
    wl_splits = set(splits["held_out_workload_id"])
    wl_protocol = set(protocol["held_out_workload_id"])
    if not (wl_edp == wl_splits == wl_protocol):
        raise ValueError("Workload sets differ between edp_dataset, splits and protocol files.")
    if (splits["test_workload_ids"].str.contains(";")).any():
        raise ValueError("A split holds out more than one workload; expected leave-ONE-workload-out.")
    return edp, protocol, splits


# ---------------------------------------------------------------- feature logic
def make_schema(edp: pd.DataFrame) -> dict:
    """Categorical levels + the fixed, ordered feature-column list."""
    tracks = sorted(edp["track"].unique())
    models = sorted(edp["model"].unique())
    cols = ["n_rows", "n_features", "feature_count", "background_size", "n_cores"]
    cols += [f"track={t}" for t in tracks] + [f"model={m}" for m in models]
    cols += [f"strategy={s}" for s in ALL_STRATEGIES]
    bad = [c for c in cols if any(f in c.lower() for f in FORBIDDEN_INPUT_SUBSTRINGS)]
    if bad:
        raise ValueError(f"Feature list contains forbidden (outcome-like) column(s): {bad}")
    return {"tracks": tracks, "models": models, "feature_columns": cols}


def build_features(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Featurise rows using ONLY RAW_INPUT_COLUMNS. feature_count / background_size are
    undefined (NaN) for the TreeSHAP track and are encoded as 0 (track one-hot disambiguates)."""
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot featurise: missing input column(s) {missing}")
    d = df[RAW_INPUT_COLUMNS]
    X = pd.DataFrame(index=d.index)
    X["n_rows"] = d["n_rows"].astype(float)
    X["n_features"] = d["n_features"].astype(float)
    X["feature_count"] = d["feature_count"].astype(float).fillna(0.0)
    X["background_size"] = d["background_size"].astype(float).fillna(0.0)
    X["n_cores"] = d["n_cores"].astype(float)
    for t in schema["tracks"]:
        X[f"track={t}"] = (d["track"] == t).astype(float)
    for m in schema["models"]:
        X[f"model={m}"] = (d["model"] == m).astype(float)
    for s in ALL_STRATEGIES:
        X[f"strategy={s}"] = (d["strategy"] == s).astype(float)
    return X[schema["feature_columns"]]


def strategy_tiebreak_rank(train_edp: pd.DataFrame) -> dict[str, int]:
    """Deterministic tie-break computed from TRAINING workloads only: strategies ranked by
    mean log10-EDP gap to the best strategy at the same (workload, n_cores). A shallow tree
    often predicts identical values for several strategies; this decides those ties without
    an arbitrary fixed order and without touching held-out data."""
    d = train_edp.copy()
    d["y"] = np.log10(d["median_edp_joule_seconds"])
    d["gap"] = d["y"] - d.groupby(["workload_id", "n_cores"])["y"].transform("min")
    mean_gap = d.groupby("strategy")["gap"].mean().reindex(ALL_STRATEGIES)
    order = mean_gap.fillna(np.inf).sort_values(kind="mergesort").index
    return {s: i for i, s in enumerate(order)}


def make_estimator(model_name: str, params: dict | None = None):
    spec = MODEL_SPECS[model_name]
    return spec["class"](**(params if params is not None else spec["params"]))


# ------------------------------------------------------------- LOWO evaluation
def run_lowo(model_name: str, params: dict, edp: pd.DataFrame, protocol: pd.DataFrame,
             splits: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-One-Workload-Out. Returns (decision-level frame, candidate-prediction frame)."""
    dec_rows, pred_frames = [], []
    for _, sp in splits.sort_values("split_id").iterrows():
        held = sp["held_out_workload_id"]
        train_ids = set(sp["train_workload_ids"].split(";"))
        test_ids = set(sp["test_workload_ids"].split(";"))

        # --- leakage guards ---
        if held in train_ids or (train_ids & test_ids) or test_ids != {held}:
            raise ValueError(f"Split {sp['split_id']}: held-out workload leaks into training set.")
        train = edp[edp["workload_id"].isin(train_ids)]
        if (train["workload_id"] == held).any() or train["workload_id"].nunique() != len(train_ids):
            raise ValueError(f"Split {sp['split_id']}: training rows inconsistent with split definition.")

        est = make_estimator(model_name, params)
        est.fit(build_features(train, schema), np.log10(train["median_edp_joule_seconds"]))
        rank = strategy_tiebreak_rank(train)

        test = protocol[(protocol["split_id"] == sp["split_id"]) & protocol["baseline_available"]].copy()
        if test.empty or (test["held_out_workload_id"] != held).any():
            raise ValueError(f"Split {sp['split_id']}: protocol rows do not match held-out workload.")

        # Selection step: features come ONLY from characteristics + n_cores + candidate strategy.
        feats = test.rename(columns={"candidate_strategy": "strategy"})[RAW_INPUT_COLUMNS]
        test["pred_log10_edp"] = est.predict(build_features(feats, schema))
        test["actual_log10_edp"] = np.log10(test["measured_median_edp"])
        test["abs_err_log10"] = (test["pred_log10_edp"] - test["actual_log10_edp"]).abs()
        test["sq_err_log10"] = (test["pred_log10_edp"] - test["actual_log10_edp"]) ** 2
        pred_frames.append(test[["split_id", "held_out_workload_id", "track", "n_cores", "candidate_strategy",
                                 "pred_log10_edp", "actual_log10_edp", "abs_err_log10", "sq_err_log10"]])

        for n_cores, grp in test.groupby("n_cores"):
            grp = grp.reset_index(drop=True)
            pr = np.round(grp["pred_log10_edp"].to_numpy(), 9)
            order = sorted(range(len(grp)), key=lambda i: (pr[i], rank[grp.loc[i, "candidate_strategy"]]))
            ch = grp.loc[order[0]]
            n_tied = int((pr == pr.min()).sum())

            # ---- decision is fixed; only now use held-out measurements to score it ----
            oracle_idx = grp["measured_median_edp"].idxmin()
            orc = grp.loc[oracle_idx]
            if not np.isclose(orc["measured_median_edp"], grp["oracle_edp"].iloc[0]):
                raise ValueError(f"Stored oracle disagrees with measured minimum at split {sp['split_id']}, "
                                 f"n_cores={n_cores}.")
            jb = grp[grp["candidate_strategy"] == "joblib"]
            dec_rows.append({
                "split_id": sp["split_id"], "held_out_workload_id": held, "track": ch["track"],
                "model": ch["model"], "n_rows": ch["n_rows"], "n_features": ch["n_features"],
                "feature_count": ch["feature_count"], "background_size": ch["background_size"],
                "n_cores": n_cores,
                "available_strategies": ";".join(grp["candidate_strategy"]),
                "predicted_strategy": ch["candidate_strategy"],
                "predicted_edp_of_choice": 10 ** ch["pred_log10_edp"],
                "n_tied_candidates": n_tied,
                "oracle_strategy": orc["candidate_strategy"],
                "scheduler_edp": ch["measured_median_edp"], "oracle_edp": orc["measured_median_edp"],
                "scheduler_time": ch["measured_median_time"], "oracle_time": orc["measured_median_time"],
                "scheduler_energy": ch["measured_median_energy"], "oracle_energy": orc["measured_median_energy"],
                "joblib_edp": jb["measured_median_edp"].iloc[0] if len(jb) else np.nan,
                "joblib_time": jb["measured_median_time"].iloc[0] if len(jb) else np.nan,
                "joblib_energy": jb["measured_median_energy"].iloc[0] if len(jb) else np.nan,
                "edp_ratio_to_oracle": ch["measured_median_edp"] / orc["measured_median_edp"],
                "edp_regret": ch["measured_median_edp"] / orc["measured_median_edp"] - 1.0,
                "time_ratio_to_oracle": ch["measured_median_time"] / orc["measured_median_time"],
                "energy_ratio_to_oracle": ch["measured_median_energy"] / orc["measured_median_energy"],
                "is_oracle_hit": bool(ch["candidate_strategy"] == orc["candidate_strategy"]),
            })
    return pd.DataFrame(dec_rows), pd.concat(pred_frames, ignore_index=True)


def fixed_policy_decisions(protocol: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """'Always use <strategy>' scored only where that strategy was actually measured."""
    r = protocol[protocol["baseline_available"] & (protocol["candidate_strategy"] == strategy)].copy()
    r["edp_ratio_to_oracle"] = r["measured_median_edp"] / r["oracle_edp"]
    r["edp_regret"] = r["edp_ratio_to_oracle"] - 1.0
    r["time_ratio_to_oracle"] = r["measured_median_time"] / r["oracle_time"]
    r["energy_ratio_to_oracle"] = r["measured_median_energy"] / r["oracle_energy"]
    r["is_oracle_hit"] = r["is_oracle"]
    return r


def decision_metrics(dec: pd.DataFrame) -> dict:
    if dec.empty:
        return {}
    ratio = dec["edp_ratio_to_oracle"]
    return {
        "n_decision_points": int(len(dec)),
        "n_oracle_hits": int(dec["is_oracle_hit"].sum()),
        "oracle_hit_rate": float(dec["is_oracle_hit"].mean()),
        "mean_edp_regret": float(dec["edp_regret"].mean()),
        "median_edp_regret": float(dec["edp_regret"].median()),
        "max_edp_regret": float(dec["edp_regret"].max()),
        "mean_edp_ratio_to_oracle": float(ratio.mean()),
        "median_edp_ratio_to_oracle": float(ratio.median()),
        "geomean_edp_ratio_to_oracle": float(np.exp(np.log(ratio).mean())),
        "frac_within_10pct_of_oracle": float((ratio <= 1.10).mean()),
        "mean_time_ratio_to_oracle": float(dec["time_ratio_to_oracle"].mean()),
        "mean_energy_ratio_to_oracle": float(dec["energy_ratio_to_oracle"].mean()),
    }


def prediction_metrics(pred: pd.DataFrame) -> dict:
    """Held-out EDP prediction error over every available held-out candidate row (log10 scale)."""
    if pred.empty:
        return {}
    y, p = pred["actual_log10_edp"], pred["pred_log10_edp"]
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n_candidate_predictions": int(len(pred)),
        "mae_log10_edp": float(pred["abs_err_log10"].mean()),
        "median_abs_err_log10_edp": float(pred["abs_err_log10"].median()),
        "rmse_log10_edp": float(np.sqrt(pred["sq_err_log10"].mean())),
        "r2_log10_edp": float(1.0 - pred["sq_err_log10"].sum() / ss_tot) if ss_tot > 0 else np.nan,
        "typical_edp_fold_error": float(10 ** pred["abs_err_log10"].mean()),  # 10^MAE, a multiplicative error
    }


def scopes(df: pd.DataFrame):
    yield "overall", df
    yield "tree", df[df["track"] == "tree"]
    yield "kernelshap", df[df["track"] == "kernelshap"]


# ------------------------------------------------------------------- artefacts
def pick_strategy(bundle: dict, track: str, model: str, n_rows: int, n_features: int, n_cores: int,
                  feature_count: float | None = None, background_size: float | None = None,
                  candidates: list[str] | None = None) -> str:
    """Deployed scheduler API: choose the strategy with the lowest predicted log10 EDP.
    Default candidates follow the benchmark: serial is excluded for KernelSHAP at n_cores > 1
    (it was only ever measured on 1 core). Pass `candidates` to override."""
    if candidates is None:
        candidates = [s for s in ALL_STRATEGIES if not (s == "serial" and track == "kernelshap" and n_cores > 1)]
    rows = pd.DataFrame([{
        "track": track, "model": model, "n_rows": n_rows, "n_features": n_features,
        "feature_count": np.nan if feature_count is None else feature_count,
        "background_size": np.nan if background_size is None else background_size,
        "n_cores": n_cores, "strategy": s} for s in candidates])
    preds = np.round(bundle["estimator"].predict(build_features(rows, bundle["schema"])), 9)
    rank = bundle["tiebreak_rank"]
    best = min(range(len(candidates)), key=lambda i: (preds[i], rank[candidates[i]]))
    return candidates[best]


def verify_availability_rule(edp: pd.DataFrame) -> bool:
    """pick_strategy's default candidate rule must match what was actually measured."""
    for (track, n_cores), g in edp.groupby(["track", "n_cores"]):
        expected = {s for s in ALL_STRATEGIES if not (s == "serial" and track == "kernelshap" and n_cores > 1)}
        if set(g["strategy"]) != expected:
            return False
    return True


def versions() -> dict:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__}


def main() -> None:
    edp, protocol, splits = load_inputs()
    schema = make_schema(edp)
    n_workloads = edp["workload_id"].nunique()

    print("=" * 78)
    print("MEMBER C — SCHEDULER TRAINING & MODEL COMPARISON (C3)")
    print("=" * 78)
    print(f"Workloads: {n_workloads} | LOWO folds: {len(splits)} | training rows available: {len(edp)}")
    print(f"Target: {TARGET_DESCRIPTION}")
    print(f"Features ({len(schema['feature_columns'])}): {schema['feature_columns']}")

    # ---------------- held-out comparison ----------------
    comp_rows, lowo = [], {}
    for name, spec in MODEL_SPECS.items():
        print(f"\nRunning 18-fold LOWO for {name} ...")
        dec, pred = run_lowo(name, spec["params"], edp, protocol, splits, schema)
        lowo[name] = (dec, pred)
        for scope, sub in scopes(dec):
            psub = pred if scope == "overall" else pred[pred["track"] == scope]
            comp_rows.append({"model": name, "scope": scope, **decision_metrics(sub), **prediction_metrics(psub),
                              "hyperparameters": json.dumps(spec["params"], sort_keys=True)})
    for scope, sub in scopes(fixed_policy_decisions(protocol, "joblib")):   # reference row, not a candidate model
        comp_rows.append({"model": "fixed_joblib (reference)", "scope": scope, **decision_metrics(sub),
                          "hyperparameters": "n/a"})
    comp = pd.DataFrame(comp_rows)

    # ---------------- model selection (held-out decision performance) ----------------
    ov = comp[comp["scope"] == "overall"].set_index("model")
    dt_regret, rf_regret = ov.loc["DecisionTree", "mean_edp_regret"], ov.loc["RandomForest", "mean_edp_regret"]
    if rf_regret < dt_regret - SIMPLICITY_MARGIN:
        selected = "RandomForest"
        reason = (f"RandomForest lowers held-out mean EDP regret by {dt_regret - rf_regret:.4f} "
                  f"(> margin {SIMPLICITY_MARGIN}).")
    else:
        selected = "DecisionTree"
        reason = (f"RandomForest does not beat DecisionTree by more than {SIMPLICITY_MARGIN} mean regret "
                  f"(DT {dt_regret:.4f} vs RF {rf_regret:.4f}); simpler/interpretable model preferred.")
    comp["selected_final_model"] = comp["model"] == selected

    lead = ["model", "scope", "selected_final_model"]
    comp = comp[lead + [c for c in comp.columns if c not in lead]]
    out = output_dir()
    comp_path = out / "scheduler_model_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # ---------------- console report ----------------
    show = ["model", "n_decision_points", "oracle_hit_rate", "mean_edp_regret", "median_edp_regret",
            "max_edp_regret", "mean_time_ratio_to_oracle", "mean_energy_ratio_to_oracle",
            "mae_log10_edp", "rmse_log10_edp", "r2_log10_edp"]
    for scope in ["overall", "tree", "kernelshap"]:
        print(f"\n--- HELD-OUT (LOWO) PERFORMANCE: {scope.upper()} ---")
        print(comp[comp["scope"] == scope][[c for c in show if c in comp.columns]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSELECTED FINAL MODEL: {selected}\n  {reason}")
    ties = lowo[selected][0]["n_tied_candidates"]
    print(f"  Decision points with tied top predictions (resolved by training-only tie-break): "
          f"{int((ties > 1).sum())}/{len(ties)}")

    # ---------------- final fit on all 18 workloads (for deployment / reproducibility) ----------------
    params = MODEL_SPECS[selected]["params"]
    final = make_estimator(selected, params)
    final.fit(build_features(edp, schema), np.log10(edp["median_edp_joule_seconds"]))
    rank_all = strategy_tiebreak_rank(edp)
    rule_ok = verify_availability_rule(edp)
    if not rule_ok:
        print("NOTE: default candidate rule in pick_strategy does not match measured availability; "
              "pass `candidates` explicitly.")

    metadata = {
        "selected_model": selected, "selection_reason": reason,
        "selection_criterion": f"min held-out mean EDP regret; RF must beat DT by > {SIMPLICITY_MARGIN}",
        "target": TARGET_DESCRIPTION, "feature_columns": schema["feature_columns"],
        "raw_input_columns": RAW_INPUT_COLUMNS, "hyperparameters": params,
        "all_model_hyperparameters": {k: v["params"] for k, v in MODEL_SPECS.items()},
        "versions": versions(), "random_state": RANDOM_STATE,
        "trained_on_workloads": sorted(edp["workload_id"].unique()), "n_training_rows": int(len(edp)),
        "tiebreak_rank": rank_all, "availability_rule_verified": bool(rule_ok),
        "validation": "18-fold Leave-One-Workload-Out (scheduler_splits.csv); the saved estimator is refit on "
                      "all 18 workloads and has NO held-out estimate of its own — report the LOWO numbers.",
        "claim_scope": "Generalisation to held-out workloads within the evaluated benchmark space only.",
    }
    bundle = {"estimator": final, "schema": schema, "tiebreak_rank": rank_all, "metadata": metadata,
              "selected_model": selected, "hyperparameters": params}
    model_path, meta_path = out / "scheduler_final_model.joblib", out / "scheduler_model_metadata.json"
    joblib.dump(bundle, model_path)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    if selected == "DecisionTree":
        print("\nFinal Decision Tree rules (fit on all 18 workloads):")
        print(export_text(final, feature_names=schema["feature_columns"], max_depth=6))

    print("\nVersions:", versions())
    print(f"\nscheduler_model_comparison.csv written to: {comp_path}")
    print(f"scheduler_final_model.joblib written to   : {model_path}")
    print(f"scheduler_model_metadata.json written to  : {meta_path}")
    print("\nCaveat: 18 independent workloads; valid claim = generalisation to held-out workloads "
          "within the evaluated benchmark space, not to arbitrary datasets.")
    print("=" * 78)


if __name__ == "__main__":
    main()