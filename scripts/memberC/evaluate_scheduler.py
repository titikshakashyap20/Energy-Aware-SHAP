"""
scripts/memberC/evaluate_scheduler.py

Member C — Infrastructure — Stage C3: held-out evaluation of the selected scheduler.

Reads, read-only:
    results/memberC/scheduler_final_model.joblib   (selected model name + hyperparameters)
    results/memberC/scheduler_model_comparison.csv (optional consistency check)
    results/memberC/edp_dataset.csv, scheduler_evaluation_protocol.csv, scheduler_splits.csv

Writes:
    results/memberC/scheduler_evaluation.csv

How the evaluation stays leakage-free
-------------------------------------
The saved final estimator was refit on ALL 18 workloads, so scoring it on those
workloads would be training-set evaluation. Instead this script re-fits the
selected model (same class, same hyperparameters) inside every one of the 18
Leave-One-Workload-Out folds, exactly as train_scheduler.py did, and scores each
fold's decisions on the held-out workload only. Strategy selection uses
workload characteristics + n_cores + candidate strategy only; measured values
are used only afterwards to score the choice. Unavailable strategies (KernelSHAP
serial at 2/4/8 cores) are never candidates and never fabricated.

Output layout (one CSV, `record_type` column):
    record_type == "summary"  : scope (overall / tree / kernelshap) x policy
                                (scheduler, each fixed strategy), vs oracle,
                                plus scheduler-vs-Joblib counts.
    record_type == "decision" : one row per held-out workload x n_cores decision
                                point, with oracle hit/miss, regret rank and
                                scheduler-vs-Joblib outcome.

Run (after train_scheduler.py):  python scripts/memberC/evaluate_scheduler.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_scheduler import (  # noqa: E402
    ALL_STRATEGIES, MEANINGFUL_MARGIN_PCT, decision_metrics, find_file, fixed_policy_decisions,
    load_inputs, output_dir, prediction_metrics, run_lowo, scopes, versions,
)

TIE_REL_TOL = 1e-9
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 0


def load_bundle() -> dict:
    try:
        path = find_file("results/memberC/scheduler_final_model.joblib")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc} Run train_scheduler.py first.") from exc
    return joblib.load(path)


def annotate_decisions(dec: pd.DataFrame) -> pd.DataFrame:
    d = dec.copy()
    d["oracle_outcome"] = np.where(d["is_oracle_hit"], "correct_oracle", "missed_oracle")
    d["regret_rank"] = d["edp_regret"].rank(method="min", ascending=False).astype(int)
    d["joblib_edp_ratio_to_oracle"] = d["joblib_edp"] / d["oracle_edp"]
    d["edp_improvement_vs_joblib_pct"] = (1.0 - d["scheduler_edp"] / d["joblib_edp"]) * 100.0

    def vs_joblib(r) -> str:
        if pd.isna(r["joblib_edp"]):
            return "joblib_unavailable"
        if r["predicted_strategy"] == "joblib":
            return "same_choice"
        rel = r["scheduler_edp"] / r["joblib_edp"]
        if rel < 1 - TIE_REL_TOL:
            return "scheduler_better"
        if rel > 1 + TIE_REL_TOL:
            return "scheduler_worse"
        return "tie"

    d["vs_joblib"] = d.apply(vs_joblib, axis=1)
    d["vs_joblib_meaningful"] = np.select(
        [(d["vs_joblib"] == "scheduler_better") & (d["edp_improvement_vs_joblib_pct"] >= MEANINGFUL_MARGIN_PCT),
         (d["vs_joblib"] == "scheduler_worse") & (-d["edp_improvement_vs_joblib_pct"] >= MEANINGFUL_MARGIN_PCT)],
        ["better_ge_margin", "worse_ge_margin"], default="within_margin")
    return d


def joblib_comparison_metrics(dec: pd.DataFrame) -> dict:
    v = dec["vs_joblib"]
    ok = dec[dec["joblib_edp"].notna()]
    ratio = (ok["scheduler_edp"] / ok["joblib_edp"]) if len(ok) else pd.Series(dtype=float)
    return {
        "n_better_than_joblib": int((v == "scheduler_better").sum()),
        "n_worse_than_joblib": int((v == "scheduler_worse").sum()),
        "n_same_choice_as_joblib": int((v == "same_choice").sum()),
        "n_tie_with_joblib": int((v == "tie").sum()),
        "n_better_ge_margin": int((dec["vs_joblib_meaningful"] == "better_ge_margin").sum()),
        "n_worse_ge_margin": int((dec["vs_joblib_meaningful"] == "worse_ge_margin").sum()),
        "geomean_scheduler_over_joblib_edp": float(np.exp(np.log(ratio).mean())) if len(ratio) else np.nan,
        "mean_pct_edp_improvement_vs_joblib": float(ok["edp_improvement_vs_joblib_pct"].mean()) if len(ok) else np.nan,
    }


def build_summary(dec: pd.DataFrame, pred: pd.DataFrame, protocol: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rows = []
    for scope, sub in scopes(dec):
        psub = pred if scope == "overall" else pred[pred["track"] == scope]
        rows.append({"record_type": "summary", "scope": scope, "policy": f"scheduler ({model_name})",
                     **decision_metrics(sub), **prediction_metrics(psub), **joblib_comparison_metrics(sub),
                     "scheduler_mean_edp_ratio_same_points": float(sub["edp_ratio_to_oracle"].mean()),
                     "scheduler_oracle_hit_rate_same_points": float(sub["is_oracle_hit"].mean())})
    for s in ALL_STRATEGIES:
        fx = fixed_policy_decisions(protocol, s)
        for scope, sub in scopes(fx):
            if sub.empty:
                continue
            keys = sub[["split_id", "n_cores"]]
            same = dec.merge(keys, on=["split_id", "n_cores"])  # scheduler on the SAME decision points
            rows.append({"record_type": "summary", "scope": scope, "policy": f"fixed_{s}", **decision_metrics(sub),
                         "scheduler_mean_edp_ratio_same_points": float(same["edp_ratio_to_oracle"].mean()),
                         "scheduler_oracle_hit_rate_same_points": float(same["is_oracle_hit"].mean())})
    return pd.DataFrame(rows)


def workload_bootstrap(dec: pd.DataFrame) -> dict | None:
    """Cluster bootstrap over WORKLOADS (decision points within a workload are correlated)."""
    if dec["joblib_edp"].isna().any():
        return None
    d = dec.assign(joblib_regret=dec["joblib_edp"] / dec["oracle_edp"] - 1.0)
    g = d.groupby("held_out_workload_id").agg(s=("edp_regret", "sum"), j=("joblib_regret", "sum"),
                                              n=("edp_regret", "size"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(g), size=(BOOTSTRAP_REPS, len(g)))
    n = g["n"].to_numpy()[idx].sum(1)
    s = g["s"].to_numpy()[idx].sum(1) / n
    j = g["j"].to_numpy()[idx].sum(1) / n
    q = lambda a: tuple(np.percentile(a, [2.5, 97.5]))
    return {"scheduler": q(s), "joblib": q(j), "diff": q(s - j), "p_sched_not_better": float((s - j >= 0).mean())}


def main() -> None:
    bundle = load_bundle()
    name, params = bundle["selected_model"], bundle["hyperparameters"]
    edp, protocol, splits = load_inputs()
    schema = bundle["schema"]

    saved_v = bundle["metadata"]["versions"]
    now_v = versions()
    if saved_v.get("scikit-learn") != now_v["scikit-learn"]:
        print(f"WARNING: model trained with scikit-learn {saved_v.get('scikit-learn')}, "
              f"running {now_v['scikit-learn']}; LOWO results may differ slightly.")

    dec, pred = run_lowo(name, params, edp, protocol, splits, schema)
    dec = annotate_decisions(dec)
    summary = build_summary(dec, pred, protocol, name)

    # Consistency with the model-comparison file written by train_scheduler.py
    try:
        comp = pd.read_csv(find_file("results/memberC/scheduler_model_comparison.csv"))
        ref = comp[(comp["model"] == name) & (comp["scope"] == "overall")]["mean_edp_regret"].iloc[0]
        mine = dec["edp_regret"].mean()
        if not np.isclose(ref, mine, rtol=1e-6, atol=1e-9):
            print(f"WARNING: mean regret {mine:.6f} differs from scheduler_model_comparison.csv ({ref:.6f}).")
    except FileNotFoundError:
        print("NOTE: scheduler_model_comparison.csv not found; skipped consistency check.")

    dec_out = dec.copy()
    dec_out.insert(0, "record_type", "decision")
    dec_out.insert(1, "scope", dec_out["track"])
    dec_out.insert(2, "policy", f"scheduler ({name})")
    out = pd.concat([summary, dec_out], ignore_index=True)
    lead = ["record_type", "scope", "policy"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out_path = output_dir() / "scheduler_evaluation.csv"
    out.to_csv(out_path, index=False)

    # ------------------------------------------------------------ console report
    pol = f"scheduler ({name})"
    print("=" * 78)
    print(f"MEMBER C — SCHEDULER EVALUATION (C3) — {name}, 18-fold Leave-One-Workload-Out")
    print("=" * 78)
    cols = ["policy", "n_decision_points", "oracle_hit_rate", "mean_edp_regret", "median_edp_regret",
            "max_edp_regret", "mean_edp_ratio_to_oracle", "mean_time_ratio_to_oracle",
            "mean_energy_ratio_to_oracle", "scheduler_mean_edp_ratio_same_points"]
    for scope in ["overall", "tree", "kernelshap"]:
        print(f"\n--- {scope.upper()} (policies scored only where actually measured) ---")
        print(summary[summary["scope"] == scope][cols].sort_values("mean_edp_ratio_to_oracle")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        s = summary[(summary["scope"] == scope) & (summary["policy"] == pol)].iloc[0]
        print(f"  Held-out EDP prediction error (log10): MAE={s['mae_log10_edp']:.3f} "
              f"(~{s['typical_edp_fold_error']:.2f}x), RMSE={s['rmse_log10_edp']:.3f}, R2={s['r2_log10_edp']:.3f}")
        print(f"  vs Joblib: better {int(s['n_better_than_joblib'])}, worse {int(s['n_worse_than_joblib'])}, "
              f"same choice {int(s['n_same_choice_as_joblib'])}, tie {int(s['n_tie_with_joblib'])} | "
              f">= {MEANINGFUL_MARGIN_PCT:.0f}% better {int(s['n_better_ge_margin'])}, "
              f"worse {int(s['n_worse_ge_margin'])} | "
              f"geomean scheduler/joblib EDP = {s['geomean_scheduler_over_joblib_edp']:.3f}")

    def line(r) -> str:
        return (f"{r.held_out_workload_id} | cores={r.n_cores} | chose {r.predicted_strategy}, oracle "
                f"{r.oracle_strategy} | regret {r.edp_regret * 100:.1f}% | vs joblib "
                f"{r.edp_improvement_vs_joblib_pct:+.1f}%")

    hits = dec[dec["is_oracle_hit"]]
    misses = dec[~dec["is_oracle_hit"]].sort_values("edp_regret", ascending=False)
    print(f"\nOracle decisions: {len(hits)} correct, {len(misses)} missed (of {len(dec)})")
    print("\nTop 10 largest EDP regrets:")
    for r in dec.sort_values("edp_regret", ascending=False).head(10).itertuples():
        print("  " + line(r))

    better = dec[dec["vs_joblib"] == "scheduler_better"].sort_values("edp_improvement_vs_joblib_pct", ascending=False)
    worse = dec[dec["vs_joblib"] == "scheduler_worse"].sort_values("edp_improvement_vs_joblib_pct")
    print(f"\nScheduler BEATS Joblib at {len(better)} decision points "
          f"({int((better['vs_joblib_meaningful'] == 'better_ge_margin').sum())} by >= {MEANINGFUL_MARGIN_PCT:.0f}%):")
    for r in better.itertuples():
        print("  " + line(r))
    print(f"\nScheduler LOSES to Joblib at {len(worse)} decision points "
          f"({int((worse['vs_joblib_meaningful'] == 'worse_ge_margin').sum())} by >= {MEANINGFUL_MARGIN_PCT:.0f}%):")
    for r in worse.itertuples():
        print("  " + line(r))

    bs = workload_bootstrap(dec)
    if bs:
        f = lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]"
        print(f"\nWorkload-level bootstrap ({BOOTSTRAP_REPS} resamples of the 18 workloads, seed {BOOTSTRAP_SEED}), "
              f"95% intervals of mean EDP regret:")
        print(f"  scheduler {f(bs['scheduler'])} | joblib {f(bs['joblib'])} | scheduler - joblib {f(bs['diff'])}")
        print(f"  share of resamples where scheduler is NOT better than joblib: {bs['p_sched_not_better']:.3f}")

    print(f"\nscheduler_evaluation.csv written to: {out_path}")
    print("\nInterpretation limit: 18 independent workloads, 72 correlated decision points. Results support "
          "generalisation to held-out workloads inside the evaluated benchmark space only, not to arbitrary "
          "datasets, models or hardware.")
    print("=" * 78)


if __name__ == "__main__":
    main()