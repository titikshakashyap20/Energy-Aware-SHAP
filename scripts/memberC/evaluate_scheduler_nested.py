"""
scripts/memberC/evaluate_scheduler_nested.py

Member C — Infrastructure — Stage C3.5: nested Leave-One-Workload-Out evaluation.

WHY THIS EXISTS
---------------
The earlier C3 result (train_scheduler.py -> evaluate_scheduler.py) compared
DecisionTree vs RandomForest on the same 18 outer folds that were then used to
report the winner's performance. The outer test folds therefore influenced model
selection, so that RandomForest number is EXPLORATORY and optimistically biased.
This script gives the unbiased estimate. Existing C3 files are not modified.

Reads, read-only:
    results/memberC/edp_dataset.csv, scheduler_evaluation_protocol.csv, scheduler_splits.csv
    results/memberC/scheduler_evaluation.csv   (only to print the exploratory C3 numbers for contrast)
Writes:
    results/memberC/scheduler_nested_evaluation.csv

PROCEDURE (per outer fold = one complete held-out workload)
-----------------------------------------------------------
1. Remove the outer workload from edp_dataset and from the protocol. Nothing
   below can see it until step 4.
2. INNER selection on the remaining 17 workloads only: inner Leave-One-Workload-Out
   (17 inner folds, each trained on 16 workloads) for BOTH models, using the same
   fixed hyperparameters as C3. Inner score per workload = mean EDP regret over that
   inner workload's decision points (the workload, not a row, is the unit).
3. Selection rule (no arbitrary threshold) — paired one-standard-error rule:
     d_w = regret_DT(w) - regret_RF(w)   for each of the 17 inner workloads
     RF is selected iff mean(d) > SE(d) = std(d, ddof=1)/sqrt(17)
     (RF better on average by more than one standard error of the paired difference);
     otherwise the simpler, interpretable DecisionTree is selected. If DT already has the
     lower inner mean regret, DT is selected.
4. OUTER evaluation: refit the selected model on all 17 workloads, choose the strategy with
   the lowest predicted log10(median EDP) among strategies actually measured at each
   (workload, n_cores), and only then score with the outer measurements.

Unavailable strategies (KernelSHAP serial at 2/4/8 cores) are never candidates and never
fabricated. The scheduler chooses STRATEGY given n_cores; it does not choose core count.

Valid claim: generalisation to held-out workloads inside the evaluated benchmark space.
(Categorical levels for the one-hot encoding come from the full dataset; this uses no
measurements and every level appears in several training workloads.)

Run (after the C2-B files exist; train_scheduler.py is NOT required):
    python scripts/memberC/evaluate_scheduler_nested.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_scheduler import (  # noqa: E402
    MEANINGFUL_MARGIN_PCT, MODEL_SPECS, RANDOM_STATE, find_file, load_inputs, make_schema,
    output_dir, run_lowo, versions,
)
from evaluate_scheduler import annotate_decisions, build_summary, workload_bootstrap  # noqa: E402

MODELS = ["DecisionTree", "RandomForest"]  # DecisionTree first = simpler/preferred on ties
LABEL = "nested-LOWO"
STATUS = "nested_unbiased"


def inner_selection(outer_wl: str, edp: pd.DataFrame, protocol: pd.DataFrame,
                    splits: pd.DataFrame, schema: dict) -> dict:
    """Choose DT vs RF using ONLY the workloads other than `outer_wl`."""
    inner_ids = sorted(set(splits["held_out_workload_id"]) - {outer_wl})
    edp_in = edp[edp["workload_id"].isin(inner_ids)]
    prot_in = protocol[protocol["held_out_workload_id"].isin(inner_ids)]
    if (edp_in["workload_id"] == outer_wl).any() or (prot_in["held_out_workload_id"] == outer_wl).any():
        raise ValueError(f"Leakage: outer workload '{outer_wl}' present in inner-selection data.")

    # Inner splits: hold out each of the 17 workloads in turn, train on the other 16.
    sp = splits[splits["held_out_workload_id"].isin(inner_ids)].copy()
    sp["train_workload_ids"] = sp["held_out_workload_id"].map(
        lambda w: ";".join(i for i in inner_ids if i != w))
    sp["train_workload_count"] = len(inner_ids) - 1
    if sp["test_workload_ids"].str.contains(";").any() or len(sp) != len(inner_ids):
        raise ValueError("Inner split construction failed.")

    per_wl, per_model = {}, {}
    for name in MODELS:
        dec, _ = run_lowo(name, MODEL_SPECS[name]["params"], edp_in, prot_in, sp, schema)
        per_wl[name] = dec.groupby("held_out_workload_id")["edp_regret"].mean().reindex(inner_ids)
        per_model[name] = {"mean_regret": float(dec["edp_regret"].mean()),
                           "hit_rate": float(dec["is_oracle_hit"].mean())}

    d = per_wl["DecisionTree"] - per_wl["RandomForest"]       # > 0 favours RandomForest
    n = len(d)
    mean_d = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    selected = "RandomForest" if (mean_d > 0 and mean_d > se) else "DecisionTree"
    return {"selected": selected, "inner_n_workloads": n, "paired_mean_diff_dt_minus_rf": mean_d,
            "paired_se": se, "inner_dt_mean_regret": per_model["DecisionTree"]["mean_regret"],
            "inner_rf_mean_regret": per_model["RandomForest"]["mean_regret"],
            "inner_dt_hit_rate": per_model["DecisionTree"]["hit_rate"],
            "inner_rf_hit_rate": per_model["RandomForest"]["hit_rate"]}


def main() -> None:
    edp, protocol, splits = load_inputs()
    schema = make_schema(edp)
    splits = splits.sort_values("split_id").reset_index(drop=True)

    sel_rows, dec_frames, pred_frames = [], [], []
    print("=" * 78)
    print("MEMBER C — NESTED LOWO SCHEDULER EVALUATION (C3.5, unbiased)")
    print("=" * 78)
    for i, sp in splits.iterrows():
        outer = sp["held_out_workload_id"]
        sel = inner_selection(outer, edp, protocol, splits, schema)
        name = sel["selected"]
        # Outer evaluation: run_lowo on THIS single split trains on the 17 other workloads only.
        dec, pred = run_lowo(name, MODEL_SPECS[name]["params"], edp, protocol, splits.iloc[[i]], schema)
        dec["selected_model"] = name
        dec_frames.append(dec)
        pred_frames.append(pred)
        sel_rows.append({"outer_split_id": int(sp["split_id"]), "outer_workload_id": outer, **sel})
        print(f"  outer {i + 1:2d}/18  {outer:<42s} -> {name:<12s} "
              f"(inner mean regret DT {sel['inner_dt_mean_regret']:.3f} / RF {sel['inner_rf_mean_regret']:.3f}; "
              f"paired diff {sel['paired_mean_diff_dt_minus_rf']:+.3f}, SE {sel['paired_se']:.3f})")

    dec = annotate_decisions(pd.concat(dec_frames, ignore_index=True))
    pred = pd.concat(pred_frames, ignore_index=True)
    selection = pd.DataFrame(sel_rows)
    if len(dec) != 72 or dec["held_out_workload_id"].nunique() != 18:
        raise ValueError(f"Expected 72 decision points over 18 workloads, got {len(dec)}.")

    summary = build_summary(dec, pred, protocol, LABEL)

    # -------------------------------------------------- CSV (record_type: selection | summary | decision)
    selection_out = selection.assign(record_type="selection")
    dec_out = dec.copy()
    dec_out.insert(0, "record_type", "decision")
    dec_out.insert(1, "scope", dec_out["track"])
    dec_out.insert(2, "policy", f"scheduler ({LABEL})")
    out = pd.concat([summary, dec_out, selection_out], ignore_index=True)
    out.insert(0, "result_status", STATUS)
    lead = ["result_status", "record_type", "scope", "policy"]
    out = out[lead + [c for c in out.columns if c not in lead]]
    out_path = output_dir() / "scheduler_nested_evaluation.csv"
    out.to_csv(out_path, index=False)

    # -------------------------------------------------- console report
    pol = f"scheduler ({LABEL})"
    print("\n[1] MODEL SELECTED ACROSS THE 18 OUTER FOLDS (inner paired one-SE rule)")
    print(selection["selected"].value_counts().reindex(MODELS).fillna(0).astype(int).to_string())

    ov = summary[(summary["scope"] == "overall") & (summary["policy"] == pol)].iloc[0]
    print("\n[2-5] NESTED-LOWO OVERALL (72 decision points)")
    print(f"  oracle hit rate          : {ov['oracle_hit_rate']:.3f}  ({int(ov['n_oracle_hits'])}/72)")
    print(f"  EDP regret mean/median/max: {ov['mean_edp_regret']:.3f} / {ov['median_edp_regret']:.3f} / "
          f"{ov['max_edp_regret']:.3f}")
    print(f"  mean EDP ratio to oracle : {ov['mean_edp_ratio_to_oracle']:.3f} "
          f"(median {ov['median_edp_ratio_to_oracle']:.3f}, geomean {ov['geomean_edp_ratio_to_oracle']:.3f})")
    print(f"  mean time ratio / energy ratio to oracle: {ov['mean_time_ratio_to_oracle']:.3f} / "
          f"{ov['mean_energy_ratio_to_oracle']:.3f}")
    print(f"  held-out log10 EDP error : MAE {ov['mae_log10_edp']:.3f}, RMSE {ov['rmse_log10_edp']:.3f}, "
          f"R2 {ov['r2_log10_edp']:.3f}")

    cols = ["policy", "n_decision_points", "oracle_hit_rate", "mean_edp_regret", "median_edp_regret",
            "max_edp_regret", "mean_time_ratio_to_oracle", "mean_energy_ratio_to_oracle"]
    for scope in ["overall", "tree", "kernelshap"]:
        print(f"\n[6-7] {scope.upper()} — nested scheduler vs fixed baselines (fixed scored only where measured)")
        print(summary[summary["scope"] == scope][cols].sort_values("mean_edp_regret")
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        s = summary[(summary["scope"] == scope) & (summary["policy"] == pol)].iloc[0]
        print(f"  vs Joblib: better {int(s['n_better_than_joblib'])}, worse {int(s['n_worse_than_joblib'])}, "
              f"same choice {int(s['n_same_choice_as_joblib'])}, tie {int(s['n_tie_with_joblib'])} | "
              f">= {MEANINGFUL_MARGIN_PCT:.0f}% better {int(s['n_better_ge_margin'])}, "
              f"worse {int(s['n_worse_ge_margin'])} | geomean scheduler/joblib EDP "
              f"{s['geomean_scheduler_over_joblib_edp']:.3f}")

    def line(r) -> str:
        return (f"{r.held_out_workload_id} | cores={r.n_cores} | [{r.selected_model}] chose "
                f"{r.predicted_strategy}, oracle {r.oracle_strategy} | regret {r.edp_regret * 100:.1f}% | "
                f"vs joblib {r.edp_improvement_vs_joblib_pct:+.1f}%")

    better = dec[dec["vs_joblib"] == "scheduler_better"].sort_values("edp_improvement_vs_joblib_pct", ascending=False)
    worse = dec[dec["vs_joblib"] == "scheduler_worse"].sort_values("edp_improvement_vs_joblib_pct")
    print(f"\n[8] Beats Joblib at {len(better)} decision points; loses at {len(worse)}.")
    print("  Losses to Joblib:")
    for r in worse.itertuples():
        print("    " + line(r))

    print("\n[9] Top 10 largest EDP regrets:")
    for r in dec.sort_values("edp_regret", ascending=False).head(10).itertuples():
        print("  " + line(r))

    by_model = dec.groupby("selected_model").agg(n=("edp_regret", "size"), hit=("is_oracle_hit", "mean"),
                                                 mean_regret=("edp_regret", "mean"))
    print("\nOuter performance split by which model the inner step selected:")
    print(by_model.to_string(float_format=lambda v: f"{v:.3f}"))

    bs = workload_bootstrap(dec)
    if bs:
        f = lambda t: f"[{t[0]:.3f}, {t[1]:.3f}]"
        print(f"\nWorkload-level bootstrap (10000 resamples of 18 workloads, seed 0), 95% intervals of mean regret:")
        print(f"  scheduler {f(bs['scheduler'])} | joblib {f(bs['joblib'])} | scheduler - joblib {f(bs['diff'])}")
        print(f"  share of resamples where scheduler is NOT better than joblib: {bs['p_sched_not_better']:.3f}")

    # [10] unbiased vs exploratory
    print("\n[10] UNBIASED NESTED RESULT vs EXPLORATORY C3 RESULT")
    try:
        c3 = pd.read_csv(find_file("results/memberC/scheduler_evaluation.csv"))
        c3 = c3[(c3["record_type"] == "summary") & (c3["scope"] == "overall") &
                (c3["policy"].str.startswith("scheduler"))].iloc[0]
        print(f"  {'':34s}{'hit rate':>10s}{'mean regret':>13s}{'median':>9s}{'max':>8s}")
        print(f"  {'EXPLORATORY C3 (selection used outer folds, BIASED)':<34s}"
              f"{c3['oracle_hit_rate']:>10.3f}{c3['mean_edp_regret']:>13.3f}"
              f"{c3['median_edp_regret']:>9.3f}{c3['max_edp_regret']:>8.3f}")
    except (FileNotFoundError, IndexError):
        print("  (scheduler_evaluation.csv not found; exploratory numbers not shown)")
    print(f"  {'NESTED C3.5 (UNBIASED, report this)':<34s}{ov['oracle_hit_rate']:>10.3f}"
          f"{ov['mean_edp_regret']:>13.3f}{ov['median_edp_regret']:>9.3f}{ov['max_edp_regret']:>8.3f}")

    print("\nVersions:", json.dumps(versions()), f"| random_state={RANDOM_STATE}")
    print(f"\nscheduler_nested_evaluation.csv written to: {out_path}")
    print("\nInterpretation limit: 18 independent workloads, 72 correlated decision points. Valid claim = "
          "generalisation to held-out workloads within the evaluated benchmark space, not arbitrary datasets.")
    print("=" * 78)


if __name__ == "__main__":
    main()