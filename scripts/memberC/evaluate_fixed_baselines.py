"""
scripts/memberC/evaluate_fixed_baselines.py

Member C — Infrastructure — Stage C2-B: fixed-strategy baseline evaluation.

Reads, read-only:
    results/memberC/scheduler_evaluation_protocol.csv

Writes:
    results/memberC/fixed_baseline_evaluation.csv

For every decision point in the protocol (held-out workload x n_cores,
built with leakage-safe LOGO splits), evaluates each of the 5 fixed
"always use strategy X" policies against the oracle (the best strategy
actually measured at that exact workload/n_cores). A baseline is only
scored at a decision point where it was actually measured
(baseline_available == True in the protocol) — never fabricated.

    edp_ratio_to_oracle         = baseline_edp / oracle_edp
    pct_edp_improvement_vs_baseline = (1 - oracle_edp / baseline_edp) * 100

Reported separately for: overall, tree, kernelshap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

ALL_STRATEGIES = ["serial", "multiprocessing", "threading", "joblib", "ray"]
MEANINGFUL_MARGIN_PCT = 10.0  # chosen threshold, documented explicitly — not implied by the data

REQUIRED_COLS = [
    "split_id", "held_out_workload_id", "track", "n_cores",
    "candidate_strategy", "measured_median_edp", "measured_median_time",
    "measured_median_energy", "oracle_strategy", "oracle_edp", "oracle_time",
    "oracle_energy", "baseline_available",
]


def find_file(relative_path: str) -> Path:
    for root in REPO_ROOT_CANDIDATES:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required input file not found: '{relative_path}'. "
        f"Looked under: {[str(r) for r in REPO_ROOT_CANDIDATES]}. "
        f"Run build_scheduler_evaluation.py first."
    )


def output_dir() -> Path:
    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out_dir.exists():
        out_dir = Path.cwd() / "results" / "memberC"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_protocol() -> pd.DataFrame:
    path = find_file("results/memberC/scheduler_evaluation_protocol.csv")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"'{path}' loaded with zero rows.")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"scheduler_evaluation_protocol.csv missing required column(s): {missing}")
    df["baseline_available"] = df["baseline_available"].astype(bool)

    # --- Fail loudly: recompute the oracle independently and compare ---
    # (defends against a hand-edited or stale protocol file; this script
    # does not trust oracle_strategy/oracle_edp blindly)
    available = df[df["baseline_available"]].copy()
    recomputed_idx = available.groupby(["split_id", "n_cores"])["measured_median_edp"].idxmin()
    recomputed_oracle = available.loc[recomputed_idx].set_index(["split_id", "n_cores"])

    stored_oracle = df.drop_duplicates(subset=["split_id", "n_cores"]).set_index(["split_id", "n_cores"])[
        ["oracle_strategy", "oracle_edp"]
    ]
    merged = stored_oracle.join(
        recomputed_oracle[["candidate_strategy", "measured_median_edp"]], how="left"
    )
    mismatch = merged[
        (merged["oracle_strategy"] != merged["candidate_strategy"])
        | (~np.isclose(merged["oracle_edp"], merged["measured_median_edp"]))
    ]
    if not mismatch.empty:
        raise ValueError(
            f"Impossible/inconsistent data: {len(mismatch)} decision point(s) in "
            f"scheduler_evaluation_protocol.csv have a stored oracle that does not "
            f"match the minimum available measured_median_edp. First offending:\n"
            f"{mismatch.head().to_string()}"
        )

    return df


def compute_baseline_rows(protocol: pd.DataFrame) -> pd.DataFrame:
    rows = protocol[protocol["baseline_available"]].copy()
    rows["edp_ratio_to_oracle"] = rows["measured_median_edp"] / rows["oracle_edp"]
    rows["pct_edp_improvement_vs_baseline"] = (
        1 - rows["oracle_edp"] / rows["measured_median_edp"]
    ) * 100.0
    rows["time_ratio_to_oracle"] = rows["measured_median_time"] / rows["oracle_time"]
    rows["energy_ratio_to_oracle"] = rows["measured_median_energy"] / rows["oracle_energy"]
    return rows


def summarize(rows: pd.DataFrame, scope_name: str) -> pd.DataFrame:
    agg = rows.groupby("candidate_strategy").agg(
        n_valid_decision_points=("edp_ratio_to_oracle", "count"),
        mean_edp_ratio_to_oracle=("edp_ratio_to_oracle", "mean"),
        median_edp_ratio_to_oracle=("edp_ratio_to_oracle", "median"),
        mean_pct_improvement_available=("pct_edp_improvement_vs_baseline", "mean"),
        median_pct_improvement_available=("pct_edp_improvement_vs_baseline", "median"),
        mean_time_ratio_to_oracle=("time_ratio_to_oracle", "mean"),
        mean_energy_ratio_to_oracle=("energy_ratio_to_oracle", "mean"),
    ).reset_index().rename(columns={"candidate_strategy": "strategy"})
    agg.insert(0, "scope", scope_name)
    return agg.sort_values("mean_edp_ratio_to_oracle").reset_index(drop=True)


def main() -> None:
    protocol = load_protocol()
    baseline_rows = compute_baseline_rows(protocol)

    summaries = [
        summarize(baseline_rows, "overall"),
        summarize(baseline_rows[baseline_rows["track"] == "tree"], "tree"),
        summarize(baseline_rows[baseline_rows["track"] == "kernelshap"], "kernelshap"),
    ]
    full_summary = pd.concat(summaries, ignore_index=True)

    out_path = output_dir() / "fixed_baseline_evaluation.csv"
    full_summary.to_csv(out_path, index=False)

    # --- Console report ---
    print("=" * 70)
    print("MEMBER C — FIXED-STRATEGY BASELINE EVALUATION (C2-B)")
    print("=" * 70)

    for scope_name in ["overall", "tree", "kernelshap"]:
        print(f"\n--- {scope_name.upper()} ---")
        sub = full_summary[full_summary["scope"] == scope_name].drop(columns="scope")
        print(sub.to_string(index=False))

    overall = full_summary[full_summary["scope"] == "overall"]
    strongest = overall.iloc[0]
    weakest = overall.iloc[-1]
    print(f"\nStrongest fixed baseline (lowest mean EDP ratio to oracle, overall): "
          f"'{strongest['strategy']}' (ratio = {strongest['mean_edp_ratio_to_oracle']:.3f}, "
          f"n = {int(strongest['n_valid_decision_points'])} decision points)")
    print(f"Weakest fixed baseline (highest mean EDP ratio to oracle, overall): "
          f"'{weakest['strategy']}' (ratio = {weakest['mean_edp_ratio_to_oracle']:.3f}, "
          f"n = {int(weakest['n_valid_decision_points'])} decision points)")

    # --- Workloads where joblib is NOT the oracle ---
    one_row_per_point = protocol.drop_duplicates(subset=["split_id", "n_cores"])
    non_joblib_oracle = one_row_per_point[one_row_per_point["oracle_strategy"] != "joblib"]
    non_joblib_workloads = sorted(non_joblib_oracle["held_out_workload_id"].unique())
    print(f"\nWorkloads where joblib is NOT the oracle at at least one n_cores decision point: "
          f"{len(non_joblib_workloads)} / {one_row_per_point['held_out_workload_id'].nunique()}")
    for wl in non_joblib_workloads:
        wl_points = non_joblib_oracle[non_joblib_oracle["held_out_workload_id"] == wl]
        detail = ", ".join(f"n_cores={r.n_cores}->{r.oracle_strategy}" for r in wl_points.itertuples())
        print(f"  - {wl}: {detail}")

    # --- Workloads where a non-joblib strategy beats joblib by a meaningful margin ---
    joblib_rows = protocol[
        (protocol["candidate_strategy"] == "joblib") & (protocol["baseline_available"])
    ][["split_id", "n_cores", "measured_median_edp"]].rename(columns={"measured_median_edp": "joblib_edp"})

    merged = one_row_per_point.merge(joblib_rows, on=["split_id", "n_cores"], how="left")
    merged["margin_pct"] = (1 - merged["oracle_edp"] / merged["joblib_edp"]) * 100.0
    meaningful = merged[
        (merged["oracle_strategy"] != "joblib")
        & merged["joblib_edp"].notna()
        & (merged["margin_pct"] >= MEANINGFUL_MARGIN_PCT)
    ]
    print(f"\nDecision points where a non-joblib strategy beats joblib by >= "
          f"{MEANINGFUL_MARGIN_PCT:.0f}% EDP margin (a fixed, stated threshold, not "
          f"implied by the data): {len(meaningful)}")
    meaningful_workloads = sorted(meaningful["held_out_workload_id"].unique())
    for wl in meaningful_workloads:
        wl_points = meaningful[meaningful["held_out_workload_id"] == wl]
        detail = ", ".join(
            f"n_cores={r.n_cores}->{r.oracle_strategy} ({r.margin_pct:.1f}% better)"
            for r in wl_points.itertuples()
        )
        print(f"  - {wl}: {detail}")

    print(f"\nfixed_baseline_evaluation.csv written to: {out_path}")
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()