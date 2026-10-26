"""
scripts/memberC/analyze_edp_landscape.py

Member C — Infrastructure — Stage C2-A: EDP landscape analysis.

Reads, read-only:
    results/memberC/edp_dataset.csv
    results/memberC/workload_summary.csv

Writes (results/memberC/):
    c2_strategy_wins.csv            (required) one row per workload: the
                                     min-median-EDP ("oracle") configuration
    c2_strategy_win_counts.csv      how many workloads each strategy wins
    c2_strategy_wins_by_track.csv   win counts, split tree vs kernelshap
    c2_strategy_wins_by_model.csv   win counts, split by model
    c2_core_effect.csv              (required) best strategy at each n_cores,
                                     per workload, + whether it changes
    c2_treeshap_size_effect.csv     best (min-over-cores) EDP per
                                     model x n_rows x strategy
    c2_kernelshap_factor_effect.csv best (min-over-cores) EDP per
                                     model x feature_count x background_size
                                     x strategy
    c2_correlations.csv             (required) time-vs-energy correlation,
                                     overall and per track
    c2_pareto_summary.csv           Pareto counts by track/strategy/overall
    c2_baseline_comparison.csv      (required) fixed-strategy EDP vs oracle,
                                     per workload x strategy
    c2_baseline_strategy_averages.csv  mean ratio/improvement per strategy

Every number printed or written here is computed directly from edp_dataset.csv
/ workload_summary.csv — no causal claims, no assumptions about WHY a pattern
appears, only what the data shows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

ALL_STRATEGIES = ["serial", "multiprocessing", "threading", "joblib", "ray"]

REQUIRED_EDP_COLS = [
    "track", "workload_id", "strategy", "n_cores", "model", "n_rows",
    "n_features", "feature_count", "background_size",
    "median_wall_time_sec", "median_energy_joules", "median_edp_joule_seconds",
    "is_pareto_optimal",
]


def find_file(relative_path: str) -> Path:
    for root in REPO_ROOT_CANDIDATES:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required input file not found: '{relative_path}'. "
        f"Looked under: {[str(r) for r in REPO_ROOT_CANDIDATES]}. "
        f"Run build_edp_dataset.py / build_workload_summary.py first."
    )


def output_dir() -> Path:
    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out_dir.exists():
        out_dir = Path.cwd() / "results" / "memberC"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    edp_path = find_file("results/memberC/edp_dataset.csv")
    ws_path = find_file("results/memberC/workload_summary.csv")

    edp = pd.read_csv(edp_path)
    ws = pd.read_csv(ws_path)

    if edp.empty:
        raise ValueError(f"'{edp_path}' loaded with zero rows.")
    if ws.empty:
        raise ValueError(f"'{ws_path}' loaded with zero rows.")

    missing = [c for c in REQUIRED_EDP_COLS if c not in edp.columns]
    if missing:
        raise ValueError(f"edp_dataset.csv missing required column(s): {missing}")

    edp["is_pareto_optimal"] = edp["is_pareto_optimal"].astype(bool)
    return edp, ws


# ---------------------------------------------------------------------------
# 1. Strategy win counts (the "oracle" configuration per workload)
# ---------------------------------------------------------------------------
def compute_strategy_wins(edp: pd.DataFrame) -> pd.DataFrame:
    idx = edp.groupby("workload_id")["median_edp_joule_seconds"].idxmin()
    winners = edp.loc[idx, [
        "workload_id", "track", "model", "n_rows", "feature_count", "background_size",
        "strategy", "n_cores",
        "median_edp_joule_seconds", "median_wall_time_sec", "median_energy_joules",
        "is_pareto_optimal",
    ]].rename(columns={
        "strategy": "winning_strategy",
        "n_cores": "winning_n_cores",
        "median_edp_joule_seconds": "winning_median_edp_joule_seconds",
        "median_wall_time_sec": "winning_median_wall_time_sec",
        "median_energy_joules": "winning_median_energy_joules",
        "is_pareto_optimal": "winner_is_pareto_optimal",
    }).sort_values(["track", "workload_id"]).reset_index(drop=True)
    return winners


# ---------------------------------------------------------------------------
# 4. Core-count effect — does the EDP-optimal strategy change with n_cores?
# ---------------------------------------------------------------------------
def compute_core_effect(edp: pd.DataFrame) -> pd.DataFrame:
    idx = edp.groupby(["workload_id", "n_cores"])["median_edp_joule_seconds"].idxmin()
    at_cores = edp.loc[idx, [
        "workload_id", "track", "n_cores", "strategy", "median_edp_joule_seconds",
    ]].rename(columns={
        "strategy": "best_strategy_at_this_n_cores",
        "median_edp_joule_seconds": "median_edp_at_this_n_cores",
    })

    changes = (
        at_cores.groupby("workload_id")["best_strategy_at_this_n_cores"]
        .nunique()
        .rename("n_unique_best_strategies_across_cores")
    )
    at_cores = at_cores.merge(changes, on="workload_id", how="left")
    at_cores["best_strategy_changes_with_cores"] = (
        at_cores["n_unique_best_strategies_across_cores"] > 1
    )
    return at_cores.sort_values(["track", "workload_id", "n_cores"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. TreeSHAP workload-size effect
# ---------------------------------------------------------------------------
def compute_treeshap_size_effect(edp: pd.DataFrame) -> pd.DataFrame:
    tree = edp[edp["track"] == "tree"]
    # "best this strategy can do" = min over cores, for each model x n_rows x strategy
    idx = tree.groupby(["model", "n_rows", "strategy"])["median_edp_joule_seconds"].idxmin()
    best = tree.loc[idx, [
        "model", "n_rows", "strategy", "n_cores", "median_edp_joule_seconds",
        "median_wall_time_sec", "median_energy_joules",
    ]].rename(columns={
        "n_cores": "best_n_cores_for_this_strategy",
    }).sort_values(["model", "n_rows", "median_edp_joule_seconds"]).reset_index(drop=True)
    return best


# ---------------------------------------------------------------------------
# 6. KernelSHAP factor effects (model, feature_count, background_size, cores)
# ---------------------------------------------------------------------------
def compute_kernelshap_factor_effect(edp: pd.DataFrame) -> pd.DataFrame:
    kernel = edp[edp["track"] == "kernelshap"]
    idx = kernel.groupby(
        ["model", "feature_count", "background_size", "strategy"]
    )["median_edp_joule_seconds"].idxmin()
    best = kernel.loc[idx, [
        "model", "feature_count", "background_size", "strategy", "n_cores",
        "median_edp_joule_seconds", "median_wall_time_sec", "median_energy_joules",
    ]].rename(columns={
        "n_cores": "best_n_cores_for_this_strategy",
    }).sort_values(
        ["model", "feature_count", "background_size", "median_edp_joule_seconds"]
    ).reset_index(drop=True)
    return best


# ---------------------------------------------------------------------------
# 7. Time-vs-energy correlation
# ---------------------------------------------------------------------------
def compute_correlations(edp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = {
        "overall": edp,
        "tree": edp[edp["track"] == "tree"],
        "kernelshap": edp[edp["track"] == "kernelshap"],
    }
    for scope_name, sub in scopes.items():
        t = sub["median_wall_time_sec"].to_numpy()
        e = sub["median_energy_joules"].to_numpy()
        pearson_r, pearson_p = stats.pearsonr(t, e)
        spearman_r, spearman_p = stats.spearmanr(t, e)
        rows.append({
            "scope": scope_name,
            "n_configurations": len(sub),
            "pearson_r": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p_value": spearman_p,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 8. Pareto summary
# ---------------------------------------------------------------------------
def compute_pareto_summary(edp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(edp)
    total_pareto = int(edp["is_pareto_optimal"].sum())
    rows.append({
        "dimension_type": "overall", "dimension_value": "all",
        "pareto_count": total_pareto, "total_count": total,
        "pareto_fraction": total_pareto / total,
    })
    for track, sub in edp.groupby("track"):
        p = int(sub["is_pareto_optimal"].sum())
        rows.append({
            "dimension_type": "track", "dimension_value": track,
            "pareto_count": p, "total_count": len(sub),
            "pareto_fraction": p / len(sub),
        })
    for strategy, sub in edp.groupby("strategy"):
        p = int(sub["is_pareto_optimal"].sum())
        rows.append({
            "dimension_type": "strategy", "dimension_value": strategy,
            "pareto_count": p, "total_count": len(sub),
            "pareto_fraction": p / len(sub),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 9. Fixed-strategy baselines vs oracle
# ---------------------------------------------------------------------------
def compute_baseline_comparison(edp: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    oracle = winners.set_index("workload_id")["winning_median_edp_joule_seconds"]
    track_lookup = winners.set_index("workload_id")["track"]

    rows = []
    for workload_id, group in edp.groupby("workload_id"):
        oracle_edp = oracle.loc[workload_id]
        track = track_lookup.loc[workload_id]
        for strategy in ALL_STRATEGIES:
            strat_rows = group[group["strategy"] == strategy]
            if strat_rows.empty:
                rows.append({
                    "workload_id": workload_id, "track": track, "strategy": strategy,
                    "fixed_strategy_edp_joule_seconds": np.nan,
                    "oracle_edp_joule_seconds": oracle_edp,
                    "edp_ratio_fixed_over_oracle": np.nan,
                    "pct_improvement_oracle_over_fixed": np.nan,
                    "evaluated": False,
                })
                continue
            fixed_edp = strat_rows["median_edp_joule_seconds"].min()
            ratio = fixed_edp / oracle_edp
            pct_improvement = (fixed_edp - oracle_edp) / fixed_edp * 100.0
            rows.append({
                "workload_id": workload_id, "track": track, "strategy": strategy,
                "fixed_strategy_edp_joule_seconds": fixed_edp,
                "oracle_edp_joule_seconds": oracle_edp,
                "edp_ratio_fixed_over_oracle": ratio,
                "pct_improvement_oracle_over_fixed": pct_improvement,
                "evaluated": True,
            })
    return pd.DataFrame(rows).sort_values(["track", "workload_id", "strategy"]).reset_index(drop=True)


def compute_baseline_strategy_averages(baseline: pd.DataFrame) -> pd.DataFrame:
    evaluated = baseline[baseline["evaluated"]]
    agg = evaluated.groupby("strategy").agg(
        n_workloads_evaluated=("workload_id", "nunique"),
        mean_edp_ratio_fixed_over_oracle=("edp_ratio_fixed_over_oracle", "mean"),
        median_edp_ratio_fixed_over_oracle=("edp_ratio_fixed_over_oracle", "median"),
        mean_pct_improvement_oracle_over_fixed=("pct_improvement_oracle_over_fixed", "mean"),
    ).reset_index().sort_values("mean_edp_ratio_fixed_over_oracle")
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    edp, ws = load_inputs()
    out_dir = output_dir()

    print("=" * 70)
    print("MEMBER C — EDP LANDSCAPE ANALYSIS (C2-A)")
    print("=" * 70)
    print(f"\nLoaded edp_dataset.csv: {len(edp)} configurations, "
          f"{edp['workload_id'].nunique()} workloads")

    # --- 1. Strategy win counts ---
    winners = compute_strategy_wins(edp)
    winners.to_csv(out_dir / "c2_strategy_wins.csv", index=False)

    win_counts = winners["winning_strategy"].value_counts().rename_axis("strategy").reset_index(name="workloads_won")
    win_counts.to_csv(out_dir / "c2_strategy_win_counts.csv", index=False)

    print("\n--- 1. STRATEGY WIN COUNTS (overall, 18 workloads) ---")
    print(win_counts.to_string(index=False))

    # --- 2. Winners by track ---
    by_track = (
        winners.groupby(["track", "winning_strategy"]).size()
        .unstack(fill_value=0)
    )
    by_track.to_csv(out_dir / "c2_strategy_wins_by_track.csv")
    print("\n--- 2. WINNERS BY TRACK ---")
    print(by_track.to_string())

    # --- 3. Winners by model ---
    by_model = (
        winners.groupby(["model", "winning_strategy"]).size()
        .unstack(fill_value=0)
    )
    by_model.to_csv(out_dir / "c2_strategy_wins_by_model.csv")
    print("\n--- 3. WINNERS BY MODEL ---")
    print(by_model.to_string())

    # --- 4. Core-count effect ---
    core_effect = compute_core_effect(edp)
    core_effect.to_csv(out_dir / "c2_core_effect.csv", index=False)
    changes_per_workload = core_effect.groupby("workload_id")["best_strategy_changes_with_cores"].first()
    n_changes = int(changes_per_workload.sum())
    n_total_wl = len(changes_per_workload)
    print(f"\n--- 4. CORE-COUNT EFFECT ---")
    print(f"Workloads where the EDP-optimal strategy CHANGES across n_cores: "
          f"{n_changes} / {n_total_wl}")
    print(f"Workloads where it STAYS THE SAME across n_cores: {n_total_wl - n_changes} / {n_total_wl}")

    # --- 5. TreeSHAP workload-size effect ---
    tree_size_effect = compute_treeshap_size_effect(edp)
    tree_size_effect.to_csv(out_dir / "c2_treeshap_size_effect.csv", index=False)
    print(f"\n--- 5. TREESHAP WORKLOAD-SIZE EFFECT ---")
    print("Best (min-over-cores) strategy per model x n_rows:")
    best_per_size = tree_size_effect.loc[
        tree_size_effect.groupby(["model", "n_rows"])["median_edp_joule_seconds"].idxmin()
    ][["model", "n_rows", "strategy", "median_edp_joule_seconds"]]
    print(best_per_size.to_string(index=False))

    # --- 6. KernelSHAP factor effects ---
    kernel_factor_effect = compute_kernelshap_factor_effect(edp)
    kernel_factor_effect.to_csv(out_dir / "c2_kernelshap_factor_effect.csv", index=False)
    print(f"\n--- 6. KERNELSHAP FACTOR EFFECTS ---")
    print("Best (min-over-cores) strategy per model x feature_count x background_size:")
    best_per_cond = kernel_factor_effect.loc[
        kernel_factor_effect.groupby(["model", "feature_count", "background_size"])["median_edp_joule_seconds"].idxmin()
    ][["model", "feature_count", "background_size", "strategy", "median_edp_joule_seconds"]]
    print(best_per_cond.to_string(index=False))

    # --- 7. Time-vs-energy correlation ---
    correlations = compute_correlations(edp)
    correlations.to_csv(out_dir / "c2_correlations.csv", index=False)
    print(f"\n--- 7. TIME vs ENERGY CORRELATION (median values, per configuration) ---")
    print(correlations.to_string(index=False))

    # --- 8. Pareto analysis ---
    pareto_summary = compute_pareto_summary(edp)
    pareto_summary.to_csv(out_dir / "c2_pareto_summary.csv", index=False)
    n_winner_pareto = int(winners["winner_is_pareto_optimal"].sum())
    print(f"\n--- 8. PARETO ANALYSIS ---")
    print(pareto_summary.to_string(index=False))
    print(f"\nWorkloads where the EDP winner is ALSO Pareto-optimal: "
          f"{n_winner_pareto} / {len(winners)}")

    # --- 9. Fixed-strategy baselines vs oracle ---
    baseline = compute_baseline_comparison(edp, winners)
    baseline.to_csv(out_dir / "c2_baseline_comparison.csv", index=False)
    baseline_avg = compute_baseline_strategy_averages(baseline)
    baseline_avg.to_csv(out_dir / "c2_baseline_strategy_averages.csv", index=False)
    print(f"\n--- 9. FIXED-STRATEGY BASELINES vs ORACLE ---")
    print("Average EDP ratio (fixed_strategy_EDP / oracle_EDP) and average %"
          " improvement of oracle over each fixed strategy, across all"
          " workloads where that strategy was evaluated:")
    print(baseline_avg.to_string(index=False))

    # --- 10. Key research findings (auto-generated, strictly from the above) ---
    print("\n" + "=" * 70)
    print("10. KEY RESEARCH FINDINGS (computed directly from the data)")
    print("=" * 70)

    top_strategy_row = win_counts.iloc[0]
    print(f"- '{top_strategy_row['strategy']}' wins the most workloads by median EDP "
          f"({int(top_strategy_row['workloads_won'])} / {len(winners)}).")

    print(f"- The EDP-optimal strategy changes across n_cores for {n_changes} of "
          f"{n_total_wl} workloads ({n_changes / n_total_wl * 100:.1f}%); it stays "
          f"fixed for the remaining {n_total_wl - n_changes}.")

    tree_top = by_track.loc["tree"].idxmax() if "tree" in by_track.index else None
    kernel_top = by_track.loc["kernelshap"].idxmax() if "kernelshap" in by_track.index else None
    if tree_top is not None and kernel_top is not None:
        same_or_diff = "the same strategy" if tree_top == kernel_top else "different strategies"
        print(f"- TreeSHAP's most-winning strategy is '{tree_top}'; KernelSHAP's is "
              f"'{kernel_top}' — {same_or_diff} lead each track.")

    overall_r = correlations.loc[correlations["scope"] == "overall", "pearson_r"].iloc[0]
    print(f"- Overall Pearson correlation between median wall time and median "
          f"energy across all {len(edp)} configurations: r = {overall_r:.4f}.")

    print(f"- {n_winner_pareto} of {len(winners)} workloads have an EDP winner that "
          f"is also Pareto-optimal.")

    closest_row = baseline_avg.iloc[0]
    farthest_row = baseline_avg.iloc[-1]
    print(f"- Of the 5 fixed-strategy baselines, '{closest_row['strategy']}' comes "
          f"closest to the oracle on average (mean EDP ratio = "
          f"{closest_row['mean_edp_ratio_fixed_over_oracle']:.3f}, i.e. oracle "
          f"improves on it by {closest_row['mean_pct_improvement_oracle_over_fixed']:.1f}% on average).")
    print(f"- '{farthest_row['strategy']}' is furthest from the oracle on average "
          f"(mean EDP ratio = {farthest_row['mean_edp_ratio_fixed_over_oracle']:.3f}, "
          f"oracle improves on it by "
          f"{farthest_row['mean_pct_improvement_oracle_over_fixed']:.1f}% on average).")

    print(f"\nAll output files written to: {out_dir}")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()