"""
scripts/memberC/build_workload_summary.py

Member C — Infrastructure — Stage: Per-workload summary.

Reads, read-only:
    results/memberC/edp_dataset.csv   (the AGGREGATED per-configuration
                                        table produced by build_edp_dataset.py)

Writes:
    results/memberC/workload_summary.csv

For each workload_id (18 expected: 6 tree + 12 kernelshap), reports the
workload's identifying context plus its best configuration by median EDP
and how many of its configurations are Pareto-optimal.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

CONTEXT_COLS = ["track", "model", "n_rows", "n_features", "feature_count", "background_size"]

REQUIRED_COLS = CONTEXT_COLS + [
    "workload_id", "strategy", "n_cores",
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
        f"Run build_edp_dataset.py before this script."
    )


def output_dir() -> Path:
    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out_dir.exists():
        out_dir = Path.cwd() / "results" / "memberC"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    edp_path = find_file("results/memberC/edp_dataset.csv")
    edp = pd.read_csv(edp_path)

    if edp.empty:
        raise ValueError(f"'{edp_path}' loaded with zero rows.")

    missing = [c for c in REQUIRED_COLS if c not in edp.columns]
    if missing:
        raise ValueError(
            f"edp_dataset.csv is missing required column(s) {missing}. "
            f"Columns present: {list(edp.columns)}."
        )

    edp["is_pareto_optimal"] = edp["is_pareto_optimal"].astype(bool)

    # Workload-identifying columns must be constant within a workload_id —
    # fail loudly if not (would mean edp_dataset.csv itself is inconsistent).
    for col in CONTEXT_COLS:
        nunique_per_wl = edp.groupby("workload_id", dropna=False)[col].nunique(dropna=False)
        inconsistent = nunique_per_wl[nunique_per_wl > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"Column '{col}' is not constant within {len(inconsistent)} "
                f"workload_id group(s) in edp_dataset.csv. First offending:\n"
                f"{inconsistent.head().to_string()}"
            )

    summary_rows = []
    for workload_id, group in edp.groupby("workload_id", dropna=False):
        best_idx = group["median_edp_joule_seconds"].idxmin()
        best_row = group.loc[best_idx]

        summary_rows.append(
            {
                "track": group["track"].iloc[0],
                "workload_id": workload_id,
                "model": group["model"].iloc[0],
                "n_rows": group["n_rows"].iloc[0],
                "n_features": group["n_features"].iloc[0],
                "feature_count": group["feature_count"].iloc[0],
                "background_size": group["background_size"].iloc[0],
                "n_strategies_evaluated": group["strategy"].nunique(),
                "n_core_counts_evaluated": group["n_cores"].nunique(),
                "n_configurations_evaluated": len(group),
                "best_strategy": best_row["strategy"],
                "best_n_cores": best_row["n_cores"],
                "best_median_edp_joule_seconds": best_row["median_edp_joule_seconds"],
                "best_median_wall_time_sec": best_row["median_wall_time_sec"],
                "best_median_energy_joules": best_row["median_energy_joules"],
                "n_pareto_optimal_configurations": int(group["is_pareto_optimal"].sum()),
            }
        )

    workload_summary = pd.DataFrame(summary_rows).sort_values(
        ["track", "workload_id"]
    ).reset_index(drop=True)

    if workload_summary["workload_id"].duplicated().any():
        raise ValueError("workload_summary has duplicate workload_id rows — grouping is broken.")

    out_path = output_dir() / "workload_summary.csv"
    workload_summary.to_csv(out_path, index=False)

    # --- Console summary ---
    print("=" * 70)
    print("MEMBER C — WORKLOAD SUMMARY")
    print("=" * 70)

    print(f"\nTotal workloads: {len(workload_summary)}")
    print("\nWorkloads per track:")
    print(workload_summary["track"].value_counts().to_string())

    print("\nWhich strategy wins the most workloads by median EDP:")
    strategy_wins = workload_summary["best_strategy"].value_counts()
    print(strategy_wins.to_string())
    top_count = strategy_wins.iloc[0]
    top_strategies = strategy_wins[strategy_wins == top_count].index.tolist()
    if len(top_strategies) == 1:
        print(f"  -> Winner: '{top_strategies[0]}' ({top_count} workload(s))")
    else:
        print(f"  -> Tie between {top_strategies} ({top_count} workload(s) each)")

    best_overall_idx = edp["median_edp_joule_seconds"].idxmin()
    worst_overall_idx = edp["median_edp_joule_seconds"].idxmax()
    best_overall = edp.loc[best_overall_idx]
    worst_overall = edp.loc[worst_overall_idx]

    print(f"\nOverall BEST median EDP across all {len(edp)} configurations:")
    print(f"  workload_id = {best_overall['workload_id']}")
    print(f"  strategy = {best_overall['strategy']}, n_cores = {best_overall['n_cores']}")
    print(f"  median_edp_joule_seconds = {best_overall['median_edp_joule_seconds']:.4f}")

    print(f"\nOverall WORST median EDP across all {len(edp)} configurations:")
    print(f"  workload_id = {worst_overall['workload_id']}")
    print(f"  strategy = {worst_overall['strategy']}, n_cores = {worst_overall['n_cores']}")
    print(f"  median_edp_joule_seconds = {worst_overall['median_edp_joule_seconds']:.4f}")

    total_pareto = int(edp["is_pareto_optimal"].sum())
    print(f"\nPareto-optimal configurations overall: {total_pareto} / {len(edp)}")

    print(f"\nWorkload summary written to: {out_path}")
    print("=" * 70)
    print("BUILD SUCCEEDED")
    print("=" * 70)


if __name__ == "__main__":
    main()