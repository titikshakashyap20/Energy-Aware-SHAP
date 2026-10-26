"""
scripts/memberC/build_edp_dataset.py

Member C — Infrastructure — Stage: Energy-Delay Product (EDP) dataset.

Reads, read-only:
    results/memberC/unified_benchmark.csv

Writes:
    results/memberC/edp_per_repeat.csv   (intermediate, per-repeat grain)
    results/memberC/edp_dataset.csv      (FINAL — aggregated per configuration)

Why two output files, when the task names "edp_dataset.csv" twice
--------------------------------------------------------------------
The per-repeat EDP values and the aggregated median/mean/std EDP values
are two different grains of data and can't both be "the" edp_dataset.csv
without one silently overwriting the other. Since build_workload_summary.py
(the next stage) needs one row per (track, workload_id, strategy, n_cores)
with median_edp_joule_seconds and a Pareto flag, that aggregated table is
what's written to the authoritative results/memberC/edp_dataset.csv. The
per-repeat table (original columns + edp_joule_seconds, before any
aggregation) is preserved too, as results/memberC/edp_per_repeat.csv, so
the "EDP computed per repeat first" step is fully visible/auditable and
nothing from that step is discarded.

Pipeline
--------
1. Load unified_benchmark.csv.
2. Keep only valid repeats (success == True). EDP is only computed from
   usable runs.
3. edp_joule_seconds = wall_time_sec * energy_joules, computed ROW BY ROW
   (i.e. per repeat) — never from aggregated time/energy.
4. Group by (track, workload_id, strategy, n_cores). For each group:
     - verify repeat_count >= 3, FAIL LOUDLY otherwise
     - verify the workload-identifying columns (model, n_rows, n_features,
       feature_count, background_size) are constant within the group,
       FAIL LOUDLY otherwise (a sign workload_id collided across distinct
       conditions)
     - compute median/mean/std of wall_time_sec, energy_joules,
       edp_joule_seconds
5. Within each workload_id, flag Pareto-optimal configurations using
   median_wall_time_sec and median_energy_joules. Dominance is checked
   ONLY among rows sharing the same workload_id.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

MIN_REPEATS = 3

GROUP_KEYS = ["track", "workload_id", "strategy", "n_cores"]
CONTEXT_COLS = ["model", "n_rows", "n_features", "feature_count", "background_size"]

REQUIRED_INPUT_COLS = GROUP_KEYS + CONTEXT_COLS + [
    "run_id", "repeat_idx", "wall_time_sec", "energy_joules", "success",
]


def find_file(relative_path: str) -> Path:
    for root in REPO_ROOT_CANDIDATES:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required input file not found: '{relative_path}'. "
        f"Looked under: {[str(r) for r in REPO_ROOT_CANDIDATES]}. "
        f"Run this script from the repo root or from scripts/memberC/, "
        f"and make sure build_unified_dataset.py has been run first."
    )


def output_dir() -> Path:
    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out_dir.exists():
        out_dir = Path.cwd() / "results" / "memberC"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def resolve_dominance_flags(group: pd.DataFrame) -> pd.Series:
    """Return a bool Series: True where the row is Pareto-optimal
    (NOT dominated by any other row in `group`), comparing
    median_wall_time_sec and median_energy_joules only.
    """
    times = group["median_wall_time_sec"].to_numpy()
    energies = group["median_energy_joules"].to_numpy()
    n = len(group)
    is_optimal = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            time_le = times[j] <= times[i]
            energy_le = energies[j] <= energies[i]
            strictly_better = (times[j] < times[i]) or (energies[j] < energies[i])
            if time_le and energy_le and strictly_better:
                is_optimal[i] = False
                break

    return pd.Series(is_optimal, index=group.index)


def main() -> None:
    unified_path = find_file("results/memberC/unified_benchmark.csv")
    df = pd.read_csv(unified_path)

    if df.empty:
        raise ValueError(f"'{unified_path}' loaded with zero rows.")

    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"unified_benchmark.csv is missing required column(s) {missing}. "
            f"Columns present: {list(df.columns)}."
        )

    # --- Step 1: keep only valid repeats ---
    df["success"] = df["success"].astype(bool)
    total_rows = len(df)
    df_valid = df[df["success"]].copy()
    invalid_rows = total_rows - len(df_valid)

    if df_valid.empty:
        raise ValueError("No valid (success == True) rows found — cannot compute EDP.")

    # Extra sanity guard beyond `success`, since EDP is meaningless on
    # non-positive time/energy even if success were mis-flagged upstream.
    bad_time = ~pd.to_numeric(df_valid["wall_time_sec"], errors="coerce").gt(0)
    bad_energy = ~pd.to_numeric(df_valid["energy_joules"], errors="coerce").gt(0)
    if (bad_time | bad_energy).any():
        raise ValueError(
            f"{int((bad_time | bad_energy).sum())} row(s) are flagged success=True "
            f"but have non-positive wall_time_sec or energy_joules. Fix upstream "
            f"data before computing EDP."
        )

    # --- Step 2: EDP per repeat, BEFORE any aggregation ---
    df_valid["edp_joule_seconds"] = df_valid["wall_time_sec"] * df_valid["energy_joules"]

    per_repeat_path = output_dir() / "edp_per_repeat.csv"
    df_valid.to_csv(per_repeat_path, index=False)

    # --- Step 3: aggregate per (track, workload_id, strategy, n_cores) ---
    grouped = df_valid.groupby(GROUP_KEYS, dropna=False)

    # Repeat-count check — FAIL LOUDLY if any group has < MIN_REPEATS.
    repeat_counts = grouped["repeat_idx"].count()
    under_min = repeat_counts[repeat_counts < MIN_REPEATS]
    if not under_min.empty:
        raise ValueError(
            f"{len(under_min)} configuration(s) have fewer than {MIN_REPEATS} "
            f"valid repeats:\n{under_min.to_string()}"
        )
    over_max = repeat_counts[repeat_counts > MIN_REPEATS]
    if not over_max.empty:
        print(
            f"WARNING: {len(over_max)} configuration(s) have MORE than "
            f"{MIN_REPEATS} valid repeats (not fatal, just unexpected):\n"
            f"{over_max.to_string()}"
        )

    # Context-column constancy check — FAIL LOUDLY if workload_id collided
    # across rows that disagree on what the workload actually is.
    for col in CONTEXT_COLS:
        nunique_per_group = grouped[col].nunique(dropna=False)
        inconsistent = nunique_per_group[nunique_per_group > 1]
        if not inconsistent.empty:
            raise ValueError(
                f"Column '{col}' is not constant within {len(inconsistent)} "
                f"(track, workload_id, strategy, n_cores) group(s) — "
                f"workload_id does not uniquely determine '{col}'. "
                f"First offending group(s):\n{inconsistent.head().to_string()}"
            )

    agg = grouped.agg(
        median_wall_time_sec=("wall_time_sec", "median"),
        mean_wall_time_sec=("wall_time_sec", "mean"),
        std_wall_time_sec=("wall_time_sec", "std"),
        median_energy_joules=("energy_joules", "median"),
        mean_energy_joules=("energy_joules", "mean"),
        std_energy_joules=("energy_joules", "std"),
        median_edp_joule_seconds=("edp_joule_seconds", "median"),
        mean_edp_joule_seconds=("edp_joule_seconds", "mean"),
        std_edp_joule_seconds=("edp_joule_seconds", "std"),
        repeat_count=("repeat_idx", "count"),
    ).reset_index()

    # Bring back the constant context columns (one value per group).
    context = grouped[CONTEXT_COLS].first().reset_index()
    edp_dataset = context.merge(agg, on=GROUP_KEYS, how="inner")

    # --- Step 4: Pareto-optimality, within workload_id only ---
    edp_dataset["is_pareto_optimal"] = False
    for workload_id, wl_group in edp_dataset.groupby("workload_id", dropna=False):
        flags = resolve_dominance_flags(wl_group)
        edp_dataset.loc[flags.index, "is_pareto_optimal"] = flags.values

    # Final column order: config identity, context, stats, pareto flag.
    column_order = (
        GROUP_KEYS
        + CONTEXT_COLS
        + [
            "median_wall_time_sec", "mean_wall_time_sec", "std_wall_time_sec",
            "median_energy_joules", "mean_energy_joules", "std_energy_joules",
            "median_edp_joule_seconds", "mean_edp_joule_seconds", "std_edp_joule_seconds",
            "repeat_count", "is_pareto_optimal",
        ]
    )
    edp_dataset = edp_dataset[column_order]

    # --- FAIL LOUDLY on anything structurally wrong before writing ---
    if edp_dataset[["median_wall_time_sec", "median_energy_joules", "median_edp_joule_seconds"]].isna().any().any():
        raise ValueError("Aggregated edp_dataset has NaN median value(s) — aggregation is broken.")
    if edp_dataset.duplicated(subset=GROUP_KEYS).any():
        raise ValueError("Aggregated edp_dataset has duplicate (track, workload_id, strategy, n_cores) rows.")

    edp_path = output_dir() / "edp_dataset.csv"
    edp_dataset.to_csv(edp_path, index=False)

    # --- Console summary ---
    print("=" * 70)
    print("MEMBER C — EDP DATASET")
    print("=" * 70)
    print(f"\nTotal rows in unified_benchmark.csv : {total_rows}")
    print(f"Valid repeats used for EDP           : {len(df_valid)}")
    print(f"Invalid/excluded repeats             : {invalid_rows}")
    print(f"\nAggregated configurations             : {len(edp_dataset)}")
    print(f"Unique workloads                      : {edp_dataset['workload_id'].nunique()}")
    print(f"Pareto-optimal configurations          : {int(edp_dataset['is_pareto_optimal'].sum())}")
    print(f"\nPer-repeat EDP table written to : {per_repeat_path}")
    print(f"Aggregated EDP dataset written to: {edp_path}")
    print("\n" + "=" * 70)
    print("BUILD SUCCEEDED")
    print("=" * 70)


if __name__ == "__main__":
    main()