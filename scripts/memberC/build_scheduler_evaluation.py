"""
scripts/memberC/build_scheduler_evaluation.py

Member C — Infrastructure — Stage C2-B: scheduler evaluation protocol.

Reads, read-only:
    results/memberC/edp_dataset.csv

Writes:
    results/memberC/scheduler_evaluation_protocol.csv
    results/memberC/scheduler_splits.csv

NO model is trained here. This script only locks down a leakage-safe
evaluation protocol: the splits, the candidate/oracle bookkeeping per
decision point, and the (not-yet-used) feature/target columns a future
scheduler would be fit and scored against.

Split design
------------
Exhaustive Leave-One-Group-Out (LOGO) over the 18 workloads. A "group"
is a workload_id (track|model|n_rows[|feature_count|background_size]) —
it never includes strategy or n_cores, so every row for a held-out
workload (any strategy, any core count) is excluded from that fold's
training set. With exactly 18 groups, LOGO is exhaustive (18 folds, one
per workload) rather than a random sample of folds, so NO random seed
is needed anywhere in this script — there is no randomness to seed.

Decision points
----------------
A decision point is (held_out_workload_id, n_cores). For each held-out
workload, the n_cores values used are whatever actually appear for that
workload in edp_dataset.csv (not a hardcoded {1,2,4,8}) — this came out
to 4 per workload in the current data, but the code does not assume it.

Candidate strategies and the serial asymmetry
-----------------------------------------------
At every decision point, each of the 5 strategies is checked against
edp_dataset.csv for an actual measured row at that exact
(workload_id, strategy, n_cores). If one exists: baseline_available=True
and the measured_* columns are filled from it. If not (this is exactly
what happens for KernelSHAP serial at n_cores in {2,4,8}, since serial
is single-core and was only ever measured at n_cores=1):
baseline_available=False and measured_* are left NaN. Nothing is
fabricated, copied down from the 1-core measurement, or silently
duplicated across core counts — the exclusion falls out of the data
lookup itself, not a special-cased branch.

Oracle
------
Computed per decision point, using ONLY the strategies with
baseline_available=True at that exact point — the oracle can never be a
strategy that wasn't actually measured there. The oracle is allowed to
use this held-out measurement (it is, by definition, "the best strategy
actually observed for this test workload") — a future scheduler must
NOT see it when predicting, but that constraint lives in how the
feature/target columns are used later, not in this file.

Target (not trained here)
---------------------------
log10_measured_median_edp is pre-computed (NaN where unavailable) as a
convenience column for a future scheduler. log10 is used because
median_edp_joule_seconds spans about 9 orders of magnitude in this
dataset (confirmed in C2-A: from ~1.5 to ~2.4 billion joule-seconds).
On the raw scale, squared-error loss would be dominated entirely by the
largest-workload rows; log10 makes relative EDP errors comparable in
scale across small and huge workloads, and turns the EDP ratios used
throughout C2 (fixed/oracle) into additive differences in log-space.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

ALL_STRATEGIES = ["serial", "multiprocessing", "threading", "joblib", "ray"]
CONTEXT_COLS = ["track", "model", "n_rows", "n_features", "feature_count", "background_size"]

REQUIRED_EDP_COLS = CONTEXT_COLS + [
    "workload_id", "strategy", "n_cores",
    "median_wall_time_sec", "median_energy_joules", "median_edp_joule_seconds",
]


def find_file(relative_path: str) -> Path:
    for root in REPO_ROOT_CANDIDATES:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required input file not found: '{relative_path}'. "
        f"Looked under: {[str(r) for r in REPO_ROOT_CANDIDATES]}. "
        f"Run build_edp_dataset.py first."
    )


def output_dir() -> Path:
    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out_dir.exists():
        out_dir = Path.cwd() / "results" / "memberC"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_edp() -> pd.DataFrame:
    path = find_file("results/memberC/edp_dataset.csv")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"'{path}' loaded with zero rows.")
    missing = [c for c in REQUIRED_EDP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"edp_dataset.csv missing required column(s): {missing}")
    if (df["median_edp_joule_seconds"] <= 0).any():
        raise ValueError("edp_dataset.csv has non-positive median_edp_joule_seconds row(s).")
    if df.duplicated(subset=["workload_id", "strategy", "n_cores"]).any():
        raise ValueError(
            "edp_dataset.csv has duplicate (workload_id, strategy, n_cores) rows — "
            "cannot build an unambiguous evaluation protocol from it."
        )
    return df


def build_workload_context(edp: pd.DataFrame) -> pd.DataFrame:
    """One row per workload_id with its fixed context columns, verified constant."""
    for col in CONTEXT_COLS:
        nunique = edp.groupby("workload_id", dropna=False)[col].nunique(dropna=False)
        bad = nunique[nunique > 1]
        if not bad.empty:
            raise ValueError(
                f"Column '{col}' is not constant within {len(bad)} workload_id "
                f"group(s) in edp_dataset.csv — workload_id does not uniquely "
                f"determine '{col}'. Offending:\n{bad.head().to_string()}"
            )
    context = edp.groupby("workload_id", dropna=False)[CONTEXT_COLS].first().reset_index()
    return context.sort_values("workload_id").reset_index(drop=True)


def build_logo_splits(workload_ids: list[str]) -> pd.DataFrame:
    """Exhaustive Leave-One-Group-Out: one split per workload. No randomness —
    every workload is held out exactly once, deterministically, in sorted order.
    """
    rows = []
    all_set = set(workload_ids)
    for split_id, held_out in enumerate(workload_ids, start=1):
        train_ids = sorted(all_set - {held_out})
        test_ids = [held_out]

        # --- Leakage guard, checked at construction time, not just asserted later ---
        if held_out in train_ids:
            raise ValueError(f"Leakage: held-out workload '{held_out}' found in its own training set.")
        if set(train_ids) | set(test_ids) != all_set:
            raise ValueError(f"Split {split_id}: train+test does not cover all workloads.")
        if set(train_ids) & set(test_ids):
            raise ValueError(f"Split {split_id}: train and test overlap.")

        rows.append({
            "split_id": split_id,
            "held_out_workload_id": held_out,
            "train_workload_count": len(train_ids),
            "test_workload_count": len(test_ids),
            "train_workload_ids": ";".join(train_ids),
            "test_workload_ids": ";".join(test_ids),
        })
    return pd.DataFrame(rows)


def build_evaluation_protocol(edp: pd.DataFrame, context: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    context_by_wl = context.set_index("workload_id")
    rows = []

    for _, split_row in splits.iterrows():
        split_id = split_row["split_id"]
        workload_id = split_row["held_out_workload_id"]
        ctx = context_by_wl.loc[workload_id]

        wl_rows = edp[edp["workload_id"] == workload_id]
        decision_n_cores = sorted(wl_rows["n_cores"].unique())
        if not decision_n_cores:
            raise ValueError(f"Workload '{workload_id}' has no n_cores values in edp_dataset.csv.")

        for n_cores in decision_n_cores:
            point_rows = wl_rows[wl_rows["n_cores"] == n_cores]

            # Build availability + measured values for all 5 candidate strategies.
            measured = {}
            for strategy in ALL_STRATEGIES:
                match = point_rows[point_rows["strategy"] == strategy]
                if len(match) > 1:
                    raise ValueError(
                        f"Impossible/inconsistent data: workload '{workload_id}', "
                        f"n_cores={n_cores}, strategy='{strategy}' has {len(match)} "
                        f"rows in edp_dataset.csv (expected at most 1)."
                    )
                if len(match) == 1:
                    r = match.iloc[0]
                    measured[strategy] = {
                        "available": True,
                        "edp": float(r["median_edp_joule_seconds"]),
                        "time": float(r["median_wall_time_sec"]),
                        "energy": float(r["median_energy_joules"]),
                    }
                else:
                    measured[strategy] = {"available": False, "edp": np.nan, "time": np.nan, "energy": np.nan}

            available_strategies = [s for s in ALL_STRATEGIES if measured[s]["available"]]
            if not available_strategies:
                raise ValueError(
                    f"Impossible decision point: workload '{workload_id}', n_cores={n_cores} "
                    f"has NO available candidate strategies at all."
                )

            oracle_strategy = min(available_strategies, key=lambda s: measured[s]["edp"])
            oracle_edp = measured[oracle_strategy]["edp"]
            oracle_time = measured[oracle_strategy]["time"]
            oracle_energy = measured[oracle_strategy]["energy"]

            for strategy in ALL_STRATEGIES:
                m = measured[strategy]
                is_oracle = m["available"] and (strategy == oracle_strategy)
                log10_edp = np.log10(m["edp"]) if m["available"] else np.nan

                rows.append({
                    "split_id": split_id,
                    "held_out_workload_id": workload_id,
                    "track": ctx["track"],
                    "model": ctx["model"],
                    "n_rows": ctx["n_rows"],
                    "n_features": ctx["n_features"],
                    "feature_count": ctx["feature_count"],
                    "background_size": ctx["background_size"],
                    "n_cores": n_cores,
                    "candidate_strategy": strategy,
                    "measured_median_edp": m["edp"],
                    "measured_median_time": m["time"],
                    "measured_median_energy": m["energy"],
                    "log10_measured_median_edp": log10_edp,
                    "oracle_strategy": oracle_strategy,
                    "oracle_edp": oracle_edp,
                    "oracle_time": oracle_time,
                    "oracle_energy": oracle_energy,
                    "is_oracle": is_oracle,
                    "baseline_available": m["available"],
                })

    protocol = pd.DataFrame(rows)

    # --- Fail loudly on structural inconsistencies ---
    available = protocol[protocol["baseline_available"]]
    if (available["measured_median_edp"] <= 0).any():
        raise ValueError("Protocol has available row(s) with non-positive measured_median_edp.")
    is_oracle_count = protocol.groupby(["split_id", "n_cores"])["is_oracle"].sum()
    if (is_oracle_count != 1).any():
        bad = is_oracle_count[is_oracle_count != 1]
        raise ValueError(
            f"Each decision point must have exactly one oracle row; found violations:\n"
            f"{bad.to_string()}"
        )

    return protocol


def main() -> None:
    edp = load_edp()
    context = build_workload_context(edp)
    workload_ids = context["workload_id"].tolist()

    if len(workload_ids) != 18:
        print(f"NOTE: found {len(workload_ids)} workloads, not the previously-confirmed 18 "
              f"— proceeding anyway since this script derives everything from the data.")

    splits = build_logo_splits(workload_ids)
    protocol = build_evaluation_protocol(edp, context, splits)

    out_dir = output_dir()
    splits_path = out_dir / "scheduler_splits.csv"
    protocol_path = out_dir / "scheduler_evaluation_protocol.csv"
    splits.to_csv(splits_path, index=False)
    protocol.to_csv(protocol_path, index=False)

    # --- Console summary (exactly what was requested) ---
    print("=" * 70)
    print("MEMBER C — SCHEDULER EVALUATION PROTOCOL (C2-B)")
    print("=" * 70)

    n_decision_points = protocol.groupby(["split_id", "n_cores"]).ngroups
    print(f"\nWorkload groups                 : {len(workload_ids)}")
    print(f"Split folds (exhaustive LOGO)    : {len(splits)}")
    print(f"Decision points (workload x cores): {n_decision_points}")
    print(f"Total protocol rows (decision points x 5 candidates): {len(protocol)}")

    print("\nCandidate strategy counts (availability across all decision points):")
    counts = protocol.groupby("candidate_strategy")["baseline_available"].agg(["sum", "count"])
    counts.columns = ["available", "total_decision_points"]
    counts["unavailable"] = counts["total_decision_points"] - counts["available"]
    print(counts.to_string())

    n_unavailable = int((~protocol["baseline_available"]).sum())
    print(f"\nTotal unavailable baseline decisions (across all candidates): {n_unavailable}")

    # --- Explicit leakage confirmation ---
    leakage_found = False
    for _, row in splits.iterrows():
        train_set = set(row["train_workload_ids"].split(";"))
        held_out = row["held_out_workload_id"]
        if held_out in train_set:
            leakage_found = True
            print(f"LEAKAGE in split {row['split_id']}: '{held_out}' found in training set!")
    if not leakage_found:
        print(f"\nConfirmed: no held-out workload appears in its corresponding training set "
              f"(checked all {len(splits)} splits).")

    print(f"\nscheduler_splits.csv written to            : {splits_path}")
    print(f"scheduler_evaluation_protocol.csv written to: {protocol_path}")
    print("\n" + "=" * 70)
    print("PROTOCOL BUILD SUCCEEDED — no model was trained in this step.")
    print("=" * 70)


if __name__ == "__main__":
    main()