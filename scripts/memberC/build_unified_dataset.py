"""
scripts/memberC/build_unified_dataset.py

Member C — Infrastructure — Stage: Build the unified benchmark dataset.

Reads, in place, with NO modification:
    results/memberA/results.csv                (TreeSHAP track, 360 rows)
    results/memberB/kernelshap_benchmark.csv   (KernelSHAP track, 612 rows)

Normalizes both into ONE canonical CSV, one row per original run — no
aggregation, no row dropping beyond what each source already contains.

Canonical schema (per the agreed spec), plus one addition explained below:

    run_id, timestamp, track, strategy, model, n_rows, n_features,
    feature_count, background_size, n_cores, cores_effective, repeat_idx,
    wall_time_sec, energy_kwh, energy_joules, machine_id, cpu_count,
    success, additivity_ok, max_additivity_error, shap_version,
    codecarbon_version, workload_id

`workload_id` is appended at the end. It is not optional — Member C's
later scheduler/EDP-analysis work needs it to group runs by "the work
being explained" independent of strategy/cores — so this script creates
it directly rather than leaving it for a later file:

    tree:       "tree|{model}|{n_rows}"
    kernelshap: "kernelshap|{model}|{feature_count}|{n_rows}|{background_size}"

Per-track mapping notes
------------------------
Member A (tree) -> canonical:
    wall_time_sec   <- wall_time_sec (raw)
    energy_joules   <- energy_joules (raw)
    energy_kwh      <- energy_joules / 3_600_000  (DERIVED; Member A's raw
                        file has no energy_kwh column, so this is computed
                        here for cross-track comparability — it is not a
                        raw value and is clearly a pure unit conversion,
                        not an estimate)
    n_features      <- n_features (raw, always 105 — post-encoding column
                        count; tree track never varies feature count as an
                        experimental factor)
    feature_count   <- left blank (not an experimental condition for tree)
    cores_effective <- n_cores (Member A's raw schema has no separate
                        "effective cores" field; n_cores is used directly)
    cpu_count       <- left blank (not logged in Member A's raw schema)
    success         <- INFERRED: wall_time_sec > 0 and energy_joules > 0
                        (Member A's raw schema has no success column, per
                        the task instructions)
    additivity_ok, max_additivity_error, shap_version, codecarbon_version
                    <- left blank (not applicable / not logged for tree)

Member B (kernelshap) -> canonical:
    wall_time_sec   <- elapsed_seconds (raw; Member B's raw column name)
    energy_kwh, energy_joules <- energy_kwh, energy_joules (raw)
    success         <- success (raw, preserved AS GIVEN — never re-derived)
    additivity_ok, max_additivity_error, shap_version, codecarbon_version
                    <- preserved as given
    feature_count   <- feature_count (raw: 15 or 30)
    n_features      <- MIRRORS feature_count. This is the one deliberate
                        duplication in the schema: n_features is meant to
                        be a track-agnostic "how many features did the
                        model see" column usable for cross-track analysis,
                        while feature_count stays as Member B's own native
                        field for provenance. Nothing is discarded by this
                        choice — both columns are populated.
    cores_effective <- cores_effective (raw; verified equal to n_cores for
                        every row in the current data, but the raw column
                        is preserved rather than assumed)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

JOULES_PER_KWH = 3_600_000

CANONICAL_COLUMNS = [
    "run_id", "timestamp", "track", "strategy", "model",
    "n_rows", "n_features", "feature_count", "background_size",
    "n_cores", "cores_effective", "repeat_idx",
    "wall_time_sec", "energy_kwh", "energy_joules",
    "machine_id", "cpu_count",
    "success", "additivity_ok", "max_additivity_error",
    "shap_version", "codecarbon_version",
    "workload_id",
]


def find_file(relative_path: str) -> Path:
    for root in REPO_ROOT_CANDIDATES:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required raw results file not found: '{relative_path}'. "
        f"Looked under: {[str(r) for r in REPO_ROOT_CANDIDATES]}. "
        f"Run this script from the repo root or from scripts/memberC/."
    )


def require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required column(s) {missing}. "
            f"Columns present: {list(df.columns)}. Stopping rather than "
            f"silently building a unified file with the wrong schema."
        )


def load_raw(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label}: raw file '{path}' loaded with zero rows.")
    return df


# ---------------------------------------------------------------------------
# Per-track normalization
# ---------------------------------------------------------------------------
def normalize_member_a(df: pd.DataFrame) -> pd.DataFrame:
    label = "Member A (tree)"
    required = [
        "run_id", "track", "strategy", "model", "n_rows", "n_features",
        "n_cores", "repeat_idx", "wall_time_sec", "energy_joules",
        "machine_id", "timestamp",
    ]
    require_columns(df, required, label)

    out = pd.DataFrame()
    out["run_id"] = df["run_id"]
    out["timestamp"] = df["timestamp"]
    out["track"] = df["track"]  # "tree"
    out["strategy"] = df["strategy"]
    out["model"] = df["model"]
    out["n_rows"] = df["n_rows"]
    out["n_features"] = df["n_features"]
    out["feature_count"] = pd.NA  # not an experimental factor for tree
    out["background_size"] = df.get("background_size", pd.NA)
    out["n_cores"] = df["n_cores"]
    out["cores_effective"] = df["n_cores"]  # no separate field in raw A
    out["repeat_idx"] = df["repeat_idx"]
    out["wall_time_sec"] = df["wall_time_sec"]
    out["energy_joules"] = df["energy_joules"]
    out["energy_kwh"] = df["energy_joules"] / JOULES_PER_KWH  # derived, see module docstring
    out["machine_id"] = df["machine_id"]
    out["cpu_count"] = pd.NA  # not logged in raw A

    time_ok = pd.to_numeric(df["wall_time_sec"], errors="coerce").gt(0)
    energy_ok = pd.to_numeric(df["energy_joules"], errors="coerce").gt(0)
    out["success"] = (time_ok & energy_ok).astype(bool)  # inferred, no raw success column

    out["additivity_ok"] = pd.NA
    out["max_additivity_error"] = pd.NA
    out["shap_version"] = pd.NA
    out["codecarbon_version"] = pd.NA

    out["workload_id"] = (
        "tree|" + out["model"].astype(str) + "|" + out["n_rows"].astype(str)
    )

    return out[CANONICAL_COLUMNS]


def normalize_member_b(df: pd.DataFrame) -> pd.DataFrame:
    label = "Member B (kernelshap)"
    required = [
        "run_id", "timestamp", "track", "strategy", "model", "feature_count",
        "n_cores", "n_rows", "background_size", "repeat_idx",
        "elapsed_seconds", "energy_kwh", "energy_joules", "success",
        "machine_id", "cpu_count",
    ]
    require_columns(df, required, label)

    out = pd.DataFrame()
    out["run_id"] = df["run_id"]
    out["timestamp"] = df["timestamp"]
    out["track"] = df["track"]  # "kernelshap"
    out["strategy"] = df["strategy"]
    out["model"] = df["model"]
    out["n_rows"] = df["n_rows"]
    out["n_features"] = df["feature_count"]  # mirrored, see module docstring
    out["feature_count"] = df["feature_count"]  # native field, preserved
    out["background_size"] = df["background_size"]
    out["n_cores"] = df["n_cores"]
    out["cores_effective"] = df.get("cores_effective", df["n_cores"])
    out["repeat_idx"] = df["repeat_idx"]
    out["wall_time_sec"] = df["elapsed_seconds"]  # per mapping instructions
    out["energy_kwh"] = df["energy_kwh"]
    out["energy_joules"] = df["energy_joules"]
    out["machine_id"] = df["machine_id"]
    out["cpu_count"] = df["cpu_count"]

    out["success"] = df["success"].astype(bool)  # preserved as given, not re-derived

    out["additivity_ok"] = df.get("additivity_ok", pd.NA)
    out["max_additivity_error"] = df.get("max_additivity_error", pd.NA)
    out["shap_version"] = df.get("shap_version", pd.NA)
    out["codecarbon_version"] = df.get("codecarbon_version", pd.NA)

    out["workload_id"] = (
        "kernelshap|"
        + out["model"].astype(str) + "|"
        + out["feature_count"].astype(str) + "|"
        + out["n_rows"].astype(str) + "|"
        + out["background_size"].astype(str)
    )

    return out[CANONICAL_COLUMNS]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    a_path = find_file("results/memberA/results.csv")
    b_path = find_file("results/memberB/kernelshap_benchmark.csv")

    df_a_raw = load_raw(a_path, "Member A")
    df_b_raw = load_raw(b_path, "Member B")

    df_a = normalize_member_a(df_a_raw)
    df_b = normalize_member_b(df_b_raw)

    unified = pd.concat([df_a, df_b], ignore_index=True)

    # --- FAIL LOUDLY on anything that would silently corrupt downstream work ---
    if unified["run_id"].isna().any():
        raise ValueError("Unified dataset has row(s) with a missing run_id.")
    if unified["run_id"].duplicated().any():
        dupes = unified.loc[unified["run_id"].duplicated(), "run_id"].tolist()
        raise ValueError(f"Unified dataset has duplicate run_id(s): {dupes[:10]}")
    if unified["track"].isna().any() or unified["model"].isna().any():
        raise ValueError("Unified dataset has row(s) with missing track or model.")
    if unified["wall_time_sec"].isna().any() or unified["energy_joules"].isna().any():
        raise ValueError(
            "Unified dataset has row(s) with missing wall_time_sec or "
            "energy_joules after normalization — mapping logic is wrong."
        )
    if not set(unified["track"].unique()) <= {"tree", "kernelshap"}:
        raise ValueError(f"Unexpected track value(s) found: {unified['track'].unique()}")

    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out_dir.exists():
        out_dir = Path.cwd() / "results" / "memberC"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "unified_benchmark.csv"
    unified.to_csv(out_path, index=False)

    # --- Console summary ---
    print("=" * 70)
    print("MEMBER C — UNIFIED BENCHMARK DATASET")
    print("=" * 70)

    print(f"\nTotal rows             : {len(unified)}")
    print("Rows per track:")
    print(unified["track"].value_counts().to_string())

    tree = unified[unified["track"] == "tree"]
    kernel = unified[unified["track"] == "kernelshap"]

    tree_configs = tree.groupby(["model", "n_rows", "strategy", "n_cores"], dropna=False).ngroups
    kernel_configs = kernel.groupby(
        ["model", "feature_count", "n_rows", "background_size", "strategy", "n_cores"], dropna=False
    ).ngroups
    print(f"\nUnique configurations — tree       : {tree_configs}")
    print(f"Unique configurations — kernelshap : {kernel_configs}")

    print(f"\nUnique workloads — tree (model x n_rows)                         : "
          f"{tree['workload_id'].nunique()}")
    print(f"Unique workloads — kernelshap (model x feature_count x n_rows x bg) : "
          f"{kernel['workload_id'].nunique()}")
    print(f"Unique workloads — total                                           : "
          f"{unified['workload_id'].nunique()}")

    failed_invalid = int((~unified["success"]).sum())
    print(f"\nFailed/invalid rows (success == False) : {failed_invalid}")

    duplicate_run_ids = int(unified["run_id"].duplicated().sum())
    print(f"Duplicate run_ids in unified dataset   : {duplicate_run_ids}")

    print(f"\nUnified dataset written to: {out_path}")
    print("=" * 70)
    print("BUILD SUCCEEDED")
    print("=" * 70)


if __name__ == "__main__":
    main()