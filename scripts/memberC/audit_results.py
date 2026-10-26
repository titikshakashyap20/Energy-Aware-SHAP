"""
scripts/memberC/audit_results.py

Member C — Infrastructure — Stage: Audit the raw Member A / Member B results.

Reads, in place, with NO modification:
    results/memberA/results.csv                (TreeSHAP track, 360 rows)
    results/memberB/kernelshap_benchmark.csv   (KernelSHAP track, 612 rows)

Validates each track's raw CSV for:
    - presence of the file and its required columns
    - row count
    - duplicate run_id values
    - duplicate logical rows (same configuration + repeat_idx appearing
      more than once)
    - missing/null values in required fields
    - invalid wall-clock time or energy values (<= 0 or non-numeric)
    - failed runs (explicit `success` column for Member B; inferred for
      Member A, which has no success column, from valid timing + energy)
    - unique configuration counts and unique workload counts
    - repeat-count consistency (every configuration should have exactly
      3 repeats; anything else is reported, not silently accepted)

Writes:
    results/memberC/validation_summary.csv

FAILS LOUDLY (raises, non-zero exit) if a raw file is missing, is empty,
or is missing a column this script depends on to do its job. Data-quality
issues found *within* an otherwise well-formed file (duplicates, nulls,
invalid values, failed runs, repeat-count mismatches) are not fatal — the
whole point of this script is to surface them in the report — but every
one of them is counted and printed so nothing is silently swallowed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution — works whether this script is run from the repo root or
# from inside scripts/memberC/.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

EXPECTED_REPEATS = 3


def find_file(relative_path: str) -> Path:
    """Locate a results file under one of the candidate repo roots.

    FAILS LOUDLY (raises FileNotFoundError) if it cannot be found anywhere
    — a missing raw file means nothing downstream can be trusted.
    """
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
    """FAIL LOUDLY if any required column is missing from df."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required column(s) {missing}. "
            f"Columns present: {list(df.columns)}. "
            f"The raw schema may have changed — stopping rather than guessing."
        )


def load_raw(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label}: raw file '{path}' loaded with zero rows.")
    return df


# ---------------------------------------------------------------------------
# Per-track audit
# ---------------------------------------------------------------------------
def audit_member_a(df: pd.DataFrame) -> dict:
    label = "Member A (tree)"
    required = [
        "run_id", "track", "strategy", "model", "n_rows", "n_features",
        "n_cores", "repeat_idx", "wall_time_sec", "energy_joules",
        "machine_id", "timestamp",
    ]
    require_columns(df, required, label)

    duplicate_run_ids = int(df["run_id"].duplicated().sum())

    config_cols = ["model", "n_rows", "strategy", "n_cores"]
    workload_cols = ["model", "n_rows"]

    duplicate_logical_rows = int(
        df.duplicated(subset=config_cols + ["repeat_idx"]).sum()
    )

    missing_required = int(df[required].isna().sum().sum())

    invalid_time = int(
        (~pd.to_numeric(df["wall_time_sec"], errors="coerce").gt(0)).sum()
    )
    invalid_energy = int(
        (~pd.to_numeric(df["energy_joules"], errors="coerce").gt(0)).sum()
    )

    # No explicit success column in the Member A schema — infer it from
    # valid (finite, positive) timing and energy, per the task instructions.
    inferred_success = (
        pd.to_numeric(df["wall_time_sec"], errors="coerce").gt(0)
        & pd.to_numeric(df["energy_joules"], errors="coerce").gt(0)
    )
    failed_runs = int((~inferred_success).sum())

    n_unique_configs = df.groupby(config_cols, dropna=False).ngroups
    n_unique_workloads = df.groupby(workload_cols, dropna=False).ngroups

    repeat_counts = df.groupby(config_cols, dropna=False)["repeat_idx"].nunique()
    configs_wrong_repeat_count = int((repeat_counts != EXPECTED_REPEATS).sum())

    return {
        "track": "memberA_tree",
        "total_rows": len(df),
        "duplicate_run_ids": duplicate_run_ids,
        "duplicate_logical_rows": duplicate_logical_rows,
        "missing_required_field_values": missing_required,
        "invalid_time_rows": invalid_time,
        "invalid_energy_rows": invalid_energy,
        "failed_or_invalid_runs": failed_runs,
        "unique_configurations": n_unique_configs,
        "unique_workloads": n_unique_workloads,
        "expected_repeats": EXPECTED_REPEATS,
        "configs_with_wrong_repeat_count": configs_wrong_repeat_count,
        "rows_passing_all_checks": len(df) - failed_runs - duplicate_logical_rows,
    }


def audit_member_b(df: pd.DataFrame) -> dict:
    label = "Member B (kernelshap)"
    required = [
        "run_id", "timestamp", "track", "strategy", "model", "feature_count",
        "n_cores", "n_rows", "background_size", "repeat_idx",
        "elapsed_seconds", "energy_joules", "success",
    ]
    require_columns(df, required, label)

    duplicate_run_ids = int(df["run_id"].duplicated().sum())

    config_cols = ["model", "feature_count", "n_rows", "background_size", "strategy", "n_cores"]
    workload_cols = ["model", "feature_count", "n_rows", "background_size"]

    duplicate_logical_rows = int(
        df.duplicated(subset=config_cols + ["repeat_idx"]).sum()
    )

    missing_required = int(df[required].isna().sum().sum())

    invalid_time = int(
        (~pd.to_numeric(df["elapsed_seconds"], errors="coerce").gt(0)).sum()
    )
    invalid_energy = int(
        (~pd.to_numeric(df["energy_joules"], errors="coerce").gt(0)).sum()
    )

    # Member B has an explicit success column — use it directly, do not
    # re-derive it.
    success_bool = df["success"].astype(bool)
    failed_runs = int((~success_bool).sum())

    # additivity_ok is KernelSHAP-specific (Shapley additivity property);
    # report it if present, but it is not required for the file to be usable.
    additivity_failures = (
        int((~df["additivity_ok"].astype(bool)).sum())
        if "additivity_ok" in df.columns
        else None
    )

    n_unique_configs = df.groupby(config_cols, dropna=False).ngroups
    n_unique_workloads = df.groupby(workload_cols, dropna=False).ngroups

    repeat_counts = df.groupby(config_cols, dropna=False)["repeat_idx"].nunique()
    configs_wrong_repeat_count = int((repeat_counts != EXPECTED_REPEATS).sum())

    return {
        "track": "memberB_kernelshap",
        "total_rows": len(df),
        "duplicate_run_ids": duplicate_run_ids,
        "duplicate_logical_rows": duplicate_logical_rows,
        "missing_required_field_values": missing_required,
        "invalid_time_rows": invalid_time,
        "invalid_energy_rows": invalid_energy,
        "failed_or_invalid_runs": failed_runs,
        "unique_configurations": n_unique_configs,
        "unique_workloads": n_unique_workloads,
        "expected_repeats": EXPECTED_REPEATS,
        "configs_with_wrong_repeat_count": configs_wrong_repeat_count,
        "rows_passing_all_checks": len(df) - failed_runs - duplicate_logical_rows,
        "additivity_check_failures": additivity_failures,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    a_path = find_file("results/memberA/results.csv")
    b_path = find_file("results/memberB/kernelshap_benchmark.csv")

    df_a = load_raw(a_path, "Member A")
    df_b = load_raw(b_path, "Member B")

    report_a = audit_member_a(df_a)
    report_b = audit_member_b(df_b)

    summary_df = pd.DataFrame([report_a, report_b])

    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out_dir.exists():
        # Mirror the candidate-root logic used for inputs, in case this is
        # being run from somewhere other than the repo root.
        out_dir = Path.cwd() / "results" / "memberC"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_summary.csv"
    summary_df.to_csv(out_path, index=False)

    # --- Console summary ---
    print("=" * 70)
    print("MEMBER C — RAW RESULTS AUDIT")
    print("=" * 70)
    for report in (report_a, report_b):
        print(f"\n--- {report['track']} ---")
        for key, value in report.items():
            if key == "track":
                continue
            print(f"  {key:32s}: {value}")

    print(f"\nValidation summary written to: {out_path}")

    # --- Fail loudly on structural inconsistencies worth stopping for ---
    # (Duplicate run_ids and duplicate logical rows mean the raw data is
    # internally inconsistent in a way that would corrupt everything built
    # on top of it — that is worth a hard stop, not just a warning line.)
    hard_failures = []
    if report_a["duplicate_run_ids"] > 0:
        hard_failures.append(f"Member A has {report_a['duplicate_run_ids']} duplicate run_id(s).")
    if report_b["duplicate_run_ids"] > 0:
        hard_failures.append(f"Member B has {report_b['duplicate_run_ids']} duplicate run_id(s).")
    if report_a["duplicate_logical_rows"] > 0:
        hard_failures.append(
            f"Member A has {report_a['duplicate_logical_rows']} duplicate "
            f"(config, repeat_idx) row(s)."
        )
    if report_b["duplicate_logical_rows"] > 0:
        hard_failures.append(
            f"Member B has {report_b['duplicate_logical_rows']} duplicate "
            f"(config, repeat_idx) row(s)."
        )

    if hard_failures:
        print("\n" + "!" * 70)
        print("AUDIT FAILED — internally inconsistent raw data:")
        for msg in hard_failures:
            print(f"  - {msg}")
        print("!" * 70)
        sys.exit(1)

    print("\nAUDIT PASSED — no duplicate run_ids or duplicate logical rows found.")
    if report_a["failed_or_invalid_runs"] or report_b["failed_or_invalid_runs"]:
        print(
            "Note: some rows are flagged failed/invalid (see counts above) — "
            "this is reported, not treated as fatal, since partial failure "
            "rates are expected information, not a sign of a broken pipeline."
        )


if __name__ == "__main__":
    main()