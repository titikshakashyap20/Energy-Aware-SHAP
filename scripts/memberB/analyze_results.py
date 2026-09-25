"""
Member B — Result Analysis
==========================

Post-processes the completed KernelSHAP benchmark.

Input:
    results/memberB/kernelshap_benchmark.csv

Output:
    results/memberB/analysis/

Responsibilities:
    1. Validate raw benchmark results.
    2. Aggregate repeated runs.
    3. Produce strategy/model/feature/background/core summaries.
    4. Produce fastest and lowest-energy configurations.
    5. Produce time-energy tradeoff data.
    6. Produce repeat-consistency statistics.

This script DOES NOT rerun KernelSHAP.
It only analyzes the existing benchmark CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

INPUT_FILE = (
    REPO_ROOT
    / "results"
    / "memberB"
    / "kernelshap_benchmark.csv"
)

OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "memberB"
    / "analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONSTANTS
# ============================================================================

TRACK = "kernelshap"

KEY_COLUMNS = [
    "strategy",
    "model",
    "feature_count",
    "n_cores",
    "n_rows",
    "background_size",
    "repeat_idx",
]

REQUIRED_COLUMNS = [
    "run_id",
    "track",
    "strategy",
    "model",
    "feature_count",
    "n_cores",
    "n_rows",
    "background_size",
    "repeat_idx",
    "elapsed_seconds",
    "energy_joules",
    "success",
]


# ============================================================================
# HELPERS
# ============================================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def mean_std(series):
    """Return mean and sample standard deviation."""
    values = pd.to_numeric(series, errors="coerce").dropna()

    if len(values) == 0:
        return np.nan, np.nan

    mean = values.mean()

    if len(values) == 1:
        std = 0.0
    else:
        std = values.std(ddof=1)

    return mean, std


def save_csv(df, filename):
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


# ============================================================================
# LOAD + VALIDATE
# ============================================================================

def load_results():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input file : {INPUT_FILE}")
    print(f"Raw rows   : {len(df)}")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    return df


def validate_results(df):
    section("Validation")

    duplicate_run_ids = df["run_id"].duplicated().sum()

    # Only successful runs participate in configuration uniqueness.
    successful = df[df["success"] == True].copy()

    duplicate_configs = (
        successful.duplicated(subset=KEY_COLUMNS).sum()
    )

    missing_elapsed = (
        successful["elapsed_seconds"].isna().sum()
    )

    missing_energy = (
        successful["energy_joules"].isna().sum()
    )

    failed_runs = (
        (df["success"] != True).sum()
    )

    print(f"Successful runs       : {len(successful)}")
    print(f"Failed runs           : {failed_runs}")
    print(f"Duplicate run IDs     : {duplicate_run_ids}")
    print(f"Duplicate configurations: {duplicate_configs}")
    print(f"Missing elapsed       : {missing_elapsed}")
    print(f"Missing energy        : {missing_energy}")

    validation = pd.DataFrame(
        [
            {
                "raw_rows": len(df),
                "successful_runs": len(successful),
                "failed_runs": failed_runs,
                "duplicate_run_ids": duplicate_run_ids,
                "duplicate_configurations": duplicate_configs,
                "missing_elapsed": missing_elapsed,
                "missing_energy": missing_energy,
            }
        ]
    )

    save_csv(validation, "validation_summary.csv")

    return successful


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def configuration_summary(df):
    grouped = (
        df.groupby(KEY_COLUMNS, dropna=False)
        .agg(
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            std_elapsed_seconds=(
                "elapsed_seconds",
                "std",
            ),
            mean_energy_joules=(
                "energy_joules",
                "mean",
            ),
            std_energy_joules=(
                "energy_joules",
                "std",
            ),
            runs=("run_id", "count"),
        )
        .reset_index()
    )

    grouped["std_elapsed_seconds"] = (
        grouped["std_elapsed_seconds"]
        .fillna(0)
    )

    grouped["std_energy_joules"] = (
        grouped["std_energy_joules"]
        .fillna(0)
    )

    grouped["mean_elapsed_minutes"] = (
        grouped["mean_elapsed_seconds"] / 60.0
    )

    grouped["mean_energy_kwh"] = (
        grouped["mean_energy_joules"] / 3_600_000.0
    )

    grouped["energy_delay_product"] = (
        grouped["mean_elapsed_seconds"]
        * grouped["mean_energy_joules"]
    )

    grouped = grouped.sort_values(
        [
            "model",
            "feature_count",
            "n_rows",
            "background_size",
            "n_cores",
            "strategy",
        ]
    )

    save_csv(
        grouped,
        "configuration_summary.csv",
    )

    return grouped


# ============================================================================
# FASTEST CONFIGURATIONS
# ============================================================================

def fastest_configurations(config):
    section("FASTEST CONFIGURATIONS")

    result = (
        config.sort_values("mean_elapsed_seconds")
        .head(20)
        .copy()
    )

    print(
        result[
            [
                "strategy",
                "model",
                "feature_count",
                "n_rows",
                "background_size",
                "n_cores",
                "mean_elapsed_seconds",
                "mean_elapsed_minutes",
                "mean_energy_joules",
            ]
        ].to_string(index=False)
    )

    save_csv(
        result,
        "fastest_configurations.csv",
    )

    return result


# ============================================================================
# LOWEST ENERGY CONFIGURATIONS
# ============================================================================

def lowest_energy_configurations(config):
    section("LOWEST-ENERGY CONFIGURATIONS")

    result = (
        config.sort_values("mean_energy_joules")
        .head(20)
        .copy()
    )

    print(
        result[
            [
                "strategy",
                "model",
                "feature_count",
                "n_rows",
                "background_size",
                "n_cores",
                "mean_elapsed_seconds",
                "mean_energy_joules",
            ]
        ].to_string(index=False)
    )

    save_csv(
        result,
        "lowest_energy_configurations.csv",
    )

    return result


# ============================================================================
# STRATEGY SUMMARY
# ============================================================================

def strategy_summary(df):
    section("STRATEGY SUMMARY")

    result = (
        df.groupby("strategy")
        .agg(
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            std_elapsed_seconds=(
                "elapsed_seconds",
                "std",
            ),
            mean_energy_kwh=(
                "energy_kwh",
                "mean",
            ),
            std_energy_kwh=(
                "energy_kwh",
                "std",
            ),
            mean_energy_joules=(
                "energy_joules",
                "mean",
            ),
            std_energy_joules=(
                "energy_joules",
                "std",
            ),
            runs=("run_id", "count"),
        )
        .reset_index()
    )

    result["std_elapsed_seconds"] = (
        result["std_elapsed_seconds"]
        .fillna(0)
    )

    result["std_energy_kwh"] = (
        result["std_energy_kwh"]
        .fillna(0)
    )

    result["std_energy_joules"] = (
        result["std_energy_joules"]
        .fillna(0)
    )

    result["mean_elapsed_minutes"] = (
        result["mean_elapsed_seconds"] / 60.0
    )

    result = result.sort_values(
        "mean_elapsed_seconds"
    )

    print(result.to_string(index=False))

    save_csv(
        result,
        "strategy_summary.csv",
    )

    return result


# ============================================================================
# MODEL SUMMARY
# ============================================================================

def model_summary(df):
    section("MODEL SUMMARY")

    result = (
        df.groupby("model")
        .agg(
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            std_elapsed_seconds=(
                "elapsed_seconds",
                "std",
            ),
            mean_energy_kwh=(
                "energy_kwh",
                "mean",
            ),
            std_energy_kwh=(
                "energy_kwh",
                "std",
            ),
            mean_energy_joules=(
                "energy_joules",
                "mean",
            ),
            std_energy_joules=(
                "energy_joules",
                "std",
            ),
            runs=("run_id", "count"),
        )
        .reset_index()
    )

    result["std_elapsed_seconds"] = (
        result["std_elapsed_seconds"].fillna(0)
    )

    result["std_energy_kwh"] = (
        result["std_energy_kwh"].fillna(0)
    )

    result["std_energy_joules"] = (
        result["std_energy_joules"].fillna(0)
    )

    result["mean_elapsed_minutes"] = (
        result["mean_elapsed_seconds"] / 60.0
    )

    print(result.to_string(index=False))

    save_csv(result, "model_summary.csv")

    return result


# ============================================================================
# FEATURE COUNT SUMMARY
# ============================================================================

def feature_summary(df):
    section("FEATURE COUNT SUMMARY")

    result = (
        df.groupby("feature_count")
        .agg(
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            std_elapsed_seconds=(
                "elapsed_seconds",
                "std",
            ),
            mean_energy_kwh=(
                "energy_kwh",
                "mean",
            ),
            std_energy_kwh=(
                "energy_kwh",
                "std",
            ),
            mean_energy_joules=(
                "energy_joules",
                "mean",
            ),
            std_energy_joules=(
                "energy_joules",
                "std",
            ),
            runs=("run_id", "count"),
        )
        .reset_index()
    )

    for column in [
        "std_elapsed_seconds",
        "std_energy_kwh",
        "std_energy_joules",
    ]:
        result[column] = result[column].fillna(0)

    result["mean_elapsed_minutes"] = (
        result["mean_elapsed_seconds"] / 60.0
    )

    print(result.to_string(index=False))

    save_csv(result, "feature_count_summary.csv")

    # Keep an additional name consistent with the broader
    # Member A-style analysis outputs.
    save_csv(
        result,
        "feature_summary.csv",
    )

    return result


# ============================================================================
# BACKGROUND SUMMARY
# ============================================================================

def background_summary(df):
    section("BACKGROUND SIZE SUMMARY")

    result = (
        df.groupby("background_size")
        .agg(
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            std_elapsed_seconds=(
                "elapsed_seconds",
                "std",
            ),
            mean_energy_kwh=(
                "energy_kwh",
                "mean",
            ),
            std_energy_kwh=(
                "energy_kwh",
                "std",
            ),
            mean_energy_joules=(
                "energy_joules",
                "mean",
            ),
            std_energy_joules=(
                "energy_joules",
                "std",
            ),
            runs=("run_id", "count"),
        )
        .reset_index()
    )

    for column in [
        "std_elapsed_seconds",
        "std_energy_kwh",
        "std_energy_joules",
    ]:
        result[column] = result[column].fillna(0)

    result["mean_elapsed_minutes"] = (
        result["mean_elapsed_seconds"] / 60.0
    )

    print(result.to_string(index=False))

    save_csv(
        result,
        "background_size_summary.csv",
    )

    save_csv(
        result,
        "background_summary.csv",
    )

    return result


# ============================================================================
# ROW SUMMARY
# ============================================================================

def row_size_summary(df):
    section("ROW SIZE SUMMARY")

    result = (
        df.groupby("n_rows")
        .agg(
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            std_elapsed_seconds=(
                "elapsed_seconds",
                "std",
            ),
            mean_energy_kwh=(
                "energy_kwh",
                "mean",
            ),
            std_energy_kwh=(
                "energy_kwh",
                "std",
            ),
            mean_energy_joules=(
                "energy_joules",
                "mean",
            ),
            std_energy_joules=(
                "energy_joules",
                "std",
            ),
            runs=("run_id", "count"),
        )
        .reset_index()
    )

    for column in [
        "std_elapsed_seconds",
        "std_energy_kwh",
        "std_energy_joules",
    ]:
        result[column] = result[column].fillna(0)

    result["mean_elapsed_minutes"] = (
        result["mean_elapsed_seconds"] / 60.0
    )

    print(result.to_string(index=False))

    save_csv(
        result,
        "row_size_summary.csv",
    )

    return result


# ============================================================================
# CORE SUMMARY
# ============================================================================

def core_summary(df):
    section("CORE COUNT SUMMARY")

    result = (
        df.groupby("n_cores")
        .agg(
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            std_elapsed_seconds=(
                "elapsed_seconds",
                "std",
            ),
            mean_energy_kwh=(
                "energy_kwh",
                "mean",
            ),
            std_energy_kwh=(
                "energy_kwh",
                "std",
            ),
            mean_energy_joules=(
                "energy_joules",
                "mean",
            ),
            std_energy_joules=(
                "energy_joules",
                "std",
            ),
            runs=("run_id", "count"),
        )
        .reset_index()
    )

    for column in [
        "std_elapsed_seconds",
        "std_energy_kwh",
        "std_energy_joules",
    ]:
        result[column] = result[column].fillna(0)

    result["mean_elapsed_minutes"] = (
        result["mean_elapsed_seconds"] / 60.0
    )

    print(result.to_string(index=False))

    save_csv(
        result,
        "cores_summary.csv",
    )

    return result


# ============================================================================
# BEST TIME PER CONDITION
# ============================================================================

def best_time_configurations(config):
    section(
        "FASTEST CONFIGURATION FOR EACH "
        "MODEL + FEATURES + ROWS + BACKGROUND"
    )

    group_cols = [
        "model",
        "feature_count",
        "n_rows",
        "background_size",
    ]

    idx = (
        config.groupby(group_cols)[
            "mean_elapsed_seconds"
        ].idxmin()
    )

    result = (
        config.loc[idx]
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    columns = [
        "model",
        "feature_count",
        "n_rows",
        "background_size",
        "strategy",
        "n_cores",
        "mean_elapsed_seconds",
        "mean_energy_joules",
    ]

    print(result[columns].to_string(index=False))

    save_csv(
        result[columns],
        "best_time_configurations.csv",
    )

    save_csv(
        result[columns],
        "fastest_configurations_by_condition.csv",
    )

    return result


# ============================================================================
# BEST ENERGY PER CONDITION
# ============================================================================

def best_energy_configurations(config):
    section(
        "LOWEST-ENERGY CONFIGURATION FOR EACH "
        "MODEL + FEATURES + ROWS + BACKGROUND"
    )

    group_cols = [
        "model",
        "feature_count",
        "n_rows",
        "background_size",
    ]

    idx = (
        config.groupby(group_cols)[
            "mean_energy_joules"
        ].idxmin()
    )

    result = (
        config.loc[idx]
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    columns = [
        "model",
        "feature_count",
        "n_rows",
        "background_size",
        "strategy",
        "n_cores",
        "mean_elapsed_seconds",
        "mean_energy_joules",
    ]

    print(result[columns].to_string(index=False))

    save_csv(
        result[columns],
        "best_energy_configurations.csv",
    )

    save_csv(
        result[columns],
        "lowest_energy_configurations_by_condition.csv",
    )

    return result


# ============================================================================
# TIME / ENERGY TRADEOFF
# ============================================================================

def time_energy_tradeoff(config):
    result = config.copy()

    result["energy_delay_product"] = (
        result["mean_elapsed_seconds"]
        * result["mean_energy_joules"]
    )

    result = result.sort_values(
        "energy_delay_product"
    )

    save_csv(
        result,
        "time_energy_tradeoff.csv",
    )

    return result


# ============================================================================
# REPEAT CONSISTENCY
# ============================================================================

def repeat_consistency(df):
    section("REPEAT CONSISTENCY — HIGHEST TIME VARIABILITY")

    group_cols = [
        "strategy",
        "model",
        "feature_count",
        "n_rows",
        "background_size",
        "n_cores",
    ]

    result = (
        df.groupby(group_cols)
        .agg(
            mean_elapsed_seconds=(
                "elapsed_seconds",
                "mean",
            ),
            std_elapsed_seconds=(
                "elapsed_seconds",
                "std",
            ),
            mean_energy_joules=(
                "energy_joules",
                "mean",
            ),
            std_energy_joules=(
                "energy_joules",
                "std",
            ),
            repeats=("run_id", "count"),
        )
        .reset_index()
    )

    result["std_elapsed_seconds"] = (
        result["std_elapsed_seconds"].fillna(0)
    )

    result["std_energy_joules"] = (
        result["std_energy_joules"].fillna(0)
    )

    result["time_cv_percent"] = np.where(
        result["mean_elapsed_seconds"] > 0,
        (
            result["std_elapsed_seconds"]
            / result["mean_elapsed_seconds"]
            * 100
        ),
        np.nan,
    )

    result["energy_cv_percent"] = np.where(
        result["mean_energy_joules"] > 0,
        (
            result["std_energy_joules"]
            / result["mean_energy_joules"]
            * 100
        ),
        np.nan,
    )

    result = result.sort_values(
        "time_cv_percent",
        ascending=False,
    )

    print(
        result.head(20).to_string(index=False)
    )

    save_csv(
        result,
        "repeat_consistency.csv",
    )

    return result


# ============================================================================
# CORRECTNESS SUMMARY
# ============================================================================

def correctness_summary(df):
    columns = [
        "strategy",
        "model",
        "feature_count",
        "n_rows",
        "background_size",
        "n_cores",
    ]

    if "additivity_ok" not in df.columns:
        result = pd.DataFrame(
            columns=columns
            + [
                "runs",
                "additivity_passes",
                "additivity_pass_rate",
                "max_additivity_error",
            ]
        )
        save_csv(result, "correctness_summary.csv")
        return result

    working = df.copy()

    working["additivity_ok"] = (
        working["additivity_ok"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    result = (
        working.groupby(columns)
        .agg(
            runs=("run_id", "count"),
            additivity_passes=(
                "additivity_ok",
                "sum",
            ),
            max_additivity_error=(
                "max_additivity_error",
                "max",
            ),
        )
        .reset_index()
    )

    result["additivity_pass_rate"] = (
        result["additivity_passes"]
        / result["runs"]
        * 100
    )

    save_csv(
        result,
        "correctness_summary.csv",
    )

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    section("MEMBER B — KERNELSHAP RESULT ANALYSIS")

    df = load_results()

    successful = validate_results(df)

    # Only successful runs are analyzed.
    df = successful.copy()

    if len(df) == 0:
        raise RuntimeError(
            "No successful benchmark runs available."
        )

    # Numeric conversion.
    numeric_columns = [
        "feature_count",
        "n_cores",
        "n_rows",
        "background_size",
        "repeat_idx",
        "elapsed_seconds",
        "energy_joules",
        "energy_kwh",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # ---------------------------------------------------------------------
    # BASIC SUMMARY
    # ---------------------------------------------------------------------

    section("BASIC SUMMARY")

    print(
        f"Strategies: "
        f"{sorted(df['strategy'].dropna().unique())}"
    )

    print(
        f"Models: "
        f"{sorted(df['model'].dropna().unique())}"
    )

    print(
        f"Feature counts: "
        f"{sorted(df['feature_count'].dropna().unique())}"
    )

    print(
        f"Core counts: "
        f"{sorted(df['n_cores'].dropna().unique())}"
    )

    print(
        f"Row sizes: "
        f"{sorted(df['n_rows'].dropna().unique())}"
    )

    print(
        f"Background sizes: "
        f"{sorted(df['background_size'].dropna().unique())}"
    )

    config = configuration_summary(df)

    # ---------------------------------------------------------------------
    # SUMMARIES
    # ---------------------------------------------------------------------

    fastest_configurations(config)
    lowest_energy_configurations(config)

    strategy_summary(df)
    model_summary(df)
    feature_summary(df)
    background_summary(df)
    row_size_summary(df)
    core_summary(df)

    best_time_configurations(config)
    best_energy_configurations(config)

    time_energy_tradeoff(config)
    repeat_consistency(df)
    correctness_summary(df)

    # ---------------------------------------------------------------------
    # FINAL
    # ---------------------------------------------------------------------

    section("ANALYSIS COMPLETE")

    print(
        f"Raw benchmark rows    : {len(df)}"
    )

    print(
        f"Successful runs       : {len(df)}"
    )

    print(
        f"Aggregated configs    : {len(config)}"
    )

    print(
        f"Output directory      : {OUTPUT_DIR}"
    )

    print()
    print("Generated analysis files:")

    for path in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()