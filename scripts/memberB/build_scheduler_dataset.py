"""
Member B - Build KernelSHAP Scheduler Dataset

Converts the completed KernelSHAP benchmark results into
one scheduler-oriented record per:

model × feature_count × n_rows × background_size
× strategy × n_cores

Repeated runs are aggregated using the median as the
primary scheduler statistic, matching Member A's scheduler
dataset methodology.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RESULTS_FILE = (
    BASE_DIR
    / "results"
    / "memberB"
    / "kernelshap_benchmark.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "memberB"
    / "analysis"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "scheduler_dataset.csv"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MEMBER B - BUILD KERNELSHAP SCHEDULER DATASET")
print("=" * 70)


# ============================================================
# LOAD RESULTS
# ============================================================

if not RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find benchmark results:\n{RESULTS_FILE}"
    )

df = pd.read_csv(RESULTS_FILE)

print(f"Raw benchmark rows: {len(df)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "strategy",
    "model",
    "feature_count",
    "n_rows",
    "background_size",
    "n_cores",
    "repeat_idx",
    "elapsed_seconds",
    "energy_joules",
    "success",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# KEEP SUCCESSFUL RUNS ONLY
# ============================================================

df["success"] = (
    df["success"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

df = df[
    df["success"]
    & df["elapsed_seconds"].notna()
    & df["energy_joules"].notna()
    & (df["elapsed_seconds"] >= 0)
    & (df["energy_joules"] >= 0)
].copy()

print(f"Valid successful benchmark rows: {len(df)}")


if df.empty:
    raise ValueError(
        "No valid successful benchmark rows were found."
    )


# ============================================================
# AGGREGATE REPEATED RUNS
#
# Median is the primary scheduler statistic.
# This follows Member A's scheduler methodology.
# ============================================================

group_columns = [
    "model",
    "feature_count",
    "n_rows",
    "background_size",
    "strategy",
    "n_cores",
]

scheduler = (
    df.groupby(group_columns)
    .agg(
        median_elapsed_seconds=(
            "elapsed_seconds",
            "median",
        ),
        median_energy_joules=(
            "energy_joules",
            "median",
        ),
        mean_elapsed_seconds=(
            "elapsed_seconds",
            "mean",
        ),
        mean_energy_joules=(
            "energy_joules",
            "mean",
        ),
        std_elapsed_seconds=(
            "elapsed_seconds",
            "std",
        ),
        std_energy_joules=(
            "energy_joules",
            "std",
        ),
        repeats=(
            "repeat_idx",
            "count",
        ),
    )
    .reset_index()
)


# ============================================================
# SAFETY FOR STANDARD DEVIATION
# ============================================================

scheduler["std_elapsed_seconds"] = (
    scheduler["std_elapsed_seconds"]
    .fillna(0)
)

scheduler["std_energy_joules"] = (
    scheduler["std_energy_joules"]
    .fillna(0)
)


# ============================================================
# DERIVED FIELDS
# ============================================================

scheduler["median_elapsed_minutes"] = (
    scheduler["median_elapsed_seconds"] / 60.0
)

scheduler["energy_per_second"] = (
    scheduler["median_energy_joules"]
    / scheduler["median_elapsed_seconds"].replace(0, pd.NA)
)

scheduler["time_energy_product"] = (
    scheduler["median_elapsed_seconds"]
    * scheduler["median_energy_joules"]
)


# ============================================================
# SORT
# ============================================================

scheduler = scheduler.sort_values(
    [
        "model",
        "feature_count",
        "n_rows",
        "background_size",
        "median_elapsed_seconds",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

scheduler.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("Scheduler dataset created:")
print(f"  {OUTPUT_FILE}")

print()
print(f"Unique scheduler configurations: {len(scheduler)}")

print()
print("Expected configurations:")
print("  2 models")
print("  2 feature counts")
print("  1 row size")
print("  3 background sizes")
print("  5 strategies")
print("  effective core configurations")
print()
print("Actual:")
print(f"  {len(scheduler)} configurations")

print()
print("=" * 70)
print("MEMBER B SCHEDULER DATASET COMPLETE")
print("=" * 70)