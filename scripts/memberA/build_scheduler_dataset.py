import os
import pandas as pd

# ============================================================
# MEMBER A - BUILD SCHEDULER DATASET
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

RESULTS_FILE = os.path.join(
    BASE_DIR, "results", "results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "results", "analysis"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "scheduler_dataset.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("MEMBER A - BUILD SCHEDULER DATASET")
print("=" * 60)

# ------------------------------------------------------------
# Load benchmark results
# ------------------------------------------------------------

df = pd.read_csv(RESULTS_FILE)

print(f"Raw benchmark rows: {len(df)}")

# ------------------------------------------------------------
# Validate required columns
# ------------------------------------------------------------

required_columns = [
    "strategy",
    "model",
    "n_rows",
    "n_cores",
    "repeat_idx",
    "wall_time_sec",
    "energy_joules",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

# ------------------------------------------------------------
# Keep only valid benchmark rows
# ------------------------------------------------------------

df = df[
    df["wall_time_sec"].notna()
    & df["energy_joules"].notna()
    & (df["wall_time_sec"] >= 0)
    & (df["energy_joules"] >= 0)
].copy()

print(f"Valid benchmark rows: {len(df)}")

# ------------------------------------------------------------
# Aggregate repeated runs
#
# We use MEDIAN as the primary scheduler statistic because
# some RandomForest runs have substantial variation.
# ------------------------------------------------------------

group_columns = [
    "model",
    "n_rows",
    "strategy",
    "n_cores",
]

scheduler = (
    df.groupby(group_columns)
    .agg(
        median_wall_time_sec=(
            "wall_time_sec",
            "median"
        ),
        median_energy_joules=(
            "energy_joules",
            "median"
        ),
        mean_wall_time_sec=(
            "wall_time_sec",
            "mean"
        ),
        mean_energy_joules=(
            "energy_joules",
            "mean"
        ),
        std_wall_time_sec=(
            "wall_time_sec",
            "std"
        ),
        std_energy_joules=(
            "energy_joules",
            "std"
        ),
        repeats=(
            "repeat_idx",
            "count"
        ),
    )
    .reset_index()
)

# Replace NaN standard deviations for safety
scheduler["std_wall_time_sec"] = (
    scheduler["std_wall_time_sec"]
    .fillna(0)
)

scheduler["std_energy_joules"] = (
    scheduler["std_energy_joules"]
    .fillna(0)
)

# ------------------------------------------------------------
# Add useful derived fields
# ------------------------------------------------------------

scheduler["median_wall_time_min"] = (
    scheduler["median_wall_time_sec"] / 60
)

# Energy per second is useful for understanding
# how energy consumption relates to execution time.
scheduler["energy_per_second"] = (
    scheduler["median_energy_joules"]
    / scheduler["median_wall_time_sec"].replace(0, pd.NA)
)

# ------------------------------------------------------------
# Sort for readability
# ------------------------------------------------------------

scheduler = scheduler.sort_values(
    [
        "model",
        "n_rows",
        "median_wall_time_sec"
    ]
).reset_index(drop=True)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

scheduler.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Scheduler dataset created:")
print(OUTPUT_FILE)

print()
print(f"Unique scheduler configurations: {len(scheduler)}")

print()
print("Expected configurations:")
print("  2 models")
print("  3 dataset sizes")
print("  5 strategies")
print("  4 core counts")
print("  = 120 configurations")

print()
print("Actual:")
print(f"  {len(scheduler)} configurations")

print()
print("=" * 60)
print("SCHEDULER DATASET COMPLETE")
print("=" * 60)