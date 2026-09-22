import pandas as pd
from pathlib import Path


# ============================================================
# MEMBER A - RESULT ANALYSIS
# ============================================================

RESULTS_FILE = Path("results/results.csv")
OUTPUT_DIR = Path("results/analysis")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("MEMBER A - RESULT ANALYSIS")
print("=" * 60)


# ============================================================
# 1. LOAD RESULTS
# ============================================================

if not RESULTS_FILE.exists():
    raise FileNotFoundError(
        f"Could not find results file: {RESULTS_FILE}"
    )

df = pd.read_csv(RESULTS_FILE)

print(f"Raw benchmark rows: {len(df)}")
print(f"Columns: {list(df.columns)}")


# ============================================================
# 2. REQUIRED COLUMNS
# ============================================================

required_columns = [
    "strategy",
    "model",
    "n_rows",
    "n_cores",
    "repeat_idx",
    "wall_time_sec",
    "energy_joules",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# 3. CLEAN DATA
# ============================================================

numeric_columns = [
    "n_rows",
    "n_cores",
    "repeat_idx",
    "wall_time_sec",
    "energy_joules",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# Remove invalid measurements
before = len(df)

df = df.dropna(
    subset=[
        "strategy",
        "model",
        "n_rows",
        "n_cores",
        "wall_time_sec",
        "energy_joules",
    ]
)

df = df[
    (df["wall_time_sec"] >= 0)
    & (df["energy_joules"] >= 0)
]

after = len(df)

print(f"Valid benchmark rows: {after}")

if before != after:
    print(f"Removed invalid rows: {before - after}")


# ============================================================
# 4. BASIC SUMMARY
# ============================================================

print()
print("=" * 60)
print("BASIC SUMMARY")
print("=" * 60)

print(f"Strategies: {sorted(df['strategy'].unique())}")
print(f"Models: {sorted(df['model'].unique())}")
print(f"Dataset sizes: {sorted(df['n_rows'].unique())}")
print(f"Core counts: {sorted(df['n_cores'].unique())}")


# ============================================================
# 5. AGGREGATE REPEATS
# ============================================================

# Each configuration has 3 repeats.
#
# We calculate:
# - mean execution time
# - std execution time
# - mean energy
# - std energy
# - number of repeats
#
# This is the main table we will use for comparisons.

group_columns = [
    "strategy",
    "model",
    "n_rows",
    "n_cores",
]

summary = (
    df.groupby(group_columns)
    .agg(
        mean_wall_time_sec=("wall_time_sec", "mean"),
        std_wall_time_sec=("wall_time_sec", "std"),
        mean_energy_joules=("energy_joules", "mean"),
        std_energy_joules=("energy_joules", "std"),
        repeats=("repeat_idx", "count"),
    )
    .reset_index()
)


# ============================================================
# 6. ADD HUMAN-READABLE TIME
# ============================================================

summary["mean_wall_time_min"] = (
    summary["mean_wall_time_sec"] / 60
)

summary["mean_wall_time_hours"] = (
    summary["mean_wall_time_sec"] / 3600
)


# ============================================================
# 7. SAVE COMPLETE SUMMARY
# ============================================================

summary_file = OUTPUT_DIR / "configuration_summary.csv"

summary.to_csv(
    summary_file,
    index=False
)

print()
print(f"Saved configuration summary:")
print(f"  {summary_file}")


# ============================================================
# 8. FASTEST CONFIGURATIONS
# ============================================================

print()
print("=" * 60)
print("FASTEST CONFIGURATIONS")
print("=" * 60)

fastest = summary.sort_values(
    "mean_wall_time_sec"
).head(20)

print(
    fastest[
        [
            "strategy",
            "model",
            "n_rows",
            "n_cores",
            "mean_wall_time_sec",
            "mean_wall_time_min",
            "mean_energy_joules",
        ]
    ].to_string(index=False)
)


# ============================================================
# 9. LOWEST ENERGY CONFIGURATIONS
# ============================================================

print()
print("=" * 60)
print("LOWEST ENERGY CONFIGURATIONS")
print("=" * 60)

lowest_energy = summary.sort_values(
    "mean_energy_joules"
).head(20)

print(
    lowest_energy[
        [
            "strategy",
            "model",
            "n_rows",
            "n_cores",
            "mean_wall_time_sec",
            "mean_energy_joules",
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. STRATEGY-LEVEL SUMMARY
# ============================================================

strategy_summary = (
    df.groupby("strategy")
    .agg(
        mean_wall_time_sec=("wall_time_sec", "mean"),
        mean_energy_joules=("energy_joules", "mean"),
        std_wall_time_sec=("wall_time_sec", "std"),
        std_energy_joules=("energy_joules", "std"),
        runs=("strategy", "count"),
    )
    .reset_index()
)

strategy_summary["mean_wall_time_min"] = (
    strategy_summary["mean_wall_time_sec"] / 60
)

strategy_summary = strategy_summary.sort_values(
    "mean_wall_time_sec"
)

strategy_file = OUTPUT_DIR / "strategy_summary.csv"

strategy_summary.to_csv(
    strategy_file,
    index=False
)

print()
print("=" * 60)
print("STRATEGY SUMMARY")
print("=" * 60)

print(
    strategy_summary.to_string(index=False)
)

print()
print(f"Saved: {strategy_file}")


# ============================================================
# 11. MODEL-LEVEL SUMMARY
# ============================================================

model_summary = (
    df.groupby("model")
    .agg(
        mean_wall_time_sec=("wall_time_sec", "mean"),
        mean_energy_joules=("energy_joules", "mean"),
        std_wall_time_sec=("wall_time_sec", "std"),
        std_energy_joules=("energy_joules", "std"),
        runs=("model", "count"),
    )
    .reset_index()
)

model_summary["mean_wall_time_min"] = (
    model_summary["mean_wall_time_sec"] / 60
)

model_file = OUTPUT_DIR / "model_summary.csv"

model_summary.to_csv(
    model_file,
    index=False
)

print()
print("=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

print(
    model_summary.to_string(index=False)
)

print()
print(f"Saved: {model_file}")


# ============================================================
# 12. DATASET-SIZE SUMMARY
# ============================================================

dataset_summary = (
    df.groupby("n_rows")
    .agg(
        mean_wall_time_sec=("wall_time_sec", "mean"),
        mean_energy_joules=("energy_joules", "mean"),
        std_wall_time_sec=("wall_time_sec", "std"),
        std_energy_joules=("energy_joules", "std"),
        runs=("n_rows", "count"),
    )
    .reset_index()
)

dataset_summary["mean_wall_time_min"] = (
    dataset_summary["mean_wall_time_sec"] / 60
)

dataset_file = OUTPUT_DIR / "dataset_size_summary.csv"

dataset_summary.to_csv(
    dataset_file,
    index=False
)

print()
print("=" * 60)
print("DATASET SIZE SUMMARY")
print("=" * 60)

print(
    dataset_summary.to_string(index=False)
)

print()
print(f"Saved: {dataset_file}")


# ============================================================
# 13. CORES SUMMARY
# ============================================================

cores_summary = (
    df.groupby("n_cores")
    .agg(
        mean_wall_time_sec=("wall_time_sec", "mean"),
        mean_energy_joules=("energy_joules", "mean"),
        std_wall_time_sec=("wall_time_sec", "std"),
        std_energy_joules=("energy_joules", "std"),
        runs=("n_cores", "count"),
    )
    .reset_index()
)

cores_summary["mean_wall_time_min"] = (
    cores_summary["mean_wall_time_sec"] / 60
)

cores_file = OUTPUT_DIR / "cores_summary.csv"

cores_summary.to_csv(
    cores_file,
    index=False
)

print()
print("=" * 60)
print("CORE COUNT SUMMARY")
print("=" * 60)

print(
    cores_summary.to_string(index=False)
)

print()
print(f"Saved: {cores_file}")


# ============================================================
# 14. BEST TIME PER CONFIGURATION
# ============================================================

# For every model + dataset size combination,
# identify the configuration with the lowest average time.

best_time = (
    summary.loc[
        summary.groupby(
            ["model", "n_rows"]
        )["mean_wall_time_sec"].idxmin()
    ]
    .sort_values(["model", "n_rows"])
)

best_time_file = OUTPUT_DIR / "best_time_configurations.csv"

best_time.to_csv(
    best_time_file,
    index=False
)

print()
print("=" * 60)
print("FASTEST CONFIGURATION FOR EACH MODEL + DATASET SIZE")
print("=" * 60)

print(
    best_time[
        [
            "model",
            "n_rows",
            "strategy",
            "n_cores",
            "mean_wall_time_sec",
            "mean_energy_joules",
        ]
    ].to_string(index=False)
)

print()
print(f"Saved: {best_time_file}")


# ============================================================
# 15. BEST ENERGY PER CONFIGURATION
# ============================================================

best_energy = (
    summary.loc[
        summary.groupby(
            ["model", "n_rows"]
        )["mean_energy_joules"].idxmin()
    ]
    .sort_values(["model", "n_rows"])
)

best_energy_file = OUTPUT_DIR / "best_energy_configurations.csv"

best_energy.to_csv(
    best_energy_file,
    index=False
)

print()
print("=" * 60)
print("LOWEST-ENERGY CONFIGURATION FOR EACH MODEL + DATASET SIZE")
print("=" * 60)

print(
    best_energy[
        [
            "model",
            "n_rows",
            "strategy",
            "n_cores",
            "mean_wall_time_sec",
            "mean_energy_joules",
        ]
    ].to_string(index=False)
)

print()
print(f"Saved: {best_energy_file}")


# ============================================================
# 16. TIME VS ENERGY TRADE-OFF DATA
# ============================================================

# This table is particularly useful for the scheduler.
#
# The scheduler will eventually need to understand:
#
#     input characteristics
#              ↓
#       expected runtime
#              ↓
#       expected energy
#              ↓
#       choose strategy + cores

tradeoff = summary.copy()

tradeoff["time_energy_product"] = (
    tradeoff["mean_wall_time_sec"]
    * tradeoff["mean_energy_joules"]
)

tradeoff_file = OUTPUT_DIR / "time_energy_tradeoff.csv"

tradeoff.to_csv(
    tradeoff_file,
    index=False
)

print()
print(f"Saved: {tradeoff_file}")


# ============================================================
# 17. CHECK REPEAT CONSISTENCY
# ============================================================

print()
print("=" * 60)
print("REPEAT CONSISTENCY")
print("=" * 60)

repeat_stats = (
    summary[
        [
            "strategy",
            "model",
            "n_rows",
            "n_cores",
            "mean_wall_time_sec",
            "std_wall_time_sec",
            "mean_energy_joules",
            "std_energy_joules",
            "repeats",
        ]
    ]
)

print(
    repeat_stats.sort_values(
        "std_wall_time_sec",
        ascending=False
    ).head(15).to_string(index=False)
)

repeat_file = OUTPUT_DIR / "repeat_consistency.csv"

repeat_stats.to_csv(
    repeat_file,
    index=False
)

print()
print(f"Saved: {repeat_file}")


# ============================================================
# 18. FINAL
# ============================================================

print()
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

print(f"Raw benchmark rows : {before}")
print(f"Valid rows         : {len(df)}")
print(f"Configurations     : {len(summary)}")
print()
print("Analysis files created in:")
print(f"  {OUTPUT_DIR.resolve()}")
print("=" * 60)