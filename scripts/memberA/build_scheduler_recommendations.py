import os
import pandas as pd

# ============================================================
# MEMBER A - BUILD SCHEDULER RECOMMENDATIONS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "analysis",
    "scheduler_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "analysis"
)

TIME_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "scheduler_time_optimal.csv"
)

ENERGY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "scheduler_energy_optimal.csv"
)

BALANCED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "scheduler_balanced.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("MEMBER A - SCHEDULER RECOMMENDATION ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# Load scheduler dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"Scheduler configurations loaded: {len(df)}")

required = [
    "model",
    "n_rows",
    "strategy",
    "n_cores",
    "median_wall_time_sec",
    "median_energy_joules",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

# ============================================================
# 1. TIME-OPTIMAL CONFIGURATION
# ============================================================

time_idx = (
    df.groupby(["model", "n_rows"])["median_wall_time_sec"]
    .idxmin()
)

time_optimal = (
    df.loc[time_idx]
    .sort_values(["model", "n_rows"])
    .reset_index(drop=True)
)

time_optimal["selection_reason"] = "Minimum median execution time"

time_optimal.to_csv(
    TIME_OUTPUT,
    index=False
)

# ============================================================
# 2. ENERGY-OPTIMAL CONFIGURATION
# ============================================================

energy_idx = (
    df.groupby(["model", "n_rows"])["median_energy_joules"]
    .idxmin()
)

energy_optimal = (
    df.loc[energy_idx]
    .sort_values(["model", "n_rows"])
    .reset_index(drop=True)
)

energy_optimal["selection_reason"] = "Minimum median energy consumption"

energy_optimal.to_csv(
    ENERGY_OUTPUT,
    index=False
)

# ============================================================
# 3. BALANCED TIME-ENERGY CONFIGURATION
#
# Normalize time and energy within each workload.
#
# balanced_score:
#     0.5 * normalized_time
#   + 0.5 * normalized_energy
#
# Lower is better.
# ============================================================

balanced = df.copy()

group_cols = ["model", "n_rows"]

def min_max_normalize(series):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            0.0,
            index=series.index
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


balanced["normalized_time"] = (
    balanced
    .groupby(group_cols)["median_wall_time_sec"]
    .transform(min_max_normalize)
)

balanced["normalized_energy"] = (
    balanced
    .groupby(group_cols)["median_energy_joules"]
    .transform(min_max_normalize)
)

balanced["balanced_score"] = (
    0.5 * balanced["normalized_time"]
    + 0.5 * balanced["normalized_energy"]
)

balanced_idx = (
    balanced.groupby(group_cols)["balanced_score"]
    .idxmin()
)

balanced_optimal = (
    balanced.loc[balanced_idx]
    .sort_values(group_cols)
    .reset_index(drop=True)
)

balanced_optimal["selection_reason"] = (
    "Best normalized time-energy trade-off"
)

balanced_optimal.to_csv(
    BALANCED_OUTPUT,
    index=False
)

# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("TIME-OPTIMAL SCHEDULER")
print("=" * 70)

print(
    time_optimal[
        [
            "model",
            "n_rows",
            "strategy",
            "n_cores",
            "median_wall_time_sec",
            "median_energy_joules",
        ]
    ].to_string(index=False)
)

print()
print("=" * 70)
print("ENERGY-OPTIMAL SCHEDULER")
print("=" * 70)

print(
    energy_optimal[
        [
            "model",
            "n_rows",
            "strategy",
            "n_cores",
            "median_wall_time_sec",
            "median_energy_joules",
        ]
    ].to_string(index=False)
)

print()
print("=" * 70)
print("BALANCED TIME-ENERGY SCHEDULER")
print("=" * 70)

print(
    balanced_optimal[
        [
            "model",
            "n_rows",
            "strategy",
            "n_cores",
            "median_wall_time_sec",
            "median_energy_joules",
            "normalized_time",
            "normalized_energy",
            "balanced_score",
        ]
    ].to_string(index=False)
)

# ============================================================
# SAVE SUMMARY
# ============================================================

summary_rows = []

for _, row in time_optimal.iterrows():
    summary_rows.append({
        "model": row["model"],
        "n_rows": row["n_rows"],
        "selection_type": "time_optimal",
        "strategy": row["strategy"],
        "n_cores": row["n_cores"],
        "median_wall_time_sec": row["median_wall_time_sec"],
        "median_energy_joules": row["median_energy_joules"],
    })

for _, row in energy_optimal.iterrows():
    summary_rows.append({
        "model": row["model"],
        "n_rows": row["n_rows"],
        "selection_type": "energy_optimal",
        "strategy": row["strategy"],
        "n_cores": row["n_cores"],
        "median_wall_time_sec": row["median_wall_time_sec"],
        "median_energy_joules": row["median_energy_joules"],
    })

for _, row in balanced_optimal.iterrows():
    summary_rows.append({
        "model": row["model"],
        "n_rows": row["n_rows"],
        "selection_type": "balanced",
        "strategy": row["strategy"],
        "n_cores": row["n_cores"],
        "median_wall_time_sec": row["median_wall_time_sec"],
        "median_energy_joules": row["median_energy_joules"],
    })

summary = pd.DataFrame(summary_rows)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "scheduler_recommendations.csv"
)

summary.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

print()
print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(TIME_OUTPUT)
print(ENERGY_OUTPUT)
print(BALANCED_OUTPUT)
print(SUMMARY_OUTPUT)

print()
print("=" * 70)
print("SCHEDULER RECOMMENDATION ANALYSIS COMPLETE")
print("=" * 70)