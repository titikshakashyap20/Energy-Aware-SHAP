"""
Member B — Build Scheduler Recommendations

Uses the Member B scheduler dataset to generate:

1. Time-optimal configurations
2. Energy-optimal configurations
3. Balanced time-energy configurations
4. Combined scheduler recommendations

The recommendation is made separately for each workload:

    model × feature_count × n_rows × background_size

This keeps Member B's KernelSHAP workload dimensions separate and
does not modify or rerun the benchmark.
"""

import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "memberB",
    "analysis",
    "scheduler_dataset.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "memberB",
    "analysis",
)

TIME_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "scheduler_time_optimal.csv",
)

ENERGY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "scheduler_energy_optimal.csv",
)

BALANCED_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "scheduler_balanced.csv",
)

SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "scheduler_recommendations.csv",
)


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("MEMBER B - KERNELSHAP SCHEDULER RECOMMENDATION ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Scheduler dataset not found:\n{INPUT_FILE}\n\n"
        "Run build_scheduler_dataset.py first."
    )

df = pd.read_csv(INPUT_FILE)

print(f"Scheduler configurations loaded: {len(df)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required = [
    "model",
    "feature_count",
    "n_rows",
    "background_size",
    "strategy",
    "n_cores",
    "median_elapsed_seconds",
    "median_energy_joules",
]

missing = [
    column
    for column in required
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df.copy()

df["median_elapsed_seconds"] = pd.to_numeric(
    df["median_elapsed_seconds"],
    errors="coerce",
)

df["median_energy_joules"] = pd.to_numeric(
    df["median_energy_joules"],
    errors="coerce",
)

df = df[
    df["median_elapsed_seconds"].notna()
    & df["median_energy_joules"].notna()
    & (df["median_elapsed_seconds"] >= 0)
    & (df["median_energy_joules"] >= 0)
].copy()

print(f"Valid scheduler configurations: {len(df)}")


# ============================================================
# WORKLOAD DEFINITION
#
# IMPORTANT:
# Do NOT compare configurations across different:
#
# - models
# - feature counts
# - row counts
# - background sizes
#
# Each workload gets its own recommendation.
# ============================================================

GROUP_COLS = [
    "model",
    "feature_count",
    "n_rows",
    "background_size",
]


# ============================================================
# 1. TIME-OPTIMAL
# ============================================================

time_idx = (
    df.groupby(GROUP_COLS)["median_elapsed_seconds"]
    .idxmin()
)

time_optimal = (
    df.loc[time_idx]
    .sort_values(GROUP_COLS)
    .reset_index(drop=True)
)

time_optimal["selection_type"] = "time_optimal"

time_optimal["selection_reason"] = (
    "Minimum median execution time"
)

time_optimal.to_csv(
    TIME_OUTPUT,
    index=False,
)


# ============================================================
# 2. ENERGY-OPTIMAL
# ============================================================

energy_idx = (
    df.groupby(GROUP_COLS)["median_energy_joules"]
    .idxmin()
)

energy_optimal = (
    df.loc[energy_idx]
    .sort_values(GROUP_COLS)
    .reset_index(drop=True)
)

energy_optimal["selection_type"] = "energy_optimal"

energy_optimal["selection_reason"] = (
    "Minimum median energy consumption"
)

energy_optimal.to_csv(
    ENERGY_OUTPUT,
    index=False,
)


# ============================================================
# 3. BALANCED TIME-ENERGY
#
# Normalize time and energy WITHIN EACH WORKLOAD.
#
# balanced_score =
#
#     0.5 * normalized_time
#   + 0.5 * normalized_energy
#
# Lower score = better balance.
# ============================================================

balanced = df.copy()


def min_max_normalize(series):
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return pd.Series(
            0.0,
            index=series.index,
        )

    return (
        (series - min_value)
        / (max_value - min_value)
    )


balanced["normalized_time"] = (
    balanced
    .groupby(GROUP_COLS)["median_elapsed_seconds"]
    .transform(min_max_normalize)
)

balanced["normalized_energy"] = (
    balanced
    .groupby(GROUP_COLS)["median_energy_joules"]
    .transform(min_max_normalize)
)

balanced["balanced_score"] = (
    0.5 * balanced["normalized_time"]
    + 0.5 * balanced["normalized_energy"]
)


balanced_idx = (
    balanced
    .groupby(GROUP_COLS)["balanced_score"]
    .idxmin()
)

balanced_optimal = (
    balanced.loc[balanced_idx]
    .sort_values(GROUP_COLS)
    .reset_index(drop=True)
)

balanced_optimal["selection_type"] = "balanced"

balanced_optimal["selection_reason"] = (
    "Best normalized time-energy trade-off"
)

balanced_optimal.to_csv(
    BALANCED_OUTPUT,
    index=False,
)


# ============================================================
# PRINT TIME-OPTIMAL RESULTS
# ============================================================

print()
print("=" * 70)
print("TIME-OPTIMAL SCHEDULER")
print("=" * 70)

print(
    time_optimal[
        [
            "model",
            "feature_count",
            "n_rows",
            "background_size",
            "strategy",
            "n_cores",
            "median_elapsed_seconds",
            "median_energy_joules",
        ]
    ].to_string(index=False)
)


# ============================================================
# PRINT ENERGY-OPTIMAL RESULTS
# ============================================================

print()
print("=" * 70)
print("ENERGY-OPTIMAL SCHEDULER")
print("=" * 70)

print(
    energy_optimal[
        [
            "model",
            "feature_count",
            "n_rows",
            "background_size",
            "strategy",
            "n_cores",
            "median_elapsed_seconds",
            "median_energy_joules",
        ]
    ].to_string(index=False)
)


# ============================================================
# PRINT BALANCED RESULTS
# ============================================================

print()
print("=" * 70)
print("BALANCED TIME-ENERGY SCHEDULER")
print("=" * 70)

print(
    balanced_optimal[
        [
            "model",
            "feature_count",
            "n_rows",
            "background_size",
            "strategy",
            "n_cores",
            "median_elapsed_seconds",
            "median_energy_joules",
            "normalized_time",
            "normalized_energy",
            "balanced_score",
        ]
    ].to_string(index=False)
)


# ============================================================
# COMBINED SUMMARY
# ============================================================

summary_columns = [
    "model",
    "feature_count",
    "n_rows",
    "background_size",
    "selection_type",
    "strategy",
    "n_cores",
    "median_elapsed_seconds",
    "median_energy_joules",
]


summary_rows = []

for _, row in time_optimal.iterrows():
    summary_rows.append(
        {
            "model": row["model"],
            "feature_count": row["feature_count"],
            "n_rows": row["n_rows"],
            "background_size": row["background_size"],
            "selection_type": "time_optimal",
            "strategy": row["strategy"],
            "n_cores": row["n_cores"],
            "median_elapsed_seconds": row[
                "median_elapsed_seconds"
            ],
            "median_energy_joules": row[
                "median_energy_joules"
            ],
        }
    )


for _, row in energy_optimal.iterrows():
    summary_rows.append(
        {
            "model": row["model"],
            "feature_count": row["feature_count"],
            "n_rows": row["n_rows"],
            "background_size": row["background_size"],
            "selection_type": "energy_optimal",
            "strategy": row["strategy"],
            "n_cores": row["n_cores"],
            "median_elapsed_seconds": row[
                "median_elapsed_seconds"
            ],
            "median_energy_joules": row[
                "median_energy_joules"
            ],
        }
    )


for _, row in balanced_optimal.iterrows():
    summary_rows.append(
        {
            "model": row["model"],
            "feature_count": row["feature_count"],
            "n_rows": row["n_rows"],
            "background_size": row["background_size"],
            "selection_type": "balanced",
            "strategy": row["strategy"],
            "n_cores": row["n_cores"],
            "median_elapsed_seconds": row[
                "median_elapsed_seconds"
            ],
            "median_energy_joules": row[
                "median_energy_joules"
            ],
        }
    )


summary = pd.DataFrame(
    summary_rows,
    columns=summary_columns,
)


summary = summary.sort_values(
    [
        "model",
        "feature_count",
        "n_rows",
        "background_size",
        "selection_type",
    ]
).reset_index(drop=True)


summary.to_csv(
    SUMMARY_OUTPUT,
    index=False,
)


# ============================================================
# VALIDATION
# ============================================================

expected_workloads = (
    df[GROUP_COLS]
    .drop_duplicates()
    .shape[0]
)

expected_summary_rows = expected_workloads * 3

print()
print("=" * 70)
print("RECOMMENDATION VALIDATION")
print("=" * 70)

print(
    f"Unique workloads              : "
    f"{expected_workloads}"
)

print(
    f"Expected recommendation rows : "
    f"{expected_summary_rows}"
)

print(
    f"Actual recommendation rows   : "
    f"{len(summary)}"
)

if len(summary) != expected_summary_rows:
    raise RuntimeError(
        "Recommendation row count does not match "
        "expected workloads × 3."
    )

print("[PASS] Every workload has 3 recommendation types.")


# ============================================================
# FILES CREATED
# ============================================================

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
print("MEMBER B SCHEDULER RECOMMENDATION ANALYSIS COMPLETE")
print("=" * 70)
