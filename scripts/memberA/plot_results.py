import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# MEMBER A - BENCHMARK VISUALIZATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

ANALYSIS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "analysis"
)

PLOT_DIR = os.path.join(
    ANALYSIS_DIR,
    "plots"
)

os.makedirs(PLOT_DIR, exist_ok=True)

SCHEDULER_FILE = os.path.join(
    ANALYSIS_DIR,
    "scheduler_dataset.csv"
)

TIME_FILE = os.path.join(
    ANALYSIS_DIR,
    "scheduler_time_optimal.csv"
)

ENERGY_FILE = os.path.join(
    ANALYSIS_DIR,
    "scheduler_energy_optimal.csv"
)

BALANCED_FILE = os.path.join(
    ANALYSIS_DIR,
    "scheduler_balanced.csv"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(SCHEDULER_FILE)
time_opt = pd.read_csv(TIME_FILE)
energy_opt = pd.read_csv(ENERGY_FILE)
balanced = pd.read_csv(BALANCED_FILE)

print("=" * 70)
print("MEMBER A - BENCHMARK VISUALIZATION")
print("=" * 70)

print(f"Scheduler rows loaded: {len(df)}")
print(f"Plot directory: {PLOT_DIR}")

# ============================================================
# HELPER
# ============================================================

def save_plot(filename):
    path = os.path.join(PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Created: {path}")


# ============================================================
# 1. EXECUTION TIME VS CORES
# ============================================================

for model in sorted(df["model"].unique()):

    for n_rows in sorted(df["n_rows"].unique()):

        subset = df[
            (df["model"] == model)
            & (df["n_rows"] == n_rows)
        ]

        plt.figure(figsize=(9, 6))

        for strategy in sorted(subset["strategy"].unique()):

            strategy_data = subset[
                subset["strategy"] == strategy
            ].sort_values("n_cores")

            plt.plot(
                strategy_data["n_cores"],
                strategy_data["median_wall_time_sec"],
                marker="o",
                label=strategy
            )

        plt.xlabel("Number of Cores")
        plt.ylabel("Median Execution Time (seconds)")
        plt.title(
            f"Execution Time vs Cores - "
            f"{model} - {n_rows:,} rows"
        )

        plt.xticks([1, 2, 4, 8])
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_plot(
            f"time_vs_cores_{model}_{n_rows}.png"
        )


# ============================================================
# 2. ENERGY VS CORES
# ============================================================

for model in sorted(df["model"].unique()):

    for n_rows in sorted(df["n_rows"].unique()):

        subset = df[
            (df["model"] == model)
            & (df["n_rows"] == n_rows)
        ]

        plt.figure(figsize=(9, 6))

        for strategy in sorted(subset["strategy"].unique()):

            strategy_data = subset[
                subset["strategy"] == strategy
            ].sort_values("n_cores")

            plt.plot(
                strategy_data["n_cores"],
                strategy_data["median_energy_joules"],
                marker="o",
                label=strategy
            )

        plt.xlabel("Number of Cores")
        plt.ylabel("Median Energy (Joules)")
        plt.title(
            f"Energy vs Cores - "
            f"{model} - {n_rows:,} rows"
        )

        plt.xticks([1, 2, 4, 8])
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_plot(
            f"energy_vs_cores_{model}_{n_rows}.png"
        )


# ============================================================
# 3. TIME VS ENERGY TRADE-OFF
# ============================================================

for model in sorted(df["model"].unique()):

    for n_rows in sorted(df["n_rows"].unique()):

        subset = df[
            (df["model"] == model)
            & (df["n_rows"] == n_rows)
        ]

        plt.figure(figsize=(9, 6))

        for strategy in sorted(subset["strategy"].unique()):

            strategy_data = subset[
                subset["strategy"] == strategy
            ]

            plt.scatter(
                strategy_data["median_wall_time_sec"],
                strategy_data["median_energy_joules"],
                label=strategy,
                s=60
            )

        plt.xlabel("Median Execution Time (seconds)")
        plt.ylabel("Median Energy (Joules)")
        plt.title(
            f"Time-Energy Trade-off - "
            f"{model} - {n_rows:,} rows"
        )

        plt.legend()
        plt.grid(True, alpha=0.3)

        save_plot(
            f"time_energy_{model}_{n_rows}.png"
        )


# ============================================================
# 4. STRATEGY-LEVEL COMPARISON
# ============================================================

strategy_summary = (
    df.groupby("strategy")
    .agg(
        mean_time=("median_wall_time_sec", "mean"),
        mean_energy=("median_energy_joules", "mean")
    )
    .reset_index()
)

plt.figure(figsize=(9, 6))

plt.bar(
    strategy_summary["strategy"],
    strategy_summary["mean_time"]
)

plt.xlabel("Strategy")
plt.ylabel("Mean Median Execution Time (seconds)")
plt.title("Overall Strategy Execution Time")

plt.xticks(rotation=20)

save_plot("overall_strategy_time.png")


plt.figure(figsize=(9, 6))

plt.bar(
    strategy_summary["strategy"],
    strategy_summary["mean_energy"]
)

plt.xlabel("Strategy")
plt.ylabel("Mean Median Energy (Joules)")
plt.title("Overall Strategy Energy Consumption")

plt.xticks(rotation=20)

save_plot("overall_strategy_energy.png")


# ============================================================
# 5. DATASET SIZE SCALING
# ============================================================

dataset_summary = (
    df.groupby(["model", "n_rows"])
    .agg(
        median_time=("median_wall_time_sec", "median"),
        median_energy=("median_energy_joules", "median")
    )
    .reset_index()
)

for model in sorted(dataset_summary["model"].unique()):

    subset = dataset_summary[
        dataset_summary["model"] == model
    ].sort_values("n_rows")

    plt.figure(figsize=(9, 6))

    plt.plot(
        subset["n_rows"],
        subset["median_time"],
        marker="o"
    )

    plt.xlabel("Dataset Size (rows)")
    plt.ylabel("Median Execution Time (seconds)")
    plt.title(
        f"Execution Time Scaling - {model}"
    )

    plt.grid(True, alpha=0.3)

    save_plot(
        f"dataset_scaling_time_{model}.png"
    )


    plt.figure(figsize=(9, 6))

    plt.plot(
        subset["n_rows"],
        subset["median_energy"],
        marker="o"
    )

    plt.xlabel("Dataset Size (rows)")
    plt.ylabel("Median Energy (Joules)")
    plt.title(
        f"Energy Scaling - {model}"
    )

    plt.grid(True, alpha=0.3)

    save_plot(
        f"dataset_scaling_energy_{model}.png"
    )


# ============================================================
# 6. SCHEDULER DECISIONS
# ============================================================

# Time-optimal scheduler

plt.figure(figsize=(11, 6))

labels = (
    time_opt["model"]
    + "\n"
    + time_opt["n_rows"].astype(str)
)

plt.bar(
    labels,
    time_opt["median_wall_time_sec"]
)

plt.xlabel("Workload")
plt.ylabel("Median Execution Time (seconds)")
plt.title("Time-Optimal Scheduler Decisions")

plt.xticks(rotation=45, ha="right")

save_plot("scheduler_time_optimal.png")


# Energy-optimal scheduler

plt.figure(figsize=(11, 6))

labels = (
    energy_opt["model"]
    + "\n"
    + energy_opt["n_rows"].astype(str)
)

plt.bar(
    labels,
    energy_opt["median_energy_joules"]
)

plt.xlabel("Workload")
plt.ylabel("Median Energy (Joules)")
plt.title("Energy-Optimal Scheduler Decisions")

plt.xticks(rotation=45, ha="right")

save_plot("scheduler_energy_optimal.png")


# ============================================================
# 7. BALANCED SCHEDULER SCORE
# ============================================================

plt.figure(figsize=(11, 6))

labels = (
    balanced["model"]
    + "\n"
    + balanced["n_rows"].astype(str)
)

plt.bar(
    labels,
    balanced["balanced_score"]
)

plt.xlabel("Workload")
plt.ylabel("Balanced Time-Energy Score")
plt.title(
    "Balanced Scheduler Decisions "
    "(Lower Score = Better Trade-off)"
)

plt.xticks(rotation=45, ha="right")

save_plot("scheduler_balanced_score.png")


# ============================================================
# 8. CORE COUNT OVERALL EFFECT
# ============================================================

core_summary = (
    df.groupby("n_cores")
    .agg(
        mean_time=("median_wall_time_sec", "mean"),
        mean_energy=("median_energy_joules", "mean")
    )
    .reset_index()
)

plt.figure(figsize=(8, 6))

plt.plot(
    core_summary["n_cores"],
    core_summary["mean_time"],
    marker="o"
)

plt.xlabel("Number of Cores")
plt.ylabel("Mean Median Execution Time (seconds)")
plt.title("Overall Effect of Core Count on Execution Time")

plt.xticks([1, 2, 4, 8])
plt.grid(True, alpha=0.3)

save_plot("overall_core_time.png")


plt.figure(figsize=(8, 6))

plt.plot(
    core_summary["n_cores"],
    core_summary["mean_energy"],
    marker="o"
)

plt.xlabel("Number of Cores")
plt.ylabel("Mean Median Energy (Joules)")
plt.title("Overall Effect of Core Count on Energy")

plt.xticks([1, 2, 4, 8])
plt.grid(True, alpha=0.3)

save_plot("overall_core_energy.png")


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print()
print(f"All plots saved to:")
print(PLOT_DIR)

print()
print("Member A visualization layer complete.")
print("=" * 70)