"""
Member B — Plot Results
=======================

Creates presentation/research plots from the completed KernelSHAP
benchmark and the analysis CSVs.

Input:
    results/memberB/kernelshap_benchmark.csv
    results/memberB/analysis/*.csv

Output:
    results/memberB/plots/*.png

This script NEVER reruns KernelSHAP.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

RESULTS_DIR = (
    REPO_ROOT
    / "results"
    / "memberB"
)

INPUT_FILE = (
    RESULTS_DIR
    / "kernelshap_benchmark.csv"
)

ANALYSIS_DIR = (
    RESULTS_DIR
    / "analysis"
)

PLOT_DIR = (
    RESULTS_DIR
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# STYLE
# ============================================================================

FIGSIZE = (10, 6)
DPI = 300


def save_plot(filename):
    path = PLOT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=DPI,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {path}")


# ============================================================================
# LOAD
# ============================================================================

def load_raw():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing benchmark file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df = df[
        df["success"].astype(str).str.lower() == "true"
    ].copy()

    return df


def load_analysis(filename):

    path = ANALYSIS_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing analysis file:\n{path}\n"
            "Run analyze_results.py first."
        )

    return pd.read_csv(path)


# ============================================================================
# 1. STRATEGY — TIME
# ============================================================================

def plot_strategy_time():

    df = load_analysis(
        "strategy_summary.csv"
    )

    df = df.sort_values(
        "mean_elapsed_seconds"
    )

    plt.figure(figsize=FIGSIZE)

    plt.bar(
        df["strategy"],
        df["mean_elapsed_seconds"],
        yerr=df["std_elapsed_seconds"],
        capsize=4,
    )

    plt.xlabel("Strategy")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title(
        "KernelSHAP: Mean Execution Time by Strategy"
    )

    save_plot(
        "strategy_mean_time.png"
    )


# ============================================================================
# 2. STRATEGY — ENERGY
# ============================================================================

def plot_strategy_energy():

    df = load_analysis(
        "strategy_summary.csv"
    )

    df = df.sort_values(
        "mean_energy_joules"
    )

    plt.figure(figsize=FIGSIZE)

    plt.bar(
        df["strategy"],
        df["mean_energy_joules"],
        yerr=df["std_energy_joules"],
        capsize=4,
    )

    plt.xlabel("Strategy")
    plt.ylabel("Mean energy (J)")
    plt.title(
        "KernelSHAP: Mean Energy Consumption by Strategy"
    )

    save_plot(
        "strategy_mean_energy.png"
    )


# ============================================================================
# 3. MODEL — TIME
# ============================================================================

def plot_model_time():

    df = load_analysis(
        "model_summary.csv"
    )

    plt.figure(figsize=FIGSIZE)

    plt.bar(
        df["model"],
        df["mean_elapsed_seconds"],
        yerr=df["std_elapsed_seconds"],
        capsize=4,
    )

    plt.xlabel("Model")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title(
        "KernelSHAP: Execution Time by Model"
    )

    save_plot(
        "model_mean_time.png"
    )


# ============================================================================
# 4. MODEL — ENERGY
# ============================================================================

def plot_model_energy():

    df = load_analysis(
        "model_summary.csv"
    )

    plt.figure(figsize=FIGSIZE)

    plt.bar(
        df["model"],
        df["mean_energy_joules"],
        yerr=df["std_energy_joules"],
        capsize=4,
    )

    plt.xlabel("Model")
    plt.ylabel("Mean energy (J)")
    plt.title(
        "KernelSHAP: Energy Consumption by Model"
    )

    save_plot(
        "model_mean_energy.png"
    )


# ============================================================================
# 5. FEATURES — TIME
# ============================================================================

def plot_feature_time():

    df = load_analysis(
        "feature_count_summary.csv"
    )

    df = df.sort_values(
        "feature_count"
    )

    plt.figure(figsize=FIGSIZE)

    plt.plot(
        df["feature_count"],
        df["mean_elapsed_seconds"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Number of features")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title(
        "KernelSHAP: Effect of Feature Count on Execution Time"
    )

    plt.xticks(
        df["feature_count"]
    )

    save_plot(
        "feature_count_vs_time.png"
    )


# ============================================================================
# 6. FEATURES — ENERGY
# ============================================================================

def plot_feature_energy():

    df = load_analysis(
        "feature_count_summary.csv"
    )

    df = df.sort_values(
        "feature_count"
    )

    plt.figure(figsize=FIGSIZE)

    plt.plot(
        df["feature_count"],
        df["mean_energy_joules"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Number of features")
    plt.ylabel("Mean energy (J)")
    plt.title(
        "KernelSHAP: Effect of Feature Count on Energy Consumption"
    )

    plt.xticks(
        df["feature_count"]
    )

    save_plot(
        "feature_count_vs_energy.png"
    )


# ============================================================================
# 7. BACKGROUND — TIME
# ============================================================================

def plot_background_time():

    df = load_analysis(
        "background_size_summary.csv"
    )

    df = df.sort_values(
        "background_size"
    )

    plt.figure(figsize=FIGSIZE)

    plt.plot(
        df["background_size"],
        df["mean_elapsed_seconds"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("KernelSHAP background size")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title(
        "KernelSHAP: Effect of Background Size on Execution Time"
    )

    plt.xticks(
        df["background_size"]
    )

    save_plot(
        "background_size_vs_time.png"
    )


# ============================================================================
# 8. BACKGROUND — ENERGY
# ============================================================================

def plot_background_energy():

    df = load_analysis(
        "background_size_summary.csv"
    )

    df = df.sort_values(
        "background_size"
    )

    plt.figure(figsize=FIGSIZE)

    plt.plot(
        df["background_size"],
        df["mean_energy_joules"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("KernelSHAP background size")
    plt.ylabel("Mean energy (J)")
    plt.title(
        "KernelSHAP: Effect of Background Size on Energy Consumption"
    )

    plt.xticks(
        df["background_size"]
    )

    save_plot(
        "background_size_vs_energy.png"
    )


# ============================================================================
# 9. CORE COUNT — TIME
# ============================================================================

def plot_core_time():

    df = load_analysis(
        "cores_summary.csv"
    )

    df = df.sort_values(
        "n_cores"
    )

    plt.figure(figsize=FIGSIZE)

    plt.plot(
        df["n_cores"],
        df["mean_elapsed_seconds"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Number of CPU cores")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title(
        "KernelSHAP: Effect of Core Count on Execution Time"
    )

    plt.xticks(
        df["n_cores"]
    )

    save_plot(
        "cores_vs_time.png"
    )


# ============================================================================
# 10. CORE COUNT — ENERGY
# ============================================================================

def plot_core_energy():

    df = load_analysis(
        "cores_summary.csv"
    )

    df = df.sort_values(
        "n_cores"
    )

    plt.figure(figsize=FIGSIZE)

    plt.plot(
        df["n_cores"],
        df["mean_energy_joules"],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Number of CPU cores")
    plt.ylabel("Mean energy (J)")
    plt.title(
        "KernelSHAP: Effect of Core Count on Energy Consumption"
    )

    plt.xticks(
        df["n_cores"]
    )

    save_plot(
        "cores_vs_energy.png"
    )


# ============================================================================
# 11. STRATEGY × MODEL — TIME
# ============================================================================

def plot_strategy_model_time():

    raw = load_raw()

    grouped = (
        raw.groupby(
            ["strategy", "model"]
        )["elapsed_seconds"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="strategy",
        columns="model",
        values="elapsed_seconds",
    )

    pivot.plot(
        kind="bar",
        figsize=FIGSIZE,
    )

    plt.xlabel("Strategy")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title(
        "KernelSHAP: Strategy vs Model Execution Time"
    )

    plt.legend(
        title="Model"
    )

    save_plot(
        "strategy_model_time.png"
    )


# ============================================================================
# 12. STRATEGY × MODEL — ENERGY
# ============================================================================

def plot_strategy_model_energy():

    raw = load_raw()

    grouped = (
        raw.groupby(
            ["strategy", "model"]
        )["energy_joules"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="strategy",
        columns="model",
        values="energy_joules",
    )

    pivot.plot(
        kind="bar",
        figsize=FIGSIZE,
    )

    plt.xlabel("Strategy")
    plt.ylabel("Mean energy (J)")
    plt.title(
        "KernelSHAP: Strategy vs Model Energy Consumption"
    )

    plt.legend(
        title="Model"
    )

    save_plot(
        "strategy_model_energy.png"
    )


# ============================================================================
# 13. 15 VS 30 FEATURES × MODEL
# ============================================================================

def plot_feature_model_time():

    raw = load_raw()

    grouped = (
        raw.groupby(
            ["model", "feature_count"]
        )["elapsed_seconds"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="model",
        columns="feature_count",
        values="elapsed_seconds",
    )

    pivot.plot(
        kind="bar",
        figsize=FIGSIZE,
    )

    plt.xlabel("Model")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title(
        "KernelSHAP: 15-Feature vs 30-Feature Execution Time"
    )

    plt.legend(
        title="Feature count"
    )

    save_plot(
        "feature_count_model_time.png"
    )


# ============================================================================
# 14. BACKGROUND × MODEL
# ============================================================================

def plot_background_model_time():

    raw = load_raw()

    grouped = (
        raw.groupby(
            ["model", "background_size"]
        )["elapsed_seconds"]
        .mean()
        .reset_index()
    )

    pivot = grouped.pivot(
        index="model",
        columns="background_size",
        values="elapsed_seconds",
    )

    pivot.plot(
        kind="bar",
        figsize=FIGSIZE,
    )

    plt.xlabel("Model")
    plt.ylabel("Mean elapsed time (seconds)")
    plt.title(
        "KernelSHAP: Background Size vs Model Execution Time"
    )

    plt.legend(
        title="Background size"
    )

    save_plot(
        "background_model_time.png"
    )


# ============================================================================
# 15. TIME VS ENERGY
# ============================================================================

def plot_time_energy():

    df = load_analysis(
        "configuration_summary.csv"
    )

    plt.figure(figsize=FIGSIZE)

    for strategy in sorted(
        df["strategy"].unique()
    ):

        subset = df[
            df["strategy"] == strategy
        ]

        plt.scatter(
            subset["mean_elapsed_seconds"],
            subset["mean_energy_joules"],
            label=strategy,
            alpha=0.75,
        )

    plt.xlabel("Mean elapsed time (seconds)")
    plt.ylabel("Mean energy (J)")
    plt.title(
        "KernelSHAP: Time–Energy Trade-off"
    )

    plt.legend(
        title="Strategy"
    )

    save_plot(
        "time_energy_tradeoff.png"
    )


# ============================================================================
# 16. EDP BY STRATEGY
# ============================================================================

def plot_edp_strategy():

    df = load_analysis(
        "configuration_summary.csv"
    )

    grouped = (
        df.groupby("strategy")[
            "energy_delay_product"
        ]
        .mean()
        .sort_values()
    )

    plt.figure(figsize=FIGSIZE)

    plt.bar(
        grouped.index,
        grouped.values,
    )

    plt.xlabel("Strategy")
    plt.ylabel("Mean Energy-Delay Product")
    plt.title(
        "KernelSHAP: Energy-Delay Product by Strategy"
    )

    save_plot(
        "strategy_energy_delay_product.png"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("MEMBER B — KERNELSHAP PLOTS")
    print("=" * 70)

    # Analysis summaries.
    plot_strategy_time()
    plot_strategy_energy()

    plot_model_time()
    plot_model_energy()

    plot_feature_time()
    plot_feature_energy()

    plot_background_time()
    plot_background_energy()

    plot_core_time()
    plot_core_energy()

    # Interaction plots.
    plot_strategy_model_time()
    plot_strategy_model_energy()

    plot_feature_model_time()
    plot_background_model_time()

    # Scheduler-relevant plot.
    plot_time_energy()
    plot_edp_strategy()

    print()
    print("=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)
    print(f"Output directory: {PLOT_DIR}")

    print()
    print("Generated plots:")

    for path in sorted(PLOT_DIR.glob("*.png")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()