"""
scripts/memberC/plot_edp_landscape.py

Member C — Infrastructure — Stage C2-A: EDP landscape plots.

Reads, read-only:
    results/memberC/edp_dataset.csv

Writes PNGs to:
    results/memberC/plots/

This script is independent of analyze_edp_landscape.py — it recomputes
whatever small aggregates each plot needs directly from edp_dataset.csv,
so either script can be run first/alone.

Plots (log scale used wherever the data spans multiple orders of
magnitude, which EDP and energy do here):

    1. strategy_win_counts_by_track.png
    2. treeshap_median_edp_by_strategy.png
    3. kernelshap_median_edp_by_strategy.png
    4. time_vs_energy_scatter.png
    5. treeshap_edp_scaling_with_rows.png
    6. kernelshap_edp_vs_background_size.png
    7. core_count_effect_on_edp.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

STRATEGIES = ["serial", "multiprocessing", "threading", "joblib", "ray"]
STRATEGY_COLORS = {
    "serial": "#444444",
    "multiprocessing": "#1f77b4",
    "threading": "#ff7f0e",
    "joblib": "#2ca02c",
    "ray": "#d62728",
}

REQUIRED_COLS = [
    "track", "workload_id", "strategy", "n_cores", "model", "n_rows",
    "feature_count", "background_size",
    "median_wall_time_sec", "median_energy_joules", "median_edp_joule_seconds",
    "is_pareto_optimal",
]


def find_file(relative_path: str) -> Path:
    for root in REPO_ROOT_CANDIDATES:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Required input file not found: '{relative_path}'. "
        f"Looked under: {[str(r) for r in REPO_ROOT_CANDIDATES]}. "
        f"Run build_edp_dataset.py first."
    )


def plots_dir() -> Path:
    out_dir = SCRIPT_DIR.parent.parent / "results" / "memberC" / "plots"
    if not out_dir.parent.exists():
        out_dir = Path.cwd() / "results" / "memberC" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_edp() -> pd.DataFrame:
    path = find_file("results/memberC/edp_dataset.csv")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"'{path}' loaded with zero rows.")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"edp_dataset.csv missing required column(s): {missing}")
    df["is_pareto_optimal"] = df["is_pareto_optimal"].astype(bool)
    return df


def winners_per_workload(edp: pd.DataFrame) -> pd.DataFrame:
    idx = edp.groupby("workload_id")["median_edp_joule_seconds"].idxmin()
    return edp.loc[idx].copy()


def best_over_cores(edp: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Min median_edp_joule_seconds over n_cores, within each group_cols combo."""
    idx = edp.groupby(group_cols)["median_edp_joule_seconds"].idxmin()
    return edp.loc[idx].copy()


# ---------------------------------------------------------------------------
# 1. Strategy win counts by track
# ---------------------------------------------------------------------------
def plot_strategy_wins_by_track(edp: pd.DataFrame, out_dir: Path) -> None:
    winners = winners_per_workload(edp)
    counts = (
        winners.groupby(["track", "strategy"]).size()
        .unstack(fill_value=0)
        .reindex(columns=STRATEGIES, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(STRATEGIES))
    width = 0.35
    tracks = counts.index.tolist()
    for i, track in enumerate(tracks):
        offset = (i - (len(tracks) - 1) / 2) * width
        ax.bar(x + offset, counts.loc[track, STRATEGIES], width, label=track)

    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_xlabel("Parallel strategy")
    ax.set_ylabel("Workloads won (minimum median EDP)")
    ax.set_title("Strategy win counts by track (EDP-optimal configuration per workload)")
    ax.legend(title="Track")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_dir / "strategy_win_counts_by_track.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Median EDP by strategy for TreeSHAP (workload-size effect visible)
# ---------------------------------------------------------------------------
def plot_treeshap_edp_by_strategy(edp: pd.DataFrame, out_dir: Path) -> None:
    tree = edp[edp["track"] == "tree"]
    best = best_over_cores(tree, ["model", "n_rows", "strategy"])
    models = sorted(best["model"].unique())
    row_sizes = sorted(best["n_rows"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = best[best["model"] == model]
        x = np.arange(len(row_sizes))
        width = 0.8 / len(STRATEGIES)
        for i, strategy in enumerate(STRATEGIES):
            vals = []
            for n_rows in row_sizes:
                match = sub[(sub["strategy"] == strategy) & (sub["n_rows"] == n_rows)]
                vals.append(match["median_edp_joule_seconds"].iloc[0] if len(match) else np.nan)
            offset = (i - (len(STRATEGIES) - 1) / 2) * width
            ax.bar(x + offset, vals, width, label=strategy, color=STRATEGY_COLORS[strategy])
        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in row_sizes])
        ax.set_xlabel("Rows explained (n_rows)")
        ax.set_yscale("log")
        ax.set_title(f"TreeSHAP — {model}")
        ax.grid(axis="y", which="both", alpha=0.3)

    axes[0].set_ylabel("Best median EDP per strategy (joule-seconds, log scale)")
    axes[-1].legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("TreeSHAP: best median EDP by strategy, across dataset size")
    fig.tight_layout()
    fig.savefig(out_dir / "treeshap_median_edp_by_strategy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Median EDP by strategy for KernelSHAP
# ---------------------------------------------------------------------------
def plot_kernelshap_edp_by_strategy(edp: pd.DataFrame, out_dir: Path) -> None:
    kernel = edp[edp["track"] == "kernelshap"]
    best = best_over_cores(kernel, ["model", "strategy"])
    models = sorted(best["model"].unique())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(models))
    width = 0.8 / len(STRATEGIES)
    for i, strategy in enumerate(STRATEGIES):
        vals = []
        for model in models:
            match = best[(best["model"] == model) & (best["strategy"] == strategy)]
            vals.append(match["median_edp_joule_seconds"].iloc[0] if len(match) else np.nan)
        offset = (i - (len(STRATEGIES) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=strategy, color=STRATEGY_COLORS[strategy])

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_xlabel("Model")
    ax.set_ylabel("Best median EDP per strategy (joule-seconds, log scale)")
    ax.set_yscale("log")
    ax.set_title("KernelSHAP: best median EDP by strategy, by model\n"
                 "(best over n_cores, aggregated across feature_count/background_size)")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "kernelshap_median_edp_by_strategy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Time vs Energy scatter, Pareto-optimal highlighted
# ---------------------------------------------------------------------------
def plot_time_vs_energy(edp: pd.DataFrame, out_dir: Path) -> None:
    tracks = ["tree", "kernelshap"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, track in zip(axes, tracks):
        sub = edp[edp["track"] == track]
        for strategy in STRATEGIES:
            s = sub[sub["strategy"] == strategy]
            if s.empty:
                continue
            non_pareto = s[~s["is_pareto_optimal"]]
            pareto = s[s["is_pareto_optimal"]]
            ax.scatter(
                non_pareto["median_wall_time_sec"], non_pareto["median_energy_joules"],
                color=STRATEGY_COLORS[strategy], alpha=0.5, s=35,
                label=strategy, marker="o",
            )
            ax.scatter(
                pareto["median_wall_time_sec"], pareto["median_energy_joules"],
                facecolors="none", edgecolors=STRATEGY_COLORS[strategy],
                s=130, linewidths=2.0, marker="o",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Median wall time (seconds, log scale)")
        ax.set_ylabel("Median energy (joules, log scale)")
        ax.set_title(f"{track}")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].legend(title="Strategy\n(open circle = Pareto-optimal)", loc="upper left", fontsize=8)
    fig.suptitle("Median wall time vs median energy, per configuration")
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_energy_scatter.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. TreeSHAP EDP scaling with row count
# ---------------------------------------------------------------------------
def plot_treeshap_edp_scaling(edp: pd.DataFrame, out_dir: Path) -> None:
    tree = edp[edp["track"] == "tree"]
    best = best_over_cores(tree, ["model", "n_rows", "strategy"])
    models = sorted(best["model"].unique())

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = best[best["model"] == model]
        for strategy in STRATEGIES:
            s = sub[sub["strategy"] == strategy].sort_values("n_rows")
            if s.empty:
                continue
            ax.plot(
                s["n_rows"], s["median_edp_joule_seconds"],
                marker="o", color=STRATEGY_COLORS[strategy], label=strategy,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rows explained (n_rows, log scale)")
        ax.set_title(f"TreeSHAP — {model}")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Best median EDP per strategy (joule-seconds, log scale)")
    axes[-1].legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("TreeSHAP: EDP scaling with dataset size")
    fig.tight_layout()
    fig.savefig(out_dir / "treeshap_edp_scaling_with_rows.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. KernelSHAP EDP variation with background size
# ---------------------------------------------------------------------------
def plot_kernelshap_edp_vs_background(edp: pd.DataFrame, out_dir: Path) -> None:
    kernel = edp[edp["track"] == "kernelshap"]
    best = best_over_cores(kernel, ["feature_count", "background_size", "strategy"])
    feature_counts = sorted(best["feature_count"].unique())

    fig, axes = plt.subplots(1, len(feature_counts), figsize=(6 * len(feature_counts), 5), sharey=True)
    if len(feature_counts) == 1:
        axes = [axes]

    for ax, fc in zip(axes, feature_counts):
        sub = best[best["feature_count"] == fc]
        for strategy in STRATEGIES:
            s = sub[sub["strategy"] == strategy].sort_values("background_size")
            if s.empty:
                continue
            ax.plot(
                s["background_size"], s["median_edp_joule_seconds"],
                marker="o", color=STRATEGY_COLORS[strategy], label=strategy,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Background sample size (log scale)")
        ax.set_title(f"KernelSHAP — feature_count = {int(fc)}")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Best median EDP per strategy (joule-seconds, log scale)\n"
                        "(median over model)")
    axes[-1].legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("KernelSHAP: EDP variation with background size")
    fig.tight_layout()
    fig.savefig(out_dir / "kernelshap_edp_vs_background_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Core-count effect on EDP, separated by track
# ---------------------------------------------------------------------------
def plot_core_count_effect(edp: pd.DataFrame, out_dir: Path) -> None:
    tracks = ["tree", "kernelshap"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    for ax, track in zip(axes, tracks):
        sub = edp[edp["track"] == track]
        # Median EDP per (strategy, n_cores), aggregated across all workloads
        # in this track (a "typical" picture; raw per-workload EDPs span
        # many orders of magnitude for tree, so we take the median across
        # workloads at each n_cores rather than the mean).
        agg = sub.groupby(["strategy", "n_cores"])["median_edp_joule_seconds"].median().reset_index()
        for strategy in STRATEGIES:
            s = agg[agg["strategy"] == strategy].sort_values("n_cores")
            if s.empty:
                continue
            ax.plot(
                s["n_cores"], s["median_edp_joule_seconds"],
                marker="o", color=STRATEGY_COLORS[strategy], label=strategy,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.set_xticklabels(["1", "2", "4", "8"])
        ax.set_yscale("log")
        ax.set_xlabel("n_cores")
        ax.set_title(f"{track}")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Median EDP across workloads (joule-seconds, log scale)")
    axes[0].legend(title="Strategy", loc="upper right", fontsize=8)
    fig.suptitle("Core-count effect on EDP, by track\n"
                 "(median across all workloads in the track, per strategy x n_cores)")
    fig.tight_layout()
    fig.savefig(out_dir / "core_count_effect_on_edp.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    edp = load_edp()
    out_dir = plots_dir()

    print("=" * 70)
    print("MEMBER C — EDP LANDSCAPE PLOTS (C2-A)")
    print("=" * 70)
    print(f"\nLoaded edp_dataset.csv: {len(edp)} configurations")

    plot_strategy_wins_by_track(edp, out_dir)
    print("  [1/7] strategy_win_counts_by_track.png")

    plot_treeshap_edp_by_strategy(edp, out_dir)
    print("  [2/7] treeshap_median_edp_by_strategy.png")

    plot_kernelshap_edp_by_strategy(edp, out_dir)
    print("  [3/7] kernelshap_median_edp_by_strategy.png")

    plot_time_vs_energy(edp, out_dir)
    print("  [4/7] time_vs_energy_scatter.png")

    plot_treeshap_edp_scaling(edp, out_dir)
    print("  [5/7] treeshap_edp_scaling_with_rows.png")

    plot_kernelshap_edp_vs_background(edp, out_dir)
    print("  [6/7] kernelshap_edp_vs_background_size.png")

    plot_core_count_effect(edp, out_dir)
    print("  [7/7] core_count_effect_on_edp.png")

    print(f"\nAll plots written to: {out_dir}")
    print("=" * 70)
    print("PLOTTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()