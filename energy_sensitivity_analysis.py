from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
FILES = [
    (ROOT / "results/memberA/results.csv", "tree", "wall_time_sec"),
    (ROOT / "results/memberB/kernelshap_benchmark.csv", "kernelshap", "elapsed_seconds"),
]
RAM_WATTS = 10.0  # Hypothetical sensitivity assumption, not measured historical RAM power.

frames = []
for path, track, time_col in FILES:
    df = pd.read_csv(path)
    df["track"] = track
    df["wall_time_sec"] = pd.to_numeric(df[time_col], errors="coerce")
    df["energy_joules"] = pd.to_numeric(df["energy_joules"], errors="coerce")
    df["n_cores"] = pd.to_numeric(df["n_cores"], errors="coerce")

    if track == "tree":
        df["workload_id"] = (
            "tree|" + df["model"].astype(str) + "|" + df["n_rows"].astype(str)
        )
        config_cols = ["workload_id", "n_cores", "repeat_idx"]
    else:
        df["workload_id"] = (
            "kernelshap|" + df["model"].astype(str) + "|"
            + df["feature_count"].astype(str) + "|"
            + df["n_rows"].astype(str) + "|"
            + df["background_size"].astype(str)
        )
        config_cols = ["workload_id", "n_cores", "repeat_idx"]

    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df = all_df[
    all_df["wall_time_sec"].gt(0)
    & all_df["energy_joules"].gt(0)
].copy()

all_df["original_edp"] = all_df["energy_joules"] * all_df["wall_time_sec"]
all_df["sensitivity_energy_j"] = (
    all_df["energy_joules"] - RAM_WATTS * all_df["wall_time_sec"]
)
all_df["sensitivity_edp"] = (
    all_df["sensitivity_energy_j"] * all_df["wall_time_sec"]
)

# Do not rank configurations with nonpositive hypothetical energy.
all_df["sensitivity_edp"] = all_df["sensitivity_edp"].where(
    all_df["sensitivity_energy_j"] > 0
)

# Compare strategies only within the same workload, core count and repeat.
group_cols = ["track", "workload_id", "n_cores", "repeat_idx", "strategy"]
summary = all_df.groupby(group_cols, dropna=False).agg(
    original_edp=("original_edp", "median"),
    sensitivity_edp=("sensitivity_edp", "median"),
    valid_sensitivity_runs=("sensitivity_edp", "count"),
).reset_index()

# For fair matched comparisons, retain only workload/core/repeat groups
# where at least two strategies have results.
match_cols = ["track", "workload_id", "n_cores", "repeat_idx"]
summary["strategy_count"] = summary.groupby(match_cols)["strategy"].transform("nunique")
matched = summary[summary["strategy_count"] >= 2].copy()

winner_cols = match_cols
def winner_table(metric, label):
    usable = matched.dropna(subset=[metric]).copy()
    idx = usable.groupby(winner_cols)[metric].idxmin()
    winners = usable.loc[idx, winner_cols + ["strategy"]].rename(
        columns={"strategy": label}
    )
    return winners

original_winners = winner_table("original_edp", "original_winner")
sensitivity_winners = winner_table("sensitivity_edp", "sensitivity_winner")
comparison = original_winners.merge(sensitivity_winners, on=winner_cols, how="outer")
comparison["winner_changed"] = (
    comparison["original_winner"] != comparison["sensitivity_winner"]
)

out_dir = ROOT / "results" / "memberC"
out_dir.mkdir(parents=True, exist_ok=True)
summary_path = out_dir / "energy_sensitivity_report.csv"
winner_path = out_dir / "energy_sensitivity_winners.csv"

summary.to_csv(summary_path, index=False)
comparison.to_csv(winner_path, index=False)

print("Sensitivity assumption:", RAM_WATTS, "W")
print("Valid runs:", len(all_df))
print("Matched workload/core/repeat groups:", len(comparison))
print("Winner changes:", int(comparison["winner_changed"].sum()))
print("Winner-change fraction:",
      round(comparison["winner_changed"].mean(), 4) if len(comparison) else "N/A")
print("Groups with no positive sensitivity estimate:",
      int(comparison["sensitivity_winner"].isna().sum()))
print("Reports written:")
print(summary_path)
print(winner_path)
print("\nReminder: sensitivity values are hypothetical, not measured CPU-only energy.")
