
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent

CSV_FILES = [
    ROOT / "results" / "memberA" / "results.csv",
    ROOT / "results" / "memberB" / "kernelshap_benchmark.csv",
]

DIAGNOSTIC_COLUMNS = [
    "cc_cpu_energy_kwh",
    "cc_ram_energy_kwh",
    "cc_gpu_energy_kwh",
    "cc_cpu_power_w",
    "cc_ram_power_w",
    "cc_cpu_utilization_pct",
    "cc_tracking_mode",
    "cc_cpu_model",
    "cc_error",
]

def main():
    for path in CSV_FILES:
        print("\n" + "=" * 65)
        print(f"FILE: {path.relative_to(ROOT)}")

        if not path.exists():
            print("NOT FOUND. Check the CSV filename.")
            continue

        df = pd.read_csv(path)
        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        print("\nEnergy-related columns:")
        energy_cols = [
            c for c in df.columns
            if "energy" in c.lower()
            or "power" in c.lower()
            or "tracking" in c.lower()
            or "utilization" in c.lower()
            or "codecarbon" in c.lower()
            or c == "cc_error"
        ]
        print(energy_cols)

        print("\nDiagnostic-field completeness:")
        for col in DIAGNOSTIC_COLUMNS:
            if col not in df.columns:
                print(f"  {col}: COLUMN ABSENT")
            else:
                values = df[col]
                populated = values.notna() & (
                    values.astype(str).str.strip() != ""
                )
                print(
                    f"  {col}: {populated.sum()}/{len(df)} populated"
                )

        time_col = next(
            (c for c in ["wall_time_sec", "elapsed_seconds"]
             if c in df.columns),
            None,
        )
        energy_col = next(
            (c for c in ["energy_joules"]
             if c in df.columns),
            None,
        )

        if time_col and energy_col:
            time = pd.to_numeric(df[time_col], errors="coerce")
            energy = pd.to_numeric(df[energy_col], errors="coerce")
            valid = (
                time.notna() & energy.notna()
                & (time > 0) & (energy > 0)
            )
            implied_power = energy[valid] / time[valid]

            print("\nBasic measurement sanity check:")
            print(f"  Valid positive time/energy rows: {valid.sum()}")
            if len(implied_power):
                print(
                    "  Implied power (energy / time), W:"
                    f"\n    median = {implied_power.median():.3f}"
                    f"\n    min    = {implied_power.min():.3f}"
                    f"\n    max    = {implied_power.max():.3f}"
                )
        else:
            print("\nCould not identify both time and energy columns.")

if __name__ == "__main__":
    main()
