"""
CodeCarbon diagnostic for Energy-Aware-SHAP.

Run from the project virtual environment:
    python diagnose_codecarbon.py

This does NOT modify your benchmark results.
It only performs a short diagnostic and prints the measurement
backend/mode that CodeCarbon exposes.
"""

import os
import platform
import sys
import time

import codecarbon
from codecarbon import EmissionsTracker

try:
    import psutil
except Exception:
    psutil = None


def cpu_work(seconds=8):
    """CPU-bound workload long enough for CodeCarbon to sample."""
    end = time.perf_counter() + seconds
    x = 0.123456789
    while time.perf_counter() < end:
        x = (x * 1.000000119 + 0.3141592653) % 1000000.0
        x = (x * 0.999999881 + 0.2718281828) % 1000000.0
    return x


def main():
    print("=" * 72)
    print("ENERGY-AWARE-SHAP — CODECARBON DIAGNOSTIC")
    print("=" * 72)
    print(f"Python          : {sys.version.split()[0]}")
    print(f"CodeCarbon      : {getattr(codecarbon, '__version__', 'unknown')}")
    print(f"OS              : {platform.platform()}")
    print(f"Processor       : {platform.processor()}")
    print(f"CPU count       : {os.cpu_count()}")

    if psutil is not None:
        print(f"psutil CPU freq : {psutil.cpu_freq()}")

    # Explicitly use machine mode because this is a diagnostic of the
    # mode used by the existing benchmark unless the benchmark proves otherwise.
    tracker = EmissionsTracker(
        project_name="energy_aware_shap_codecarbon_diagnostic",
        tracking_mode="machine",
        measure_power_secs=1,
        save_to_file=False,
        log_level="error",
    )

    tracker.start()
    start = time.perf_counter()
    result = cpu_work(8)
    wall = time.perf_counter() - start
    tracker.stop()

    data = getattr(tracker, "final_emissions_data", None)

    print("\n--- MEASUREMENT ---")
    print(f"Diagnostic wall time       : {wall:.3f} s")
    print(f"Energy consumed            : {getattr(data, 'energy_consumed', None)} kWh")
    print(f"CPU energy                 : {getattr(data, 'cpu_energy', None)} kWh")
    print(f"RAM energy                 : {getattr(data, 'ram_energy', None)} kWh")
    print(f"CPU power                  : {getattr(data, 'cpu_power', None)} W")
    print(f"RAM power                  : {getattr(data, 'ram_power', None)} W")
    print(f"CPU utilization            : {getattr(data, 'cpu_utilization_percent', None)} %")
    print(f"RAM utilization            : {getattr(data, 'ram_utilization_percent', None)} %")
    print(f"Tracking mode              : {getattr(data, 'tracking_mode', None)}")
    print(f"CPU model                  : {getattr(data, 'cpu_model', None)}")
    print(f"Duration recorded          : {getattr(data, 'duration', None)} s")
    print(f"Result checksum            : {result:.6f}")

    print("\n--- RAW FINAL DATA OBJECT ---")
    if data is not None:
        # Print only public attributes so we can see backend-specific fields
        # without dumping internal/private implementation details.
        attrs = {}
        for name in dir(data):
            if name.startswith("_"):
                continue
            try:
                value = getattr(data, name)
            except Exception:
                continue
            if callable(value):
                continue
            attrs[name] = value
        for name in sorted(attrs):
            print(f"{name}: {attrs[name]}")

    print("\n" + "=" * 72)
    print("IMPORTANT: This diagnostic does not prove what the OLD benchmark used.")
    print("It tells us what the CURRENT CodeCarbon 3.3.1 installation can expose.")
    print("Send the COMPLETE terminal output back here.")
    print("=" * 72)


if __name__ == "__main__":
    main()
