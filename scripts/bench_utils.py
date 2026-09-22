"""
Shared benchmarking infrastructure for the Energy-Aware SHAP Scheduler project.
Used identically by Member A (TreeSHAP) and Member B (KernelSHAP) tracks.
Contains NO model-, dataset-, or SHAP-specific logic.
"""

import atexit
import csv
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from codecarbon import EmissionsTracker

CSV_FIELDS = [
    "run_id", "track", "strategy", "model", "n_rows", "n_features",
    "background_size", "n_cores", "repeat_idx", "wall_time_sec",
    "energy_joules", "machine_id", "timestamp",
]

DEFAULT_CSV_PATH = Path(__file__).resolve().parents[1] / "results" / "results.csv"

_tracker = None  # process-wide singleton — hardware detection runs once


def _get_tracker():
    """
    Lazily create ONE EmissionsTracker for the whole benchmark run.
    Hardware detection (including the Windows physical-CPU subprocess call
    that was hanging) happens here, once, instead of on every
    run_and_measure() call.
    """
    global _tracker
    if _tracker is None:
        _tracker = EmissionsTracker(
            measure_power_secs=1,
            save_to_file=False,
            log_level="error",
            allow_multiple_runs=True,
        )
        _tracker.start()
        atexit.register(_shutdown_tracker)
    return _tracker


def _shutdown_tracker():
    """Ensure the tracker is cleanly stopped when the process exits."""
    global _tracker
    if _tracker is not None:
        try:
            _tracker.stop()
        except Exception:
            pass
        _tracker = None


def make_chunks(rows, n_cores):
    """
    Split `rows` into up to n_cores roughly-equal chunks.
    Handles n_cores=1 and avoids empty chunks when len(rows) < n_cores.
    Works for both lists and pandas DataFrames.
    """
    n_cores = max(1, int(n_cores))
    n = len(rows)
    if n_cores == 1 or n <= 1:
        return [rows]

    n_chunks = min(n_cores, n)
    base, remainder = divmod(n, n_chunks)
    chunks = []
    start = 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        end = start + size
        chunk = rows.iloc[start:end] if hasattr(rows, "iloc") else rows[start:end]
        chunks.append(chunk)
        start = end
    return chunks


def get_machine_id():
    """Machine identifier derived from the hostname — never hard-coded."""
    return socket.gethostname()


def get_timestamp():
    """UTC timestamp, ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def run_and_measure(fn, *, strategy, track, model, n_rows, n_features,
                     n_cores, repeat_idx, background_size=None,
                     csv_path=DEFAULT_CSV_PATH):
    """
    Generic timing + energy wrapper.

    fn : zero-argument callable
        The caller closes over its own chunk/model/n_cores before passing
        this in — bench_utils never touches SHAP, sklearn, or XGBoost.

    Runs fn(), times it with time.perf_counter(), measures energy via a
    CodeCarbon task on the shared tracker, appends one row to the shared
    CSV, and returns that row.
    """
    tracker = _get_tracker()
    task_name = f"{track}-{strategy}-{model}-{n_rows}r-{n_cores}c-rep{repeat_idx}"

    tracker.start_task(task_name)
    start_time = time.perf_counter()
    try:
        fn()
    finally:
        wall_time_sec = time.perf_counter() - start_time
        emissions_data = tracker.stop_task()

    energy_joules = emissions_data.energy_consumed * 3_600_000  # kWh -> J

    row = {
        "run_id": str(uuid.uuid4()),
        "track": track,
        "strategy": strategy,
        "model": model,
        "n_rows": n_rows,
        "n_features": n_features,
        "background_size": background_size,
        "n_cores": n_cores,
        "repeat_idx": repeat_idx,
        "wall_time_sec": wall_time_sec,
        "energy_joules": energy_joules,
        "machine_id": get_machine_id(),
        "timestamp": get_timestamp(),
    }

    _append_row(row, csv_path)
    return row


def _append_row(row, csv_path):
    """Append one row to the shared CSV, writing the header if the file is new."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)