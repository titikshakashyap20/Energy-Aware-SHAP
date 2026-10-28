"""
Shared benchmarking infrastructure for the Energy-Aware SHAP Scheduler project.

Used identically by Member A (TreeSHAP) and Member B (KernelSHAP) tracks.

Contains NO model-, dataset-, or SHAP-specific logic.

IMPORTANT:
- Member A's existing run_and_measure() behavior is preserved.
- Member B Stage 5 uses measure_call() when it needs the measured function's
  return value for validation.
"""

import atexit
import csv
import importlib.metadata
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
    # Diagnostic schema for new Member A records. Historical rows are left blank.
    "runtime_sec", "requested_cores", "effective_cores", "workload_id",
    "configuration_id", "run_status", "error", "energy_units",
    "energy_backend", "tracking_mode", "cpu_energy_joules", "ram_energy_joules",
    "gpu_energy_joules", "total_energy_joules", "cpu_energy_status",
    "ram_energy_status", "gpu_energy_status", "python_version", "codecarbon_version",
    "shap_version", "model_library_version",
]

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parents[1] / "results" / "results.csv"
)

_tracker = None


def _get_tracker():
    """
    Lazily create ONE EmissionsTracker for the whole benchmark run.

    This is the same tracker configuration used by the existing Member A
    infrastructure. Hardware detection happens once per process.
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

        chunk = (
            rows.iloc[start:end]
            if hasattr(rows, "iloc")
            else rows[start:end]
        )

        chunks.append(chunk)
        start = end

    return chunks


def get_machine_id():
    """Machine identifier derived from the hostname — never hard-coded."""
    return socket.gethostname()


def get_timestamp():
    """UTC timestamp, ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# NEW: generic measurement API for Member B
# ---------------------------------------------------------------------------

_CC_EXTRA_FIELDS = (
    "cpu_energy",
    "ram_energy",
    "gpu_energy",
    "cpu_power",
    "ram_power",
    "gpu_power",
    "cpu_utilization_percent",
    "tracking_mode",
    "cpu_model",
    "cpu_count",
    "country_iso_code",
    "codecarbon_version",
)


def measure_call(fn, *, task_name, raise_on_error=True):
    """
    Measure exactly one zero-argument callable.

    IMPORTANT FOR MEMBER B:
    The return value of fn() is preserved so benchmark.py can validate the
    KernelSHAP output AFTER the measured interval.

    The measured interval contains only:

        tracker.start_task()
        timer start
        fn()
        timer stop
        tracker.stop_task()

    Validation, CSV writing, Ray cleanup, etc. happen outside this function.

    Returns:
        (result, measurement_dict)

    measurement_dict contains:
        wall_time_sec
        energy_kwh
        emissions_kg
        success
        error
        codecarbon_error

    plus available CodeCarbon component fields.

    This function does NOT replace run_and_measure().
    """

    tracker = _get_tracker()

    result = None
    error = ""
    success = True

    tracker.start_task(task_name)
    start_time = time.perf_counter()

    try:
        result = fn()

    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"

        if raise_on_error:
            raise

    finally:
        wall_time_sec = time.perf_counter() - start_time

        try:
            emissions_data = tracker.stop_task()
        except Exception as exc:
            emissions_data = None
            cc_error = f"{type(exc).__name__}: {exc}"
        else:
            cc_error = (
                ""
                if emissions_data is not None
                else "stop_task returned None"
            )

    def _num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    measurement = {
        "wall_time_sec": wall_time_sec,
        "energy_kwh": _num(
            getattr(emissions_data, "energy_consumed", None)
        ),
        "emissions_kg": _num(
            getattr(emissions_data, "emissions", None)
        ),
        "success": success,
        "error": error,
        "codecarbon_error": cc_error,
    }

    for name in _CC_EXTRA_FIELDS:
        measurement[name] = getattr(
            emissions_data,
            name,
            None,
        )

    return result, measurement


# ---------------------------------------------------------------------------
# EXISTING MEMBER A API
# ---------------------------------------------------------------------------

def run_and_measure(
    fn,
    *,
    strategy,
    track,
    model,
    n_rows,
    n_features,
    n_cores,
    repeat_idx,
    background_size=None,
    csv_path=DEFAULT_CSV_PATH,
):
    """
    Generic timing + energy wrapper.

    THIS FUNCTION IS KEPT COMPATIBLE WITH THE ORIGINAL MEMBER A VERSION.

    Member A can continue using this function exactly as before.

    fn : zero-argument callable
        The caller closes over its own chunk/model/n_cores before passing
        this in — bench_utils never touches SHAP, sklearn, or XGBoost.

    Runs fn(), times it with time.perf_counter(), measures energy via a
    CodeCarbon task on the shared tracker, appends one row to the shared
    CSV, and returns that row.
    """

    tracker = _get_tracker()

    task_name = (
        f"{track}-{strategy}-{model}-"
        f"{n_rows}r-{n_cores}c-rep{repeat_idx}"
    )

    tracker.start_task(task_name)

    start_time = time.perf_counter()
    run_error = None
    run_exception = None

    try:
        fn()
    except Exception as exc:
        run_exception = exc
        run_error = f"{type(exc).__name__}: {exc}"

    finally:
        wall_time_sec = time.perf_counter() - start_time
        emissions_data = tracker.stop_task()

    energy_joules = (
        emissions_data.energy_consumed * 3_600_000
    )

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
        "runtime_sec": wall_time_sec,
        "requested_cores": n_cores,
        "effective_cores": n_cores,
        "workload_id": f"{track}|{model}|{n_rows}" if background_size is None else f"{track}|{model}|{n_features}|{n_rows}|{background_size}",
        "configuration_id": f"{track}|{model}|{n_rows}|{strategy}|{n_cores}|{repeat_idx}",
        "run_status": "failed" if run_error else "success",
        "error": run_error or "",
        "energy_units": "joules",
        "energy_backend": "CodeCarbon",
        "tracking_mode": "unknown",
        "cpu_energy_joules": None,
        "ram_energy_joules": None,
        "gpu_energy_joules": None,
        "total_energy_joules": energy_joules,
        "cpu_energy_status": "unavailable",
        "ram_energy_status": "unavailable",
        "gpu_energy_status": "unsupported_or_unavailable",
        "python_version": __import__("platform").python_version(),
        "codecarbon_version": _package_version("codecarbon"),
        "shap_version": _package_version("shap"),
        "model_library_version": (_package_version("xgboost") if "xgb" in str(model).lower() else _package_version("scikit-learn")),
    }

    _append_row(row, csv_path)

    if run_error:
        raise run_exception

    return row


def _append_row(row, csv_path):
    """Append one row to the shared CSV, writing the header if the file is new."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    fieldnames = list(CSV_FIELDS)

    # Preserve old rows while extending an old-format Member A CSV on the first
    # append using the diagnostic schema. Missing historical fields stay blank.
    if file_exists and csv_path.stat().st_size:
        with open(csv_path, "r", newline="") as existing:
            reader = csv.DictReader(existing)
            old_fields = reader.fieldnames or []
            old_rows = list(reader)
        if old_fields != CSV_FIELDS:
            fieldnames = list(dict.fromkeys(old_fields + CSV_FIELDS))
            with open(csv_path, "w", newline="") as migrated:
                writer = csv.DictWriter(migrated, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(old_rows)
            file_exists = True
        else:
            fieldnames = old_fields

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
