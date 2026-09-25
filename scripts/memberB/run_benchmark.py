"""
Member B - Stage 5: Benchmark / Measurement Layer
==================================================

KernelSHAP benchmark for the Energy-Aware SHAP Scheduler project.

IMPORTANT:
- Uses the EXISTING shared scripts/bench_utils.py.
- Does NOT replace or modify Member A's bench_utils.py.
- Uses bench_utils.run_and_measure() for timing + CodeCarbon energy.
- Does not contain model-, dataset-, or SHAP-specific logic in bench_utils.
- Stage 4 parallel strategies are treated as black-box run_* functions.

Strategies:
    serial
    multiprocessing
    threading
    joblib
    ray

Models:
    SVM-RBF
    LogisticRegression

Feature counts:
    15
    30

Full benchmark:
    cores       = [1, 2, 4, 8]
    rows        = 100
    background  = [25, 50, 100]
    repeats     = 3

Typical usage from repository root:

    python scripts/memberB/benchmark.py --inspect-memberb

    python scripts/memberB/benchmark.py --smoke-test

    python scripts/memberB/benchmark.py --dry-run

    python scripts/memberB/benchmark.py

    python scripts/memberB/benchmark.py --full --dry-run

    python scripts/memberB/benchmark.py --full --max-runs 10

Existing successful runs are skipped automatically.
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

import argparse
import csv
import gc
import importlib
import inspect
import math
import os
import random
import shutil
import socket
import sys
import tempfile
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path


# ============================================================================
# PATH SETUP
# ============================================================================

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
REPO_ROOT = SCRIPTS_DIR.parent

# scripts/ must be importable so that:
#     import bench_utils
# works from this file.
#
# memberB/ is also added so that:
#     import parallel_strategies
# works.

for _p in (str(HERE), str(SCRIPTS_DIR)):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


# ============================================================================
# PROJECT CONSTANTS
# ============================================================================

TRACK = "kernelshap"

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "memberB"
    / "kernelshap_benchmark.csv"
)

ALL_MODELS = [
    "SVM-RBF",
    "LogisticRegression",
]

ALL_FEATURES = [
    15,
    30,
]

ALL_STRATEGIES = [
    "serial",
    "multiprocessing",
    "threading",
    "joblib",
    "ray",
]

STRATEGY_FUNC_NAMES = {
    "serial": "run_serial",
    "multiprocessing": "run_multiprocessing",
    "threading": "run_threading",
    "joblib": "run_joblib",
    "ray": "run_ray",
}


# ============================================================================
# BENCHMARK GRIDS
# ============================================================================

# Official project core grid.
FULL_CORES = [
    1,
    2,
    4,
    8,
]

# Full benchmark.
FULL = {
    "cores": FULL_CORES,
    "n_rows": 100,
    "background": [25, 50, 100],
    "repeats": 3,
    "models": ALL_MODELS,
    "features": ALL_FEATURES,
}

# Small pilot benchmark.
PILOT = {
    "cores": [1, 2, 4],
    "n_rows": 16,
    "background": [25],
    "repeats": 1,
    "models": ALL_MODELS,
    "features": [15],
}


# ============================================================================
# OUTPUT CSV
# ============================================================================

CSV_FIELDS = [
    "run_id",
    "timestamp",
    "track",
    "strategy",
    "model",
    "feature_count",
    "n_cores",
    "cores_effective",
    "n_rows",
    "background_size",
    "repeat_idx",
    "elapsed_seconds",
    "energy_kwh",
    "energy_joules",
    "emissions_kg",
    "success",
    "error",
    "output_shape",
    "additivity_ok",
    "max_additivity_error",
    "cc_cpu_energy_kwh",
    "cc_ram_energy_kwh",
    "cc_gpu_energy_kwh",
    "cc_cpu_power_w",
    "cc_ram_power_w",
    "cc_gpu_power_w",
    "cc_cpu_utilization_pct",
    "cc_tracking_mode",
    "cc_cpu_model",
    "cc_error",
    "ray_leftover_pids",
    "pin_blas_threads",
    "blas_threads_per_worker",
    "machine_id",
    "cpu_count",
    "seed",
    "shap_version",
    "codecarbon_version",
]


class AdapterError(RuntimeError):
    """Member B data/model interface could not be connected."""


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def log(msg=""):
    print(msg, flush=True)


def warn(msg):
    print(
        f"WARNING: {msg}",
        file=sys.stderr,
        flush=True,
    )


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clean_text(value, limit=400):
    return " ".join(str(value).split())[:limit]


def cell(value):
    """
    Convert None / NaN to an empty CSV cell.
    """
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return value


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and math.isfinite(value)
    )


# ============================================================================
# CSV WRITER
# ============================================================================

class ResultWriter:

    def __init__(self, path):
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._check_or_create_header()

    def _check_or_create_header(self):

        if (
            not self.path.exists()
            or self.path.stat().st_size == 0
        ):
            with open(
                self.path,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:

                csv.writer(f).writerow(
                    CSV_FIELDS
                )

            return

        with open(
            self.path,
            newline="",
            encoding="utf-8",
        ) as f:

            header = next(
                csv.reader(f),
                [],
            )

        if header != CSV_FIELDS:

            raise SystemExit(
                f"ERROR: {self.path} already exists "
                "with a different column layout.\n"
                f"Existing: {header}\n"
                f"Expected: {CSV_FIELDS}\n"
                "Refusing to append."
            )

    def append(self, row):

        with open(
            self.path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    cell(row.get(column))
                    for column in CSV_FIELDS
                ]
            )

            f.flush()
            os.fsync(f.fileno())


# ============================================================================
# RESUME SUPPORT
# ============================================================================

def run_key(
    strategy,
    model,
    feature_count,
    n_cores,
    n_rows,
    background_size,
    repeat_idx,
):

    return (
        str(strategy),
        str(model),
        int(feature_count),
        int(n_cores),
        int(n_rows),
        int(background_size),
        int(repeat_idx),
    )


def load_done_keys(path):

    done = set()

    path = Path(path)

    if not path.exists():
        return done

    with open(
        path,
        newline="",
        encoding="utf-8",
    ) as f:

        for row in csv.DictReader(f):

            try:

                success = str(
                    row["success"]
                ).strip().lower()

                if success not in (
                    "true",
                    "1",
                ):
                    continue

                done.add(
                    run_key(
                        row["strategy"],
                        row["model"],
                        row["feature_count"],
                        row["n_cores"],
                        row["n_rows"],
                        row["background_size"],
                        row["repeat_idx"],
                    )
                )

            except (
                KeyError,
                ValueError,
                TypeError,
            ):
                continue

    return done


# ============================================================================
# MEMBER B ADAPTER
# ============================================================================

DATA_CANDIDATES = [
    "load_memberB_data",
    "prepare_data",
    "load_and_prepare",
    "get_data",
    "load_data",
]

FEATURE_CANDIDATES = [
    "select_features",
    "select_top_features",
    "get_top_features",
    "top_k_features",
    "get_feature_subset",
    "select_top_k",
]

MODEL_CANDIDATES = {

    "SVM-RBF": [
        "train_svm",
        "train_svm_rbf",
        "train_svc",
        "build_svm",
        "fit_svm",
    ],

    "LogisticRegression": [
        "train_logistic_regression",
        "train_logreg",
        "train_logistic",
        "train_lr",
        "build_logistic_regression",
        "fit_logistic_regression",
    ],
}


def _public_callables(module):

    output = []

    for name, obj in vars(module).items():

        if name.startswith("_"):
            continue

        if not callable(obj):
            continue

        if getattr(
            obj,
            "__module__",
            None,
        ) != module.__name__:
            continue

        try:
            signature = inspect.signature(obj)

            output.append(
                f"{name}{signature}"
            )

        except (
            TypeError,
            ValueError,
        ):
            output.append(name)

    return sorted(output)


def _pick(module, candidates, what):

    for name in candidates:

        if hasattr(module, name):

            return getattr(
                module,
                name,
            )

    raise AdapterError(
        f"Could not find {what} in "
        f"{module.__name__}.py.\n"
        f"Tried: {candidates}\n"
        f"Available: {_public_callables(module)}"
    )


# ============================================================================
# WORKLOAD
# ============================================================================

class Workload:

    def __init__(
        self,
        model,
        rows,
        background,
        feature_count,
        model_name,
    ):

        self.model = model
        self.rows = rows
        self.background = background
        self.feature_count = feature_count
        self.model_name = model_name


class WorkloadFactory:

    def __init__(self, seed):

        self.seed = seed

        self._data = None
        self._features = {}
        self._models = {}
        self._workloads = {}

    # ---------------------------------------------------------------------

    def _load(self):
        if self._data is None:
            dl = importlib.import_module("data_loading")

            # Member B's actual data-loading API
            if hasattr(dl, "load_memberB_data"):
                data = dl.load_memberB_data(
                    test_size=0.2,
                    random_state=self.seed,
                    n_top=15,
                )

                # MemberBData contains the already-prepared 15- and 30-feature
                # train/test datasets.
                self._data = (
                    data.X_train_30,
                    data.X_test_30,
                    data.y_train,
                    data.y_test,
                )
            else:
                # Generic fallback for older layouts.
                fn = _pick(
                    dl,
                    DATA_CANDIDATES,
                    "data preparation function",
                )

                out = fn()

                if not (
                    isinstance(out, (tuple, list))
                    and len(out) == 4
                ):
                    raise AdapterError(
                        f"{fn.__name__}() should return "
                        "(X_train, X_test, y_train, y_test); "
                        f"got {type(out).__name__}. "
                        "Adjust WorkloadFactory._load()."
                    )

                self._data = tuple(out)

        return self._data

    # ---------------------------------------------------------------------

    def _feature_columns(self, feature_count):
        if feature_count in self._features:
            return self._features[feature_count]

        dl = importlib.import_module("data_loading")

        # Member B already defines the official feature subsets.
        if hasattr(dl, "load_memberB_data"):
            data = dl.load_memberB_data(
                test_size=0.2,
                random_state=self.seed,
                n_top=15,
            )

            if feature_count == 15:
                cols = list(data.feature_names_15)

            elif feature_count == 30:
                cols = list(data.feature_names_30)

            else:
                raise AdapterError(
                    f"Member B supports feature counts 15 and 30; "
                    f"got {feature_count}"
                )

            if len(cols) != feature_count:
                raise AdapterError(
                    f"Member B feature list contains {len(cols)} columns, "
                    f"expected {feature_count}"
                )

            self._features[feature_count] = cols
            return cols

        # Generic fallback for older layouts.
        X_train, _, y_train, _ = self._load()

        if X_train.shape[1] == feature_count:
            cols = list(X_train.columns)
        else:
            mod = importlib.import_module("models")

            try:
                sel = _pick(
                    mod,
                    FEATURE_CANDIDATES,
                    "feature-selection function",
                )
            except AdapterError:
                sel = _pick(
                    dl,
                    FEATURE_CANDIDATES,
                    "feature-selection function",
                )

            res = sel(
                X_train,
                y_train,
                feature_count,
            )

            cols = (
                list(res.columns)
                if hasattr(res, "columns")
                else list(res)
            )

            if cols and not isinstance(cols[0], str):
                cols = [X_train.columns[i] for i in cols]

        if len(cols) != feature_count:
            raise AdapterError(
                f"feature selection gave {len(cols)} columns, "
                f"expected {feature_count}"
            )

        self._features[feature_count] = cols
        return cols

    # ---------------------------------------------------------------------

    def _model(
        self,
        model_name,
        feature_count,
    ):

        key = (
            model_name,
            feature_count,
        )

        if key in self._models:

            return self._models[key]

        (
            X_train,
            _,
            y_train,
            _,
        ) = self._load()

        columns = self._feature_columns(
            feature_count
        )

        models_module = importlib.import_module(
            "models"
        )

        trainer = _pick(
            models_module,
            MODEL_CANDIDATES[model_name],
            f"{model_name} training function",
        )

        log(
            f"  Training {model_name} "
            f"with {feature_count} features..."
        )

        self._models[key] = trainer(
            X_train[columns],
            y_train,
        )

        return self._models[key]

    # ---------------------------------------------------------------------

    def get(
        self,
        model_name,
        feature_count,
        n_rows,
        background_size,
    ):

        key = (
            model_name,
            feature_count,
            n_rows,
            background_size,
        )

        if key in self._workloads:

            return self._workloads[key]

        (
            X_train,
            X_test,
            _,
            _,
        ) = self._load()

        columns = self._feature_columns(
            feature_count
        )

        model = self._model(
            model_name,
            feature_count,
        )

        rows = X_test.sample(
            n=min(
                n_rows,
                len(X_test),
            ),
            random_state=self.seed,
        )[columns]

        background = X_train.sample(
            n=min(
                background_size,
                len(X_train),
            ),
            random_state=self.seed + 1,
        )[columns]

        workload = Workload(
            model=model,
            rows=rows,
            background=background,
            feature_count=feature_count,
            model_name=model_name,
        )

        self._workloads[key] = workload

        return workload


# ============================================================================
# STRATEGY LOADING
# ============================================================================

def load_strategy_funcs():

    parallel_strategies = importlib.import_module(
        "parallel_strategies"
    )

    functions = {}

    for strategy, function_name in STRATEGY_FUNC_NAMES.items():

        if not hasattr(
            parallel_strategies,
            function_name,
        ):

            raise AdapterError(
                f"parallel_strategies."
                f"{function_name} not found.\n"
                f"Available: "
                f"{_public_callables(parallel_strategies)}"
            )

        functions[strategy] = getattr(
            parallel_strategies,
            function_name,
        )

    metadata = {
        "pin_blas_threads": getattr(
            parallel_strategies,
            "PIN_BLAS_THREADS",
            None,
        ),

        "blas_threads_per_worker": getattr(
            parallel_strategies,
            "BLAS_THREADS_PER_WORKER",
            None,
        ),
    }

    return functions, metadata


# ============================================================================
# INSPECTION
# ============================================================================

def inspect_memberb():

    log(
        f"Member B directory: {HERE}"
    )

    for module_name in (
        "data_loading",
        "models",
        "explain",
        "parallel_strategies",
    ):

        try:

            module = importlib.import_module(
                module_name
            )

        except Exception as exc:

            log(
                f"\n[{module_name}] IMPORT FAILED: "
                f"{type(exc).__name__}: {exc}"
            )

            continue

        log(
            f"\n[{module_name}] "
            f"({getattr(module, '__file__', '?')})"
        )

        for signature in _public_callables(
            module
        ):

            log(
                f"  def {signature}"
            )

        constants = {
            key: value
            for key, value
            in vars(module).items()
            if (
                key.isupper()
                and isinstance(
                    value,
                    (
                        int,
                        float,
                        str,
                        bool,
                        type(None),
                    ),
                )
            )
        }

        if constants:

            log(
                f"  constants: {constants}"
            )


# ============================================================================
# OUTPUT VALIDATION
# ============================================================================

def validate_output(
    output,
    workload,
    tolerance,
    skip_additivity,
):

    import numpy as np

    if not isinstance(
        output,
        np.ndarray,
    ):

        return (
            False,
            "",
            None,
            None,
            f"output is "
            f"{type(output).__name__}, "
            "not ndarray",
        )

    shape_string = str(
        tuple(output.shape)
    )

    n_rows = len(
        workload.rows
    )

    n_features = (
        workload.feature_count
    )

    # KernelSHAP expected shape:
    #
    # (n_rows, n_features, n_classes)

    if (
        output.ndim != 3
        or output.shape[0] != n_rows
        or output.shape[1] != n_features
    ):

        return (
            False,
            shape_string,
            None,
            None,
            f"shape {shape_string} != "
            f"({n_rows}, {n_features}, "
            "n_classes)",
        )

    classes = getattr(
        workload.model,
        "classes_",
        None,
    )

    if (
        classes is not None
        and output.shape[2] != len(classes)
    ):

        return (
            False,
            shape_string,
            None,
            None,
            f"n_classes {output.shape[2]} "
            f"!= len(model.classes_) "
            f"{len(classes)}",
        )

    if not np.isfinite(
        output
    ).all():

        return (
            False,
            shape_string,
            None,
            None,
            "non-finite SHAP values",
        )

    # ---------------------------------------------------------
    # Additivity
    # ---------------------------------------------------------

    if skip_additivity:

        return (
            True,
            shape_string,
            None,
            None,
            "",
        )

    if not hasattr(
        workload.model,
        "predict_proba",
    ):

        return (
            True,
            shape_string,
            None,
            None,
            "additivity not checked "
            "(model has no predict_proba)",
        )

    try:

        predictions = np.asarray(
            workload.model.predict_proba(
                workload.rows
            )
        )

        background_predictions = np.asarray(
            workload.model.predict_proba(
                workload.background
            )
        )

        base = background_predictions.mean(
            axis=0
        )

        reconstructed = (
            base[None, :]
            + output.sum(axis=1)
        )

        error = float(
            np.max(
                np.abs(
                    reconstructed
                    - predictions
                )
            )
        )

    except Exception as exc:

        return (
            True,
            shape_string,
            None,
            None,
            "additivity not evaluated: "
            f"{type(exc).__name__}: {exc}",
        )

    if error > tolerance:

        return (
            False,
            shape_string,
            False,
            error,
            f"additivity error "
            f"{error:.3g} > tolerance "
            f"{tolerance:g}",
        )

    return (
        True,
        shape_string,
        True,
        error,
        "",
    )


# ============================================================================
# RAY PROCESS MANAGEMENT
# ============================================================================

def ray_pids():

    try:
        import psutil
    except ImportError:
        return set()

    current_pid = os.getpid()

    found = set()

    for process in psutil.process_iter(
        [
            "pid",
            "name",
            "cmdline",
        ]
    ):

        try:

            pid = process.info["pid"]

            if pid == current_pid:
                continue

            name = (
                process.info["name"]
                or ""
            ).lower().replace(
                ".exe",
                "",
            )

            command = " ".join(
                process.info["cmdline"]
                or []
            ).lower()

            if (
                name in (
                    "raylet",
                    "gcs_server",
                )
                or name.startswith(
                    "ray::"
                )
                or "ray/_private"
                in command
                or "ray\\_private"
                in command
                or "default_worker.py"
                in command
            ):

                found.add(pid)

        except Exception:
            continue

    return found


def ray_cleanup(
    before,
    grace=10.0,
):

    try:

        import psutil

    except ImportError:

        return []

    try:

        import ray

        if ray.is_initialized():

            warn(
                "Ray still initialized "
                "after run; shutting down."
            )

            ray.shutdown()

    except ImportError:

        return []

    deadline = (
        time.time()
        + grace
    )

    leftover = (
        ray_pids()
        - before
    )

    while (
        leftover
        and time.time() < deadline
    ):

        time.sleep(0.5)

        leftover = (
            ray_pids()
            - before
        )

    if leftover:

        warn(
            "Ray processes still alive: "
            f"{sorted(leftover)}"
        )

        processes = []

        for pid in leftover:

            try:

                processes.append(
                    psutil.Process(pid)
                )

            except psutil.Error:
                pass

        for process in processes:

            try:

                process.terminate()

            except psutil.Error:
                pass

        _, alive = psutil.wait_procs(
            processes,
            timeout=5,
        )

        for process in alive:

            try:

                process.kill()

            except psutil.Error:
                pass

    return sorted(leftover)


# ============================================================================
# VERSION INFORMATION
# ============================================================================

def versions_info():

    versions = {}

    for package in (
        "shap",
        "codecarbon",
    ):

        try:

            versions[package] = (
                importlib.import_module(
                    package
                ).__version__
            )

        except Exception:

            versions[package] = ""

    return versions


# ============================================================================
# ONE MEASURED RUN
# ============================================================================

def execute_one(
    strategy,
    model_name,
    feature_count,
    n_cores,
    n_rows,
    background_size,
    repeat_idx,
    *,
    factory,
    funcs,
    meta,
    writer,
    seed,
    tolerance,
    skip_additivity,
    machine_id,
    versions,
):
    """
    Execute one complete Stage 5 benchmark run.

    The actual strategy call is measured through the EXISTING
    bench_utils.run_and_measure() API.

    Because the existing run_and_measure() API does not return
    the function result, the SHAP output is captured in a closure.
    """

    import bench_utils

    cores_effective = (
        1
        if strategy == "serial"
        else n_cores
    )

    row = {

        "run_id": str(
            uuid.uuid4()
        ),

        "timestamp": utc_now(),

        "track": TRACK,

        "strategy": strategy,

        "model": model_name,

        "feature_count": feature_count,

        "n_cores": n_cores,

        "cores_effective": cores_effective,

        "n_rows": n_rows,

        "background_size": background_size,

        "repeat_idx": repeat_idx,

        "elapsed_seconds": None,

        "energy_kwh": None,

        "energy_joules": None,

        "emissions_kg": None,

        "success": False,

        "error": "",

        "output_shape": "",

        "additivity_ok": None,

        "max_additivity_error": None,

        "cc_cpu_energy_kwh": None,

        "cc_ram_energy_kwh": None,

        "cc_gpu_energy_kwh": None,

        "cc_cpu_power_w": None,

        "cc_ram_power_w": None,

        "cc_gpu_power_w": None,

        "cc_cpu_utilization_pct": None,

        "cc_tracking_mode": None,

        "cc_cpu_model": None,

        "cc_error": "",

        "ray_leftover_pids": 0,

        "pin_blas_threads": meta.get(
            "pin_blas_threads"
        ),

        "blas_threads_per_worker": meta.get(
            "blas_threads_per_worker"
        ),

        "machine_id": machine_id,

        "cpu_count": os.cpu_count(),

        "seed": seed,

        "shap_version": versions[
            "shap"
        ],

        "codecarbon_version": versions[
            "codecarbon"
        ],
    }

    # ------------------------------------------------------------------
    # Build workload OUTSIDE measured interval.
    # ------------------------------------------------------------------

    try:

        workload = factory.get(
            model_name,
            feature_count,
            n_rows,
            background_size,
        )

    except AdapterError:

        raise

    except Exception as exc:

        row["error"] = clean_text(
            f"workload: "
            f"{type(exc).__name__}: {exc}"
        )

        writer.append(row)

        return row

    # ------------------------------------------------------------------
    # Strategy function.
    # ------------------------------------------------------------------

    strategy_function = funcs[
        strategy
    ]

    # Capture output because existing run_and_measure()
    # only returns its measurement row.

    captured = {
        "output": None
    }

    def measured_call():

        captured["output"] = strategy_function(
            workload.rows,
            workload.model,
            workload.background,
            n_cores,
        )

    # ------------------------------------------------------------------
    # Ray process snapshot BEFORE measured run.
    # ------------------------------------------------------------------

    ray_before = (
        ray_pids()
        if strategy == "ray"
        else None
    )

    gc.collect()

    # ------------------------------------------------------------------
    # Existing bench_utils measurement.
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    # This is deliberately using run_and_measure(), because this is the
    # exact shared API already used for Member A.
    #
    # The temporary CSV prevents the generic Member A CSV schema from being
    # mixed into Member B's richer benchmark CSV.
    # ------------------------------------------------------------------

    temporary_csv = (
        Path(tempfile.gettempdir())
        / "energy_aware_shap_memberB_measurement.csv"
    )

    task_error = None
    measurement_row = None

    try:

        measurement_row = (
            bench_utils.run_and_measure(
                measured_call,
                strategy=strategy,
                track=TRACK,
                model=model_name,
                n_rows=n_rows,
                n_features=feature_count,
                n_cores=n_cores,
                repeat_idx=repeat_idx,
                background_size=background_size,
                csv_path=temporary_csv,
            )
        )

    except Exception as exc:

        task_error = (
            f"{type(exc).__name__}: {exc}"
        )

    # ------------------------------------------------------------------
    # Ray cleanup OUTSIDE measured interval.
    # ------------------------------------------------------------------

    if ray_before is not None:

        try:

            leftover = ray_cleanup(
                ray_before
            )

            row["ray_leftover_pids"] = len(
                leftover
            )

        except Exception as exc:

            warn(
                "Ray cleanup failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Measurement data.
    # ------------------------------------------------------------------

    if measurement_row is not None:

        row["elapsed_seconds"] = (
            measurement_row.get(
                "wall_time_sec"
            )
        )

        row["energy_joules"] = (
            measurement_row.get(
                "energy_joules"
            )
        )

        # Existing bench_utils stores energy in joules.
        # Convert back to kWh only for the richer Member B CSV.

        if is_finite_number(
            row["energy_joules"]
        ):

            row["energy_kwh"] = (
                row["energy_joules"]
                / 3_600_000
            )

    # ------------------------------------------------------------------
    # Validation.
    # ------------------------------------------------------------------

    problems = []

    if task_error is not None:

        problems.append(
            task_error
        )

    else:

        output = captured[
            "output"
        ]

        try:

            (
                valid,
                shape_string,
                additivity_ok,
                max_error,
                message,
            ) = validate_output(
                output,
                workload,
                tolerance,
                skip_additivity,
            )

            row["output_shape"] = (
                shape_string
            )

            row["additivity_ok"] = (
                additivity_ok
            )

            row["max_additivity_error"] = (
                max_error
            )

            if not valid:

                problems.append(
                    f"validation: {message}"
                )

            elif message:

                warn(
                    f"{strategy}/{model_name}: "
                    f"{message}"
                )

        except Exception as exc:

            problems.append(
                "validation exception: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

    # ------------------------------------------------------------------
    # Ray leftovers are a failure condition.
    # ------------------------------------------------------------------

    if row["ray_leftover_pids"]:

        problems.append(
            f"{row['ray_leftover_pids']} "
            "Ray process(es) remained after run"
        )

    # ------------------------------------------------------------------
    # Success.
    # ------------------------------------------------------------------

    row["success"] = (
        len(problems) == 0
    )

    row["error"] = clean_text(
        " | ".join(problems)
    )

    # ------------------------------------------------------------------
    # Write the rich Member B row.
    # ------------------------------------------------------------------

    writer.append(row)

    return row


# ============================================================================
# SUMMARY
# ============================================================================

def print_row_summary(row):

    energy = row.get(
        "energy_kwh"
    )

    elapsed = row.get(
        "elapsed_seconds"
    )

    if is_finite_number(
        energy
    ):

        energy_text = (
            f"{energy:.3e}"
        )

    else:

        energy_text = "n/a"

    if is_finite_number(
        elapsed
    ):

        elapsed_text = (
            f"{elapsed:.2f}s"
        )

    else:

        elapsed_text = "n/a"

    message = (
        f"   -> success={row['success']} "
        f"elapsed={elapsed_text} "
        f"energy_kwh={energy_text}"
    )

    if row.get("error"):

        message += (
            f" error={row['error']}"
        )

    log(message)


# ============================================================================
# PLAN
# ============================================================================

def build_plan(
    models,
    features,
    strategies,
    cores,
    backgrounds,
    n_rows,
    repeats,
    serial_all_cores,
    shuffle,
    seed,
):

    plan = []

    # Repeat-major ordering.
    #
    # This ensures repeat 0 covers the complete grid before repeat 1.
    # This is useful if a long benchmark is interrupted.

    for repeat in range(
        repeats
    ):

        block = []

        for background in backgrounds:

            for model in models:

                for feature_count in features:

                    group = []

                    for strategy in strategies:

                        if (
                            strategy == "serial"
                            and not serial_all_cores
                        ):

                            core_list = [1]

                        else:

                            core_list = cores

                        for core_count in core_list:

                            group.append(
                                (
                                    strategy,
                                    model,
                                    feature_count,
                                    core_count,
                                    n_rows,
                                    background,
                                    repeat,
                                )
                            )

                    if shuffle:

                        random.Random(
                            f"{seed}-{repeat}-"
                            f"{background}-{model}-"
                            f"{feature_count}"
                        ).shuffle(group)

                    block.extend(
                        group
                    )

        plan.extend(
            block
        )

    return plan


# ============================================================================
# CONFIGURATION
# ============================================================================

def resolve_config(args):

    profile = (
        FULL
        if args.full
        else PILOT
    )

    config = {

        "models": (
            args.models
            or profile["models"]
        ),

        "features": (
            args.features
            or profile["features"]
        ),

        "strategies": (
            args.strategies
            or ALL_STRATEGIES
        ),

        "cores": (
            args.cores
            or profile["cores"]
        ),

        "backgrounds": (
            args.background_size
            or profile["background"]
        ),

        "n_rows": (
            args.n_rows
            or profile["n_rows"]
        ),

        "repeats": (
            args.repeats
            or profile["repeats"]
        ),
    }

    cpu_count = (
        os.cpu_count()
        or 1
    )

    if not args.allow_oversubscribe:

        kept = [
            core
            for core in config["cores"]
            if core <= cpu_count
        ]

        dropped = [
            core
            for core in config["cores"]
            if core > cpu_count
        ]

        if dropped:

            warn(
                f"Machine has {cpu_count} "
                f"CPU cores; dropping "
                f"n_cores={dropped}."
            )

        if not kept:

            raise SystemExit(
                "ERROR: no requested "
                "core count fits this machine."
            )

        config["cores"] = kept

    return config


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_benchmark(args):

    config = resolve_config(
        args
    )

    plan = build_plan(
        config["models"],
        config["features"],
        config["strategies"],
        config["cores"],
        config["backgrounds"],
        config["n_rows"],
        config["repeats"],
        args.serial_all_cores,
        args.shuffle,
        args.seed,
    )

    profile_name = (
        "FULL"
        if args.full
        else "PILOT"
    )

    log(
        f"Profile: {profile_name}"
    )

    log(
        f"Output: {args.output}"
    )

    log(
        f"Grid: "
        f"models={config['models']} "
        f"features={config['features']} "
        f"strategies={config['strategies']} "
        f"cores={config['cores']} "
        f"background={config['backgrounds']} "
        f"rows={config['n_rows']} "
        f"repeats={config['repeats']}"
    )

    if (
        "serial"
        in config["strategies"]
        and not args.serial_all_cores
    ):

        log(
            "Serial is run once per condition "
            "with n_cores=1."
        )

    # ------------------------------------------------------------------
    # Resume.
    # ------------------------------------------------------------------

    done = load_done_keys(
        args.output
    )

    todo = [
        item
        for item in plan
        if run_key(*item)
        not in done
    ]

    log(
        f"Planned runs: {len(plan)} "
        f"| already successful: "
        f"{len(plan) - len(todo)} "
        f"| remaining: {len(todo)}"
    )

    if args.max_runs:

        todo = todo[
            :args.max_runs
        ]

        log(
            f"--max-runs {args.max_runs}: "
            f"running {len(todo)} run(s)"
        )

    # ------------------------------------------------------------------
    # Dry run.
    # ------------------------------------------------------------------

    if args.dry_run:

        for index, item in enumerate(
            todo,
            1,
        ):

            (
                strategy,
                model,
                feature_count,
                n_cores,
                n_rows,
                background,
                repeat,
            ) = item

            log(
                f"{index:4d}. "
                f"strategy={strategy} "
                f"model={model} "
                f"features={feature_count} "
                f"cores={n_cores} "
                f"rows={n_rows} "
                f"bg={background} "
                f"repeat={repeat}"
            )

        return 0

    if not todo:

        log(
            "Nothing to do."
        )

        return 0

    # ------------------------------------------------------------------
    # Initialize.
    # ------------------------------------------------------------------

    writer = ResultWriter(
        args.output
    )

    functions, metadata = (
        load_strategy_funcs()
    )

    log(
        "Stage 4 BLAS pinning: "
        f"PIN_BLAS_THREADS="
        f"{metadata['pin_blas_threads']} "
        f"BLAS_THREADS_PER_WORKER="
        f"{metadata['blas_threads_per_worker']}"
    )

    factory = WorkloadFactory(
        args.seed
    )

    versions = versions_info()

    machine_id = (
        socket.gethostname()
    )

    # ------------------------------------------------------------------
    # Fail fast on adapter problems.
    # ------------------------------------------------------------------

    first = todo[0]

    factory.get(
        first[1],
        first[2],
        first[4],
        first[5],
    )

    # ------------------------------------------------------------------
    # Main benchmark loop.
    # ------------------------------------------------------------------

    successful = 0
    failed = 0

    start = time.time()

    try:

        for index, item in enumerate(
            todo,
            1,
        ):

            (
                strategy,
                model,
                feature_count,
                cores,
                rows,
                background,
                repeat,
            ) = item

            eta_text = ""

            completed = (
                successful
                + failed
            )

            if completed:

                seconds_per_run = (
                    time.time()
                    - start
                ) / completed

                if seconds_per_run > 0:

                    remaining_seconds = (
                        seconds_per_run
                        * (
                            len(todo)
                            - index
                        )
                    )

                    remaining_minutes = (
                        remaining_seconds
                        / 60.0
                    )

                    eta_text = (
                        f" | ETA ~"
                        f"{remaining_minutes:.1f} min"
                    )

            log(
                f"\n[{index}/{len(todo)}] "
                f"START strategy={strategy} "
                f"model={model} "
                f"features={feature_count} "
                f"cores={cores} "
                f"rows={rows} "
                f"bg={background} "
                f"repeat={repeat}"
                f"{eta_text}"
            )

            row = execute_one(
                strategy,
                model,
                feature_count,
                cores,
                rows,
                background,
                repeat,
                factory=factory,
                funcs=functions,
                meta=metadata,
                writer=writer,
                seed=args.seed,
                tolerance=args.additivity_tol,
                skip_additivity=args.skip_additivity,
                machine_id=machine_id,
                versions=versions,
            )

            print_row_summary(
                row
            )

            if row["success"]:

                successful += 1

            else:

                failed += 1

            if (
                args.cooldown
                and index < len(todo)
            ):

                time.sleep(
                    args.cooldown
                )

    except KeyboardInterrupt:

        log(
            "\nInterrupted "
            "(Ctrl+C). "
            "Results so far are saved. "
            "Re-run to resume."
        )

    finally:

        # Safety cleanup if anything leaves Ray initialized.
        try:

            import ray

            if ray.is_initialized():

                ray.shutdown()

        except ImportError:

            pass

    log(
        f"\nDone: "
        f"{successful} succeeded, "
        f"{failed} failed."
    )

    log(
        f"Results: {args.output}"
    )

    return (
        0
        if failed == 0
        else 2
    )


# ============================================================================
# SMOKE TEST
# ============================================================================

def run_smoke_test(args):

    import bench_utils

    results = []

    def check(
        name,
        condition,
        detail="",
    ):

        passed = bool(
            condition
        )

        results.append(
            (
                name,
                passed,
                detail,
            )
        )

        log(
            f"  "
            f"[{'PASS' if passed else 'FAIL'}] "
            f"{name}"
            + (
                f" - {detail}"
                if detail
                else ""
            )
        )

    temporary_directory = Path(
        tempfile.mkdtemp(
            prefix="eashap_memberB_smoke_"
        )
    )

    csv_path = (
        temporary_directory
        / "smoke.csv"
    )

    log(
        f"Smoke test temporary CSV: "
        f"{csv_path}"
    )

    try:

        # ==============================================================
        # 1. Existing bench_utils API
        # ==============================================================

        log(
            "\n1. Existing bench_utils API"
        )

        check(
            "bench_utils.run_and_measure exists",
            hasattr(
                bench_utils,
                "run_and_measure",
            ),
        )

        check(
            "bench_utils.make_chunks exists",
            hasattr(
                bench_utils,
                "make_chunks",
            ),
        )

        # ==============================================================
        # 2. Strategy loading
        # ==============================================================

        log(
            "\n2. Stage 4 strategy loading"
        )

        functions, metadata = (
            load_strategy_funcs()
        )

        for strategy in ALL_STRATEGIES:

            check(
                f"{strategy} strategy loaded",
                strategy in functions
                and callable(
                    functions[strategy]
                ),
            )

        # ==============================================================
        # 3. Workload
        # ==============================================================

        log(
            "\n3. Building tiny workload"
        )

        factory = WorkloadFactory(
            args.seed
        )

        model_name = (
            "LogisticRegression"
        )

        feature_count = 15
        n_rows = 4
        background_size = 10

        workload = factory.get(
            model_name,
            feature_count,
            n_rows,
            background_size,
        )

        check(
            "workload built",
            workload.rows.shape
            == (
                n_rows,
                feature_count,
            )
            and len(
                workload.background
            )
            == background_size,
            (
                f"rows={tuple(workload.rows.shape)} "
                f"background="
                f"{tuple(workload.background.shape)}"
            ),
        )

        # ==============================================================
        # 4. One real serial run
        # ==============================================================

        log(
            "\n4. Serial end-to-end run"
        )

        writer = ResultWriter(
            csv_path
        )

        versions = versions_info()

        row = execute_one(
            "serial",
            model_name,
            feature_count,
            1,
            n_rows,
            background_size,
            0,
            factory=factory,
            funcs=functions,
            meta=metadata,
            writer=writer,
            seed=args.seed,
            tolerance=args.additivity_tol,
            skip_additivity=args.skip_additivity,
            machine_id=socket.gethostname(),
            versions=versions,
        )

        check(
            "serial run succeeded",
            row["success"],
            row["error"],
        )

        check(
            "elapsed time recorded",
            is_finite_number(
                row["elapsed_seconds"]
            )
            and row["elapsed_seconds"] > 0,
            str(
                row["elapsed_seconds"]
            ),
        )

        check(
            "output shape recorded",
            row["output_shape"].startswith(
                f"({n_rows}, {feature_count},"
            ),
            row["output_shape"],
        )

        # ==============================================================
        # 5. CSV
        # ==============================================================

        with open(
            csv_path,
            newline="",
            encoding="utf-8",
        ) as f:

            written_rows = list(
                csv.DictReader(f)
            )

        check(
            "CSV row written",
            len(written_rows) == 1,
        )

        check(
            "CSV contains required columns",
            all(
                column in written_rows[0]
                for column in (
                    "timestamp",
                    "strategy",
                    "model",
                    "feature_count",
                    "n_cores",
                    "n_rows",
                    "background_size",
                    "elapsed_seconds",
                    "energy_kwh",
                    "energy_joules",
                    "success",
                    "error",
                )
            ),
        )

        # ==============================================================
        # 6. Resume
        # ==============================================================

        check(
            "resume index detects successful run",
            run_key(
                "serial",
                model_name,
                feature_count,
                1,
                n_rows,
                background_size,
                0,
            )
            in load_done_keys(
                csv_path
            ),
        )

        # ==============================================================
        # 7. Intentional failure
        # ==============================================================

        log(
            "\n5. Intentional failure handling"
        )

        original_threading = functions[
            "threading"
        ]

        def intentional_failure(
            *args,
            **kwargs,
        ):

            raise RuntimeError(
                "intentional smoke-test failure"
            )

        functions["threading"] = (
            intentional_failure
        )

        failed_row = execute_one(
            "threading",
            model_name,
            feature_count,
            2,
            n_rows,
            background_size,
            0,
            factory=factory,
            funcs=functions,
            meta=metadata,
            writer=writer,
            seed=args.seed,
            tolerance=args.additivity_tol,
            skip_additivity=args.skip_additivity,
            machine_id=socket.gethostname(),
            versions=versions,
        )

        check(
            "failed run recorded",
            (
                not failed_row["success"]
                and "intentional"
                in failed_row["error"]
            ),
            failed_row["error"],
        )

        check(
            "failed run not considered successful for resume",
            run_key(
                "threading",
                model_name,
                feature_count,
                2,
                n_rows,
                background_size,
                0,
            )
            not in load_done_keys(
                csv_path
            ),
        )

        functions[
            "threading"
        ] = original_threading

        # ==============================================================
        # 8. Real threading run
        # ==============================================================

        real_threading = execute_one(
            "threading",
            model_name,
            feature_count,
            2,
            n_rows,
            background_size,
            1,
            factory=factory,
            funcs=functions,
            meta=metadata,
            writer=writer,
            seed=args.seed,
            tolerance=args.additivity_tol,
            skip_additivity=args.skip_additivity,
            machine_id=socket.gethostname(),
            versions=versions,
        )

        check(
            "benchmark continues after failure",
            real_threading["success"],
            real_threading["error"],
        )

        # ==============================================================
        # 9. Ray
        # ==============================================================

        log(
            "\n6. Ray cleanup"
        )

        if (
            os.cpu_count() or 1
        ) >= 2:

            ray_cores = 2

        else:

            ray_cores = 1

        before_ray = ray_pids()

        ray_row = execute_one(
            "ray",
            model_name,
            feature_count,
            ray_cores,
            n_rows,
            background_size,
            0,
            factory=factory,
            funcs=functions,
            meta=metadata,
            writer=writer,
            seed=args.seed,
            tolerance=args.additivity_tol,
            skip_additivity=args.skip_additivity,
            machine_id=socket.gethostname(),
            versions=versions,
        )

        check(
            "ray run succeeded",
            ray_row["success"],
            ray_row["error"],
        )

        time.sleep(1)

        try:

            import ray

            ray_not_initialized = (
                not ray.is_initialized()
            )

        except ImportError:

            ray_not_initialized = True

        check(
            "Ray runtime shut down",
            ray_not_initialized,
        )

        remaining_ray = (
            ray_pids()
            - before_ray
        )

        check(
            "no Ray processes remain",
            not remaining_ray,
            str(
                sorted(
                    remaining_ray
                )
            ),
        )

        # ==============================================================
        # 10. CSV count
        # ==============================================================

        with open(
            csv_path,
            newline="",
            encoding="utf-8",
        ) as f:

            all_rows = list(
                csv.DictReader(f)
            )

        # serial success
        # threading intentional failure
        # threading real run
        # ray
        #
        # = 4 rows

        check(
            "all smoke-test runs written",
            len(all_rows) == 4,
            f"{len(all_rows)} rows",
        )

    except AdapterError as exc:

        check(
            "Member B adapter wiring",
            False,
            str(exc),
        )

    except Exception as exc:

        check(
            "smoke test completed without unexpected exception",
            False,
            f"{type(exc).__name__}: {exc}",
        )

    finally:

        try:

            import ray

            if ray.is_initialized():

                ray.shutdown()

        except ImportError:

            pass

        shutil.rmtree(
            temporary_directory,
            ignore_errors=True,
        )

    failures = sum(
        1
        for _, passed, _
        in results
        if not passed
    )

    log(
        f"\nSMOKE TEST "
        f"{'PASSED' if failures == 0 else 'FAILED'} "
        f"("
        f"{len(results) - failures}"
        f"/{len(results)} checks)"
    )

    return (
        0
        if failures == 0
        else 1
    )


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description=(
            "Stage 5 KernelSHAP benchmark "
            "for Member B."
        )
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run fast end-to-end "
            "self-check."
        ),
    )

    parser.add_argument(
        "--inspect-memberb",
        action="store_true",
        help=(
            "Show functions and constants "
            "found in Member B modules."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Use full benchmark grid: "
            "cores 1/2/4/8, background "
            "25/50/100, 100 rows, "
            "3 repeats."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print benchmark plan "
            "without running."
        ),
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help=(
            "Run at most N remaining "
            "runs."
        ),
    )

    parser.add_argument(
        "--repeats",
        type=int,
        help=(
            "Measured repeats per "
            "condition."
        ),
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=ALL_STRATEGIES,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODELS,
    )

    parser.add_argument(
        "--features",
        nargs="+",
        type=int,
        choices=ALL_FEATURES,
    )

    parser.add_argument(
        "--cores",
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--n-rows",
        type=int,
        help=(
            "Number of instances "
            "explained per run."
        ),
    )

    parser.add_argument(
        "--background-size",
        nargs="+",
        type=int,
        help=(
            "KernelSHAP background "
            "sizes."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            f"Output CSV. Default: "
            f"{DEFAULT_OUTPUT}"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Fixed seed for selecting "
            "rows/background."
        ),
    )

    parser.add_argument(
        "--serial-all-cores",
        action="store_true",
        help=(
            "Also execute serial once "
            "for every requested core count."
        ),
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help=(
            "Shuffle strategy/core order "
            "within each condition."
        ),
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.0,
        help=(
            "Seconds to wait between "
            "benchmark runs."
        ),
    )

    parser.add_argument(
        "--allow-oversubscribe",
        action="store_true",
        help=(
            "Allow requested core counts "
            "above os.cpu_count()."
        ),
    )

    parser.add_argument(
        "--additivity-tol",
        type=float,
        default=1e-2,
        help=(
            "Maximum allowed KernelSHAP "
            "additivity error."
        ),
    )

    parser.add_argument(
        "--skip-additivity",
        action="store_true",
        help=(
            "Skip additivity validation."
        ),
    )

    return parser.parse_args(
        argv
    )


# ============================================================================
# MAIN
# ============================================================================

def main(argv=None):

    args = parse_args(
        argv
    )

    try:

        if args.inspect_memberb:

            inspect_memberb()

            return 0

        if args.smoke_test:

            return run_smoke_test(
                args
            )

        return run_benchmark(
            args
        )

    except AdapterError as exc:

        print(
            f"\nADAPTER ERROR: {exc}",
            file=sys.stderr,
        )

        return 3


if __name__ == "__main__":

    sys.exit(
        main()
    )