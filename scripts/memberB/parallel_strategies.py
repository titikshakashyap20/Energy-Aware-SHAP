"""
Stage 4: Five execution strategies for the SAME KernelSHAP workload.

Member B — Energy-Aware SHAP

Strategies:
    1. serial
    2. multiprocessing
    3. threading
    4. joblib/loky
    5. ray

All strategies call the SAME explain_chunk() function.

The only intended difference is HOW the same workload is scheduled.

Stage 5 is responsible for timing and energy measurement.
Therefore this file contains NO CodeCarbon measurement.

Important compatibility decisions:
    - KernelSHAP may return a custom ExplanationResult, shap.Explanation,
      or ndarray depending on the installed explain.py / SHAP API.
      _extract_shap_values() normalizes these to one ndarray format.
    - DataFrames are converted to NumPy before entering KernelSHAP. This
      avoids SHAP 0.52 interacting incorrectly with sklearn Pipeline's
      feature_names_in_ attribute.
    - KernelSHAP is stochastic. We therefore do NOT require bit-for-bit
      equality between independent strategy runs.
    - BLAS/OpenMP threads are pinned to one per worker so n_cores controls
      the intended parallelism.
    - multiprocessing always uses spawn.
    - Ray startup/shutdown happens inside run_ray(), so Stage 5 measures it.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from itertools import repeat
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


# ============================================================================
# IMPORT PATHS
# ============================================================================

_THIS_DIR = Path(__file__).resolve().parent

for _p in (_THIS_DIR.parent, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


from explain import explain_chunk  # noqa: E402


# ============================================================================
# FAIRNESS SETTINGS
# ============================================================================

# Important:
# Without limiting BLAS/OpenMP, "2 cores" could actually mean:
#
#     2 workers × several BLAS threads per worker
#
# That would make comparisons between strategies unfair.
PIN_BLAS_THREADS = True
BLAS_THREADS_PER_WORKER = 1


# ============================================================================
# INPUT NORMALIZATION
# ============================================================================

def _to_numpy(value):
    """
    Convert pandas DataFrame / Series or other array-like input to ndarray.

    KernelSHAP + sklearn Pipeline + SHAP 0.52 can fail when a DataFrame-fitted
    Pipeline is handed directly to KernelExplainer because SHAP attempts to
    manipulate feature_names_in_.

    Converting the workload arrays here gives every strategy the same
    numerical input while leaving the fitted model untouched.
    """
    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "to_numpy"):
        return value.to_numpy()

    return np.asarray(value)


# ============================================================================
# SHAP OUTPUT NORMALIZATION
# ============================================================================

def _extract_shap_values(result):
    """
    Normalize explain_chunk() output to:

        ndarray of shape
        (n_rows, n_features, n_classes)

    Supported outputs:

    1. ndarray
    2. ExplanationResult with .shap_values
    3. ExplanationResult with .values
    4. shap.Explanation with .values
    5. dict containing "shap_values" or "values"

    The benchmark layer only needs the SHAP array.
    expected_value/base values are intentionally not propagated through the
    parallel strategy layer because Stage 5 benchmarks the explanation
    workload itself.
    """

    # ---------------------------------------------------------
    # Case 1: already ndarray
    # ---------------------------------------------------------
    if isinstance(result, np.ndarray):
        values = result

    # ---------------------------------------------------------
    # Case 2: custom ExplanationResult
    #
    # Your current explain.py appears to use this.
    # ---------------------------------------------------------
    elif hasattr(result, "shap_values"):
        values = result.shap_values

    # ---------------------------------------------------------
    # Case 3: shap.Explanation
    # ---------------------------------------------------------
    elif hasattr(result, "values"):
        values = result.values

    # ---------------------------------------------------------
    # Case 4: dictionary-style result
    # ---------------------------------------------------------
    elif isinstance(result, dict):
        if "shap_values" in result:
            values = result["shap_values"]
        elif "values" in result:
            values = result["values"]
        else:
            raise TypeError(
                "explain_chunk() returned a dictionary but it contains "
                "neither 'shap_values' nor 'values'. "
                f"Keys: {list(result.keys())}"
            )

    else:
        raise TypeError(
            "explain_chunk() returned an unsupported object: "
            f"{type(result).__name__}. "
            "Expected ndarray, ExplanationResult with .shap_values/.values, "
            "shap.Explanation, or a compatible dictionary."
        )

    values = np.asarray(values)

    # ---------------------------------------------------------
    # SHAP 0.52 + predict_proba should normally give:
    #
    #     (n_rows, n_features, n_classes)
    #
    # We deliberately reject ambiguous 2-D output here because silently
    # inventing a class dimension could corrupt the benchmark.
    # ---------------------------------------------------------
    if values.ndim != 3:
        raise TypeError(
            "Normalized SHAP output must be 3-dimensional "
            "(n_rows, n_features, n_classes), "
            f"but got shape {values.shape}."
        )

    return values


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _split(rows, n_cores):
    """
    Use the project's shared bench_utils.make_chunks().
    """

    from bench_utils import make_chunks

    return make_chunks(rows, n_cores)


@contextmanager
def _blas_limit():
    """
    Limit BLAS/OpenMP thread pools.

    This context is used inside the actual execution worker wherever possible.
    """

    if PIN_BLAS_THREADS:
        with threadpool_limits(limits=BLAS_THREADS_PER_WORKER):
            yield
    else:
        yield


def _worker(chunk, model, background):
    """
    Shared unit of work.

    Every strategy eventually executes this function.

    The workload is normalized to NumPy before explain_chunk() so that the
    DataFrame/SHAP/Pipeline compatibility issue does not change the workload
    between strategies.
    """

    chunk_np = _to_numpy(chunk)
    background_np = _to_numpy(background)

    with _blas_limit():
        result = explain_chunk(
            chunk_np,
            model,
            background_np,
        )

    return _extract_shap_values(result)


def _ray_worker(chunk, model, background):
    """
    Ray-specific worker.

    Ray can expose NumPy arrays through read-only object-store views.
    Some sklearn/libsvm operations expect writable buffers, so private copies
    are made inside the Ray worker.

    This copy is part of the real Ray execution cost and therefore correctly
    remains inside the Stage 5 measurement window.
    """

    chunk_copy = copy.deepcopy(chunk)
    model_copy = copy.deepcopy(model)
    background_copy = copy.deepcopy(background)

    return _worker(
        chunk_copy,
        model_copy,
        background_copy,
    )


def _combine(results, chunks):
    """
    Validate and concatenate chunk results.

    Results are already in chunk submission order because:
        - executor.map()
        - joblib.Parallel()
        - ray.get(list_of_refs)

    preserve that order.
    """

    if len(results) != len(chunks):
        raise RuntimeError(
            f"Received {len(results)} results for {len(chunks)} chunks."
        )

    arrays = []

    for i, (result, chunk) in enumerate(zip(results, chunks)):

        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"Chunk {i}: expected ndarray after normalization, "
                f"got {type(result).__name__}."
            )

        if result.ndim != 3:
            raise TypeError(
                f"Chunk {i}: expected 3-D SHAP array, "
                f"got shape {result.shape}."
            )

        if result.shape[0] != len(chunk):
            raise RuntimeError(
                f"Chunk {i}: received {result.shape[0]} explanations "
                f"for {len(chunk)} input rows."
            )

        arrays.append(result)

    if not arrays:
        raise RuntimeError("No SHAP result chunks were produced.")

    return np.concatenate(arrays, axis=0)


# ============================================================================
# EXECUTOR FACTORIES
# ============================================================================

def _process_pool(n_workers):
    """
    Always use spawn.

    This makes multiprocessing behavior consistent with Windows and avoids
    giving Linux/macOS an accidental fork advantage.
    """

    return ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=multiprocessing.get_context("spawn"),
    )


def _thread_pool(n_workers):
    return ThreadPoolExecutor(
        max_workers=n_workers,
        thread_name_prefix="shap",
    )


def _loky_parallel(n_workers):
    return Parallel(
        n_jobs=n_workers,
        backend="loky",
    )


@contextmanager
def _ray_session(n_cores):
    """
    Start and stop a fresh local Ray runtime.

    IMPORTANT:
    Ray startup and shutdown happen inside run_ray().
    Therefore Stage 5 can measure them.
    """

    import ray

    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        num_cpus=n_cores,
        include_dashboard=False,
        logging_level=logging.ERROR,
        log_to_driver=False,
    )

    try:
        yield ray

    finally:
        ray.shutdown()


# ============================================================================
# FIVE STRATEGIES
# ============================================================================

def run_serial(rows, model, background, n_cores):
    """
    Serial baseline.

    n_cores is accepted only for API compatibility.
    """

    del n_cores

    return _combine(
        [
            _worker(
                rows,
                model,
                background,
            )
        ],
        [rows],
    )


def run_multiprocessing(rows, model, background, n_cores):
    """
    ProcessPoolExecutor with explicit spawn.
    """

    chunks = _split(rows, n_cores)

    with _process_pool(len(chunks)) as pool:

        results = list(
            pool.map(
                _worker,
                chunks,
                repeat(model),
                repeat(background),
            )
        )

    return _combine(results, chunks)


def run_threading(rows, model, background, n_cores):
    """
    ThreadPoolExecutor.

    BLAS limits are applied around the complete thread pool because BLAS
    thread limits are process-wide.
    """

    chunks = _split(rows, n_cores)

    with _blas_limit():

        with _thread_pool(len(chunks)) as pool:

            results = list(
                pool.map(
                    _worker,
                    chunks,
                    repeat(model),
                    repeat(background),
                )
            )

    return _combine(results, chunks)


def run_joblib(rows, model, background, n_cores):
    """
    joblib using the loky backend.

    At n_cores=1, joblib intentionally uses its SequentialBackend.
    This is genuine joblib behavior and is documented rather than artificially
    changed.
    """

    chunks = _split(rows, n_cores)

    results = _loky_parallel(len(chunks))(
        delayed(_worker)(
            chunk,
            model,
            background,
        )
        for chunk in chunks
    )

    return _combine(results, chunks)


def run_ray(rows, model, background, n_cores):
    """
    Ray strategy.

    Ray initialization, execution, result collection and shutdown all occur
    inside this function so Stage 5 measures the complete Ray overhead.
    """

    chunks = _split(rows, n_cores)

    with _ray_session(n_cores) as ray:

        model_ref = ray.put(model)
        background_ref = ray.put(background)

        task = ray.remote(num_cpus=1)(_ray_worker)

        refs = [
            task.remote(
                chunk,
                model_ref,
                background_ref,
            )
            for chunk in chunks
        ]

        results = ray.get(refs)

        # Must combine BEFORE shutting Ray down.
        combined = _combine(
            results,
            chunks,
        )

    return combined


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

STRATEGIES = {
    "serial": run_serial,
    "multiprocessing": run_multiprocessing,
    "threading": run_threading,
    "joblib": run_joblib,
    "ray": run_ray,
}


# ============================================================================
# SMOKE-TEST FIXTURE
# ============================================================================

def _build_fixture(
    n_rows,
    n_background,
    use_dataframe,
    seed=42,
):
    """
    Build a tiny Breast Cancer fixture.

    The model uses the SAME explicit calibration approach as Member B's
    corrected models.py:

        StandardScaler
        ->
        SVC(RBF)
        ->
        CalibratedClassifierCV

    This avoids the deprecated SVC(probability=True) API.
    """

    import pandas as pd

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    X, y = load_breast_cancer(
        as_frame=True,
        return_X_y=True,
    )

    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=seed,
    )

    # Feature selection uses training data only.
    rf = RandomForestClassifier(
        n_estimators=50,
        random_state=seed,
    )

    rf.fit(
        X_train,
        y_train,
    )

    importance = pd.Series(
        rf.feature_importances_,
        index=X_train.columns,
    )

    top15 = (
        importance
        .sort_values(ascending=False)
        .head(15)
        .index
    )

    X_train = X_train[top15]
    X_test = X_test[top15]

    base_svc = SVC(
        kernel="rbf",
        random_state=seed,
    )

    calibrated_svc = CalibratedClassifierCV(
        estimator=base_svc,
        method="sigmoid",
        cv=5,
        ensemble=False,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", calibrated_svc),
        ]
    )

    # Fit on NumPy for the smoke test.
    # This ensures the fixture itself never introduces the DataFrame feature
    # name issue we are testing against.
    model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
    )

    probabilities = model.predict_proba(
        X_test.to_numpy()
    )[:, 1]

    # Select rows whose predictions are sufficiently distinguishable so the
    # additivity test can detect accidental row reordering/duplication.
    picked = []

    for i, probability in enumerate(probabilities):

        if all(
            abs(probability - probabilities[j]) >= 1e-4
            for j in picked
        ):
            picked.append(i)

        if len(picked) == n_rows:
            break

    if len(picked) < n_rows:
        raise RuntimeError(
            "Could not find enough distinguishable test rows."
        )

    rows_df = X_test.iloc[picked]

    background_df = X_train.sample(
        n=n_background,
        random_state=seed,
    )

    if use_dataframe:
        rows = rows_df
        background = background_df
    else:
        rows = rows_df.to_numpy()
        background = background_df.to_numpy()

    return rows, background, model


# ============================================================================
# PROCESS / THREAD PROBES
# ============================================================================

def _identity_probe(_i):
    """
    Tiny task used to verify where work actually runs.
    """

    time.sleep(0.3)

    return (
        os.getpid(),
        threading.get_ident(),
        threading.current_thread().name,
    )


def _ray_pids():
    try:
        import psutil

        return {
            process.pid
            for process in psutil.process_iter()
        }

    except ImportError:
        return set()


def _ray_leftovers(before_pids):
    """
    Find Ray processes that appeared after before_pids and remain alive.
    """

    try:
        import psutil

    except ImportError:
        return None

    leftovers = []

    for process in psutil.process_iter(
        ["pid", "name", "cmdline"]
    ):

        if process.info["pid"] in before_pids:
            continue

        name = (
            process.info["name"] or ""
        ).lower()

        command = " ".join(
            process.info["cmdline"] or []
        ).lower()

        if (
            "raylet" in name
            or "gcs_server" in name
            or "default_worker.py" in command
            or "ray/_private" in command.replace("\\", "/")
        ):
            leftovers.append(
                (
                    process.info["pid"],
                    name,
                )
            )

    return leftovers


# ============================================================================
# SMOKE TEST
# ============================================================================

def _smoke_test(
    use_dataframe=False,
):
    """
    Small functional test.

    This is NOT the actual benchmark.

    It checks:
        - SHAP output shape
        - finite values
        - additivity
        - ordering
        - all five strategies
        - n_cores 1 and 2
        - rows < cores
        - multiprocessing spawn
        - threading
        - joblib/loky
        - Ray startup/shutdown
        - no deprecated SVC probability=True
    """

    N_ROWS = 20
    N_BACKGROUND = 10

    failures = []

    def check(ok, message):

        print(
            f"  [{'PASS' if ok else 'FAIL'}] {message}"
        )

        if not ok:
            failures.append(message)

    print("=" * 72)
    print(
        "Stage 4 smoke test: parallel_strategies "
        "(KernelSHAP, SVM-RBF, 15 features)"
    )

    print(
        f"input type: "
        f"{'DataFrame' if use_dataframe else 'ndarray'} "
        f"| rows={N_ROWS} "
        f"background={N_BACKGROUND} "
        f"| pid={os.getpid()}"
    )

    print(
        f"BLAS pinning: {PIN_BLAS_THREADS} "
        f"({BLAS_THREADS_PER_WORKER} thread/worker)"
    )

    print("=" * 72)

    rows, background, model = _build_fixture(
        N_ROWS,
        N_BACKGROUND,
        use_dataframe,
    )

    n_features = rows.shape[1]
    n_classes = len(model.classes_)

    expected_shape = (
        N_ROWS,
        n_features,
        n_classes,
    )

    # ---------------------------------------------------------
    # Convert ONLY for validation/prediction.
    # Strategy functions themselves accept both DataFrame and ndarray.
    # ---------------------------------------------------------

    rows_np = _to_numpy(rows)
    background_np = _to_numpy(background)

    p_rows = model.predict_proba(rows_np)
    p_background = model.predict_proba(background_np)

    base = p_background.mean(axis=0)

    target = p_rows - base

    ADD_TOL = 1e-5

    def additivity_error(phi):

        return float(
            np.abs(
                phi.sum(axis=1) - target
            ).max()
        )

    # ---------------------------------------------------------
    # Serial baseline
    # ---------------------------------------------------------

    print("\n--- serial baseline ---")

    try:

        serial = run_serial(
            rows,
            model,
            background,
            1,
        )

        check(
            serial.shape == expected_shape,
            f"serial shape {serial.shape} == {expected_shape}",
        )

        check(
            np.isfinite(serial).all(),
            "serial: all SHAP values finite",
        )

        serial_error = additivity_error(serial)

        check(
            serial_error < ADD_TOL,
            f"serial additivity error "
            f"{serial_error:.2e} < {ADD_TOL:.0e}",
        )

    except Exception as exc:

        check(
            False,
            f"serial raised {type(exc).__name__}: {exc}",
        )

        serial = None

    if serial is None:
        print("\nSmoke test cannot continue without serial baseline.")
        sys.exit(1)

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # KernelSHAP is stochastic.
    #
    # We DO NOT compare serial vs serial or strategy outputs for exact
    # equality. Instead, every strategy must satisfy the mathematical
    # additivity relationship independently.
    # ---------------------------------------------------------

    print(
        "\nKernelSHAP randomness policy: "
        "exact cross-run SHAP equality is NOT required; "
        "shape, finiteness and additivity are required."
    )

    # ---------------------------------------------------------
    # Strategy tests
    # ---------------------------------------------------------

    outputs = {}

    for n_cores in (1, 2):

        print(
            f"\n--- n_cores = {n_cores} ---"
        )

        chunks = _split(
            rows,
            n_cores,
        )

        print(
            f"chunks produced: {len(chunks)}"
        )

        for name, function in STRATEGIES.items():

            start = time.perf_counter()

            try:

                phi = function(
                    rows,
                    model,
                    background,
                    n_cores,
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

            except Exception as exc:

                check(
                    False,
                    f"{name}: raised "
                    f"{type(exc).__name__}: {exc}",
                )

                continue

            outputs[
                (name, n_cores)
            ] = phi

            print(
                f"  {name:18s} "
                f"ok ({elapsed:5.1f}s, informational only; "
                f"chunks={len(chunks)})"
            )

            check(
                isinstance(phi, np.ndarray)
                and phi.shape == expected_shape,
                f"{name}: shape {phi.shape} == "
                f"{expected_shape}",
            )

            check(
                np.isfinite(phi).all(),
                f"{name}: all SHAP values finite",
            )

            error = additivity_error(phi)

            check(
                error < ADD_TOL,
                f"{name}: additivity error "
                f"{error:.2e} < {ADD_TOL:.0e}",
            )

    # ---------------------------------------------------------
    # Edge case: fewer rows than cores
    # ---------------------------------------------------------

    print(
        "\n--- edge case: "
        "3 rows, n_cores = 4 ---"
    )

    rows3 = (
        rows.iloc[:3]
        if hasattr(rows, "iloc")
        else rows[:3]
    )

    target3 = target[:3]

    for name, function in STRATEGIES.items():

        try:

            phi = function(
                rows3,
                model,
                background,
                4,
            )

            expected3 = (
                3,
                n_features,
                n_classes,
            )

            error3 = float(
                np.abs(
                    phi.sum(axis=1)
                    - target3
                ).max()
            )

            check(
                phi.shape == expected3,
                f"{name}: 3 rows / 4 cores "
                f"-> shape {phi.shape}",
            )

            check(
                error3 < ADD_TOL,
                f"{name}: 3 rows / 4 cores "
                f"-> additivity error "
                f"{error3:.2e}",
            )

        except Exception as exc:

            check(
                False,
                f"{name}: edge case raised "
                f"{type(exc).__name__}: {exc}",
            )

    # ---------------------------------------------------------
    # Multiprocessing probe
    # ---------------------------------------------------------

    print(
        "\n--- execution-context probes ---"
    )

    parent_pid = os.getpid()
    parent_thread = threading.get_ident()

    with _process_pool(2) as pool:

        process_results = list(
            pool.map(
                _identity_probe,
                range(4),
            )
        )

        start_method = (
            pool._mp_context
            .get_start_method()
        )

    worker_pids = {
        result[0]
        for result in process_results
    }

    check(
        parent_pid not in worker_pids
        and len(worker_pids) >= 2,
        "multiprocessing: separate worker processes",
    )

    check(
        start_method == "spawn",
        f"multiprocessing: start method = "
        f"{start_method!r}",
    )

    # ---------------------------------------------------------
    # Threading probe
    # ---------------------------------------------------------

    with _thread_pool(2) as pool:

        thread_results = list(
            pool.map(
                _identity_probe,
                range(4),
            )
        )

    thread_ids = {
        result[1]
        for result in thread_results
    }

    thread_pids = {
        result[0]
        for result in thread_results
    }

    check(
        thread_pids == {parent_pid}
        and len(thread_ids) >= 2
        and parent_thread not in thread_ids,
        "threading: separate worker threads "
        "inside parent process",
    )

    # ---------------------------------------------------------
    # joblib probe
    # ---------------------------------------------------------

    try:

        import joblib
        from joblib._parallel_backends import (
            LokyBackend,
            SequentialBackend,
        )

        with _loky_parallel(2) as parallel:

            joblib_results = parallel(
                delayed(_identity_probe)(i)
                for i in range(4)
            )

            backend = parallel._backend

        joblib_pids = {
            result[0]
            for result in joblib_results
        }

        check(
            isinstance(
                backend,
                LokyBackend,
            )
            and parent_pid not in joblib_pids,
            "joblib n_jobs=2: loky worker processes",
        )

        with _loky_parallel(1) as parallel:

            backend_one = parallel._backend

        check(
            isinstance(
                backend_one,
                SequentialBackend,
            ),
            "joblib n_jobs=1: SequentialBackend "
            "(documented behavior)",
        )

        print(
            f"joblib version: {joblib.__version__}"
        )

    except Exception as exc:

        check(
            False,
            f"joblib probe raised "
            f"{type(exc).__name__}: {exc}",
        )

    # ---------------------------------------------------------
    # Ray probe
    # ---------------------------------------------------------

    try:

        import ray

        before_ray = _ray_pids()

        with _ray_session(2) as ray_runtime:

            task = ray_runtime.remote(
                num_cpus=1
            )(_identity_probe)

            ray_results = ray_runtime.get(
                [
                    task.remote(i)
                    for i in range(4)
                ]
            )

            ray_running = (
                ray_runtime.is_initialized()
            )

        ray_worker_pids = {
            result[0]
            for result in ray_results
        }

        check(
            ray_running,
            "ray: runtime initialized during session",
        )

        check(
            parent_pid not in ray_worker_pids,
            "ray: work executed in worker process",
        )

        check(
            not ray.is_initialized(),
            "ray: runtime shut down after session",
        )

        # Check actual run_ray cleanup.
        run_ray(
            rows,
            model,
            background,
            2,
        )

        leftovers = []

        for _ in range(20):

            leftovers = _ray_leftovers(
                before_ray
            )

            if not leftovers:
                break

            time.sleep(0.5)

        if leftovers is None:

            print(
                "  [SKIP] Ray process cleanup check: "
                "psutil not installed"
            )

        else:

            check(
                not leftovers,
                "ray: no Ray processes left "
                f"after shutdown {leftovers or ''}",
            )

        check(
            not ray.is_initialized(),
            "ray: runtime still shut down "
            "after run_ray()",
        )

    except Exception as exc:

        check(
            False,
            f"ray probe raised "
            f"{type(exc).__name__}: {exc}",
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 72
    )

    if failures:

        print(
            f"STAGE 4 SMOKE TEST: FAILED "
            f"({len(failures)} check(s))"
        )

        for failure in failures:
            print(
                "  -",
                failure,
            )

        sys.exit(1)

    print(
        "STAGE 4 SMOKE TEST: ALL CHECKS PASSED"
    )

    print(
        "=" * 72
    )


# ============================================================================
# WINDOWS-SAFE ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    _smoke_test(
        use_dataframe=(
            "--dataframe"
            in sys.argv
        )
    )