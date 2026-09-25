"""
Step 4: Five parallelism strategy wrappers around explain_chunk() (Step 3).

NOTE: _make_chunks() below is a TEMPORARY placeholder with the exact signature
Member C's bench_utils.make_chunks(rows, n_cores) is expected to have. Once
that lands, replace the import line below and delete _make_chunks() — do not
keep two competing chunking implementations.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
from joblib import Parallel, delayed
import ray

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_utils import make_chunks
from explain import explain_chunk
from data_loading import prepare_data
from models import train_random_forest, train_xgboost




def _worker(chunk, model):
    """Top-level, picklable worker — required for multiprocessing on Windows."""
    return explain_chunk(chunk, model)


def run_serial(rows, model, n_cores=1):
    """Baseline: no chunking, no parallel infra."""
    return [explain_chunk(rows, model)]


def run_multiprocessing(rows, model, n_cores=1):
    chunks = make_chunks(rows, n_cores)
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        return list(executor.map(_worker, chunks, [model] * len(chunks)))


def run_threading(rows, model, n_cores=1):
    chunks = make_chunks(rows, n_cores)
    with ThreadPoolExecutor(max_workers=n_cores) as executor:
        return list(executor.map(_worker, chunks, [model] * len(chunks)))


def run_joblib(rows, model, n_cores=1):
    chunks = make_chunks(rows, n_cores)
    return Parallel(n_jobs=n_cores, backend="loky")(
        delayed(explain_chunk)(chunk, model) for chunk in chunks
    )


def run_ray(rows, model, n_cores=1):
    chunks = make_chunks(rows, n_cores)

    ray.shutdown()  # clean context so cluster matches n_cores exactly
    ray.init(num_cpus=n_cores, ignore_reinit_error=True, logging_level="ERROR")

    # NOTE: num_cpus caps how many Ray *tasks* run concurrently, but it does
    # NOT stop each individual task from spawning its own internal threads
    # (NumPy/sklearn/XGBoost commonly use BLAS/OpenMP thread pools that ignore
    # Ray entirely). At n_cores=1 this could mean a single Ray task quietly
    # uses more than 1 physical core underneath, which would make "n_cores=1"
    # not really mean 1 core for Ray specifically. This same risk applies to
    # multiprocessing/joblib/threading too, since they call the same
    # explain_chunk() with the same underlying BLAS/OpenMP libraries — so any
    # fix (e.g. pinning OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1) needs to be
    # applied at the harness level, identically across all 5 strategies, not
    # patched into Ray alone. Flagging for Member C's bench_utils / grid runner.
    model_ref = ray.put(model)          # serialize model once, not per task
    explain_remote = ray.remote(explain_chunk)
    futures = [explain_remote.remote(chunk, model_ref) for chunk in chunks]
    results = ray.get(futures)

    ray.shutdown()
    return results


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prepare_data()
    chunk = X_test.iloc[:20]

    rf = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    strategies = {
        "serial": run_serial,
        "multiprocessing": run_multiprocessing,
        "threading": run_threading,
        "joblib": run_joblib,
        "ray": run_ray,
    }

    for model_name, model in [("RandomForest", rf), ("XGBoost", xgb)]:
        for strat_name, strat_fn in strategies.items():
            for n_cores in (1, 2):
                results = strat_fn(chunk, model, n_cores=n_cores)
                shapes = [np.asarray(r).shape for r in results]
                print(f"{model_name:12s} | {strat_name:15s} | cores={n_cores} "
                      f"| n_chunks={len(results)} | shapes={shapes}")