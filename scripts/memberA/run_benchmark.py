"""
Member A — Step 5: TreeSHAP benchmark grid runner.

Official grid:
row sizes {1000, 10000, full} x cores {1,2,4,8}
x models {RandomForest, XGBoost}
x strategies {serial, multiprocessing, threading, joblib, ray}
x 3 repeats
= 360 runs.

The runner is resume-safe: configurations already present in results.csv
are skipped, so an interrupted benchmark can be restarted safely.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_utils import run_and_measure, DEFAULT_CSV_PATH
from data_loading import prepare_data, get_explanation_pool
from models import train_random_forest, train_xgboost
from parallel_strategies import (
    run_serial,
    run_multiprocessing,
    run_threading,
    run_joblib,
    run_ray,
)

import pandas as pd


SEED = 42
TRACK = "tree"

# Official benchmark
SMOKE_TEST = False

if SMOKE_TEST:
    ROW_SIZES = [1000]
    CORE_COUNTS = [1, 2]
    REPEATS = 1
else:
    ROW_SIZES = [1000, 10000, "full"]
    CORE_COUNTS = [1, 2, 4, 8]
    REPEATS = 3


STRATEGIES = {
    "serial": run_serial,
    "multiprocessing": run_multiprocessing,
    "threading": run_threading,
    "joblib": run_joblib,
    "ray": run_ray,
}


def get_row_subsets(pool):
    """Create deterministic row subsets reused across all configurations."""
    subsets = {}

    for size in ROW_SIZES:
        if size == "full":
            subsets[size] = pool
        else:
            subsets[size] = pool.sample(
                n=size,
                random_state=SEED,
            )

    return subsets


def load_completed_runs(csv_path):
    """
    Load existing benchmark results so completed configurations
    can be skipped when resuming.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        return set()

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return set()

    required = {
        "track",
        "strategy",
        "model",
        "n_rows",
        "n_cores",
        "repeat_idx",
    }

    if not required.issubset(df.columns):
        return set()

    completed = set()

    for _, row in df.iterrows():
        if row["track"] != TRACK:
            continue

        completed.add(
            (
                str(row["strategy"]),
                str(row["model"]),
                int(row["n_rows"]),
                int(row["n_cores"]),
                int(row["repeat_idx"]),
            )
        )

    return completed


def main():
    total = (
        len(ROW_SIZES)
        * len(CORE_COUNTS)
        * len(STRATEGIES)
        * 2
        * REPEATS
    )

    print(f"SMOKE_TEST = {SMOKE_TEST}")
    print(
        f"Grid: row_sizes={ROW_SIZES}, "
        f"cores={CORE_COUNTS}, repeats={REPEATS}"
    )
    print(f"Total official configurations: {total}\n")

    completed = load_completed_runs(DEFAULT_CSV_PATH)

    print(f"Previously completed configurations: {len(completed)}")
    print(f"Remaining configurations: {total - len(completed)}\n")

    # -------------------------
    # Load benchmark data
    # -------------------------

    pool = get_explanation_pool()
    n_features = pool.shape[1]
    row_subsets = get_row_subsets(pool)

    # -------------------------
    # Train models once
    # -------------------------

    print("Preparing training data...")
    X_train, X_test, y_train, y_test = prepare_data()

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)

    print("Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)

    models = {
        "RandomForest": rf,
        "XGBoost": xgb,
    }

    # -------------------------
    # Run benchmark
    # -------------------------

    attempted = 0
    skipped = 0
    failed = 0

    for row_size in ROW_SIZES:

        selected_rows = row_subsets[row_size]
        n_rows = len(selected_rows)

        for model_name, model in models.items():

            for strategy_name, strategy_fn in STRATEGIES.items():

                for n_cores in CORE_COUNTS:

                    for repeat_idx in range(REPEATS):

                        key = (
                            strategy_name,
                            model_name,
                            n_rows,
                            n_cores,
                            repeat_idx,
                        )

                        if key in completed:
                            skipped += 1
                            print(
                                f"SKIP rows={n_rows} "
                                f"model={model_name} "
                                f"strategy={strategy_name} "
                                f"cores={n_cores} "
                                f"repeat={repeat_idx}"
                            )
                            continue

                        attempted += 1

                        label = (
                            f"rows={n_rows} "
                            f"model={model_name} "
                            f"strategy={strategy_name} "
                            f"cores={n_cores} "
                            f"repeat={repeat_idx}"
                        )

                        print(f"\nSTART {label}", flush=True)

                        start = time.perf_counter()

                        fn = lambda: strategy_fn(
                            selected_rows,
                            model,
                            n_cores=n_cores,
                        )

                        try:
                            run_and_measure(
                                fn,
                                strategy=strategy_name,
                                track=TRACK,
                                model=model_name,
                                n_rows=n_rows,
                                n_features=n_features,
                                n_cores=n_cores,
                                repeat_idx=repeat_idx,
                                background_size=None,
                            )

                            elapsed = time.perf_counter() - start

                            print(
                                f"DONE {label} "
                                f"elapsed={elapsed:.2f} sec",
                                flush=True,
                            )

                            completed.add(key)

                        except Exception as e:
                            failed += 1

                            print(
                                f"FAILED {label} "
                                f"-> {type(e).__name__}: {e}",
                                flush=True,
                            )

    print("\n" + "=" * 60)
    print("BENCHMARK FINISHED")
    print("=" * 60)
    print(f"Total configurations : {total}")
    print(f"Attempted             : {attempted}")
    print(f"Skipped               : {skipped}")
    print(f"Failed                : {failed}")
    print(f"Successful/available  : {total - failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()