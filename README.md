Energy-Aware SHAP

Project Overview

This project benchmarks different parallelization strategies for SHAP explanations with the goal of studying the trade-off between execution time and energy consumption.

The project contains two complementary benchmark tracks:

Member A — Energy-Aware TreeSHAP

Member B — Energy-Aware KernelSHAP

Both tracks use the same overall benchmarking philosophy and parallelization strategies, while targeting different SHAP explainers.

Member A — Energy-Aware TreeSHAP Benchmark

Member A implemented the original benchmarking and scheduler-analysis pipeline for TreeSHAP.

Models

RandomForest

XGBoost

Dataset Sizes

1,000 rows

10,000 rows

48,842 rows

Parallelization Strategies

Serial

Multiprocessing

Threading

Joblib

Ray

Core Counts

1

2

4

8

Benchmark Metrics

Each benchmark records:

Wall-clock execution time

Energy consumption

Energy in Joules

CodeCarbon measurements

SHAP output/correctness information

The benchmark contains 360 valid runs covering 120 unique configurations with 3 repeats per configuration.

Member A Pipeline

1. scripts/memberA/run_benchmark.py

Runs the TreeSHAP benchmark across the configured models, dataset sizes, strategies, core counts, and repeats.

2. scripts/memberA/analyze_results.py

Analyzes the raw benchmark results and generates:

configuration summary

strategy summary

model summary

dataset-size summary

core-count summary

fastest configurations

lowest-energy configurations

time-energy trade-offs

repeat consistency

3. scripts/memberA/build_scheduler_dataset.py

Converts benchmark results into a scheduler-oriented dataset containing one record for each:

model × dataset-size × strategy × core-count

This produces 120 scheduler configurations.

4. scripts/memberA/build_scheduler_recommendations.py

Builds scheduler recommendations based on:

time-optimal configuration

energy-optimal configuration

balanced time-energy configuration

5. scripts/memberA/plot_results.py

Generates visualizations for:

execution time vs number of cores

energy vs number of cores

time-energy trade-offs

strategy comparisons

dataset scaling

scheduler recommendations

core-count comparisons

Member B — Energy-Aware KernelSHAP Benchmark

Member B extends the project with a separate KernelSHAP benchmark track while keeping the same project architecture and analysis philosophy.

Models

SVM-RBF

LogisticRegression

Feature Counts

15 features

30 features

Evaluation Rows

100 rows

Background Dataset Sizes

25

50

100

Parallelization Strategies

Serial

Multiprocessing

Threading

Joblib

Ray

Core Counts

1

2

4

8

Serial execution is evaluated with one effective core.

Benchmark Repeats

Each configuration is repeated 3 times.

The completed Member B benchmark contains:

612 successful runs

0 failed runs

204 unique scheduler configurations

Member B Pipeline

1. scripts/memberB/data_loading.py

Loads and prepares the data used by the KernelSHAP benchmark.

2. scripts/memberB/models.py

Defines and trains the models used by the KernelSHAP benchmark.

3. scripts/memberB/explain.py

Runs KernelSHAP explanations and performs the required explanation/correctness checks.

4. scripts/memberB/parallel_strategies.py

Implements the KernelSHAP execution strategies:

Serial

Multiprocessing

Threading

Joblib

Ray

5. scripts/memberB/run_benchmark.py

Runs the complete KernelSHAP benchmark.

Example:

python scripts\memberB\run_benchmark.py --full

The benchmark output is:

results/memberB/kernelshap_benchmark.csv

6. scripts/memberB/validate_results.py

Validates the benchmark output, including successful/failed runs and configuration consistency.

7. scripts/memberB/analyze_results.py

Analyzes the completed KernelSHAP benchmark and generates summaries for:

configuration

strategy

model

feature count

background size

row size

core count

fastest configurations

lowest-energy configurations

time-energy trade-offs

repeat consistency

correctness/additivity

scheduler-oriented comparisons

8. scripts/memberB/plot_results.py

Generates visualizations for the KernelSHAP benchmark, including:

strategy execution time

strategy energy consumption

strategy-model comparisons

feature-count effects

background-size effects

core-count effects

model comparisons

time-energy trade-offs

9. scripts/memberB/build_scheduler_dataset.py

Converts the validated KernelSHAP benchmark into a scheduler-oriented dataset.

The completed Member B scheduler dataset contains:

204 unique scheduler configurations.

Output:

results/memberB/analysis/scheduler_dataset.csv

10. scripts/memberB/build_scheduler_recommendations.py

Builds KernelSHAP scheduler recommendations for each workload based on:

time-optimal configuration

energy-optimal configuration

balanced time-energy configuration

The completed recommendation output contains:

36 recommendation rows = 12 workloads × 3 recommendation types.

Outputs:

results/memberB/analysis/scheduler_time_optimal.csv
results/memberB/analysis/scheduler_energy_optimal.csv
results/memberB/analysis/scheduler_balanced.csv
results/memberB/analysis/scheduler_recommendations.csv

Scheduler Outputs

The scheduler analysis provides configuration choices based on measured benchmark data rather than assuming that one parallelization strategy is always optimal.

For both tracks, the scheduler analysis considers:

Time-optimal configuration — lowest measured execution time for the workload.

Energy-optimal configuration — lowest measured energy consumption.

Balanced configuration — selected using normalized time and energy measurements.

These outputs form the basis for the final scheduler implementation.

Repository Structure

Energy-Aware-SHAP/
│
├── scripts/
│   ├── bench_utils.py
│   │
│   ├── memberA/
│   │   ├── analyze_results.py
│   │   ├── build_scheduler_dataset.py
│   │   ├── build_scheduler_recommendations.py
│   │   ├── data_loading.py
│   │   ├── explain.py
│   │   ├── models.py
│   │   ├── parallel_strategies.py
│   │   ├── plot_results.py
│   │   └── run_benchmark.py
│   │
│   └── memberB/
│       ├── analyze_results.py
│       ├── build_scheduler_dataset.py
│       ├── build_scheduler_recommendations.py
│       ├── data_loading.py
│       ├── explain.py
│       ├── models.py
│       ├── parallel_strategies.py
│       ├── plot_results.py
│       ├── run_benchmark.py
│       └── validate_results.py
│
├── results/
│   ├── memberA/
│   │   ├── results.csv
│   │   └── analysis/
│   │
│   └── memberB/
│       ├── kernelshap_benchmark.csv
│       ├── analysis/
│       └── plots/
│
└── README.md

Important Output Locations

Member A

Raw benchmark:

results/memberA/results.csv

Processed analysis:

results/memberA/analysis/

Plots:

results/memberA/analysis/plots/

Member B

Raw benchmark:

results/memberB/kernelshap_benchmark.csv

Processed analysis:

results/memberB/analysis/

Plots:

results/memberB/plots/

Project Architecture

Member B is implemented as a separate benchmark track rather than replacing Member A.

                 Energy-Aware SHAP
                         │
             ┌───────────┴───────────┐
             │                       │
          Member A                Member B
          TreeSHAP                KernelSHAP
             │                       │
      Benchmark pipeline      Benchmark pipeline
             │                       │
       Analysis + plots       Analysis + plots
             │                       │
       Scheduler dataset      Scheduler dataset
             │                       │
       Recommendations        Recommendations
             └───────────┬───────────┘
                         │
                  Final Scheduler

The two tracks remain independently reproducible while providing compatible scheduler-oriented outputs.

Current Project Status

✅Member A benchmark

✅Member A analysis

✅Member A plots

✅Member A scheduler dataset

✅Member A scheduler recommendations

✅Member B benchmark

✅Member B validation

✅Member B analysis

✅Member B plots

✅Member B scheduler dataset

✅Member B scheduler recommendations

Final unified scheduler implementation

Final scheduler evaluation

Final project comparison and reporting