# Energy-Aware SHAP

## Project Overview

This project benchmarks different parallelization strategies for SHAP TreeExplainer with the goal of studying the trade-off between execution time and energy consumption.

## Member A - Energy-Aware TreeSHAP Benchmark

Member A implemented the benchmarking and scheduler-analysis pipeline.

### Models

- RandomForest
- XGBoost

### Dataset Sizes

- 1,000 rows
- 10,000 rows
- 48,842 rows

### Parallelization Strategies

- Serial
- Multiprocessing
- Threading
- Joblib
- Ray

### Core Counts

- 1
- 2
- 4
- 8

### Benchmark Metrics

Each benchmark records:

- Wall-clock execution time
- Energy consumption in Joules

The benchmark contains 360 valid runs covering 120 unique configurations with 3 repeats per configuration.

## Member A Pipeline

### 1. `scripts/memberA/analyze_results.py`

Analyzes the raw benchmark results and generates:

- configuration summary
- strategy summary
- model summary
- dataset-size summary
- core-count summary
- fastest configurations
- lowest-energy configurations
- time-energy trade-offs
- repeat consistency

### 2. `scripts/memberA/build_scheduler_dataset.py`

Converts the benchmark results into a scheduler-oriented dataset containing one record for each model × dataset-size × strategy × core-count configuration.

This produces 120 scheduler configurations.

### 3. `scripts/memberA/build_scheduler_recommendations.py`

Builds scheduler recommendations based on:

- time-optimal configuration
- energy-optimal configuration
- balanced time-energy configuration

### 4. `scripts/memberA/plot_results.py`

Generates visualizations for:

- execution time vs number of cores
- energy vs number of cores
- time-energy trade-offs
- strategy comparisons
- dataset scaling
- scheduler recommendations
- core-count comparisons

## Important Output

The main raw benchmark file is:

`results/results.csv`

The processed analysis files are located in:

`results/analysis/`

The generated plots are located in:

`results/analysis/plots/`

## Member B Compatibility

Member B should build on the existing project structure and should not redesign or replace the Member A benchmark pipeline.

Member B outputs should remain compatible with the existing benchmark results and project architecture.