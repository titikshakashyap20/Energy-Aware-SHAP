# Member C research results

Scheduler regret averaged 0.1640; Joblib regret averaged 0.1601. The paired scheduler-minus-Joblib difference was 0.0040 (workload bootstrap 95% percentile CI -0.0160 to 0.0294; Wilcoxon signed-rank W=39.0, two-sided p=1, n=18 workloads).

The interval includes zero and the test is not significant. This does not establish equivalence. The observations are 18 held-out workloads from one nested LOWO evaluation; the folds are not independent full-pipeline retrainings, and inference is limited to this benchmark space. Four decision points were averaged within each workload before paired inference.

The 10 W RAM sensitivity analysis had 216 matched groups and 27 winner flips. Among flips, the original best-versus-runner-up EDP gap had median 2.92% and ranged 0.22%–130.11%. The assumed adjustment is E_adj = E_measured − 10 W × runtime and EDP_adj = E_adj × runtime; nonpositive adjusted energy is excluded. This hypothetical adjustment is not measured CPU-only energy. Historical energy component diagnostics are incomplete and GPU energy was unsupported/unavailable, not a measured zero.

The benchmark-optimal strategy is the lowest EDP among observed configurations and serves only as an evaluation reference; it is not known in advance for a new workload. See results_summary.json and the generated tables for machine-readable estimates.
