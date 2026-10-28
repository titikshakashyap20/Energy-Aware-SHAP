"""Run the reproducible Member C analysis on existing CSV data; never runs benchmarks."""
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parent
required = ["results/memberA/results.csv", "results/memberB/kernelshap_benchmark.csv",
            "results/memberC/unified_benchmark.csv", "results/memberC/edp_dataset.csv",
            "results/memberC/scheduler_nested_evaluation.csv", "results/memberC/scheduler_evaluation_protocol.csv",
            "results/memberC/energy_sensitivity_report.csv", "results/memberC/energy_sensitivity_winners.csv"]
missing = [p for p in required if not (ROOT/p).is_file()]
if missing:
    raise SystemExit("Missing required analysis inputs: " + ", ".join(missing))
steps = [[sys.executable, "scripts/memberC/audit_results.py"],
         [sys.executable, "scripts/memberC/finalize_research.py"]]
for command in steps:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
print("Complete. Outputs: results/memberC/*.csv, results/memberC/plots/*, results_summary.json")
