"""Reproduce Member C workload-level statistics, sensitivity tables, figures and summary."""
from __future__ import annotations
import hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/memberC"
PLOTS = OUT / "plots"
SEED, N_BOOT = 20261028, 10_000

def bootstrap_mean_ci(x, seed):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return np.quantile(draws, [.025, .975])

def stats_outputs():
    nested = pd.read_csv(OUT / "scheduler_nested_evaluation.csv")
    d = nested[nested.record_type.eq("decision")].copy()
    d["held_out_workload_id"] = d.held_out_workload_id.fillna(d.get("workload_id"))
    if d.held_out_workload_id.isna().any():
        d["held_out_workload_id"] = d.split_id.map(lambda x: f"split-{int(x)}")
    # Four decision points are averaged within each of the 18 held-out workloads.
    paired = d.groupby("held_out_workload_id").agg(
        scheduler_regret=("edp_regret", "mean"), joblib_regret=("joblib_edp_ratio_to_oracle", lambda s: np.nan),
        scheduler_decision_points=("edp_regret", "size"),
    )
    # Joblib regret is measured EDP/oracle - 1; source decision rows store its ratio.
    paired["joblib_regret"] = d.assign(joblib_regret=d.joblib_edp_ratio_to_oracle - 1).groupby("held_out_workload_id").joblib_regret.mean()
    paired = paired.dropna(subset=["scheduler_regret", "joblib_regret"])
    paired["difference_scheduler_minus_joblib"] = paired.scheduler_regret - paired.joblib_regret
    paired.to_csv(OUT / "scheduler_joblib_paired_differences.csv", index_label="workload_id")
    diff = paired.difference_scheduler_minus_joblib.to_numpy()
    ci = bootstrap_mean_ci(diff, SEED)
    w = wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="approx")
    sig = pd.DataFrame([{
        "n_workloads": len(paired), "n_zero_differences": int(np.isclose(diff, 0).sum()), "n_nonzero_differences": int((~np.isclose(diff, 0)).sum()), "mean_scheduler_regret": paired.scheduler_regret.mean(),
        "mean_joblib_regret": paired.joblib_regret.mean(), "mean_difference_scheduler_minus_joblib": diff.mean(),
        "difference_ci95_percentile_lower": ci[0], "difference_ci95_percentile_upper": ci[1],
        "wilcoxon_statistic": w.statistic, "wilcoxon_p_value": w.pvalue,
        "zero_difference_handling": "Wilcoxon zero_method=wilcox: omit zero differences",
        "ties_handling": "average ranks for tied absolute differences; two-sided asymptotic normal approximation",
        "bootstrap_resamples": N_BOOT, "bootstrap_seed": SEED,
        "bootstrap_interval": "percentile 2.5th and 97.5th quantiles; paired workloads resampled as units",
        "interpretation": "Inference is conditional on these held-out workloads and nested evaluation; nonsignificance is not equivalence.",
    }])
    sig.to_csv(OUT / "scheduler_joblib_significance.csv", index=False)

    # Per strategy: retain measured decision points, aggregate four cores within workload.
    base = pd.read_csv(OUT / "scheduler_evaluation_protocol.csv")
    rows=[]
    for i, (strategy, g) in enumerate(base[base.baseline_available.astype(bool)].groupby("candidate_strategy")):
        q=g.copy(); q["regret"] = q.measured_median_edp / q.oracle_edp - 1
        q["hit"] = q.candidate_strategy.eq(q.oracle_strategy).astype(float)
        wl=q.groupby("held_out_workload_id").agg(regret=("regret","mean"),hit=("hit","mean"),observations=("regret","size"))
        rlo,rhi=bootstrap_mean_ci(wl.regret,SEED+i); hlo,hhi=bootstrap_mean_ci(wl.hit,SEED+100+i)
        rows.append({"strategy":strategy,"mean_regret":wl.regret.mean(),"regret_ci_lower":rlo,"regret_ci_upper":rhi,
          "mean_hit_rate":wl.hit.mean(),"hit_rate_ci_lower":hlo,"hit_rate_ci_upper":hhi,
          "n_eligible_workloads":len(wl),"n_observations":int(wl.observations.sum()),"bootstrap_resamples":N_BOOT,"random_seed":SEED+i,
          "regret_definition":"measured median EDP / observed oracle EDP - 1, averaged over available core decision points within workload",
          "hit_rate_definition":"fraction of available workload/core decision points where strategy equals observed EDP-minimizing strategy"})
    fixed=pd.DataFrame(rows).sort_values("strategy")
    fixed.to_csv(OUT/"fixed_strategy_confidence_intervals.csv",index=False)
    return paired,sig,fixed

def sensitivity_outputs():
    w=pd.read_csv(OUT/"energy_sensitivity_winners.csv")
    report=pd.read_csv(OUT/"energy_sensitivity_report.csv")
    rows=[]
    for (track,cores),g in w.groupby(["track","n_cores"]):
        original=g.original_winner.value_counts(); adjusted=g.sensitivity_winner.value_counts()
        rows.append({"track":track,"n_cores":cores,"matched_comparisons":len(g),"winner_flips":int(g.winner_changed.sum()),
          "winner_flip_percent":100*g.winner_changed.mean(),
          "original_winner_counts_json":json.dumps(original.to_dict(),sort_keys=True),
          "sensitivity_winner_counts_json":json.dumps(adjusted.to_dict(),sort_keys=True)})
    by=pd.DataFrame(rows); by.to_csv(OUT/"energy_sensitivity_by_track_core.csv",index=False)
    detail=w[w.winner_changed].copy()
    # Compare original winner's EDP against best alternative among the same matched group.
    vals=[]
    for r in detail.itertuples():
        g=report[(report.track==r.track)&(report.workload_id==r.workload_id)&(report.n_cores==r.n_cores)&(report.repeat_idx==r.repeat_idx)]
        edp=g.original_edp.sort_values().to_numpy()
        vals.append((edp[1]-edp[0], (edp[1]-edp[0])/edp[0]*100 if edp[0] else np.nan))
    detail["original_best_vs_runner_up_edp_gap_joule_seconds"]=[x[0] for x in vals]
    detail["original_best_vs_runner_up_gap_percent"]=[x[1] for x in vals]
    detail.to_csv(OUT/"energy_sensitivity_flip_details.csv",index=False)
    return by,detail

def figure_outputs(fixed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(exist_ok=True)
    raw=pd.read_csv(OUT/"unified_benchmark.csv")
    raw=raw[raw.success.astype(str).str.lower().isin(["true","1"])].copy()
    raw["edp"] = raw.wall_time_sec*raw.energy_joules
    edp=pd.read_csv(OUT/"edp_dataset.csv")
    # small multiples by workload; point/interval summaries across repeats, no cross-workload lines
    workloads=sorted(edp.workload_id.unique()); ncol=3; nrow=int(np.ceil(len(workloads)/ncol))
    fig,axs=plt.subplots(nrow,ncol,figsize=(15,3.2*nrow),squeeze=False)
    for ax,wid in zip(axs.flat,workloads):
        g=raw[raw.workload_id==wid].groupby(["strategy","n_cores"]).edp.agg(["median","min","max"]).reset_index()
        for s,ss in g.groupby("strategy"):
            ss=ss.sort_values("n_cores"); ax.errorbar(ss.n_cores,ss["median"],yerr=[ss["median"]-ss["min"],ss["max"]-ss["median"]],marker="o",label=s,capsize=2)
        ax.set(title=wid,yscale="log",xlabel="Core count",ylabel="EDP (J·s)")
    for ax in axs.flat[len(workloads):]: ax.remove()
    handles,labels=axs.flat[0].get_legend_handles_labels(); fig.legend(handles,labels,loc="upper center",ncol=5)
    fig.suptitle("EDP by core count and workload (median; min–max across repeats)",y=1.01); fig.tight_layout()
    savefig(fig,"edp_vs_core_count")
    # heatmap observed best strategy by workload/core
    idx=edp.groupby(["workload_id","n_cores"]).median_edp_joule_seconds.idxmin()
    win=edp.loc[idx].pivot(index="workload_id",columns="n_cores",values="strategy")
    order={x:i for i,x in enumerate(sorted(edp.strategy.unique()))}; enc=win.apply(lambda c:c.map(order))
    cmap=plt.get_cmap("tab10",max(5,len(order)))
    fig,ax=plt.subplots(figsize=(8,8)); masked=np.ma.masked_invalid(enc.to_numpy(dtype=float)); ax.imshow(masked,aspect="auto",cmap=cmap,vmin=-.5,vmax=len(order)-.5)
    ax.set_xticks(np.arange(len(enc.columns)),enc.columns); ax.set_yticks(np.arange(len(enc.index)),enc.index)
    for yi,wid in enumerate(enc.index):
        for xi,col in enumerate(enc.columns):
            ax.text(xi+.5,yi+.5,win.loc[wid,col] if pd.notna(win.loc[wid,col]) else "Unavailable",ha="center",va="center",fontsize=7)
    ax.set_title("Observed lowest EDP strategy (not a guarantee for unseen workloads)"); ax.set_xlabel("Evaluated core count"); ax.set_ylabel("Workload")
    savefig(fig,"best_strategy_heatmap")
    fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(len(fixed)); means=fixed.mean_regret.to_numpy()
    ax.errorbar(x,means,yerr=[means-fixed.regret_ci_lower, fixed.regret_ci_upper-means],fmt="o",capsize=5)
    ax.set_xticks(x,fixed.strategy); ax.set_ylabel("Mean EDP regret (ratio − 1)"); ax.set_title("Fixed strategy regret (workload bootstrap 95% CI)"); ax.grid(axis="y",alpha=.3)
    savefig(fig,"fixed_strategy_regret_ci")
    fig,ax=plt.subplots(figsize=(8,6))
    for s,g in raw.groupby("strategy"):
        ax.scatter(g.wall_time_sec,g.energy_joules,s=16,alpha=.55,label=s)
    ax.set(xscale="log",yscale="log",xlabel="Runtime (s, per run)",ylabel="Energy (J, per run)",title="Runtime versus total energy (individual successful runs)"); ax.legend()
    savefig(fig,"runtime_vs_energy")

def savefig(fig,name):
    fig.savefig(PLOTS/f"{name}.png",dpi=300,bbox_inches="tight"); fig.savefig(PLOTS/f"{name}.pdf",bbox_inches="tight"); import matplotlib.pyplot as plt; plt.close(fig)

def summary_outputs(paired,sig,fixed,by,details):
    files={"memberA":"results/memberA/results.csv","memberB":"results/memberB/kernelshap_benchmark.csv","unified":"results/memberC/unified_benchmark.csv"}
    raws={k:pd.read_csv(ROOT/v) for k,v in files.items()}
    unified=raws["unified"]
    summary={"generated_utc":datetime.now(timezone.utc).isoformat(),"python_version":platform.python_version(),"provenance_sha256":{k:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in files.items()},
      "benchmark_counts":{"valid_runs":int(unified.success.astype(str).str.lower().isin(["true","1"]).sum()),"run_counts_by_track":{k:len(v) for k,v in raws.items() if k!="unified"},
        "configurations_by_track":{"memberA":int(raws["memberA"].groupby(["model","n_rows","strategy","n_cores"]).ngroups),"memberB":int(raws["memberB"].groupby(["model","feature_count","n_rows","background_size","strategy","n_cores"]).ngroups)},
        "workloads":int(unified.workload_id.nunique()),"workloads_by_track":{str(k):int(v.workload_id.nunique()) for k,v in unified.groupby("track")}},
      "scheduler_vs_joblib":sig.iloc[0].to_dict(),"nested_scheduler_overall":pd.read_csv(OUT/"scheduler_nested_evaluation.csv").query("record_type == 'summary' and scope == 'overall'").iloc[0][["n_decision_points","n_oracle_hits","oracle_hit_rate","mean_edp_regret","median_edp_regret","mean_edp_ratio_to_oracle","geomean_edp_ratio_to_oracle"]].to_dict(),"fixed_strategy_estimates":fixed.to_dict(orient="records"),
      "sensitivity":{"matched_comparisons":int(len(pd.read_csv(OUT/"energy_sensitivity_winners.csv"))),"winner_flips":int(pd.read_csv(OUT/"energy_sensitivity_winners.csv").winner_changed.sum()),"breakdown":by.to_dict(orient="records"),"flip_details_count":len(details),"formula":"adjusted_energy = measured_total_energy_j - 10 W * runtime_s; adjusted EDP = adjusted_energy * runtime_s; nonpositive adjusted energy excluded from adjusted winner ranking","warning":"Hypothetical sensitivity only; not measured CPU-only energy."},
      "energy_measurement_limitations":"Historical total energy has heterogeneous/limited component diagnostics; GPU energy was unsupported/unavailable in relevant measurements and is not treated as measured zero.","inference_limit":"18 held-out workloads in one nested LOWO evaluation; folds are not independent retrainings; results do not establish general scheduler superiority or equivalence."}
    # Raw audit numbers with explicit nullable values.
    summary["raw_quality"]={}
    for k,df in raws.items():
        summary["raw_quality"][k]={"rows":len(df),"missing_cells":int(df.isna().sum().sum()),"duplicate_run_ids":int(df.run_id.duplicated().sum()) if "run_id" in df else None,"failed_or_invalid":int((~df.success.astype(str).str.lower().isin(["true","1"])).sum()) if "success" in df else None}
    audit_path=OUT/"validation_summary.csv"
    if audit_path.exists(): summary["raw_quality"]["audit_report"]=pd.read_csv(audit_path).to_dict(orient="records")
    measured={"valid_runs":972,"memberA_runs":360,"memberB_runs":612,"configurations":324,"workloads":18}
    actual={"valid_runs":summary["benchmark_counts"]["valid_runs"],"memberA_runs":len(raws["memberA"]),"memberB_runs":len(raws["memberB"]),"configurations":sum(summary["benchmark_counts"]["configurations_by_track"].values()),"workloads":summary["benchmark_counts"]["workloads"]}
    summary["expected_count_checks"]={k:{"expected_reference":v,"actual":actual[k],"matches":actual[k]==v} for k,v in measured.items()}
    def clean(value):
        if isinstance(value,dict): return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [clean(v) for v in value]
        if isinstance(value,(np.floating,float)): return None if not np.isfinite(value) else float(value)
        if isinstance(value,(np.integer,)): return int(value)
        if pd.isna(value): return None
        return value
    (ROOT/"results_summary.json").write_text(json.dumps(clean(summary),indent=2,allow_nan=False)+"\n")
    st=sig.iloc[0]
    report=("# Member C research results\n\n"
      f"Scheduler regret averaged {st.mean_scheduler_regret:.4f}; Joblib regret averaged {st.mean_joblib_regret:.4f}. "
      f"The paired scheduler-minus-Joblib difference was {st.mean_difference_scheduler_minus_joblib:.4f} "
      f"(workload bootstrap 95% percentile CI {st.difference_ci95_percentile_lower:.4f} to {st.difference_ci95_percentile_upper:.4f}; "
      f"Wilcoxon signed-rank W={st.wilcoxon_statistic:.1f}, two-sided p={st.wilcoxon_p_value:.4g}, n={int(st.n_workloads)} workloads).\n\n"
      "The interval includes zero and the test is not significant. This does not establish equivalence. "
      "The observations are 18 held-out workloads from one nested LOWO evaluation; the folds are not independent full-pipeline retrainings, "
      "and inference is limited to this benchmark space. Four decision points were averaged within each workload before paired inference.\n\n"
      f"The 10 W RAM sensitivity analysis had {len(pd.read_csv(OUT/'energy_sensitivity_winners.csv'))} matched groups and {int(pd.read_csv(OUT/'energy_sensitivity_winners.csv').winner_changed.sum())} winner flips. "
      f"Among flips, the original best-versus-runner-up EDP gap had median {details.original_best_vs_runner_up_gap_percent.median():.2f}% "
      f"and ranged {details.original_best_vs_runner_up_gap_percent.min():.2f}%–{details.original_best_vs_runner_up_gap_percent.max():.2f}%. "
      "The assumed adjustment is E_adj = E_measured − 10 W × runtime and EDP_adj = E_adj × runtime; nonpositive adjusted energy is excluded. "
      "This hypothetical adjustment is not measured CPU-only energy. Historical energy component diagnostics are incomplete and GPU energy was unsupported/unavailable, not a measured zero.\n\n"
      "The benchmark-optimal strategy is the lowest EDP among observed configurations and serves only as an evaluation reference; it is not known in advance for a new workload. See results_summary.json and the generated tables for machine-readable estimates.")
    (OUT/"memberc_research_report.md").write_text(report+"\n")

def main():
    paired,sig,fixed=stats_outputs(); by,details=sensitivity_outputs(); figure_outputs(fixed); summary_outputs(paired,sig,fixed,by,details)
    print("Generated workload-level statistics, sensitivity tables, PNG/PDF figures, results_summary.json")
if __name__=="__main__": main()
