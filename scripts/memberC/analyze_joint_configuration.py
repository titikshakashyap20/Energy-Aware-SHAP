"""
scripts/memberC/analyze_joint_configuration.py

Member C — Infrastructure — Stage C4: joint (strategy + core-count) configuration ANALYSIS.

Reads, read-only:
    results/memberC/edp_dataset.csv
Writes (new files only; C3 / C3.5 outputs are not touched):
    results/memberC/joint_configuration_winners.csv
    results/memberC/joint_configuration_analysis.csv
    results/memberC/joint_configuration_summary.csv

THIS IS NOT A PREDICTIVE SCHEDULER EVALUATION.
Everything here is a MEASURED ORACLE: what could be achieved if the benchmark
outcome were known in advance. It is an upper bound on any predictive joint
scheduler. No model is trained, no splitting is done, nothing is fabricated.

Definitions (workload W, core budget M in {1,2,4,8})
----------------------------------------------------
  joblib@M              measured joblib at exactly M cores ("always joblib, use all cores")
  strategy-only oracle  best measured strategy at exactly M cores  (the C3 / C3.5 oracle)
  joint oracle          best measured (strategy, k) over all k <= M
  core-selection only   best measured joblib over k <= M
  improvement %         (1 - candidate_edp / joblib@M_edp) * 100
  time / energy ratio   the chosen configuration's own median time / median energy
                        divided by joblib@M's (EDP is median of per-repeat EDP, so it is
                        not exactly median_time * median_energy)
At M = 1 the joint and strategy-only oracles coincide by construction.

Serial caveat (two variants, both reported)
-------------------------------------------
KernelSHAP serial exists only at 1 core and is never fabricated. TreeSHAP serial was
measured at 1/2/4/8 "cores" but its runtime does not change with the label (repeated
single-core runs; differences are noise), so letting a joint oracle pick serial@8 would
reward noise. Therefore:
  as_measured        every measured row used literally
  serial_1core_only  serial counts only as a 1-core configuration in the JOINT oracle
                     (strategy-only oracle and joblib@M stay as measured, matching C3)

Recommendation rule (declared before running)
---------------------------------------------
  saving  = 1 - geomean(joint_edp / joblib@M_edp) over all 72 workload x budget points
  breadth = share of the 72 points with >= 10% improvement (same margin as C2-B)
  A strong   : saving >= 20% AND breadth >= 50%
  C weak     : saving <  10% OR  breadth <  25%
  B moderate : otherwise
  BUILD JOINT SCHEDULER only if case A holds under BOTH variants; otherwise DO NOT BUILD.

Run:  python scripts/memberC/analyze_joint_configuration.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

ALL_STRATEGIES = ["serial", "multiprocessing", "threading", "joblib", "ray"]
STRAT_RANK = {s: i for i, s in enumerate(ALL_STRATEGIES)}
VARIANTS = ["as_measured", "serial_1core_only"]
OBJECTIVES = {"edp": "median_edp_joule_seconds", "time": "median_wall_time_sec", "energy": "median_energy_joules"}
MARGIN_PCT = 10.0
STRONG_SAVING, STRONG_BREADTH = 0.20, 0.50
WEAK_SAVING, WEAK_BREADTH = 0.10, 0.25
CORE_STRATEGIES = ["multiprocessing", "threading", "joblib", "ray"]  # serial is core-count invariant

CONTEXT = ["track", "model", "n_rows", "n_features", "feature_count", "background_size"]
REQUIRED = CONTEXT + ["workload_id", "strategy", "n_cores", "median_wall_time_sec", "median_energy_joules",
                      "median_edp_joule_seconds", "std_edp_joule_seconds", "mean_edp_joule_seconds"]


def find_file(rel: str) -> Path:
    for root in REPO_ROOT_CANDIDATES:
        p = root / rel
        if p.is_file():
            return p
    raise FileNotFoundError(f"Required input file not found: '{rel}'. Looked under {[str(r) for r in REPO_ROOT_CANDIDATES]}.")


def output_dir() -> Path:
    out = SCRIPT_DIR.parent.parent / "results" / "memberC"
    if not out.exists():
        out = Path.cwd() / "results" / "memberC"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_edp() -> pd.DataFrame:
    df = pd.read_csv(find_file("results/memberC/edp_dataset.csv"))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"edp_dataset.csv missing column(s): {missing}")
    if df.duplicated(["workload_id", "strategy", "n_cores"]).any():
        raise ValueError("Duplicate (workload_id, strategy, n_cores) rows.")
    if (df["median_edp_joule_seconds"] <= 0).any():
        raise ValueError("Non-positive median EDP.")
    ks = df[(df["track"] == "kernelshap") & (df["strategy"] == "serial") & (df["n_cores"] > 1)]
    if len(ks):
        raise ValueError("KernelSHAP serial found at >1 core; expected 1-core only (never fabricate).")
    return df


def configs_for(edp: pd.DataFrame, variant: str) -> pd.DataFrame:
    c = edp.copy()
    if variant == "serial_1core_only":
        c = c[~((c["strategy"] == "serial") & (c["n_cores"] > 1))]
    c["strat_rank"] = c["strategy"].map(STRAT_RANK)
    return c


def best_config(c: pd.DataFrame, key: str) -> pd.Series:
    """Lowest `key`; deterministic ties -> fewer cores, then strategy order."""
    return c.sort_values([key, "n_cores", "strat_rank"], kind="mergesort").iloc[0]


def build_analysis(edp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANTS:
        cfg_all = configs_for(edp, variant)
        for wl, g_full in edp.groupby("workload_id"):
            g_joint = cfg_all[cfg_all["workload_id"] == wl]
            g_full = g_full.assign(strat_rank=g_full["strategy"].map(STRAT_RANK))
            ctx = g_full.iloc[0]
            for M in sorted(g_full.loc[g_full["strategy"] == "joblib", "n_cores"].unique()):
                jb = g_full[(g_full["strategy"] == "joblib") & (g_full["n_cores"] == M)]
                if len(jb) != 1:
                    raise ValueError(f"joblib@{M} missing/duplicated for {wl}")
                jb = jb.iloc[0]
                so = best_config(g_full[g_full["n_cores"] == M], "median_edp_joule_seconds")
                jo = best_config(g_joint[g_joint["n_cores"] <= M], "median_edp_joule_seconds")
                cj = best_config(g_full[(g_full["strategy"] == "joblib") & (g_full["n_cores"] <= M)],
                                 "median_edp_joule_seconds")
                E = "median_edp_joule_seconds"; T = "median_wall_time_sec"; N = "median_energy_joules"
                rows.append({
                    "variant": variant, "workload_id": wl, **{k: ctx[k] for k in CONTEXT}, "max_cores": M,
                    "joblib_edp": jb[E], "joblib_time": jb[T], "joblib_energy": jb[N],
                    "strategy_only_strategy": so["strategy"], "strategy_only_edp": so[E],
                    "joint_strategy": jo["strategy"], "joint_cores": int(jo["n_cores"]), "joint_edp": jo[E],
                    "joint_time": jo[T], "joint_energy": jo[N],
                    "joblib_best_cores": int(cj["n_cores"]), "joblib_best_cores_edp": cj[E],
                    "joint_over_joblib_edp": jo[E] / jb[E],
                    "pct_improvement_joint_vs_joblib": (1 - jo[E] / jb[E]) * 100,
                    "joint_time_ratio_vs_joblib": jo[T] / jb[T],
                    "joint_energy_ratio_vs_joblib": jo[N] / jb[N],
                    "strategy_only_over_joblib_edp": so[E] / jb[E],
                    "pct_improvement_strategy_only_vs_joblib": (1 - so[E] / jb[E]) * 100,
                    "strategy_only_time_ratio_vs_joblib": so[T] / jb[T],
                    "strategy_only_energy_ratio_vs_joblib": so[N] / jb[N],
                    "core_selection_only_over_joblib_edp": cj[E] / jb[E],
                    "pct_improvement_core_selection_only": (1 - cj[E] / jb[E]) * 100,
                    "joint_over_strategy_only_edp": jo[E] / so[E],
                    "pct_incremental_joint_vs_strategy_only": (1 - jo[E] / so[E]) * 100,
                    "joint_uses_fewer_than_max_cores": bool(jo["n_cores"] < M),
                    "joint_equals_joblib_at_max": bool(jo["strategy"] == "joblib" and jo["n_cores"] == M),
                })
    return pd.DataFrame(rows)


def build_winners(edp: pd.DataFrame) -> pd.DataFrame:
    """Unconstrained (all measured cores) winner per workload and objective."""
    rows = []
    for variant in VARIANTS:
        cfg = configs_for(edp, variant)
        for wl, g in cfg.groupby("workload_id"):
            ctx = g.iloc[0]
            for obj, col in OBJECTIVES.items():
                b = best_config(g, col)
                rows.append({"variant": variant, "workload_id": wl, **{k: ctx[k] for k in CONTEXT},
                             "objective": obj, "best_strategy": b["strategy"], "best_cores": int(b["n_cores"]),
                             "median_edp": b["median_edp_joule_seconds"], "median_time": b["median_wall_time_sec"],
                             "median_energy": b["median_energy_joules"]})
    return pd.DataFrame(rows)


def scopes(df: pd.DataFrame):
    yield "overall", df
    yield "tree", df[df["track"] == "tree"]
    yield "kernelshap", df[df["track"] == "kernelshap"]
    for m in sorted(df["model"].unique()):
        yield f"model:{m}", df[df["model"] == m]


def subsets(d: pd.DataFrame):
    yield "all_budgets", d
    yield "budget_ge2", d[d["max_cores"] >= 2]
    for k in sorted(d["max_cores"].unique()):
        yield f"budget_{k}", d[d["max_cores"] == k]


def gsave(ratio: pd.Series) -> float:
    return float((1 - np.exp(np.log(ratio).mean())) * 100) if len(ratio) else np.nan


def headroom(d: pd.DataFrame) -> dict:
    n = len(d)
    if n == 0:
        return {}
    mp = d["max_cores"] > 1
    return {
        "n_points": n, "n_workloads": int(d["workload_id"].nunique()),
        "joint_mean_pct_improvement_vs_joblib": float(d["pct_improvement_joint_vs_joblib"].mean()),
        "joint_median_pct_improvement_vs_joblib": float(d["pct_improvement_joint_vs_joblib"].median()),
        "joint_geomean_saving_pct_vs_joblib": gsave(d["joint_over_joblib_edp"]),
        "joint_share_ge_margin_vs_joblib": float((d["pct_improvement_joint_vs_joblib"] >= MARGIN_PCT).mean()),
        "joint_share_joblib_at_max_is_optimal": float(d["joint_equals_joblib_at_max"].mean()),
        "joint_mean_time_ratio_vs_joblib": float(d["joint_time_ratio_vs_joblib"].mean()),
        "joint_mean_energy_ratio_vs_joblib": float(d["joint_energy_ratio_vs_joblib"].mean()),
        "strategy_only_geomean_saving_pct_vs_joblib": gsave(d["strategy_only_over_joblib_edp"]),
        "strategy_only_share_ge_margin_vs_joblib": float((d["pct_improvement_strategy_only_vs_joblib"] >= MARGIN_PCT).mean()),
        "core_selection_only_geomean_saving_pct": gsave(d["core_selection_only_over_joblib_edp"]),
        "core_selection_only_share_ge_margin": float((d["pct_improvement_core_selection_only"] >= MARGIN_PCT).mean()),
        "incremental_joint_over_strategy_only_geomean_saving_pct": gsave(d["joint_over_strategy_only_edp"]),
        "incremental_share_ge_margin": float((d["pct_incremental_joint_vs_strategy_only"] >= MARGIN_PCT).mean()),
        "share_joint_uses_fewer_than_max_cores_(budget>1)": float(d.loc[mp, "joint_uses_fewer_than_max_cores"].mean()) if mp.any() else np.nan,
    }


def classify(saving: float, breadth: float) -> str:
    if saving >= STRONG_SAVING * 100 and breadth >= STRONG_BREADTH:
        return "A_strong"
    if saving < WEAK_SAVING * 100 or breadth < WEAK_BREADTH:
        return "C_weak"
    return "B_moderate"


def main() -> None:
    edp = load_edp()
    analysis = build_analysis(edp)
    winners = build_winners(edp)
    out = output_dir()

    summary = []
    def add(section, **kw):
        summary.append({"section": section, **kw})

    # ---- headroom (joint / strategy-only / core-selection-only vs fixed joblib@M) ----
    for variant in VARIANTS:
        av = analysis[analysis["variant"] == variant]
        for scope, sub in scopes(av):
            for sname, s in subsets(sub):
                add("headroom", variant=variant, scope=scope, subset=sname, **headroom(s))

    # ---- who wins, where (unconstrained, per objective) ----
    for variant in VARIANTS:
        w = winners[winners["variant"] == variant]
        for scope, sub in scopes(w):
            n = sub["workload_id"].nunique()
            for obj, so in sub.groupby("objective"):
                for col, section in [("best_strategy", "wins_by_strategy"), ("best_cores", "wins_by_cores")]:
                    for item, cnt in so[col].value_counts().items():
                        add(section, variant=variant, scope=scope, objective=obj, item=str(item),
                            count=int(cnt), share=cnt / n)
            e = sub[sub["objective"] == "edp"]
            for (s, k), cnt in e.groupby(["best_strategy", "best_cores"]).size().items():
                add("strategy_x_cores_edp_unconstrained", variant=variant, scope=scope, item=f"{s}|{k}",
                    count=int(cnt), share=cnt / n)
        # per workload x budget (72 decision points)
        av = analysis[analysis["variant"] == variant]
        for scope, sub in scopes(av):
            for col, section in [("joint_strategy", "budget_wins_by_strategy"), ("joint_cores", "budget_wins_by_cores")]:
                for item, cnt in sub[col].value_counts().items():
                    add(section, variant=variant, scope=scope, objective="edp", item=str(item),
                        count=int(cnt), share=cnt / len(sub))

    # ---- core-count matters? interaction with strategy (serial excluded: core-count invariant) ----
    inter_rows = []
    for wl, g in edp.groupby("workload_id"):
        best_k = {}
        for s in CORE_STRATEGIES:
            gs = g[g["strategy"] == s].sort_values(["median_edp_joule_seconds", "n_cores"])
            best_k[s] = int(gs.iloc[0]["n_cores"])
        so_win = {M: best_config(g[g["n_cores"] == M].assign(strat_rank=g["strategy"].map(STRAT_RANK)),
                                 "median_edp_joule_seconds")["strategy"] for M in sorted(g["n_cores"].unique())}
        inter_rows.append({"workload_id": wl, "track": g["track"].iloc[0], "model": g["model"].iloc[0],
                           "n_distinct_best_k_across_strategies": len(set(best_k.values())),
                           "n_distinct_strategy_only_winners_across_budgets": len(set(so_win.values()))})
    inter = pd.DataFrame(inter_rows)
    for scope, sub in [("overall", inter), ("tree", inter[inter["track"] == "tree"]),
                       ("kernelshap", inter[inter["track"] == "kernelshap"])]:
        n = len(sub)
        a = int((sub["n_distinct_best_k_across_strategies"] > 1).sum())
        b = int((sub["n_distinct_strategy_only_winners_across_budgets"] > 1).sum())
        add("interaction", scope=scope, item="workloads where best core count differs across strategies",
            count=a, share=a / n)
        add("interaction", scope=scope, item="workloads where strategy-only winner changes with core budget",
            count=b, share=b / n)

    # ---- recommendation under the pre-declared rule ----
    verdicts = {}
    for variant in VARIANTS:
        o = analysis[analysis["variant"] == variant]
        saving = gsave(o["joint_over_joblib_edp"])
        breadth = float((o["pct_improvement_joint_vs_joblib"] >= MARGIN_PCT).mean())
        verdicts[variant] = (saving, breadth, classify(saving, breadth))
        add("recommendation", variant=variant, scope="overall", item="geomean_saving_pct_vs_joblib", value=saving)
        add("recommendation", variant=variant, scope="overall", item="share_points_ge_10pct", value=breadth)
        add("recommendation", variant=variant, scope="overall", item="case", value=verdicts[variant][2])
    order = {"A_strong": 0, "B_moderate": 1, "C_weak": 2}
    final_case = max((v[2] for v in verdicts.values()), key=lambda c: order[c])
    recommendation = "BUILD JOINT SCHEDULER" if final_case == "A_strong" else "DO NOT BUILD JOINT SCHEDULER"
    add("recommendation", scope="overall", item="final_case_(conservative_of_both_variants)", value=final_case)
    add("recommendation", scope="overall", item="final_recommendation", value=recommendation)

    summary_df = pd.DataFrame(summary)
    lead = ["section", "variant", "scope", "subset", "objective", "item"]
    summary_df = summary_df[[c for c in lead if c in summary_df.columns] +
                            [c for c in summary_df.columns if c not in lead]]
    winners.to_csv(out / "joint_configuration_winners.csv", index=False)
    analysis.to_csv(out / "joint_configuration_analysis.csv", index=False)
    summary_df.to_csv(out / "joint_configuration_summary.csv", index=False)

    # ------------------------------------------------------------------ console
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    fmt = lambda v: f"{v:.2f}"
    print("=" * 90)
    print("MEMBER C — C4 JOINT STRATEGY + CORE-COUNT ANALYSIS  (MEASURED ORACLE, not a predictive result)")
    print("=" * 90)
    nwl = edp["workload_id"].nunique()
    cv = (edp["std_edp_joule_seconds"] / edp["mean_edp_joule_seconds"]).median()
    print(f"Workloads: {nwl} | decision points per variant: {len(analysis) // 2} | configurations: {len(edp)}")
    print(f"Typical run-to-run noise: median std/mean of per-repeat EDP = {cv:.1%} (3 repeats per configuration).")
    print("Oracles pick the minimum of many noisy medians, so oracle gains are optimistic (winner's curse).")

    # sanity: reproduce C2 facts from the same data
    a0 = analysis[analysis["variant"] == "as_measured"]
    n21 = int(((a0["strategy_only_strategy"] != "joblib") & (a0["pct_improvement_strategy_only_vs_joblib"] >= MARGIN_PCT)).sum())
    print(f"\nSanity vs C2: strategy-only oracle beats joblib by >=10% at {n21}/72 points (C2 reported 21/72); "
          f"strategy-only winner changes across budgets for "
          f"{int((inter['n_distinct_strategy_only_winners_across_budgets'] > 1).sum())}/18 workloads (C2: 12/18).")

    print("\n--- [1] JOINT WINNERS per workload (unconstrained, objective = EDP; variant as_measured) ---")
    wt = winners[(winners["variant"] == "as_measured") & (winners["objective"] == "edp")]
    wt2 = winners[(winners["variant"] == "serial_1core_only") & (winners["objective"] == "edp")].set_index("workload_id")
    show = wt[["workload_id", "best_strategy", "best_cores", "median_edp", "median_time", "median_energy"]].copy()
    show["conservative_winner"] = show["workload_id"].map(lambda w: f"{wt2.loc[w, 'best_strategy']}@{wt2.loc[w, 'best_cores']}")
    print(show.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    for obj in ["time", "energy"]:
        wo = winners[(winners["variant"] == "as_measured") & (winners["objective"] == obj)]
        same = int(((wo.set_index("workload_id")[["best_strategy", "best_cores"]]
                     == wt.set_index("workload_id")[["best_strategy", "best_cores"]]).all(axis=1)).sum())
        print(f"  workloads where the {obj}-optimal configuration equals the EDP-optimal one: {same}/18")

    for variant in VARIANTS:
        print(f"\n--- [3-4] WHO WINS (EDP, unconstrained) — {variant} ---")
        w = winners[(winners["variant"] == variant) & (winners["objective"] == "edp")]
        print("by strategy :", w["best_strategy"].value_counts().to_dict())
        print("by cores    :", w["best_cores"].value_counts().sort_index().to_dict())
        print("strategy x cores:", w.groupby(["best_strategy", "best_cores"]).size().to_dict())
        for scope in ["tree", "kernelshap"]:
            ws = w[w["track"] == scope]
            print(f"  {scope:<10s} strategy {ws['best_strategy'].value_counts().to_dict()} | cores "
                  f"{ws['best_cores'].value_counts().sort_index().to_dict()}")
        print("  by model:")
        for m, ws in w.groupby("model"):
            print(f"    {m:<20s} strategy {ws['best_strategy'].value_counts().to_dict()} | cores "
                  f"{ws['best_cores'].value_counts().sort_index().to_dict()}")
    n_fewer = int((winners[(winners['variant'] == 'as_measured') & (winners['objective'] == 'edp')]['best_cores'] < 8).sum())
    print(f"\nWorkloads whose EDP-optimal core count is < 8: {n_fewer}/18. "
          f"Distinct optimal core counts: {[int(k) for k in sorted(winners[(winners['variant']=='as_measured')&(winners['objective']=='edp')]['best_cores'].unique())]}")
    print(f"Best core count differs across strategies (interaction) for "
          f"{int((inter['n_distinct_best_k_across_strategies'] > 1).sum())}/18 workloads (serial excluded).")

    cols = ["scope", "subset", "n_points", "joint_geomean_saving_pct_vs_joblib", "joint_mean_pct_improvement_vs_joblib",
            "joint_share_ge_margin_vs_joblib", "joint_mean_time_ratio_vs_joblib", "joint_mean_energy_ratio_vs_joblib",
            "strategy_only_geomean_saving_pct_vs_joblib", "core_selection_only_geomean_saving_pct",
            "incremental_joint_over_strategy_only_geomean_saving_pct"]
    hd = summary_df[summary_df["section"] == "headroom"]
    for variant in VARIANTS:
        print(f"\n--- [2] HEADROOM vs FIXED JOBLIB@budget — {variant} (EDP saving %, geomean) ---")
        h = hd[(hd["variant"] == variant) & (hd["scope"].isin(["overall", "tree", "kernelshap"]))]
        print(h[cols].to_string(index=False, float_format=fmt))

    print("\n--- [5] DECISION (rule declared in script header) ---")
    for variant, (sv, br, case) in verdicts.items():
        print(f"  {variant:<20s} geomean EDP saving vs Joblib@budget = {sv:.1f}% | points with >=10% saving = {br:.1%} | case {case}")
    print(f"  thresholds: A needs saving >= {STRONG_SAVING:.0%} and breadth >= {STRONG_BREADTH:.0%}; "
          f"C if saving < {WEAK_SAVING:.0%} or breadth < {WEAK_BREADTH:.0%}")
    print("\n  Reminder: these are MEASURED-ORACLE upper bounds. A predictive scheduler can only do worse. C3.5's "
          "strategy-only scheduler captured essentially none of its (smaller) oracle headroom over Joblib.")
    print(f"\nFINAL RECOMMENDATION: {recommendation}   (case {final_case}, conservative of both variants)")
    print(f"\nWritten: {out / 'joint_configuration_winners.csv'}\n         {out / 'joint_configuration_analysis.csv'}"
          f"\n         {out / 'joint_configuration_summary.csv'}")
    print("=" * 90)


if __name__ == "__main__":
    main()