"""
scripts/memberC/evaluate_core_cap.py

Member C — Infrastructure — Stage C4.5: predictive CORE-CAP feasibility test (Joblib only).

Question: given a core budget M, can a SIMPLE workload-aware rule predict, BEFORE running,
when using fewer than M cores is beneficial?  Feasibility experiment only — not a scheduler.

Reads, read-only:
    results/memberC/edp_dataset.csv
    results/memberC/joint_configuration_analysis.csv   (optional cross-check of the descriptive numbers)
Writes (new files only):
    results/memberC/core_cap_evaluation.csv     one row per (budget, rule, workload)
    results/memberC/core_cap_summary.csv        descriptive / classification / performance / noise / decision

Target (derived from measurements; used ONLY for training-workload labels and post-hoc scoring)
  ratio_k  = joblib_EDP(k) / joblib_EDP(M)  for k in {1,2,4,8}, k <= M     (Joblib rows ONLY)
  label    = 1 if min_{k<M} ratio_k <= 0.90 (>= 10% saving), else 0         (fixed, not revisited)
  best_ratio = min_{k<=M} ratio_k  (continuous oracle quantity)

Inputs a rule may use: track, model, n_rows, n_features, feature_count, background_size, M.
NOT used: strategy, measured EDP/time/energy of the held-out workload, optimal core count,
oracle labels, Pareto status. (M is constant within a budget analysis.)

Rules (all pre-declared; nothing is tuned)
  A  always use M cores
  B  best single fixed core count on the TRAINING workloads (min geomean ratio_k; ties -> larger k)
  C  PRIMARY interpretable rule: (track, model) family majority. If more than half of the training
     workloads in the held-out workload's family have label 1, cap to the k that minimises geomean
     ratio_k over that family's positive training workloads; otherwise use M.
  D  secondary: DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, random_state=0) on the
     allowed features; if it predicts a cap, use the k minimising geomean ratio_k over all positive
     training workloads.
  ORACLE reference: best measured Joblib core count <= M (uses held-out data; upper bound only).

Validation: 18-fold Leave-One-Workload-Out. Each fold derives the rule from the other 17 complete
workloads; the held-out workload's measurements are read only after the prediction is fixed.
No row-level splitting anywhere.

Decision (declared before running; applied to rule C at M = 8)
  useful       : held-out geomean EDP saving vs Joblib@8 >= 10%  AND  <= 3/18 workloads >= 10% worse
  not confined : >= 2 models each with >= 2 workloads improved by >= 10%
  BUILD CORE-CAP SCHEDULER                  : useful AND not confined
  DO NOT BUILD CORE-CAP SCHEDULER           : geomean saving < 3% OR the harm test fails
  SIGNAL EXISTS BUT DATA IS TOO LIMITED — DO NOT BUILD YET : everything else

Caveats: 18 independent workloads; EDP noise ~30% with 3 repeats; the oracle takes a minimum of
noisy medians, so oracle savings are optimistic. This experiment does not support claims about
arbitrary datasets, models or hardware.

Run:  python scripts/memberC/evaluate_core_cap.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [SCRIPT_DIR.parent.parent, Path.cwd()]

CORES = [1, 2, 4, 8]
BUDGETS = [8, 4]               # 8 primary, 4 secondary
PRIMARY_BUDGET = 8
TARGET_SAVING = 0.10           # label threshold (fixed)
HARM_RATIO = 1.10              # "harmed" = EDP at least 10% worse than Joblib@M
USEFUL_SAVING_PCT, SOME_SAVING_PCT, MAX_HARM_WORKLOADS = 10.0, 3.0, 3
MIN_MODELS, MIN_PER_MODEL = 2, 2
PRIMARY_RULE = "C_family_majority"
RULES = ["A_always_max", "B_best_fixed_cores", "C_family_majority", "D_tree_depth2", "ORACLE_joblib_core_cap"]
CONTEXT = ["track", "model", "n_rows", "n_features", "feature_count", "background_size"]
TREE_RANDOM_STATE = 0


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


def load_joblib_table() -> pd.DataFrame:
    edp = pd.read_csv(find_file("results/memberC/edp_dataset.csv"))
    need = CONTEXT + ["workload_id", "strategy", "n_cores", "median_edp_joule_seconds", "median_wall_time_sec",
                      "median_energy_joules", "mean_edp_joule_seconds", "std_edp_joule_seconds"]
    miss = [c for c in need if c not in edp.columns]
    if miss:
        raise ValueError(f"edp_dataset.csv missing {miss}")
    j = edp[edp["strategy"] == "joblib"]                       # Joblib rows ONLY
    if j.duplicated(["workload_id", "n_cores"]).any():
        raise ValueError("Duplicate joblib rows.")
    rows = []
    for wl, g in j.groupby("workload_id"):
        if sorted(g["n_cores"]) != CORES:
            raise ValueError(f"{wl}: joblib not measured at exactly {CORES} (got {sorted(g['n_cores'])}).")
        g = g.set_index("n_cores")
        r = {"workload_id": wl, **{c: g.iloc[0][c] for c in CONTEXT}}
        for k in CORES:
            r[f"edp_{k}"] = g.loc[k, "median_edp_joule_seconds"]
            r[f"time_{k}"] = g.loc[k, "median_wall_time_sec"]
            r[f"energy_{k}"] = g.loc[k, "median_energy_joules"]
            r[f"cv_{k}"] = g.loc[k, "std_edp_joule_seconds"] / g.loc[k, "mean_edp_joule_seconds"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("workload_id").reset_index(drop=True)


def ratios(W: pd.DataFrame, M: int) -> dict[int, pd.Series]:
    return {k: W[f"edp_{k}"] / W[f"edp_{M}"] for k in CORES if k <= M}


def best_k(ratio_by_k: dict[int, pd.Series], idx, candidates: list[int]) -> int:
    """k in `candidates` minimising geomean ratio over rows `idx`; ties -> larger k."""
    best, best_v = None, None
    for k in sorted(candidates, reverse=True):
        v = float(np.exp(np.log(ratio_by_k[k].loc[idx]).mean()))
        if best_v is None or v < best_v - 1e-12:
            best, best_v = k, v
    return best


def tree_features(W: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=W.index)
    X["is_kernelshap"] = (W["track"] == "kernelshap").astype(float)
    for m in ["LogisticRegression", "RandomForest", "SVM-RBF", "XGBoost"]:
        X[f"model={m}"] = (W["model"] == m).astype(float)
    X["n_rows"] = W["n_rows"].astype(float)
    X["n_features"] = W["n_features"].astype(float)
    X["feature_count"] = W["feature_count"].astype(float).fillna(0.0)
    X["background_size"] = W["background_size"].astype(float).fillna(0.0)
    return X


def run_budget(W: pd.DataFrame, M: int) -> pd.DataFrame:
    rb = ratios(W, M)
    ks = sorted(rb)
    caps = [k for k in ks if k < M]
    ratio_df = pd.DataFrame({k: rb[k] for k in ks})
    label = (ratio_df[caps].min(axis=1) <= 1 - TARGET_SAVING + 1e-12).astype(int)
    best_ratio = ratio_df.min(axis=1)
    oracle_k = ratio_df.apply(lambda r: max(k for k in ks if r[k] <= r.min() + 1e-12), axis=1)  # ties -> larger k
    X = tree_features(W)
    out = []
    for i in W.index:
        train = W.index[W.index != i]                          # complete held-out workload removed
        assert W.loc[i, "workload_id"] not in set(W.loc[train, "workload_id"])
        y_tr = label.loc[train]
        pos_tr = train[y_tr.loc[train] == 1]
        fam = train[(W.loc[train, "track"] == W.loc[i, "track"]) & (W.loc[train, "model"] == W.loc[i, "model"])]

        choice = {"A_always_max": (False, M)}
        kB = best_k(rb, train, ks)
        choice["B_best_fixed_cores"] = (kB < M, kB)
        pos_fam = fam[label.loc[fam] == 1]
        if len(fam) and len(pos_fam) / len(fam) > 0.5:
            choice["C_family_majority"] = (True, best_k(rb, pos_fam, caps))
        else:
            choice["C_family_majority"] = (False, M)
        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, random_state=TREE_RANDOM_STATE)
        tree.fit(X.loc[train], y_tr)
        if int(tree.predict(X.loc[[i]])[0]) == 1 and len(pos_tr):
            choice["D_tree_depth2"] = (True, best_k(rb, pos_tr, caps))
        else:
            choice["D_tree_depth2"] = (False, M)

        # ---- predictions fixed above; held-out measurements are used only below ----
        choice["ORACLE_joblib_core_cap"] = (int(oracle_k.loc[i]) < M, int(oracle_k.loc[i]))
        for rule in RULES:
            cap, k = choice[rule]
            ratio = float(rb[k].loc[i])
            cv_m, cv_k = W.loc[i, f"cv_{M}"], W.loc[i, f"cv_{k}"]
            # rough standard error of log(ratio): relative SEs of two means from 3 repeats, combined in quadrature
            se = float(np.sqrt((cv_m ** 2 + cv_k ** 2) / 3)) if k != M else np.nan
            out.append({
                "budget": M, "rule": rule, "workload_id": W.loc[i, "workload_id"],
                **{c: W.loc[i, c] for c in CONTEXT},
                "true_cap_label": int(label.loc[i]), "predicted_cap": int(cap), "chosen_cores": k,
                "oracle_cores": int(oracle_k.loc[i]), "joblib_edp_at_M": W.loc[i, f"edp_{M}"],
                "chosen_edp": W.loc[i, f"edp_{k}"], "edp_ratio_vs_joblib_at_M": ratio,
                "pct_saving_vs_joblib_at_M": (1 - ratio) * 100,
                "time_ratio_vs_joblib_at_M": W.loc[i, f"time_{k}"] / W.loc[i, f"time_{M}"],
                "energy_ratio_vs_joblib_at_M": W.loc[i, f"energy_{k}"] / W.loc[i, f"energy_{M}"],
                "oracle_best_ratio": float(best_ratio.loc[i]),
                "approx_se_of_log_ratio": se,
                "saving_exceeds_2se_heuristic": bool(k != M and (-np.log(ratio)) > 2 * se),
            })
    return pd.DataFrame(out)


def cls_metrics(d: pd.DataFrame) -> dict:
    y, p = d["true_cap_label"].to_numpy(), d["predicted_cap"].to_numpy()
    tp, fp, fn, tn = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum()), \
        int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    return {"n": len(d), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "accuracy": (tp + tn) / len(d),
            "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan,
            "true_opportunities": int(y.sum()), "opportunities_detected": tp}


def perf_metrics(d: pd.DataFrame) -> dict:
    r = d["edp_ratio_vs_joblib_at_M"]
    return {"n_workloads": len(d), "mean_edp_ratio": float(r.mean()), "median_edp_ratio": float(r.median()),
            "geomean_edp_ratio": float(np.exp(np.log(r).mean())),
            "geomean_saving_pct": float((1 - np.exp(np.log(r).mean())) * 100),
            "mean_saving_pct": float(d["pct_saving_vs_joblib_at_M"].mean()),
            "mean_time_ratio": float(d["time_ratio_vs_joblib_at_M"].mean()),
            "mean_energy_ratio": float(d["energy_ratio_vs_joblib_at_M"].mean()),
            "share_improved_ge10pct": float((r <= 1 - TARGET_SAVING + 1e-12).mean()),
            "share_harmed_ge10pct": float((r >= HARM_RATIO - 1e-12).mean()),
            "n_improved_ge10pct": int((r <= 1 - TARGET_SAVING + 1e-12).sum()),
            "n_harmed_ge10pct": int((r >= HARM_RATIO - 1e-12).sum())}


def scopes(d: pd.DataFrame):
    yield "overall", d
    yield "tree", d[d["track"] == "tree"]
    yield "kernelshap", d[d["track"] == "kernelshap"]
    for m in ["RandomForest", "XGBoost", "SVM-RBF", "LogisticRegression"]:
        yield f"model:{m}", d[d["model"] == m]


def main() -> None:
    W = load_joblib_table()
    out = output_dir()
    pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)
    print("=" * 92)
    print("MEMBER C — C4.5 PREDICTIVE CORE-CAP FEASIBILITY (Joblib only, nested LOWO)")
    print("=" * 92)
    print(f"Workloads: {len(W)} | joblib measured at cores {CORES} for every workload | primary budget M=8, secondary M=4")

    # ---------------------------------------------------------------- PART 3 descriptive
    r8 = ratios(W, 8)
    D = W[["workload_id", "track", "model", "n_rows", "feature_count", "background_size"]].copy()
    for k in CORES:
        D[f"joblib_edp@{k}"] = W[f"edp_{k}"]
    ratio8 = pd.DataFrame(r8)
    D["best_joblib_cores"] = ratio8.apply(lambda r: max(k for k in CORES if r[k] <= r.min() + 1e-12), axis=1)
    D["best_joblib_edp"] = W[[f"edp_{k}" for k in CORES]].min(axis=1)
    D["saving_vs_joblib@8_pct"] = (1 - ratio8.min(axis=1)) * 100
    D["label_cap>=10%"] = (D["saving_vs_joblib@8_pct"] >= 10 - 1e-9).astype(int)
    print("\n--- PART 3: DESCRIPTIVE TABLE (measured Joblib EDP, budget 8) ---")
    show = D.copy(); show["workload_id"] = show["workload_id"].str.replace("kernelshap|", "ks|", regex=False)
    print(show.drop(columns=["track"]).to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    pos = D[D["label_cap>=10%"] == 1]
    print(f"\nWorkloads with >=10% saving from a Joblib core cap at M=8: {len(pos)}/18 "
          f"(KernelSHAP {int((pos['track']=='kernelshap').sum())}/12, TreeSHAP {int((pos['track']=='tree').sum())}/6)")
    print("By model (positives / workloads | mean saving %):")
    for m, g in D.groupby("model"):
        print(f"  {m:<20s} {int(g['label_cap>=10%'].sum())}/{len(g)} | mean saving {g['saving_vs_joblib@8_pct'].mean():.1f}% | "
              f"best cores {g['best_joblib_cores'].tolist()}")
    top = pos["model"].value_counts()
    print(f"Share of all positives coming from one model ({top.index[0]}): {top.iloc[0] / max(len(pos),1):.0%}"
          f" -> the signal is {'essentially one model/track pattern' if top.iloc[0] / max(len(pos),1) >= 0.6 else 'spread across models'}.")
    X_full = tree_features(W)
    lab8 = pd.Series(D["label_cap>=10%"].to_numpy(), index=W.index)
    print("\n(Descriptive only — depth-2 tree fit on all 18 workloads to show what it would split on; NOT an evaluation):")
    print(export_text(DecisionTreeClassifier(max_depth=2, min_samples_leaf=3, random_state=TREE_RANDOM_STATE).fit(X_full, lab8),
                      feature_names=list(X_full.columns)))

    # optional cross-check with C4 output
    try:
        c4 = pd.read_csv(find_file("results/memberC/joint_configuration_analysis.csv"))
        c4 = c4[(c4["variant"] == "as_measured") & (c4["max_cores"] == 8)].set_index("workload_id")
        diff = (c4["pct_improvement_core_selection_only"].reindex(W["workload_id"]).to_numpy() - D["saving_vs_joblib@8_pct"].to_numpy())
        print(f"Cross-check vs C4 core-selection-only saving at M=8: max abs difference = {np.nanmax(np.abs(diff)):.2e} pct points")
    except FileNotFoundError:
        print("NOTE: joint_configuration_analysis.csv not found; cross-check skipped.")

    # ---------------------------------------------------------------- PART 5-7 nested LOWO
    eval_rows = [run_budget(W, M) for M in BUDGETS]
    ev = pd.concat(eval_rows, ignore_index=True)
    ev.to_csv(out / "core_cap_evaluation.csv", index=False)

    summary = []
    for _, r in D.iterrows():
        summary.append({"section": "descriptive_workload", "budget": 8, "item": r["workload_id"],
                        "best_joblib_cores": r["best_joblib_cores"], "saving_vs_joblib_at_M_pct": r["saving_vs_joblib@8_pct"],
                        "true_cap_label": r["label_cap>=10%"]})
    for M in BUDGETS:
        for rule in RULES:
            d = ev[(ev["budget"] == M) & (ev["rule"] == rule)]
            for scope, s in scopes(d):
                summary.append({"section": "performance", "budget": M, "rule": rule, "scope": scope, **perf_metrics(s)})
            summary.append({"section": "classification", "budget": M, "rule": rule, "scope": "overall", **cls_metrics(d)})
            for scope in ["tree", "kernelshap"]:
                summary.append({"section": "classification", "budget": M, "rule": rule, "scope": scope,
                                **cls_metrics(d[d["track"] == scope])})
        # noise check (descriptive): oracle opportunities and the primary rule's realised savings
        orc = ev[(ev["budget"] == M) & (ev["rule"] == "ORACLE_joblib_core_cap")]
        prim = ev[(ev["budget"] == M) & (ev["rule"] == PRIMARY_RULE) & (ev["predicted_cap"] == 1)]
        for name, d in [("oracle_opportunities", orc[orc["true_cap_label"] == 1]),
                        ("primary_rule_predicted_caps", prim)]:
            sv = d["pct_saving_vs_joblib_at_M"]
            summary.append({"section": "noise_check", "budget": M, "item": name, "n_caps": len(d),
                            "n_ge10pct": int((sv >= 10).sum()), "n_ge20pct": int((sv >= 20).sum()),
                            "n_ge30pct": int((sv >= 30).sum()),
                            "n_exceeding_2se_heuristic": int(d["saving_exceeds_2se_heuristic"].sum()),
                            "n_harmful_caps_lt_0pct": int((sv < 0).sum())})
    summ_df_noise_cv = float(np.median(np.concatenate([W[f"cv_{k}"].to_numpy() for k in CORES])))

    # ------------------------------------------------------------ console report
    for M in BUDGETS:
        tag = "PRIMARY" if M == PRIMARY_BUDGET else "SECONDARY"
        print(f"\n{'=' * 92}\nBUDGET M={M} ({tag}) — nested LOWO, 18 held-out workloads\n{'=' * 92}")
        n_pos = int(ev[(ev['budget'] == M) & (ev['rule'] == 'A_always_max')]['true_cap_label'].sum())
        print(f"True >=10% core-cap opportunities at M={M}: {n_pos}/18")
        print("\n[A] CORE-CAP CLASSIFICATION (overall)")
        rows = []
        for rule in RULES[1:4]:
            c = cls_metrics(ev[(ev["budget"] == M) & (ev["rule"] == rule)])
            rows.append({"rule": rule, **{k: c[k] for k in ["accuracy", "precision", "recall", "tp", "fp", "fn", "tn", "opportunities_detected", "true_opportunities"]}})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("  (confusion = [[TN FP],[FN TP]]; precision NaN when no cap was predicted; rule A never predicts a cap)")
        print("\n[B] ACTUAL PERFORMANCE vs Joblib@M (ratio < 1 is better; overall)")
        rows = []
        for rule in RULES:
            p = perf_metrics(ev[(ev["budget"] == M) & (ev["rule"] == rule)])
            rows.append({"rule": rule, **{k: p[k] for k in ["mean_edp_ratio", "median_edp_ratio", "geomean_edp_ratio",
                                                             "mean_saving_pct", "geomean_saving_pct", "mean_time_ratio",
                                                             "mean_energy_ratio", "n_improved_ge10pct", "n_harmed_ge10pct"]}})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        for title, keys in [("[C] TRACK BREAKDOWN", ["tree", "kernelshap"]),
                            ("[D] MODEL BREAKDOWN", [f"model:{m}" for m in ["RandomForest", "XGBoost", "SVM-RBF", "LogisticRegression"]])]:
            print(f"\n{title} (geomean saving % | improved>=10% / harmed>=10% | n)")
            for rule in ["B_best_fixed_cores", "C_family_majority", "D_tree_depth2", "ORACLE_joblib_core_cap"]:
                d = ev[(ev["budget"] == M) & (ev["rule"] == rule)]
                cells = []
                for scope, s in scopes(d):
                    if scope in keys:
                        p = perf_metrics(s)
                        cells.append(f"{scope.replace('model:','')}: {p['geomean_saving_pct']:.1f}% | {p['n_improved_ge10pct']}/{p['n_harmed_ge10pct']} | {p['n_workloads']}")
                print(f"  {rule:<24s} " + "  ||  ".join(cells))
        print("\n[Noise check — descriptive; 10% label threshold unchanged]")
        for name in ["oracle_opportunities", "primary_rule_predicted_caps"]:
            s = [x for x in summary if x["section"] == "noise_check" and x["budget"] == M and x["item"] == name][0]
            print(f"  {name:<28s} caps={s['n_caps']}  >=10%: {s['n_ge10pct']}  >=20%: {s['n_ge20pct']}  >=30%: {s['n_ge30pct']} "
                  f"| exceed ~2 SE (rough): {s['n_exceeding_2se_heuristic']} | caps that HURT (<0%): {s['n_harmful_caps_lt_0pct']}")
        print("  Per-workload choices (primary rule C):")
        pr = ev[(ev["budget"] == M) & (ev["rule"] == PRIMARY_RULE)]
        for r in pr.itertuples():
            print(f"    {r.workload_id:<42s} true={r.true_cap_label} pred={r.predicted_cap} chose {r.chosen_cores} "
                  f"(oracle {r.oracle_cores}) ratio {r.edp_ratio_vs_joblib_at_M:.2f}")
    print(f"\nMedian per-config EDP noise (std/mean) over joblib rows: {summ_df_noise_cv:.1%}. Savings of 10-20% are within noise "
          f"for individual workloads; ~2 SE flag is a rough heuristic (3 repeats).")

    # ---------------------------------------------------------------- PART 8 decision
    pr = ev[(ev["budget"] == PRIMARY_BUDGET) & (ev["rule"] == PRIMARY_RULE)]
    pm = perf_metrics(pr)
    improved = pr[pr["edp_ratio_vs_joblib_at_M"] <= 1 - TARGET_SAVING + 1e-12]
    models_ok = int((improved["model"].value_counts() >= MIN_PER_MODEL).sum())
    useful = pm["geomean_saving_pct"] >= USEFUL_SAVING_PCT and pm["n_harmed_ge10pct"] <= MAX_HARM_WORKLOADS
    not_confined = models_ok >= MIN_MODELS
    if useful and not_confined:
        decision = "BUILD CORE-CAP SCHEDULER"
    elif pm["geomean_saving_pct"] < SOME_SAVING_PCT or pm["n_harmed_ge10pct"] > MAX_HARM_WORKLOADS:
        decision = "DO NOT BUILD CORE-CAP SCHEDULER"
    else:
        decision = "SIGNAL EXISTS BUT DATA IS TOO LIMITED — DO NOT BUILD YET"
    summary.append({"section": "decision", "budget": PRIMARY_BUDGET, "rule": PRIMARY_RULE, "item": "final_decision",
                    "decision": decision, "geomean_saving_pct": pm["geomean_saving_pct"],
                    "n_harmed_ge10pct": pm["n_harmed_ge10pct"], "n_models_with_ge2_improved": models_ok,
                    "useful": useful, "not_confined": not_confined})
    pd.DataFrame(summary).to_csv(out / "core_cap_summary.csv", index=False)

    print(f"\n{'=' * 92}\nDECISION (primary rule {PRIMARY_RULE}, M={PRIMARY_BUDGET}, held-out)\n{'=' * 92}")
    print(f"  geomean EDP saving vs Joblib@8 : {pm['geomean_saving_pct']:.1f}%   (useful needs >= {USEFUL_SAVING_PCT:.0f}%)")
    print(f"  workloads >=10% better / worse : {pm['n_improved_ge10pct']} / {pm['n_harmed_ge10pct']}  (harm limit {MAX_HARM_WORKLOADS})")
    print(f"  models with >=2 improved       : {models_ok}  (not confined needs >= {MIN_MODELS})")
    print(f"  useful={useful} | not_confined={not_confined}")
    print(f"\nFINAL DECISION: {decision}")
    print("\nCaveats: 18 independent workloads; ~30% EDP noise; oracle savings are optimistic; no claim about arbitrary "
          "datasets/models/hardware. Rules were pre-declared and not tuned.")
    print(f"\nWritten: {out / 'core_cap_evaluation.csv'}\n         {out / 'core_cap_summary.csv'}")
    print("=" * 92)


if __name__ == "__main__":
    main()