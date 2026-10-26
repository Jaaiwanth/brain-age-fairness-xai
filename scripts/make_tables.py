"""Print the numbers behind every table of the paper from the saved result files.

Nothing is re-trained: each table is read from the result files written by the scripts named in
README ("Paper tables and figures"). Run the analysis scripts first, then:

Usage: python scripts/make_tables.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def j(path):
    return json.loads(path.read_text())


def head(title):
    print(f"\n=== {title} ===")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    P = ap.parse_args().project_dir
    R = P / "results"

    head("Table I - subjects per site (release/splits.csv, ages from the split tables)")
    s = pd.read_csv(HERE.parent / "release" / "splits.csv")
    ages = pd.concat([pd.read_csv(P / f"ixi_{x}.csv")[["IXI_ID", "AGE"]] for x in ["train", "val", "test"]])
    s = s.merge(ages.rename(columns={"AGE": "age"}), on="IXI_ID")
    s["heldout"] = s.split.isin(["val", "test"])
    print(s.groupby("site").agg(n=("split", "size"), mean_age=("age", "mean"), heldout_n=("heldout", "sum"),
                                test_n=("split", lambda x: (x == "test").sum())).round(1).to_string())

    head("Table II - accuracy")
    p5 = j(R / "phase5" / "summary.json")
    for k in ["raw", "corrected"]:
        print(f"main model, test ({k}): MAE {p5[k]['metrics']['mae']:.2f} {np.round(p5[k]['ci95']['mae'], 2)} r {p5[k]['metrics']['r']:.3f}")
    cvj = j(R / "phase10j_cv_analysis" / "summary.json")
    rn = j(R / "phase11b_revision_numbers" / "summary.json")["cv_seed0_oof_mae"]
    print(f"CV out-of-fold seed0: MAE {cvj['per_seed']['seed0']['mae']:.2f} {rn['ci95']} r {cvj['per_seed']['seed0']['r']:.3f}")
    e = cvj["locked_test_fold_ensemble"]
    print(f"CV ensemble ({e['n_models']} models), test: MAE {e['mae']:.2f} r {e['r']:.3f}")
    z = j(R / "phase3" / "summary.json")["val_raw"]
    print(f"zero-shot, validation: MAE {z['mae']:.2f} r {z['r']:.3f}")

    head("Table III - group effects and verdicts (margin 2)")
    v = pd.read_csv(R / "phase10d_verdicts" / "graded_verdicts.csv")
    print(v[["analysis", "comparison", "outcome", "est", "ci95", "mde80", "p_holm", "verdict"]].round(2).to_string(index=False))

    head("Table S1 - verdicts under margins 1.5 / 2 / 2.5")
    vs = {m: pd.read_csv(R / "phase10d_verdicts" / f) for m, f in
          [(1.5, "graded_verdicts_margin1.5.csv"), (2.0, "graded_verdicts.csv"), (2.5, "graded_verdicts_margin2.5.csv")]}
    t = vs[2.0][["analysis", "comparison", "outcome"]].copy()
    for m, d in vs.items():
        t[f"margin {m}"] = d.verdict
    print(t.to_string(index=False))

    head("Table IV - site probe, held-out (balanced accuracy)")
    pr = j(R / "phase9a_site_probe" / "summary.json")["probe"]
    for k in ["demographics", "intensity", "sfcn_pretrained", "sfcn_finetuned_resid"]:
        print(f"{k}: {pr[k]['balanced_accuracy']:.3f} [{pr[k]['ci_low']:.3f}, {pr[k]['ci_high']:.3f}]")
    er = j(R / "phase10f_site_erasure" / "summary.json")
    print("permutation null:", {k: round(x, 3) for k, x in er["probe_permutation_null"].items()})
    print("CV within-fold mean:", round(cvj["site_probe_within_fold_bal_acc"]["mean"], 3))

    head("Table V - explanations tested (held-out unless noted)")
    for k, x in j(R / "phase9b_stress_test" / "summary.json")["headline_vs_own_site_control"].items():
        print(f"appearance {k}: {x['mean_delta']:+.2f}")
    for k, x in j(R / "phase9d_blur_to_iop" / "summary.json")["groups"].items():
        print(f"blur to IOP, {k}:", {kk: vv for kk, vv in x.items() if kk != "half"})
    print(pd.read_csv(R / "phase10b_age_robustness" / "iop_offset_robustness.csv").round(2).to_string(index=False))
    for k, x in j(R / "phase10e_cohort" / "summary.json")["iop_offset_with_cohort_covariates"].items():
        print(f"cohort covariates, {k}: {x}")
    for x in er["versions"]:
        print(f"{x['features']}: offset {x['iop_vs_guys']:+.2f} {np.round(x['ci'], 2)} MAE {x['heldout_mae']:.2f} "
              f"r {x.get('heldout_r_age_pred', float('nan')):.3f} IOP gap SD {x.get('iop_heldout_bag_sd', float('nan')):.2f}")
    for k, x in j(R / "phase11a_review_checks" / "summary.json")["zero_shot"].items():
        est = x.get("iop_vs_guys", x.get("iop_minus_guys"))
        print(f"zero-shot, {k}: {est:+.2f} {np.round(x.get('ci_bootstrap', x.get('ci')), 2)} n={x.get('n', x.get('n_pairs'))}")

    head("Table VI - deletion test")
    print(pd.DataFrame(j(R / "phase10g_deletion_baselines" / "summary.json")["summary"]).to_string(index=False))

    head("Table VII - 90% coverage, CV out-of-fold, Wilson CIs")
    print(pd.DataFrame(j(R / "phase11b_revision_numbers" / "summary.json")["coverage_90_wilson"]).to_string(index=False))
    print(pd.DataFrame(j(R / "phase11a_review_checks" / "summary.json")["split_conformal_90"]).to_string(index=False))

    head("Table VIII - passport (release/passport.json)")
    for k, c in j(HERE.parent / "release" / "passport.json")["checks"].items():
        print(f"{k}: {c['verdict']}")


if __name__ == "__main__":
    main()
