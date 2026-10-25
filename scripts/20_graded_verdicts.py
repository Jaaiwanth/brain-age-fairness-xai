"""Phase 10d - Pre-specified, graded fairness verdicts (equivalence tests, power, multiplicity).

Replaces binary PASS/FAIL with a protocol. For each group comparison (age-adjusted OLS, HC3):
    margin  = 2 years (smallest difference of interest; about half the model's MAE, fixed in advance)
    PASS          equivalence shown: the 90% CI lies inside +/- margin (two one-sided tests, alpha 0.05)
    INCONCLUSIVE  the 95% CI includes 0 but equivalence is not shown
    CONCERN       the 95% CI excludes 0 but the effect is within the margin, or it does not replicate
    FAIL          the 95% CI excludes 0, |effect| >= margin, and the effect replicates in an independent split
Also reported: the minimum detectable effect at 80% power (2.80 x SE) and Holm-adjusted p-values
across every group comparison in the family.

Splits: primary = the locked test set; replication = the validation set (independent subjects).
If the Phase 10a cross-validation is present, the same verdicts are also computed on all CV
out-of-sample predictions (primary) with the locked test set as replication.

Usage: python scripts/20_graded_verdicts.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

MARGIN = 2.0
TERMS = {"Male vs Female": "C(sex, Treatment('Female'))[T.Male]",
         "HH vs Guys": "C(site, Treatment('Guys'))[T.HH]",
         "IOP vs Guys": "C(site, Treatment('Guys'))[T.IOP]"}
FORMULA = "{} ~ C(sex, Treatment('Female')) + C(site, Treatment('Guys')) + age"


def effects(df):
    rows = []
    for outcome in ["bag", "abs_error"]:
        m = smf.ols(FORMULA.format(outcome), data=df).fit(cov_type="HC3")
        for name, t in TERMS.items():
            est, se = float(m.params[t]), float(m.bse[t])
            rows.append({"comparison": name, "outcome": outcome, "est": est, "se": se, "p": float(m.pvalues[t]),
                         "ci95": [est - 1.96 * se, est + 1.96 * se], "ci90": [est - 1.645 * se, est + 1.645 * se],
                         "p_tost": float(max(norm.sf((est + MARGIN) / se), norm.cdf((est - MARGIN) / se))),
                         "mde80": 2.80 * se})
    return pd.DataFrame(rows)


def verdict(r, rep):
    lo, hi = r.ci95
    if -MARGIN < r.ci90[0] and r.ci90[1] < MARGIN:
        return "PASS"
    if lo <= 0 <= hi:
        return "INCONCLUSIVE"
    replicated = np.sign(rep.est) == np.sign(r.est) and not (rep.ci95[0] <= 0 <= rep.ci95[1])
    if abs(r.est) >= MARGIN and replicated:
        return "FAIL"
    return "CONCERN"


def grade(primary, replication, label):
    a, b = effects(primary), effects(replication)
    a["p_holm"] = multipletests(a.p, method="holm")[1]
    a["verdict"] = [verdict(r, b.iloc[i]) for i, r in a.iterrows()]
    a["replication_est"], a["replication_ci95"] = b.est, b.ci95
    a.insert(0, "analysis", label)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    P = p.parse_args().project_dir
    bc = json.loads((P / "results" / "phase5" / "summary.json").read_text())["bias_correction"]
    feats = np.load(P / "results" / "phase9a_site_probe" / "features.npz", allow_pickle=True)
    pred = pd.DataFrame({"IXI_ID": feats["ids"].astype(int), "pred": feats["pred"]})
    def split(name):
        d = pd.read_csv(P / f"ixi_{name}.csv")[["IXI_ID", "site", "AGE", "sex_label"]]
        d = d.rename(columns={"AGE": "age", "sex_label": "sex"}).merge(pred, on="IXI_ID")
        d["bag"] = (d.pred - bc["intercept"]) / bc["slope"] - d.age
        d["abs_error"] = (d.pred - d.age).abs()
        return d
    test, val = split("test"), split("val")
    out = [grade(test, val, "main model: test (replication: validation)")]

    cv_dir = P / "results" / "phase10_cv" / "seed0"
    folds = sorted(cv_dir.glob("fold*/outer_predictions.csv"))
    if len(folds) == 5:
        parts = []
        for f in folds:
            o, iv = pd.read_csv(f), pd.read_csv(f.parent / "inner_val_predictions.csv")
            a, b = np.polyfit(iv.age, iv.pred, 1)
            parts.append(o.assign(bag=(o.pred - b) / a - o.age, abs_error=(o.pred - o.age).abs()))
        cv = pd.concat(parts)
        tparts = []
        for f in folds:
            t, iv = pd.read_csv(f.parent / "test_predictions.csv"), pd.read_csv(f.parent / "inner_val_predictions.csv")
            a, b = np.polyfit(iv.age, iv.pred, 1)
            tparts.append(t.assign(cbag=(t.pred - b) / a - t.age))
        ens = pd.concat(tparts).groupby(["IXI_ID", "site", "sex", "age"], as_index=False).agg(pred=("pred", "mean"), bag=("cbag", "mean"))
        ens["abs_error"] = (ens.pred - ens.age).abs()
        out.append(grade(cv, ens, f"5-fold CV out-of-sample n={len(cv)} (replication: locked test, fold ensemble)"))

    res = pd.concat(out, ignore_index=True)
    o = P / "results" / "phase10d_verdicts"; o.mkdir(parents=True, exist_ok=True)
    res.to_csv(o / "graded_verdicts.csv", index=False)
    for _, r in res.iterrows():
        print(f"{r.analysis[:40]:40s} {r.comparison:15s} {r.outcome:9s} {r.est:+.2f} "
              f"95%[{r.ci95[0]:+.2f},{r.ci95[1]:+.2f}] MDE {r.mde80:.2f} p_holm {r.p_holm:.3f} TOST p {r.p_tost:.3f} -> {r.verdict}")


if __name__ == "__main__":
    main()
