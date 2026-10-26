"""Phase 11a - Robustness checks requested in review.

1. Zero-shot IOP offset vs the head's 42-year floor: re-estimated with a B-spline of age, in 1:1
   age- and sex-matched IOP-Guys pairs, and restricted to ages 50-80.
2. Conformal coverage with calibration and evaluation on disjoint subjects: calibrate on CV folds
   0-2 and evaluate on folds 3-4, then swap; the number of calibration points per site is reported.

Usage: python scripts/30_review_checks.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

T = "C(site, Treatment('Guys'))[T.IOP]"


def boot_ci(d, formula, rng, n=2000):
    boots = []
    for _ in range(n):
        s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _, g in d.groupby("site")])
        boots.append(smf.ols(formula, data=s).fit().params[T])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def matched_pairs(d, rng, caliper=3.0):
    """Greedy 1:1 nearest-age matching of IOP to Guys within sex, without replacement."""
    pairs = []
    for sex, g in d.groupby("sex"):
        iop = g[g.site == "IOP"].sample(frac=1, random_state=0)
        guys = g[g.site == "Guys"].copy()
        for _, r in iop.iterrows():
            if guys.empty:
                break
            j = (guys.age - r.age).abs().idxmin()
            if abs(guys.age[j] - r.age) <= caliper:
                pairs.append((r.pred - guys.pred[j], r.age - guys.age[j]))
                guys = guys.drop(j)
    diff = np.array([p[0] for p in pairs])
    boots = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    return {"n_pairs": len(diff), "mean_age_diff": float(np.mean([p[1] for p in pairs])),
            "iop_minus_guys": float(diff.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def zero_shot(P, rng):
    z = pd.read_csv(P / "results" / "phase3" / "zero_shot_predictions.csv").rename(columns={"predicted_age": "pred"})
    z = z[z.site.isin(["Guys", "IOP", "HH"])]
    out = {}
    specs = [("linear age, 42-82", z[(z.age >= 42) & (z.age <= 82)], "pred ~ C(site, Treatment('Guys')) + C(sex) + age"),
             ("B-spline age (4 df), 42-82", z[(z.age >= 42) & (z.age <= 82)],
              "pred ~ C(site, Treatment('Guys')) + C(sex) + bs(age, df=4)"),
             ("linear age, 50-80", z[(z.age >= 50) & (z.age <= 80)], "pred ~ C(site, Treatment('Guys')) + C(sex) + age"),
             ("B-spline age (4 df), 50-80", z[(z.age >= 50) & (z.age <= 80)],
              "pred ~ C(site, Treatment('Guys')) + C(sex) + bs(age, df=4)")]
    for name, d, f in specs:
        m = smf.ols(f, data=d).fit(cov_type="HC3")
        out[name] = {"n": len(d), "n_iop": int((d.site == "IOP").sum()), "iop_vs_guys": float(m.params[T]),
                     "ci_bootstrap": boot_ci(d, f, rng), "p": float(m.pvalues[T])}
    for lo, hi in [(42, 82), (50, 80)]:
        out[f"1:1 age/sex-matched pairs, {lo}-{hi}"] = matched_pairs(z[(z.age >= lo) & (z.age <= hi)], rng)
    return out


def conformal_q(res, level):
    n = len(res)
    return np.quantile(res, min(1.0, np.ceil((n + 1) * level) / n), method="higher")


def split_conformal(P, level=0.9):
    rows = []
    for seed_dir in sorted((P / "results" / "phase10_cv").glob("seed*")):
        oof = pd.concat([pd.read_csv(f) for f in sorted(seed_dir.glob("fold*/outer_predictions.csv"))], ignore_index=True)
        if oof.fold.nunique() != 5:
            continue
        oof["res"] = (oof.pred - oof.age).abs()
        for cal_folds in [{0, 1, 2}, {3, 4}]:
            cal, ev = oof[oof.fold.isin(cal_folds)], oof[~oof.fold.isin(cal_folds)]
            q_pool = conformal_q(cal.res.to_numpy(), level)
            q_site = {s: conformal_q(g.res.to_numpy(), level) for s, g in cal.groupby("site")}
            for site, g in ev.groupby("site"):
                rows.append({"seed": seed_dir.name, "calibrate_on": "folds " + ",".join(str(k + 1) for k in sorted(cal_folds)),
                             "site": site, "n_cal": int((cal.site == site).sum()), "n_eval": len(g),
                             "pooled_cov": float((g.res <= q_pool).mean()), "pooled_halfwidth": float(q_pool),
                             "site_cov": float((g.res <= q_site[site]).mean()), "site_halfwidth": float(q_site[site])})
    rows = pd.DataFrame(rows)
    pooled = rows.groupby("site").apply(lambda g: pd.Series({
        "n_cal_per_split": g.n_cal.mean(), "n_eval_total_per_seed": g.n_eval.sum() / g.seed.nunique(),
        "pooled_cov": np.average(g.pooled_cov, weights=g.n_eval), "site_cov": np.average(g.site_cov, weights=g.n_eval),
        "site_halfwidth": g.site_halfwidth.mean()}), include_groups=False)
    return rows, pooled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    P = p.parse_args().project_dir
    rng = np.random.default_rng(0)
    o = P / "results" / "phase11a_review_checks"; o.mkdir(parents=True, exist_ok=True)
    zs = zero_shot(P, rng)
    rows, pooled = split_conformal(P)
    rows.to_csv(o / "split_conformal_rows.csv", index=False)
    out = {"zero_shot": zs, "split_conformal_90": pooled.round(3).reset_index().to_dict("records")}
    (o / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
