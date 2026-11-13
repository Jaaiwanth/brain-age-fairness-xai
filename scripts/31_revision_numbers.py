"""Phase 11b - Interval estimates added in revision.

1. Wilson 95% CIs for the coverage values of the uncertainty table: the model's own 90% range
   (seed 0, out-of-fold) and split conformal on disjoint folds (Phase 11a; coverage averaged over the
   three seeds, with n = subjects per site, so the interval reflects subject sampling only).
2. Bootstrap 95% CI (2000 resamples of subjects) for the out-of-fold MAE of CV seed 0.

Usage: python scripts/31_revision_numbers.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint


def wilson(p, n):
    lo, hi = proportion_confint(round(p * n), n, method="wilson")
    return [round(float(lo), 3), round(float(hi), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    P = ap.parse_args().project_dir
    R = P / "results"
    out = {"coverage_90_wilson": []}

    rel = pd.read_csv(R / "phase10i_uncertainty" / "reliability_by_site.csv")
    own = rel[(rel.method == "model's own distribution") & (rel.level == 0.9)]
    for _, r in own.iterrows():
        out["coverage_90_wilson"].append({"interval": "model's own 90% range", "site": r.site, "n": int(r.n),
                                          "coverage": round(r.coverage, 3), "ci95": wilson(r.coverage, int(r.n))})
    sc = pd.read_csv(R / "phase11a_review_checks" / "split_conformal_rows.csv")
    for col, name in [("pooled_cov", "split conformal, pooled"), ("site_cov", "split conformal, per site")]:
        for site, g in sc.groupby("site"):
            n = int(g.n_eval.sum() / g.seed.nunique())
            cov = float(np.average(g[col], weights=g.n_eval))
            out["coverage_90_wilson"].append({"interval": name, "site": site, "n": n,
                                              "coverage": round(cov, 3), "ci95": wilson(cov, n)})

    oof = pd.concat([pd.read_csv(f) for f in sorted((R / "phase10_cv" / "seed0").glob("fold*/outer_predictions.csv"))])
    err = (oof.pred - oof.age).abs().to_numpy()
    rng = np.random.default_rng(0)
    boots = [rng.choice(err, len(err)).mean() for _ in range(2000)]
    out["cv_seed0_oof_mae"] = {"n": len(err), "mae": round(float(err.mean()), 3),
                               "ci95": [round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2)]}

    o = R / "phase11b_revision_numbers"; o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
