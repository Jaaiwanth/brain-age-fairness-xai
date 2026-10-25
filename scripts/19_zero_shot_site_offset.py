"""Phase 10c - Does the IOP offset exist before fine-tuning? (test of the overfitting explanation)

The UK Biobank SFCN never saw IXI, so every IXI subject is out-of-sample for it. Its head only covers
ages 42-82, so the comparison is restricted to subjects aged 42-82. The age-adjusted IOP-vs-Guys gap
is estimated (raw prediction ~ age + site + sex, and Cole-corrected gap with the correction fitted on
the 42-82 train subjects), on all subjects and on the held-out (val + test) subjects only.

Usage: python scripts/19_zero_shot_site_offset.py --project-dir D:/ML_Project
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, statsmodels.formula.api as smf

p = argparse.ArgumentParser(); p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
P = p.parse_args().project_dir
rng = np.random.default_rng(0)
z = pd.read_csv(P / "results" / "phase3" / "zero_shot_predictions.csv").rename(columns={"predicted_age": "pred"})
z = z[(z.age >= 42) & (z.age <= 82)]
a, b = np.polyfit(z[z.split == "train"].age, z[z.split == "train"].pred, 1)
z["gap"] = (z.pred - b) / a - z.age
F = "{} ~ C(site, Treatment('Guys')) + C(sex) + age"
T = "C(site, Treatment('Guys'))[T.IOP]"
out = {}
for name, d in [("all subjects 42-82", z), ("held-out (val+test) 42-82", z[z.split != "train"])]:
    for outcome in ["pred", "gap"]:
        m = smf.ols(F.format(outcome), data=d).fit(cov_type="HC3")
        boots = []
        for _ in range(2000):
            s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _, g in d.groupby("site")])
            boots.append(smf.ols(F.format(outcome), data=s).fit().params[T])
        out[f"{name} | {outcome}"] = {"n": len(d), "n_iop": int((d.site == "IOP").sum()), "iop_vs_guys": float(m.params[T]),
                                      "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                      "p": float(m.pvalues[T])}
o = P / "results" / "phase10c_zero_shot_offset"; o.mkdir(parents=True, exist_ok=True)
(o / "summary.json").write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
