"""Phase 10b - Is the IOP offset an age artefact? Robustness of the site effect to age handling.

IOP recruited younger people, and brain-age gaps depend on age, so the IOP offset could in principle
come from how age is handled. Each check below re-estimates the age-adjusted IOP-vs-Guys
difference in brain-age gap:

1. Bias-correction variants: (a) none (raw gap, linear age covariate), (b) Cole correction fitted on
   validation (the paper's choice), (c) Cole correction fitted on train, (d) "age as covariate"
   (raw prediction ~ age + site + sex; Le et al. 2018).
2. Nonlinear age: a B-spline of age (4 df) instead of the linear term.
3. Age matching: (i) Guys/HH restricted to IOP's age range; (ii) 1:1 nearest-neighbour matching of
   each IOP subject to a Guys subject of the same sex (caliper 3 years, without replacement).
4. Native brain volume as an extra covariate (registration removes head size from the model input).
5. Edge of the label range: prediction error for the youngest subjects by site.

Prediction sources: the main model (held-out = val + test; test alone) and, when present, the
5-fold cross-validated predictions from Phase 10a (every train+val subject out-of-sample).

Usage:
    python scripts/18_age_robustness.py --project-dir D:/ML_Project
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SITES = ["Guys", "HH", "IOP"]
COLORS = {"Guys": "#2a78d6", "HH": "#eb6834", "IOP": "#1baf7a"}
N_BOOT = 2000


def parse_args():
    p = argparse.ArgumentParser(description="Age robustness of the IOP offset.")
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    p.add_argument("--checkpoint-dir", type=Path,
                   default=Path("G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official"))
    return p.parse_args()


def cole(fit_df):
    a, b = np.polyfit(fit_df.age, fit_df.pred, 1)
    return lambda d: (d.pred - b) / a


def iop_effect(df, formula, outcome, rng, boot=True):
    """IOP-vs-Guys coefficient with HC3 p-value and a bootstrap CI (resampling within site)."""
    d = df.assign(y=outcome(df) if callable(outcome) else df[outcome])
    m = smf.ols(formula.replace("OUT", "y"), data=d).fit(cov_type="HC3")
    term = [t for t in m.params.index if "IOP" in t][0]
    out = {"n": len(d), "n_iop": int((d.site == "IOP").sum()), "iop_vs_guys": float(m.params[term]),
           "p": float(m.pvalues[term])}
    if boot:
        draws = []
        groups = [g for _, g in d.groupby("site")]
        for _ in range(N_BOOT):
            s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for g in groups])
            try:
                draws.append(smf.ols(formula.replace("OUT", "y"), data=s).fit().params[term])
            except Exception:
                pass
        out["ci"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return out


def nn_match(df, rng, caliper=3.0):
    iop = df[df.site == "IOP"]
    pool = df[df.site == "Guys"].copy()
    keep = []
    for _, r in iop.sample(frac=1, random_state=0).iterrows():
        c = pool[(pool.sex == r.sex)]
        if c.empty:
            continue
        j = (c.age - r.age).abs().idxmin()
        if abs(pool.loc[j, "age"] - r.age) <= caliper:
            keep += [r.name, j]
            pool = pool.drop(j)
    return df.loc[keep]


def run_checks(df, corr_val, corr_train, rng, label):
    rows = []
    base = "OUT ~ C(site, Treatment('Guys')) + C(sex, Treatment('Female'))"
    variants = [
        ("1a  no correction (raw gap), linear age", base + " + age", lambda d: d.pred - d.age),
        ("1b  Cole correction fitted on val [paper]", base + " + age", lambda d: corr_val(d) - d.age),
        ("1c  Cole correction fitted on train", base + " + age", lambda d: corr_train(d) - d.age),
        ("1d  age as covariate (prediction ~ age)", base + " + age", "pred"),
        ("2   spline age (4 df), val correction", base + " + bs(age, df=4)", lambda d: corr_val(d) - d.age),
        ("4   + native brain volume, val correction", base + " + age + nbv", lambda d: corr_val(d) - d.age),
    ]
    for name, f, out in variants:
        rows.append({"source": label, "check": name, **iop_effect(df, f, out, rng)})
    lo, hi = df[df.site == "IOP"].age.min(), df[df.site == "IOP"].age.max()
    rng_df = df[(df.age >= lo) & (df.age <= hi)]
    rows.append({"source": label, "check": f"3i  restricted to IOP age range {lo:.0f}-{hi:.0f}",
                 **iop_effect(rng_df, base + " + age", lambda d: corr_val(d) - d.age, rng)})
    m = nn_match(df, rng)
    rows.append({"source": label, "check": "3ii 1:1 age/sex-matched IOP vs Guys (caliper 3 y)",
                 **iop_effect(m, "OUT ~ C(site, Treatment('Guys'))", lambda d: corr_val(d) - d.age, rng)})
    return rows


def main():
    args = parse_args()
    P = args.project_dir
    rng = np.random.default_rng(0)
    out_dir = P / "results" / "phase10b_age_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    qc = pd.read_csv(P / "phase2_v2_qc.csv")[["IXI_ID", "native_brain_volume_ml"]].rename(
        columns={"native_brain_volume_ml": "nbv"})
    feats = np.load(P / "results" / "phase9a_site_probe" / "features.npz", allow_pickle=True)
    pred = pd.DataFrame({"IXI_ID": feats["ids"].astype(int), "pred": feats["pred"]})
    splits = []
    for s in ["train", "val", "test"]:
        d = pd.read_csv(P / f"ixi_{s}.csv")[["IXI_ID", "site", "AGE", "sex_label"]]
        splits.append(d.assign(split=s))
    meta = pd.concat(splits).rename(columns={"AGE": "age", "sex_label": "sex"})
    df = meta.merge(pred, on="IXI_ID").merge(qc, on="IXI_ID")
    df["nbv"] = (df.nbv - df.nbv.mean()) / df.nbv.std()

    corr_val, corr_train = cole(df[df.split == "val"]), cole(df[df.split == "train"])
    rows = []
    rows += run_checks(df[df.split.isin(["val", "test"])], corr_val, corr_train, rng, "main model, held-out (val+test)")
    rows += run_checks(df[df.split == "test"], corr_val, corr_train, rng, "main model, test only")

    cv_dir = P / "results" / "phase10_cv" / "seed0"
    folds = sorted(cv_dir.glob("fold*/outer_predictions.csv"))
    if len(folds) == 5:
        cv = pd.concat([pd.read_csv(f) for f in folds]).merge(qc, on="IXI_ID")
        cv["nbv"] = (cv.nbv - cv.nbv.mean()) / cv.nbv.std()
        iv = pd.concat([pd.read_csv(f.parent / "inner_val_predictions.csv") for f in folds])
        # Each fold's correction is fitted on its own inner-validation subjects.
        parts = []
        for k, g in cv.groupby("fold"):
            c = cole(iv[iv.fold == k])
            parts.append(g.assign(pred=lambda d, c=c: d.pred, cv_corr=c(g)))
        cv = pd.concat(parts)
        corr_cv = lambda d: d.cv_corr
        rows += run_checks(cv, corr_cv, corr_cv, rng, f"5-fold CV, all train+val (n={len(cv)})")

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "iop_offset_robustness.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(res[["source", "check", "n", "n_iop", "iop_vs_guys", "ci", "p"]].to_string(index=False))

    # Native brain volume by site (age/sex adjusted) and the youngest-subject edge check.
    raw = meta.merge(pd.read_csv(P / "phase2_v2_qc.csv")[["IXI_ID", "native_brain_volume_ml"]], on="IXI_ID")
    m = smf.ols("native_brain_volume_ml ~ C(site, Treatment('Guys')) + C(sex) + age", data=raw).fit(cov_type="HC3")
    nbv = {t: [float(m.params[t]), float(m.pvalues[t])] for t in m.params.index if "site" in t}
    held = df[df.split.isin(["val", "test"])].copy()
    held["corr_gap"] = corr_val(held) - held.age
    young = held[held.age < 30].groupby("site").agg(n=("age", "size"), mean_gap=("corr_gap", "mean"),
                                                    mean_raw_gap=("pred", lambda s: 0)).drop(columns="mean_raw_gap")
    held["raw_gap"] = held.pred - held.age
    young["mean_raw_gap"] = held[held.age < 30].groupby("site").raw_gap.mean()
    old = held[held.age >= 30].groupby("site").corr_gap.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in SITES:
        d = held[held.site == s]
        axes[0].scatter(d.age, d.corr_gap, s=18, color=COLORS[s], label=f"{s} (n={len(d)})", alpha=.8)
        g = d.sort_values("age")
        axes[0].plot(g.age, g.corr_gap.rolling(9, center=True, min_periods=4).mean(), color=COLORS[s], lw=2)
    axes[0].axhline(0, color="k", lw=.8); axes[0].axvline(30, color="grey", ls=":", lw=1)
    axes[0].set(xlabel="age (years)", ylabel="corrected brain-age gap (years)",
                title="Held-out gap by age and site (line: running mean)")
    axes[0].legend(frameon=False, fontsize=8)
    r = res[res.source == res.source.iloc[0]]
    y = np.arange(len(r))[::-1]
    axes[1].errorbar(r.iop_vs_guys, y, xerr=[r.iop_vs_guys - r.ci.str[0], r.ci.str[1] - r.iop_vs_guys],
                     fmt="o", color=COLORS["IOP"], capsize=3)
    axes[1].axvline(0, color="k", lw=.8)
    axes[1].set_yticks(y, [c.split("  ", 1)[-1].strip()[:42] for c in r.check], fontsize=8)
    axes[1].set(xlabel="IOP vs Guys gap (years, 95% CI)", title="IOP offset under every age handling")
    fig.tight_layout()
    fig.savefig(out_dir / "age_robustness.png", dpi=130)

    summary = {"native_brain_volume_ml_site_effects_age_sex_adjusted": nbv,
               "youngest_under_30_heldout": young.reset_index().to_dict("records"),
               "heldout_30_plus_mean_gap": old.to_dict()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
