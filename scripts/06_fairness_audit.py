"""Phase 6 - Fairness audit of the frozen model's test-set predictions.

Questions, each controlling for chronological age (Rule 5) with 95% CIs (Rule 6):
    1. Does the brain-age gap (BAG) differ by sex or by acquisition site?
       Model: bag_corrected ~ sex + site + age
    2. Is the model less accurate for some groups?
       Model: |error| ~ sex + site + age      (raw predictions)

CIs: nonparametric bootstrap over subjects (5,000 resamples), alongside HC3
robust OLS intervals. Subgroups are small (IOP n = 9), so intervals are wide;
report them, do not over-read point estimates.

Wording (Rules 2-4): differences between sites are "acquisition-site effects";
no clinical interpretation of BAG.

Usage:
    python scripts/06_fairness_audit.py --project-dir D:/ML_Project
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

IN_COLAB = Path("/content").exists()
DEFAULT_PROJECT_DIR = (
    Path("/content/drive/MyDrive/ML_Project") if IN_COLAB else Path("G:/My Drive/ML_Project")
)
N_BOOT = 5000
FORMULAS = {
    "bag_corrected": "bag_corrected ~ C(sex, Treatment('Female')) + C(site, Treatment('Guys')) + age",
    "abs_error": "abs_error ~ C(sex, Treatment('Female')) + C(site, Treatment('Guys')) + age",
}


def tidy_name(term):
    return (term.replace("C(sex, Treatment('Female'))[T.Male]", "Male vs Female")
                .replace("C(site, Treatment('Guys'))[T.HH]", "HH vs Guys")
                .replace("C(site, Treatment('Guys'))[T.IOP]", "IOP vs Guys")
                .replace("age", "Age (per year)"))


def fit(df, formula):
    return smf.ols(formula, data=df).fit(cov_type="HC3")


def bootstrap_coefs(df, formula, rng):
    draws = []
    for _ in range(N_BOOT):
        s = df.iloc[rng.integers(0, len(df), len(df))]
        if s.site.nunique() < df.site.nunique() or s.sex.nunique() < 2:
            continue   # a resample missing a whole group cannot estimate its effect
        draws.append(fit(s, formula).params)
    return pd.DataFrame(draws)


def group_ci(values, rng):
    v = np.asarray(values)
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return np.percentile(means, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser(description="Phase 6 fairness audit.")
    p.add_argument("--project-dir", type=Path,
                   default=Path(os.environ.get("BRAINAGE_PROJECT_DIR", DEFAULT_PROJECT_DIR)))
    args = p.parse_args()
    in_path = args.project_dir / "results" / "phase5" / "test_predictions.csv"
    out_dir = args.project_dir / "results" / "phase6"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df["abs_error"] = (df.predicted_age - df.age).abs()
    rng = np.random.default_rng(0)
    print(f"Test predictions: {len(df)} subjects")
    print(pd.crosstab(df.site, df.sex, margins=True).to_string(), "\n")

    # ---- Per-group descriptives ----
    rows = []
    for col in ["sex", "site"]:
        for g, d in df.groupby(col):
            mae_ci, bag_ci = group_ci(d.abs_error, rng), group_ci(d.bag_corrected, rng)
            rows.append({"grouping": col, "group": g, "n": len(d), "mean_age": d.age.mean(),
                         "mae": d.abs_error.mean(), "mae_ci_low": mae_ci[0], "mae_ci_high": mae_ci[1],
                         "mean_bag_corrected": d.bag_corrected.mean(),
                         "bag_ci_low": bag_ci[0], "bag_ci_high": bag_ci[1]})
    groups = pd.DataFrame(rows)
    groups.to_csv(out_dir / "group_summary.csv", index=False)
    print("Per-group (unadjusted) - MAE and mean corrected BAG, 95% bootstrap CI:")
    for _, r in groups.iterrows():
        print(f"  {r.group:6s} n={r.n:2d} age {r.mean_age:4.1f} | MAE {r.mae:4.2f} "
              f"({r.mae_ci_low:.2f}-{r.mae_ci_high:.2f}) | BAG {r.mean_bag_corrected:+5.2f} "
              f"({r.bag_ci_low:+.2f} to {r.bag_ci_high:+.2f})")

    # ---- Age-adjusted models ----
    results = {}
    coef_rows = []
    for outcome, formula in FORMULAS.items():
        model = fit(df, formula)
        boot = bootstrap_coefs(df, formula, rng)
        hc3 = model.conf_int()
        print(f"\nAge-adjusted model: {outcome} ~ sex + site + age  (n={len(df)}, "
              f"{len(boot)} valid bootstrap resamples)")
        for term in model.params.index:
            if term == "Intercept":
                continue
            lo, hi = np.percentile(boot[term], [2.5, 97.5])
            row = {"outcome": outcome, "term": tidy_name(term), "estimate": model.params[term],
                   "boot_ci_low": lo, "boot_ci_high": hi,
                   "hc3_ci_low": hc3.loc[term, 0], "hc3_ci_high": hc3.loc[term, 1],
                   "hc3_p": model.pvalues[term], "ci_excludes_zero": bool(lo > 0 or hi < 0)}
            coef_rows.append(row)
            flag = "  <- CI excludes 0" if row["ci_excludes_zero"] else ""
            print(f"  {row['term']:16s} {row['estimate']:+6.2f} yrs  bootstrap 95% CI "
                  f"[{lo:+.2f}, {hi:+.2f}]  (HC3 [{row['hc3_ci_low']:+.2f}, {row['hc3_ci_high']:+.2f}], "
                  f"p={row['hc3_p']:.3f}){flag}")
        results[outcome] = {"r2": model.rsquared, "n": int(model.nobs)}
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(out_dir / "age_adjusted_effects.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(
        {"input": str(in_path), "n": len(df), "n_boot": N_BOOT, "models": results,
         "formulas": FORMULAS}, indent=2))

    # ---- Figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, outcome, title in [(axes[0], "bag_corrected", "Brain-age gap (corrected)"),
                               (axes[1], "abs_error", "Absolute error")]:
        c = coefs[(coefs.outcome == outcome) & (coefs.term != "Age (per year)")].iloc[::-1]
        y = np.arange(len(c))
        ax.errorbar(c.estimate, y, xerr=[c.estimate - c.boot_ci_low, c.boot_ci_high - c.estimate],
                    fmt="o", capsize=4)
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.set_yticks(y, c.term)
        ax.set_xlabel("Age-adjusted difference (years), 95% bootstrap CI")
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_dir / "age_adjusted_effects.png", dpi=130)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
