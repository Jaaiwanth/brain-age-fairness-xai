"""Investigate the male-vs-female brain-age error disparity on Phase 3 zero-shot results.

Usage: python analysis/sex_disparity.py <results.csv> <label> [--out analysis/output]

All analysis choices are fixed up front (age bins, site x age-bin strata, tests) and
applied identically to every model; nothing is tuned toward a hypothesis.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

AGE_BINS = [20, 30, 40, 50, 60, 70, 100]
AGE_LABELS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
MIN_N = 10  # cells with fewer subjects of either sex are flagged unstable
RNG = np.random.default_rng(0)
N_BOOT = 5000
N_PERM = 5000
OUT = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.append(s)


def metrics(df):
    e, ae = df["error"].to_numpy(), df["abs_error"].to_numpy()
    return pd.Series({"n": len(df), "MAE": ae.mean(), "RMSE": np.sqrt((e ** 2).mean()), "ME": e.mean()})


def boot_ci(x, fn, n=N_BOOT):
    x = np.asarray(x)
    vals = [fn(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(vals, [2.5, 97.5])


def cohens_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def cliffs_delta(a, b):
    # P(a>b) - P(a<b) via Mann-Whitney U
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def compare(df, col, title):
    m, f = df.loc[df.sex == "Male", col].to_numpy(), df.loc[df.sex == "Female", col].to_numpy()
    diff = m.mean() - f.mean()
    ci = boot_ci(np.arange(len(df)), lambda i: df[col].to_numpy()[i][(df.sex.to_numpy()[i] == "Male")].mean()
                 - df[col].to_numpy()[i][(df.sex.to_numpy()[i] == "Female")].mean(), n=2000)
    say(f"  {title}: M-F mean diff = {diff:+.2f} yrs (95% bootstrap CI {ci[0]:+.2f}, {ci[1]:+.2f})")
    say(f"    Welch t p={stats.ttest_ind(m, f, equal_var=False).pvalue:.4g} | "
        f"Mann-Whitney p={stats.mannwhitneyu(m, f).pvalue:.4g} | "
        f"Cohen d={cohens_d(m, f):+.2f} | Cliff delta={cliffs_delta(m, f):+.2f}")
    say(f"    Shapiro normality p (M/F) = {stats.shapiro(m).pvalue:.3g} / {stats.shapiro(f).pvalue:.3g}")


def stratified_perm(df, strata_cols, col="abs_error"):
    """Within-stratum M-F difference (weighted by stratum size), p-value by permuting sex within strata."""
    g = df.groupby(strata_cols, observed=True)
    idx, sexes, vals = [], [], df[col].to_numpy()
    for _, sub in g:
        if (sub.sex == "Male").any() and (sub.sex == "Female").any():
            idx.append(sub.index.to_numpy())
    pos = {k: i for i, k in enumerate(df.index)}
    groups = [np.array([pos[i] for i in ix]) for ix in idx]
    isM = (df.sex == "Male").to_numpy()
    w = np.array([len(g_) for g_ in groups], float)

    def stat(is_m):
        d = [vals[g_][is_m[g_]].mean() - vals[g_][~is_m[g_]].mean() for g_ in groups]
        return float(np.average(d, weights=w))

    obs = stat(isM)
    null = []
    for _ in range(N_PERM):
        p = isM.copy()
        for g_ in groups:
            p[g_] = RNG.permutation(isM[g_])
        null.append(stat(p))
    null = np.array(null)
    return obs, float((np.abs(null) >= abs(obs)).mean()), len(groups), int(sum(len(g_) for g_ in groups))


def main(path, label, out):
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    df = df.dropna(subset=["predicted_age"]).reset_index(drop=True)
    df["age_bin"] = pd.cut(df.actual_age, AGE_BINS, labels=AGE_LABELS, right=False)

    say(f"\n{'=' * 78}\nMODEL: {label}   (file: {Path(path).name})\n{'=' * 78}")
    say(f"Subjects: {len(df)} | sites: {df.site.value_counts().to_dict()} | sex: {df.sex.value_counts().to_dict()}")

    # 1. Verify
    say("\n[1] Verify: error metrics by sex (95% bootstrap CI for MAE)")
    tab = df.groupby("sex").apply(metrics, include_groups=False)
    for s in tab.index:
        ae = df.loc[df.sex == s, "abs_error"].to_numpy()
        lo, hi = boot_ci(ae, np.mean)
        tab.loc[s, "MAE_CI"] = f"[{lo:.2f}, {hi:.2f}]"
    say(tab.round(2).to_string())

    # 2. Age distribution
    say("\n[2] Age distribution by sex")
    say(df.groupby("sex").actual_age.describe().round(2).to_string())
    am, af = df.loc[df.sex == "Male", "actual_age"], df.loc[df.sex == "Female", "actual_age"]
    say(f"  KS test p={stats.ks_2samp(am, af).pvalue:.4g} | Mann-Whitney p={stats.mannwhitneyu(am, af).pvalue:.4g}")
    cnt = pd.crosstab(df.age_bin, df.sex)
    say("\n  Per age bin x sex:")
    pb = df.groupby(["age_bin", "sex"], observed=True).apply(metrics, include_groups=False).round(2)
    pb["flag"] = np.where(pb["n"] < MIN_N, "UNSTABLE(n<%d)" % MIN_N, "")
    say(pb.to_string())
    # direct standardisation: female MAE re-weighted to male age-bin mix (bins with both sexes)
    mae_bin = df.groupby(["age_bin", "sex"], observed=True).abs_error.mean().unstack()
    nm = cnt["Male"]
    ok = mae_bin.dropna().index
    wts = nm[ok] / nm[ok].sum()
    mm, ff = (mae_bin.loc[ok, "Male"] * wts).sum(), (mae_bin.loc[ok, "Female"] * wts).sum()
    say(f"\n  Age-standardised MAE (male age-bin mix weights): Male {mm:.2f} vs Female {ff:.2f}  (M-F {mm - ff:+.2f})")
    say(f"  Raw (unadjusted) M-F MAE gap: {tab.loc['Male', 'MAE'] - tab.loc['Female', 'MAE']:+.2f}")

    # 3. Site
    say("\n[3] Site confounding")
    ct = pd.crosstab(df.site, df.sex)
    ct["pct_male"] = (100 * ct["Male"] / ct.sum(axis=1)).round(1)
    say(ct.to_string())
    say(f"  chi2 (sex x site) p={stats.chi2_contingency(pd.crosstab(df.site, df.sex))[1]:.4g}")
    say("\n  Site-level error (all subjects):")
    say(df.groupby("site").apply(metrics, include_groups=False).round(2).to_string())
    sb = df.groupby(["site", "sex"]).apply(metrics, include_groups=False).round(2)
    sb["flag"] = np.where(sb["n"] < MIN_N, "UNSTABLE", "")
    say("\n  Site x sex:")
    say(sb.to_string())
    ssd = df.groupby("site").actual_age.agg(["mean", "std"]).round(1)
    say("\n  Mean age by site (site-age confound check):")
    say(ssd.to_string())

    # 4. Adjusted models
    say("\n[4] Regression (HC3 robust SE). Dependent: abs_error and signed error. Not causal.")
    df["male"] = (df.sex == "Male").astype(int)
    for dv in ["abs_error", "error"]:
        say(f"\n  DV = {dv}")
        for name, f in [("unadjusted", f"{dv} ~ male"),
                        ("+ age (spline)", f"{dv} ~ male + bs(actual_age, df=4)"),
                        ("+ site", f"{dv} ~ male + C(site)"),
                        ("+ age + site", f"{dv} ~ male + bs(actual_age, df=4) + C(site)"),
                        ("+ age*sex + site", f"{dv} ~ male * bs(actual_age, df=4) + C(site)")]:
            r = smf.ols(f, df).fit(cov_type="HC3")
            if "age*sex" in name:
                inter = [k for k in r.params.index if k.startswith("male:")]
                w = r.wald_test(" , ".join(f"{k} = 0" for k in inter), scalar=True)
                say(f"    {name:18s} sex x age interaction joint Wald p={float(w.pvalue):.4g}")
            else:
                ci = r.conf_int().loc["male"]
                say(f"    {name:18s} male coef={r.params['male']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] p={r.pvalues['male']:.4g}  (R2={r.rsquared:.3f})")
    obs, p, ns, nused = stratified_perm(df, ["site", "age_bin"])
    say(f"\n  Stratified permutation (sex permuted within site x age-bin; {ns} strata with both sexes, n={nused}):")
    say(f"    within-stratum M-F abs_error diff = {obs:+.2f} yrs, permutation p={p:.4g}")

    # 5. Sample size flags
    say("\n[5] Small-cell flags (either sex n<%d)" % MIN_N)
    small = pd.crosstab([df.site, df.age_bin], df.sex)
    small = small[(small["Male"] < MIN_N) | (small["Female"] < MIN_N)]
    say(f"  {len(small)} of {len(pd.crosstab([df.site, df.age_bin], df.sex))} site x age-bin cells are small; "
        "treat per-cell sex comparisons there as unstable.")

    # 6/7. Behaviour + significance
    say("\n[6] Model behaviour")
    for s, sub in df.groupby("sex"):
        sl = stats.linregress(sub.actual_age, sub.predicted_age)
        say(f"  {s}: mean error {sub.error.mean():+.2f}; predicted-vs-actual slope={sl.slope:.3f} "
            f"(1.0 = ideal, 0 = constant prediction), r={sl.rvalue:.3f}; pred SD={sub.predicted_age.std():.2f}")
    say("  Error-vs-age Spearman by sex: " + ", ".join(
        f"{s}: rho={stats.spearmanr(sub.actual_age, sub.error)[0]:+.2f}" for s, sub in df.groupby("sex")))
    say("\n[7] Unadjusted significance tests")
    compare(df, "abs_error", "abs_error")
    compare(df, "error", "signed error (BAG)")

    # Plots
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    col = {"Male": "#4C78A8", "Female": "#E45756"}
    for s, sub in df.groupby("sex"):
        ax[0, 0].scatter(sub.actual_age, sub.predicted_age, s=14, alpha=0.55, c=col[s], label=f"{s} (n={len(sub)})")
        sl = stats.linregress(sub.actual_age, sub.predicted_age)
        xs = np.array([20, 86])
        ax[0, 0].plot(xs, sl.intercept + sl.slope * xs, c=col[s], lw=2)
        ax[0, 1].scatter(sub.actual_age, sub.error, s=14, alpha=0.55, c=col[s])
        ax[1, 0].hist(sub.error, bins=25, alpha=0.55, color=col[s], label=s, density=True)
        ax[1, 1].hist(sub.actual_age, bins=15, alpha=0.55, color=col[s], label=s, density=True)
    ax[0, 0].plot([20, 86], [20, 86], "k--", lw=1)
    ax[0, 0].set(title="Predicted vs actual, by sex", xlabel="Actual age", ylabel="Predicted age"); ax[0, 0].legend()
    ax[0, 1].axhline(0, c="k", lw=1); ax[0, 1].set(title="Signed error vs actual age", xlabel="Actual age", ylabel="Pred - actual")
    ax[1, 0].set(title="Brain-age delta distribution", xlabel="Pred - actual"); ax[1, 0].legend()
    ax[1, 1].set(title="Age distribution by sex", xlabel="Actual age"); ax[1, 1].legend()
    mb = df.groupby(["age_bin", "sex"], observed=True).abs_error.agg(["mean", "count"]).reset_index()
    for k, (s, sub) in enumerate(mb.groupby("sex")):
        xs = np.arange(len(AGE_LABELS))[sub.age_bin.cat.codes.to_numpy()] + (k - 0.5) * 0.38
        ax[0, 2].bar(xs, sub["mean"], 0.38, color=col[s], label=s)
        for x, n in zip(xs, sub["count"]):
            ax[0, 2].text(x, 0.4, f"n={n}", ha="center", rotation=90, color="white", fontsize=8)
    ax[0, 2].set_xticks(range(len(AGE_LABELS)), AGE_LABELS)
    ax[0, 2].set(title="MAE by age bin x sex", ylabel="MAE (yrs)"); ax[0, 2].legend()
    sx = df.groupby(["site", "sex"]).abs_error.mean().unstack()
    sx.plot.bar(ax=ax[1, 2], color=[col["Female"], col["Male"]], rot=0)
    ax[1, 2].set(title="MAE by site x sex", ylabel="MAE (yrs)")
    fig.suptitle(f"{label}: sex disparity investigation", y=1.0)
    fig.tight_layout()
    fig.savefig(out / f"sex_disparity_{label}.png", dpi=140, bbox_inches="tight")
    (out / f"sex_disparity_{label}.txt").write_text("\n".join(OUT), encoding="utf8")
    OUT.clear()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("label"); ap.add_argument("--out", default="analysis/output")
    a = ap.parse_args()
    main(a.csv, a.label, Path(a.out))
