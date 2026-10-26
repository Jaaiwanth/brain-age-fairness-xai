"""Phase 10h - Descriptive figures requested in review, plus volume-normalised attribution.

1. Age distribution per site (train and held-out): makes the recruitment-age confound visible.
2. Brain-age gap per subject by site and by sex (held-out), not only mean differences.
3. Attribution density: each coarse region's share of attribution divided by its share of brain
   volume (1 = attribution proportional to size), by site, with age/sex-adjusted site differences.

Usage: python scripts/24_extra_figures.py --project-dir D:/ML_Project
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gc", HERE / "07_gradcam.py")
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)
ft = gc.ft
SITES = ["Guys", "HH", "IOP"]
COLORS = {"Guys": "#2a78d6", "HH": "#eb6834", "IOP": "#1baf7a"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    P = p.parse_args().project_dir
    out = P / "results" / "phase10h_figures"; out.mkdir(parents=True, exist_ok=True)
    meta = pd.concat([pd.read_csv(P / f"ixi_{s}.csv").assign(split=s) for s in ["train", "val", "test"]])
    meta = meta[~meta.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)]
    bc = json.loads((P / "results" / "phase5" / "summary.json").read_text())["bias_correction"]
    f = np.load(P / "results" / "phase9a_site_probe" / "features.npz", allow_pickle=True)
    meta = meta.merge(pd.DataFrame({"IXI_ID": f["ids"].astype(int), "pred": f["pred"]}), on="IXI_ID")
    meta["bag"] = (meta.pred - bc["intercept"]) / bc["slope"] - meta.AGE
    held = meta[meta.split != "train"]

    # 1 + 2
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 0.8]})
    bins = np.arange(18, 92, 4)
    for s in SITES:
        ax[0].hist(meta[meta.site == s].AGE, bins=bins, histtype="step", lw=2, color=COLORS[s],
                   label=f"{s} (n={(meta.site == s).sum()}, mean {meta[meta.site == s].AGE.mean():.1f})")
    ax[0].set(xlabel="age (years)", ylabel="subjects", title="Age distribution by site (all 485)")
    ax[0].legend(frameon=False, fontsize=8)
    rng = np.random.default_rng(0)
    for i, s in enumerate(SITES):
        d = held[held.site == s].bag
        ax[1].scatter(i + rng.uniform(-.18, .18, len(d)), d, s=14, color=COLORS[s], alpha=.75)
        ax[1].plot([i - .28, i + .28], [d.median()] * 2, color="k", lw=2)
    ax[1].axhline(0, color="grey", lw=.8, ls=":")
    ax[1].set_xticks(range(3), SITES); ax[1].set(ylabel="corrected brain-age gap (years)",
                                                 title="Held-out gap per subject (bar: median)")
    for i, s in enumerate(["Female", "Male"]):
        d = held[held.sex_label == s].bag
        ax[2].scatter(i + rng.uniform(-.18, .18, len(d)), d, s=14, color="#6b6f76", alpha=.6)
        ax[2].plot([i - .28, i + .28], [d.median()] * 2, color="k", lw=2)
    ax[2].axhline(0, color="grey", lw=.8, ls=":")
    ax[2].set_xticks(range(2), ["Female", "Male"]); ax[2].set(title="By sex")
    ax[2].set_ylim(ax[1].get_ylim())
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "age_distribution_and_gaps.png", dpi=300)

    # 3. Volume-normalised attribution.
    fine, names, to_coarse, coarse_names = gc.build_atlas()
    coarse = to_coarse[fine]
    roi = pd.read_csv(P / "results" / "phase7" / "roi_attribution_coarse.csv")
    roi = roi[roi.split != "train"]
    vol = []
    for i in roi.IXI_ID:
        v = np.load(ft.volume_path(P / ft.DEFAULT_VOLUME_SUBDIR, int(i)))
        cnt = np.bincount(coarse[v > 0], minlength=len(coarse_names)).astype(float)
        vol.append(cnt / cnt.sum())
    vol = pd.DataFrame(vol, columns=coarse_names, index=roi.index)
    dens = roi[coarse_names] / vol
    dens = pd.concat([roi[["IXI_ID", "site", "sex", "age"]], dens], axis=1)
    rows = []
    for c in coarse_names:
        d = dens.rename(columns={c: "y"})
        m = smf.ols("y ~ C(site, Treatment('Guys')) + C(sex) + age", data=d).fit(cov_type="HC3")
        for s in ["HH", "IOP"]:
            t = f"C(site, Treatment('Guys'))[T.{s}]"
            rows.append({"region": c, "site": s, "mean_density": float(d.y.mean()), "volume_share": float(vol[c].mean()),
                         "est": float(m.params[t]), "p": float(m.pvalues[t])})
    r = pd.DataFrame(rows)
    r["q_fdr"] = multipletests(r.p, method="fdr_bh")[1]
    r.to_csv(out / "attribution_density_site_effects.csv", index=False)
    print(r.round(3).to_string())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(coarse_names))
    for k, s in enumerate(SITES):
        ax.bar(x + (k - 1) * .27, dens[dens.site == s][coarse_names].mean(), .27, color=COLORS[s], label=s)
    ax.axhline(1, color="k", lw=.8, ls=":")
    ax.set_xticks(x, [c.replace(" and ", " & ") for c in coarse_names], rotation=25, ha="right", fontsize=8)
    ax.set(ylabel="attribution share / volume share", title="Attribution density by region and site (held-out)")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "attribution_density.png", dpi=300)


if __name__ == "__main__":
    main()
