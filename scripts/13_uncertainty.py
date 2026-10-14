"""Phase 9f - Does the model know when it is unsure? Predictive uncertainty by site.

SFCN outputs a probability for every 1-year age bin, so each prediction comes with a spread:
    predictive SD   sqrt(sum_k p_k (c_k - mean)^2)
    90% interval    5th-95th percentile of the predicted distribution (bins treated as uniform)
These are the model's own uncertainty, not calibrated intervals; the question is whether they rise
where the model is worse (IOP, and above all IOP when it was never seen in training).

A. Main Phase 4 model, held-out subjects (val + test), from the cached Phase 9a features passed
   through the model's own age head (no model run): SD ~ site + age + sex (HC3); Spearman of SD
   with absolute error; 90%-interval coverage of the true age by site; |error| / SD.
B. Leave-one-site-out models (Phase 6b) without IOP and without HH: each model's unseen-site
   subjects vs its seen-site validation subjects: SD and coverage, SD ~ unseen + age (HC3).
Raw (not bias-corrected) predictions are used throughout, since the intervals come from the raw
distribution. Bootstrap CIs, 5,000 resamples.

Usage:
    python scripts/13_uncertainty.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo \
        --checkpoint-root "G:/My Drive/BrainAge_Project/checkpoints"
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

SITES = ["Guys", "HH", "IOP"]
N_BOOT = 5000


def parse_args():
    p = argparse.ArgumentParser(description="Predictive uncertainty by site.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--checkpoint-root", type=Path, default=None,
                   help="Folder holding sfcn_finetune_official/ and loso_without_*/")
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--loso-sites", nargs="*", default=["IOP", "HH"])
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def describe(prob, centers, edges):
    mean = prob @ centers
    sd = np.sqrt((prob * (centers[None] - mean[:, None]) ** 2).sum(1))
    cdf = np.cumsum(prob, 1)
    def quantile(q):
        k = np.minimum((cdf < q).sum(1), prob.shape[1] - 1)
        prev = np.where(k > 0, cdf[np.arange(len(k)), np.maximum(k - 1, 0)], 0.0)
        frac = (q - prev) / np.maximum(prob[np.arange(len(k)), k], 1e-12)
        return edges[k] + frac * (edges[k + 1] - edges[k])
    return mean, sd, quantile(0.05), quantile(0.95)


def boot_mean_ci(x, rng):
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def add_metrics(d):
    d["abs_error"] = (d.pred - d.AGE).abs()
    d["covered"] = ((d.AGE >= d.lo90) & (d.AGE <= d.hi90)).astype(float)
    d["z"] = d.abs_error / d.sd
    return d


def group_table(d, by, rng):
    rows = []
    for g, x in d.groupby(by):
        rows.append({by: g, "n": len(x), "mean_sd": x.sd.mean(), "sd_ci": boot_mean_ci(x.sd, rng),
                     "mae": x.abs_error.mean(), "coverage90": x.covered.mean(),
                     "coverage_ci": boot_mean_ci(x.covered, rng), "mean_abs_error_over_sd": x.z.mean()})
    return pd.DataFrame(rows)


@torch.no_grad()
def run_model(model, df, volume_dir):
    ds = ft.IXIDataset(df.rename(columns={"age": "AGE"}), volume_dir, ft.CONFIG["input_shape"], augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    probs, ids = [], []
    for volumes, _, batch_ids in loader:
        probs.append(torch.exp(model(volumes)[0].reshape(volumes.size(0), -1).float()).numpy())
        ids += batch_ids.tolist()
    return np.concatenate(probs), ids


def main():
    args = parse_args()
    project_dir = args.project_dir
    ck_root = args.checkpoint_root or project_dir / "checkpoints"
    volume_dir = args.volume_dir or project_dir / ft.DEFAULT_VOLUME_SUBDIR
    feat_path = args.features or project_dir / "results" / "phase9a_site_probe" / "features.npz"
    out_dir = args.out_dir or project_dir / "results" / "phase9f_uncertainty"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)
    summary = {}

    # ---------------- A. main model, held-out subjects ----------------
    ckpt = torch.load(ck_root / "sfcn_finetune_official" / "best.pt", map_location="cpu", weights_only=False)
    edges, centers = ft.make_bins(ckpt["config"]["bin_range"], ckpt["config"]["bin_step"])
    sd_ = ckpt["model_state_dict"]
    W = sd_["classifier.conv_6.weight"].reshape(sd_["classifier.conv_6.weight"].shape[0], -1).double().numpy()
    b = sd_["classifier.conv_6.bias"].double().numpy()
    splits = {s: ft.load_split(project_dir, s) for s in ["train", "val", "test"]}
    df = pd.concat([d[~d.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].assign(split=s) for s, d in splits.items()],
                   ignore_index=True)
    feats = np.load(feat_path)
    assert np.array_equal(feats["ids"], df.IXI_ID.to_numpy())
    logits = feats["sfcn_finetuned"].astype(np.float64) @ W.T + b
    prob = np.exp(logits - logits.max(1, keepdims=True))
    prob /= prob.sum(1, keepdims=True)
    df["pred"], df["sd"], df["lo90"], df["hi90"] = describe(prob, centers, edges)
    assert np.abs(df.pred - feats["pred"]).max() < 0.05
    df = add_metrics(df)
    held = df[df.split != "train"].copy()
    held.to_csv(out_dir / "main_model_uncertainty.csv", index=False)

    tab = group_table(held, "site", rng)
    fit = smf.ols("sd ~ C(site, Treatment('Guys')) + AGE + C(sex_label)", data=held).fit(cov_type="HC3")
    rho, p_rho = spearmanr(held.sd, held.abs_error)
    t = "C(site, Treatment('Guys'))[T.IOP]"
    summary["main_model_heldout"] = {
        "n": len(held), "by_site": tab.to_dict("records"),
        "overall_coverage90": float(held.covered.mean()), "spearman_sd_vs_abs_error": [float(rho), float(p_rho)],
        "sd_iop_vs_guys_adjusted": {"estimate": float(fit.params[t]), "ci": list(map(float, fit.conf_int().loc[t])),
                                    "p": float(fit.pvalues[t])}}
    print(f"A. Main model, held-out (n={len(held)}). Overall 90%-interval coverage {100 * held.covered.mean():.0f}%;"
          f" Spearman(SD, |error|) {rho:+.2f} (p={p_rho:.3f})")
    for _, r in tab.iterrows():
        print(f"   {r.site:5s} n={r.n:3d} | predictive SD {r.mean_sd:.2f} yrs | MAE {r.mae:.2f} | "
              f"90% coverage {100 * r.coverage90:.0f}% [{100 * r.coverage_ci[0]:.0f}, {100 * r.coverage_ci[1]:.0f}] "
              f"| |error|/SD {r.mean_abs_error_over_sd:.2f}")
    print(f"   SD, IOP vs Guys (age/sex-adjusted): {fit.params[t]:+.2f} yrs "
          f"[{fit.conf_int().loc[t][0]:+.2f}, {fit.conf_int().loc[t][1]:+.2f}]")

    # ---------------- B. leave-one-site-out models ----------------
    ft.ensure_sfcn_repo(args.sfcn_repo)
    loso_frames = []
    summary["loso"] = {}
    for site in args.loso_sites:
        d = ck_root / f"loso_without_{site}"
        c = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
        e2, c2 = ft.make_bins(c["config"]["bin_range"], c["config"]["bin_step"])
        model = ft.build_model(args.sfcn_repo, len(c2), torch.device("cpu"))
        model.load_state_dict(c["model_state_dict"])
        model.eval()
        subj = pd.concat([pd.read_csv(d / "heldout_predictions.csv").assign(unseen=1),
                          pd.read_csv(d / "val_predictions.csv").assign(unseen=0)], ignore_index=True)
        print(f"\nB. Model trained without {site}: running {len(subj)} subjects ...", flush=True)
        p2, ids = run_model(model, subj, volume_dir)
        assert ids == subj.IXI_ID.tolist()
        x = subj.rename(columns={"age": "AGE"}).copy()
        x["pred"], x["sd"], x["lo90"], x["hi90"] = describe(p2.astype(np.float64), c2, e2)
        gap = np.abs(x.pred - x.predicted_age).max()
        print(f"   re-run reproduces the saved predictions (max diff {gap:.3f} yrs)")
        x = add_metrics(x).assign(model=f"without_{site}")
        loso_frames.append(x)
        f2 = smf.ols("sd ~ unseen + AGE", data=x).fit(cov_type="HC3")
        tab2 = group_table(x.assign(group=np.where(x.unseen == 1, f"{site} (unseen)", "seen sites")), "group", rng)
        summary["loso"][site] = {"by_group": tab2.to_dict("records"),
                                 "sd_unseen_effect": {"estimate": float(f2.params["unseen"]),
                                                      "ci": list(map(float, f2.conf_int().loc["unseen"])),
                                                      "p": float(f2.pvalues["unseen"])},
                                 "spearman_sd_vs_abs_error": float(spearmanr(x.sd, x.abs_error)[0])}
        for _, r in tab2.iterrows():
            print(f"   {r.group:14s} n={r.n:3d} | SD {r.mean_sd:.2f} | MAE {r.mae:.2f} | 90% coverage "
                  f"{100 * r.coverage90:.0f}% [{100 * r.coverage_ci[0]:.0f}, {100 * r.coverage_ci[1]:.0f}] | "
                  f"|error|/SD {r.mean_abs_error_over_sd:.2f}")
        print(f"   SD unseen vs seen (age-adjusted): {f2.params['unseen']:+.2f} yrs "
              f"[{f2.conf_int().loc['unseen'][0]:+.2f}, {f2.conf_int().loc['unseen'][1]:+.2f}]")
    if loso_frames:
        pd.concat(loso_frames).to_csv(out_dir / "loso_uncertainty.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for i, s in enumerate(SITES):
        g = held[held.site == s]
        ax.scatter(g.sd, g.abs_error, s=14, color=f"C{i}", label=s)
    ax.set(xlabel="Model's own uncertainty (predictive SD, yrs)", ylabel="Actual |error| (yrs)",
           title=f"Main model, held-out: does uncertainty track error?\nSpearman {rho:+.2f}")
    ax.legend()
    ax = axes[1]
    labels, cov, cis, sds = [], [], [], []
    for _, r in tab.iterrows():
        labels.append(f"{r.site}\n(main)")
        cov.append(100 * r.coverage90); cis.append([100 * v for v in r.coverage_ci]); sds.append(r.mean_sd)
    for site in args.loso_sites:
        for r in summary["loso"][site]["by_group"]:
            if "unseen" in r["group"]:
                labels.append(f"{site}\n(unseen)")
                cov.append(100 * r["coverage90"]); cis.append([100 * v for v in r["coverage_ci"]]); sds.append(r["mean_sd"])
    cov, cis = np.array(cov), np.array(cis)
    xx = np.arange(len(labels))
    ax.bar(xx, cov, color=[f"C{i}" for i in range(3)] + ["C2", "C1"][:len(args.loso_sites)], alpha=0.8)
    ax.errorbar(xx, cov, yerr=[cov - cis[:, 0], cis[:, 1] - cov], fmt="none", ecolor="k", capsize=4)
    ax.axhline(90, color="k", ls="--", lw=1, label="nominal 90%")
    ax.set_xticks(xx, labels)
    ax.set(ylim=(0, 105), ylabel="True age inside the model's 90% range (%)", title="Coverage of the model's own 90% range")
    ax.legend(loc="lower left")
    ax = axes[2]
    ax.bar(xx, sds, color=[f"C{i}" for i in range(3)] + ["C2", "C1"][:len(args.loso_sites)], alpha=0.8)
    ax.set_xticks(xx, labels)
    ax.set(ylabel="Mean predictive SD (yrs)", title="How unsure the model says it is")
    plt.tight_layout()
    fig.savefig(out_dir / "uncertainty.png", dpi=130)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
