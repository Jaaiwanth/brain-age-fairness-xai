"""Phase 9e - Remove the site fingerprint from the model's features (ComBat) and re-predict age.

Phase 9a: the 64-d pooled features that the age head reads identify the acquisition site (91%
after removing age/sex). Here those features are harmonised with ComBat (Johnson et al. 2007;
Fortin et al. 2018) - location/scale version, no empirical Bayes - preserving age, age^2 and sex.
Site parameters are estimated on TRAIN subjects only and applied to everyone. Age is then
re-predicted by passing the harmonised features through the model's own, unchanged age head
(classifier.conv_6 -> softmax -> expected age), so nothing is retrained.

Compared, original vs harmonised, on held-out subjects (val + test, n = 146):
    - site probe balanced accuracy (same probe as Phase 9a; chance 33%)
    - MAE and r (raw), and the bias-corrected BAG (correction refitted on val for each version)
    - bag_corrected ~ sex + site + age, HC3, 95% bootstrap CIs (site effects vs Guys)

Rule 7: the frozen Phase 4 model and its Phase 5 test result stand. ComBat has no tuning
parameters; this is a secondary analysis, reported side by side with the original on the same
subjects, and nothing here is selected using test labels.
Caveat: ComBat removes ALL site-associated feature differences not explained by age and sex,
including any genuine cohort differences, so a smaller site offset after harmonisation does not by
itself show the offset was caused by the scanner.

Usage:
    python scripts/12_harmonise_features.py --project-dir D:/ML_Project \
        --checkpoint "G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official/best.pt"
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ft = _load("ft", "04_finetune_sfcn.py")
sp = _load("sp", "08_site_probe.py")

SITES = ["Guys", "HH", "IOP"]
N_BOOT = 2000


def parse_args():
    p = argparse.ArgumentParser(description="ComBat harmonisation of SFCN features.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--features", type=Path, default=None,
                   help="Defaults to <project-dir>/results/phase9a_site_probe/features.npz")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def combat_fit(Y, site, covars, train):
    """Location/scale ComBat preserving covariates; parameters from train rows only."""
    S = pd.get_dummies(pd.Categorical(site, categories=SITES)).to_numpy(float)
    D = np.column_stack([S, covars])
    coef, *_ = np.linalg.lstsq(D[train], Y[train], rcond=None)
    gamma, beta = coef[:len(SITES)], coef[len(SITES):]
    n = S[train].sum(0)
    alpha = (n[:, None] * gamma).sum(0) / n.sum()  # pooled grand mean
    resid = Y[train] - D[train] @ coef
    sigma = np.sqrt((resid ** 2).mean(0))
    # gamma holds each site's full intercept (no separate intercept column), so the standardised
    # residual is (Y - X beta - S gamma) / sigma; alpha is only added back when re-composing.
    z = resid / np.where(sigma > 1e-8, sigma, 1.0)
    delta = np.stack([z[S[train][:, k] == 1].std(0) for k in range(len(SITES))])
    return {"alpha": alpha, "beta": beta, "gamma": gamma, "sigma": sigma, "delta": delta}


def combat_apply(Y, site, covars, p):
    """Features that are constant in train (inactive ReLU channels, sigma = 0) are left unchanged."""
    S = pd.get_dummies(pd.Categorical(site, categories=SITES)).to_numpy(float)
    live = p["sigma"] > 1e-8
    out = Y.copy()
    z = (Y[:, live] - covars @ p["beta"][:, live] - S @ p["gamma"][:, live]) / p["sigma"][live]
    delta = np.maximum(S @ p["delta"][:, live], 1e-8)
    out[:, live] = p["sigma"][live] * z / delta + p["alpha"][live] + covars @ p["beta"][:, live]
    return out


def head_predict(E, weight, bias, centers):
    logits = E @ weight.T + bias
    logp = logits - logits.max(1, keepdims=True)
    prob = np.exp(logp) / np.exp(logp).sum(1, keepdims=True)
    return prob @ centers


def site_effects(d, rng):
    formula = "bag_corrected ~ C(sex_label) + C(site, Treatment('Guys')) + AGE"
    fit = smf.ols(formula, data=d).fit(cov_type="HC3")
    terms = {"HH vs Guys": "C(site, Treatment('Guys'))[T.HH]", "IOP vs Guys": "C(site, Treatment('Guys'))[T.IOP]",
             "Male vs Female": "C(sex_label)[T.Male]"}
    boots = {k: [] for k in terms}
    for _ in range(N_BOOT):
        b = d.iloc[rng.integers(0, len(d), len(d))]
        if b.site.nunique() < 3:
            continue
        f = smf.ols(formula, data=b).fit()
        for k, t in terms.items():
            boots[k].append(f.params[t])
    return {k: {"estimate": float(fit.params[t]), "ci": [float(np.percentile(boots[k], 2.5)),
                                                          float(np.percentile(boots[k], 97.5))],
                "p_hc3": float(fit.pvalues[t])} for k, t in terms.items()}


def main():
    args = parse_args()
    project_dir = args.project_dir
    checkpoint = args.checkpoint or project_dir / "checkpoints" / "sfcn_finetune_official" / "best.pt"
    feat_path = args.features or project_dir / "results" / "phase9a_site_probe" / "features.npz"
    out_dir = args.out_dir or project_dir / "results" / "phase9e_harmonisation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    splits = {s: ft.load_split(project_dir, s) for s in ["train", "val", "test"]}
    df = pd.concat([d[~d.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].assign(split=s) for s, d in splits.items()],
                   ignore_index=True)
    feats = np.load(feat_path)
    assert np.array_equal(feats["ids"], df.IXI_ID.to_numpy()), "feature cache does not match the splits"
    E = feats["sfcn_finetuned"].astype(np.float64)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    W = sd["classifier.conv_6.weight"].reshape(sd["classifier.conv_6.weight"].shape[0], -1).double().numpy()
    b = sd["classifier.conv_6.bias"].double().numpy()
    _, centers = ft.make_bins(ckpt["config"]["bin_range"], ckpt["config"]["bin_step"])
    check = np.abs(head_predict(E, W, b, centers) - feats["pred"]).max()
    print(f"Age head re-applied to cached features reproduces the model's predictions (max diff {check:.4f} yrs)")
    assert check < 0.05

    train = (df.split == "train").to_numpy()
    covars = np.column_stack([df.AGE, df.AGE ** 2, (df.sex_label == "Male").astype(float)])
    params = combat_fit(E, df.site.to_numpy(), covars, train)
    E_h = combat_apply(E, df.site.to_numpy(), covars, params)
    print(f"ComBat fitted on {train.sum()} train subjects; site shifts (mean |gamma - alpha| per feature): "
          + ", ".join(f"{s} {np.abs(params['gamma'][k] - params['alpha']).mean():.3f}" for k, s in enumerate(SITES)))

    versions = {"original": E, "harmonised": E_h}
    results, held_frames = {}, []
    y = df.site.to_numpy()
    for name, X in versions.items():
        pred = head_predict(X, W, b, centers)
        d = df.assign(predicted_age=pred)
        val = d[d.split == "val"]
        a_, b_ = np.polyfit(val.AGE, val.predicted_age, 1)
        d["bag_corrected"] = (d.predicted_age - b_) / a_ - d.AGE
        held = d[d.split != "train"].copy()
        err = held.predicted_age - held.AGE
        probe = sp.probe(sp.residualise(X, covars, train), y, train, rng)
        results[name] = {
            "site_probe_balanced_accuracy": float(probe["balanced_accuracy"]),
            "site_probe_ci": [probe["ci_low"], probe["ci_high"]],
            "heldout_mae_raw": float(err.abs().mean()), "heldout_r": float(np.corrcoef(held.AGE, held.predicted_age)[0, 1]),
            "test_mae_raw": float(err[held.split == "test"].abs().mean()),
            "bias_correction": {"slope": float(a_), "intercept": float(b_)},
            "mean_bag_corrected_by_site": held.groupby("site").bag_corrected.mean().round(3).to_dict(),
            "effects": site_effects(held, rng),
        }
        held_frames.append(held.assign(version=name))
    pd.concat(held_frames)[["IXI_ID", "split", "site", "sex_label", "AGE", "version", "predicted_age",
                            "bag_corrected"]].to_csv(out_dir / "heldout_predictions.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(
        {"method": "ComBat location/scale (no empirical Bayes), covariates age, age^2, sex; fitted on train",
         "n_train": int(train.sum()), "n_heldout": int((~train).sum()), "results": results}, indent=2))

    print(f"\nHeld-out subjects (val + test, n={(~train).sum()}):")
    print(f"{'':32s}{'original':>22s}{'harmonised':>22s}")
    o, h = results["original"], results["harmonised"]
    print(f"{'Site probe (age/sex removed)':32s}{100 * o['site_probe_balanced_accuracy']:21.1f}%"
          f"{100 * h['site_probe_balanced_accuracy']:21.1f}%")
    print(f"{'MAE raw (held-out)':32s}{o['heldout_mae_raw']:22.2f}{h['heldout_mae_raw']:22.2f}")
    print(f"{'MAE raw (test only)':32s}{o['test_mae_raw']:22.2f}{h['test_mae_raw']:22.2f}")
    print(f"{'r':32s}{o['heldout_r']:22.3f}{h['heldout_r']:22.3f}")
    for k in ["IOP vs Guys", "HH vs Guys", "Male vs Female"]:
        fmt = lambda e: f"{e['estimate']:+.2f} [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]"
        print(f"{'BAG ' + k:32s}{fmt(o['effects'][k]):>22s}{fmt(h['effects'][k]):>22s}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names, cols = ["original", "harmonised"], ["C7", "C2"]
    ax = axes[0]
    vals = [100 * results[n]["site_probe_balanced_accuracy"] for n in names]
    ci = np.array([[100 * c for c in results[n]["site_probe_ci"]] for n in names])
    ax.bar(names, vals, color=cols)
    ax.errorbar(names, vals, yerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)], fmt="none", ecolor="k", capsize=4)
    ax.axhline(100 / 3, color="k", ls="--", lw=1, label="chance")
    ax.set(ylim=(0, 105), ylabel="Site identified (balanced acc., %)", title="Site fingerprint in the features")
    ax.legend()
    ax = axes[1]
    terms = ["IOP vs Guys", "HH vs Guys", "Male vs Female"]
    yy = np.arange(len(terms))
    for i, n in enumerate(names):
        e = np.array([results[n]["effects"][t]["estimate"] for t in terms])
        c = np.array([results[n]["effects"][t]["ci"] for t in terms])
        ax.errorbar(e, yy + (0.15 if i else -0.15), xerr=[e - c[:, 0], c[:, 1] - e], fmt="o", capsize=4,
                    color=cols[i], label=n)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(yy, terms)
    ax.set(xlabel="Corrected brain-age gap difference (yrs), age-adjusted", title="Group effects (held-out)")
    ax.legend()
    ax = axes[2]
    vals = [results[n]["heldout_mae_raw"] for n in names]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center")
    ax.set(ylabel="MAE (years)", title="Accuracy (held-out)")
    plt.tight_layout()
    fig.savefig(out_dir / "harmonisation.png", dpi=300)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
