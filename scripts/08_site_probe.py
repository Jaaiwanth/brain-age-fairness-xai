"""Phase 9a - Does the age model carry a scanner fingerprint? (site probe)

The model was trained only to predict age. This tests whether its internal summary of a
brain (the 64 numbers after global average pooling, just before the age head) also
identifies the acquisition site - i.e. whether site information is present in the
representation the age prediction is read from.

For every subject (train/val/test), the pretrained UK Biobank SFCN and the fine-tuned Phase 4
SFCN are run once (no training, no gradients). Then a linear probe (multinomial logistic
regression, class-balanced, regularisation chosen by 5-fold CV on the TRAIN split only) is fitted
on train subjects and scored on held-out subjects (val + test) for these feature sets:

    demographics      age, age^2, sex            -> how much site follows from who was recruited
    intensity         33 brain-intensity quantiles -> is site visible in the input images?
    sfcn_pretrained   64-d embedding, UK Biobank SFCN (never saw IXI)
    sfcn_finetuned    64-d embedding, fine-tuned SFCN
    *_resid           same embeddings with age, age^2 and sex regressed out (fit on train),
                      so the probe cannot succeed by reading age or sex

Metric: balanced accuracy (chance = 33.3% for three sites), 95% bootstrap CI over held-out
subjects. Exploratory link to the brain-age gap: on held-out subjects, does the probe's
P(IOP) relate to the bias-corrected BAG, overall and within site (bag ~ p_iop + site + age)?

The probe is a measurement tool; nothing about the age model is changed or selected here,
and the test subjects are used only to score the probe (Rule 7).

Usage:
    python scripts/08_site_probe.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo \
        --checkpoint "G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official/best.pt"
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

SITES = ["Guys", "HH", "IOP"]
N_QUANTILES = 33
N_BOOT = 2000


def parse_args():
    p = argparse.ArgumentParser(description="Site probe on SFCN embeddings.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Only the first N subjects per split.")
    return p.parse_args()


def load_pretrained(repo_dir):
    from dp_model.model_files.sfcn import SFCN
    model = SFCN()
    state = torch.load(repo_dir / ft.SFCN_WEIGHTS_RELPATH, map_location="cpu")
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model.eval()


def add_pool_hook(model, store, key):
    model.classifier.average_pool.register_forward_hook(
        lambda m, i, o: store.__setitem__(key, o.flatten(1).detach().numpy()))


@torch.no_grad()
def extract(df, volume_dir, models, ft_centers):
    """Embeddings from each model, fine-tuned predicted age and intensity quantiles."""
    store = {}
    for name, m in models.items():
        add_pool_hook(m, store, name)
    ds = ft.IXIDataset(df, volume_dir, ft.CONFIG["input_shape"], augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    out = {name: [] for name in models} | {"intensity": [], "pred": [], "ids": []}
    qs = np.linspace(1, 99, N_QUANTILES)
    start, done = time.time(), 0
    for volumes, _, ids in loader:
        for name, m in models.items():
            log_probs = m(volumes)[0].reshape(volumes.size(0), -1)
            out[name].append(store[name])
            if name == "sfcn_finetuned":
                out["pred"] += ft.predict_ages(log_probs, ft_centers).tolist()
        for v in volumes.numpy()[:, 0]:
            out["intensity"].append(np.percentile(v[v > 0], qs))
        out["ids"] += ids.tolist()
        done += volumes.size(0)
        if done % 40 < 4 or done == len(ds):
            rate = (time.time() - start) / done
            print(f"  {done}/{len(ds)} | {rate:.1f}s/subject | ETA {rate * (len(ds) - done) / 60:.0f} min",
                  flush=True)
    return {k: (np.concatenate(v) if k in models else np.asarray(v)) for k, v in out.items()}


def residualise(X, covars, train_mask):
    """Remove the linear effect of covariates (fit on train only)."""
    C = np.column_stack([np.ones(len(covars)), covars])
    beta, *_ = np.linalg.lstsq(C[train_mask], X[train_mask], rcond=None)
    return X - C @ beta


def probe(X, y, train_mask, rng):
    clf = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 11), cv=5, class_weight="balanced", scoring="balanced_accuracy",
        max_iter=10000))
    clf.fit(X[train_mask], y[train_mask])
    test = ~train_mask
    pred, yt = clf.predict(X[test]), y[test]
    proba = clf.predict_proba(X[test])
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(yt), len(yt))
        if len(set(yt[i])) == len(SITES):
            boots.append(balanced_accuracy_score(yt[i], pred[i]))
    return {
        "balanced_accuracy": balanced_accuracy_score(yt, pred),
        "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5)),
        "accuracy": float((pred == yt).mean()),
        **{f"recall_{s}": r for s, r in zip(SITES, recall_score(yt, pred, labels=SITES, average=None))},
        "confusion": confusion_matrix(yt, pred, labels=SITES).tolist(),
        "C": float(clf[-1].C_[0]),
        "p_iop": proba[:, list(clf.classes_).index("IOP")],
    }


def main():
    args = parse_args()
    project_dir = args.project_dir
    checkpoint = args.checkpoint or project_dir / "checkpoints" / "sfcn_finetune_official" / "best.pt"
    volume_dir = args.volume_dir or project_dir / ft.DEFAULT_VOLUME_SUBDIR
    out_dir = args.out_dir or project_dir / "results" / "phase9a_site_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)

    splits = {s: ft.load_split(project_dir, s) for s in ["train", "val", "test"]}
    ft.check_labels_against_metadata(project_dir, splits)
    frames = []
    for s, d in splits.items():
        d = d[~d.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)]
        frames.append((d.head(args.limit) if args.limit else d).assign(split=s))
    df = pd.concat(frames, ignore_index=True)

    ft.ensure_sfcn_repo(args.sfcn_repo)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, centers = ft.make_bins(ckpt["config"]["bin_range"], ckpt["config"]["bin_step"])
    finetuned = ft.build_model(args.sfcn_repo, len(centers), torch.device("cpu"))
    finetuned.load_state_dict(ckpt["model_state_dict"])
    models = {"sfcn_pretrained": load_pretrained(args.sfcn_repo), "sfcn_finetuned": finetuned.eval()}
    print(f"Subjects: {len(df)} {df.split.value_counts().to_dict()} | sites {df.site.value_counts().to_dict()}")

    cache = out_dir / "features.npz"
    if cache.exists() and np.array_equal(np.load(cache)["ids"], df.IXI_ID.to_numpy()):
        feats = dict(np.load(cache))
        print(f"Loaded cached features from {cache}")
    else:
        feats = extract(df, volume_dir, models, torch.tensor(centers, dtype=torch.float32))
        assert np.array_equal(feats["ids"], df.IXI_ID.to_numpy())
        np.savez_compressed(cache, **feats)

    # Bias-corrected BAG with the Phase 5 (validation-fitted) correction.
    bc = json.loads((project_dir / "results" / "phase5" / "summary.json").read_text())["bias_correction"]
    df["predicted_age"] = feats["pred"]
    df["bag_corrected"] = (df.predicted_age - bc["intercept"]) / bc["slope"] - df.AGE

    y = df.site.to_numpy()
    train = (df.split == "train").to_numpy()
    covars = np.column_stack([df.AGE, df.AGE ** 2, (df.sex_label == "Male").astype(float)])
    feature_sets = {
        "demographics": covars,
        "intensity": feats["intensity"],
        "sfcn_pretrained": feats["sfcn_pretrained"],
        "sfcn_finetuned": feats["sfcn_finetuned"],
        "sfcn_pretrained_resid": residualise(feats["sfcn_pretrained"], covars, train),
        "sfcn_finetuned_resid": residualise(feats["sfcn_finetuned"], covars, train),
    }

    rows, held = [], df[~train].copy()
    for name, X in feature_sets.items():
        r = probe(np.asarray(X, dtype=float), y, train, rng)
        if name == "sfcn_finetuned_resid":
            held["p_iop"] = r["p_iop"]
        r.pop("p_iop")
        rows.append({"features": name, **r})
    res = pd.DataFrame(rows)
    res.drop(columns="confusion").to_csv(out_dir / "probe_results.csv", index=False)

    # Exploratory: is the model's "IOP-likeness" related to the brain-age gap?
    rho, rho_p = spearmanr(held.p_iop, held.bag_corrected)
    within = smf.ols("bag_corrected ~ p_iop + C(site) + AGE", data=held).fit(cov_type="HC3")
    link = {"spearman_rho": float(rho), "spearman_p": float(rho_p),
            "within_site_slope_per_unit_p": float(within.params["p_iop"]),
            "within_site_ci": [float(x) for x in within.conf_int().loc["p_iop"]],
            "within_site_p": float(within.pvalues["p_iop"])}
    held[["IXI_ID", "split", "site", "sex_label", "AGE", "predicted_age", "bag_corrected", "p_iop"]].to_csv(
        out_dir / "heldout_p_iop.csv", index=False)

    summary = {"n_train": int(train.sum()), "n_heldout": int((~train).sum()),
               "chance_balanced_accuracy": 1 / len(SITES), "bias_correction": bc,
               "probe": {r["features"]: {k: v for k, v in r.items() if k != "features"} for r in rows},
               "p_iop_vs_bag": link}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\nSite probe on held-out subjects (n={(~train).sum()}), chance = 33.3% balanced accuracy")
    for r in rows:
        print(f"  {r['features']:24s} {100 * r['balanced_accuracy']:5.1f}% "
              f"[{100 * r['ci_low']:.1f}, {100 * r['ci_high']:.1f}] | recall Guys {100 * r['recall_Guys']:.0f}%"
              f" HH {100 * r['recall_HH']:.0f}% IOP {100 * r['recall_IOP']:.0f}%")
    cm = res.set_index("features").loc["sfcn_finetuned_resid", "confusion"]
    print("Confusion (fine-tuned, age/sex removed), rows = true, cols = predicted", SITES)
    for s, row in zip(SITES, cm):
        print(f"  {s:5s} {row}")
    print(f"P(IOP) vs corrected BAG: Spearman rho {rho:+.2f} (p={rho_p:.3f}); within site, "
          f"BAG change per 0->1 P(IOP): {link['within_site_slope_per_unit_p']:+.2f} yrs "
          f"[{link['within_site_ci'][0]:+.2f}, {link['within_site_ci'][1]:+.2f}] (p={link['within_site_p']:.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"demographics": "Age + sex only", "intensity": "Image intensities",
              "sfcn_pretrained": "SFCN pretrained", "sfcn_finetuned": "SFCN fine-tuned",
              "sfcn_pretrained_resid": "SFCN pretrained\n(age/sex removed)",
              "sfcn_finetuned_resid": "SFCN fine-tuned\n(age/sex removed)"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    x = np.arange(len(res))
    ba = 100 * res.balanced_accuracy
    ax.bar(x, ba, color=["0.6", "0.6", "C0", "C1", "C0", "C1"])
    ax.errorbar(x, ba, yerr=[ba - 100 * res.ci_low, 100 * res.ci_high - ba], fmt="none", ecolor="k", capsize=4)
    ax.axhline(100 / 3, color="k", ls="--", lw=1, label="chance (33%)")
    ax.set_xticks(x, [labels[f] for f in res.features], fontsize=8, rotation=20, ha="right")
    ax.set(ylim=(0, 105), ylabel="Site identified correctly\n(balanced accuracy, %)",
           title="Can the acquisition site be read from the model's features?")
    ax.legend(loc="upper left")
    ax = axes[1]
    cm = np.array(res.set_index("features").loc["sfcn_finetuned_resid", "confusion"])
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(len(SITES)):
        for j in range(len(SITES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] / cm[i].sum() > 0.5 else "black")
    ax.set_xticks(range(len(SITES)), SITES)
    ax.set_yticks(range(len(SITES)), SITES)
    ax.set(xlabel="Site predicted from features", ylabel="True site",
           title="Fine-tuned SFCN, age/sex removed\n(held-out subjects)")
    plt.tight_layout()
    fig.savefig(out_dir / "site_probe.png", dpi=130)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
