"""Phase 9b - Acquisition stress test: does changing only the scanner "look" change brain age?

Phase 9a showed the acquisition site is readable from the model's features. This tests whether
the age prediction RESPONDS to acquisition appearance, by changing the image appearance of the
same brains while leaving anatomy untouched, then predicting age again with the frozen Phase 4
model. Held-out subjects only (val + test); nothing is fitted to them.

1. Site intensity swap (main test). Each subject's brain-intensity distribution is mapped onto
   the average distribution of a target site (quantile / histogram matching, 101 quantiles).
   Target distributions come from TRAIN subjects only and are standardised to a common age and
   sex mix, so they differ in appearance, not in the ages each site recruited. Matching to the
   subject's own site is the control: any larger change is due to the other site's look.
2. Single-property sweeps on a 40-subject subset: contrast (gamma), noise, blur and a smooth
   intensity non-uniformity, two strengths each.

Every perturbed volume is re-normalised the SFCN way (divide by the full-box mean), as the
original inputs are. Outcome: change in raw predicted age (perturbed - original), paired per
subject, 95% bootstrap CI. (Dividing by the bias-correction slope, 0.83, converts it to
corrected-BAG years.)

Usage:
    python scripts/09_acquisition_stress_test.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo \
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
import torch
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

SITES = ["Guys", "HH", "IOP"]
QS = np.linspace(0, 100, 101)
N_SWEEP = 40
N_BOOT = 5000
SWEEPS = {
    "contrast gamma 0.8": ("gamma", 0.8), "contrast gamma 1.25": ("gamma", 1.25),
    "noise 2%": ("noise", 0.02), "noise 5%": ("noise", 0.05),
    "blur sigma 1 voxel": ("blur", 1.0), "blur sigma 1.5 voxels": ("blur", 1.5),
    "non-uniformity +10%": ("bias", 0.10), "non-uniformity -10%": ("bias", -0.10),
}


def parse_args():
    p = argparse.ArgumentParser(description="Acquisition stress test.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Only the first N held-out subjects.")
    return p.parse_args()


def load_normalised(volume_dir, ixi_id):
    v = np.load(ft.volume_path(volume_dir, ixi_id)).astype(np.float32)
    return v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)


def renormalise(v):
    return (v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)).astype(np.float32)


def match_to(v, target_q):
    out = np.zeros_like(v)
    brain = v > 0
    out[brain] = np.interp(v[brain], np.percentile(v[brain], QS), target_q)
    return renormalise(np.clip(out, 0, None))


def perturb(v, kind, level, rng):
    brain = v > 0
    out = v.copy()
    if kind == "gamma":
        m = v[brain].max()
        out[brain] = m * (v[brain] / m) ** level
    elif kind == "noise":
        out[brain] = v[brain] + rng.normal(0, level * v[brain].mean(), brain.sum())
    elif kind == "blur":
        out = gaussian_filter(v, level) * brain
    elif kind == "bias":
        ramp = np.linspace(-1, 1, v.shape[2], dtype=np.float32)[None, None, :]
        out = v * (1 + level * ramp)
    return renormalise(np.clip(out, 0, None) * brain)


class Predictor:
    def __init__(self, model, centers):
        self.model, self.centers, self.n = model, torch.tensor(centers, dtype=torch.float32), 0

    @torch.no_grad()
    def __call__(self, vols):
        x = torch.from_numpy(np.stack(vols)[:, None])
        self.n += len(vols)
        return ft.predict_ages(self.model(x)[0].reshape(len(vols), -1), self.centers).tolist()


def boot_ci(d, rng):
    d = np.asarray(d)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    args = parse_args()
    project_dir = args.project_dir
    checkpoint = args.checkpoint or project_dir / "checkpoints" / "sfcn_finetune_official" / "best.pt"
    volume_dir = args.volume_dir or project_dir / ft.DEFAULT_VOLUME_SUBDIR
    out_dir = args.out_dir or project_dir / "results" / "phase9b_stress_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)

    splits = {s: ft.load_split(project_dir, s) for s in ["train", "val", "test"]}
    ft.check_labels_against_metadata(project_dir, splits)
    splits = {s: d[~d.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True) for s, d in splits.items()}
    held = pd.concat([splits["val"].assign(split="val"), splits["test"].assign(split="test")],
                     ignore_index=True)
    if args.limit:
        held = held.groupby("site", group_keys=False).head(max(1, args.limit // 3)).reset_index(drop=True)

    # Target intensity distributions per site, from TRAIN subjects only. Sites recruited different
    # ages (IOP train mean ~41 vs ~48), and tissue proportions change with age, so each site's
    # reference is age/sex-standardised: per site, regress the quantile vector on age, age^2 and sex
    # and evaluate at the pooled train mean. The references then differ by site appearance only.
    print("Building age/sex-standardised site intensity references from train subjects...", flush=True)
    tr = splits["train"]
    Q = np.stack([np.percentile(v[v > 0], QS) for v in (load_normalised(volume_dir, i) for i in tr.IXI_ID)])
    cov = np.column_stack([np.ones(len(tr)), tr.AGE, tr.AGE ** 2, (tr.sex_label == "Male").astype(float)])
    common = cov.mean(axis=0)
    target_q = {}
    for s in SITES:
        m = (tr.site == s).to_numpy()
        beta, *_ = np.linalg.lstsq(cov[m], Q[m], rcond=None)
        target_q[s] = np.maximum.accumulate(common @ beta)  # keep the quantile function monotone
    np.savez(out_dir / "site_reference_quantiles.npz", **target_q)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, centers = ft.make_bins(ckpt["config"]["bin_range"], ckpt["config"]["bin_step"])
    ft.ensure_sfcn_repo(args.sfcn_repo)
    model = ft.build_model(args.sfcn_repo, len(centers), torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    predict = Predictor(model.eval(), centers)

    sweep_ids = set(held.sample(n=min(N_SWEEP, len(held)), random_state=0).IXI_ID)
    print(f"Held-out subjects: {len(held)} {held.site.value_counts().to_dict()} | sweep subset {len(sweep_ids)}")

    rows, start = [], time.time()
    for k, r in held.iterrows():
        v = load_normalised(volume_dir, r.IXI_ID)
        conds = {"original": v} | {f"match {s}": match_to(v, target_q[s]) for s in SITES}
        if r.IXI_ID in sweep_ids:
            conds |= {name: perturb(v, kind, lvl, rng) for name, (kind, lvl) in SWEEPS.items()}
        names = list(conds)
        preds = []
        for i in range(0, len(names), 4):
            preds += predict([conds[n] for n in names[i:i + 4]])
        base = preds[0]
        for n, p in zip(names, preds):
            rows.append({"IXI_ID": r.IXI_ID, "site": r.site, "sex": r.sex_label, "age": r.AGE,
                         "condition": n, "predicted_age": p, "delta": p - base})
        done = k + 1
        if done % 10 == 0 or done == len(held):
            rate = (time.time() - start) / done
            print(f"  {done}/{len(held)} | {rate:.1f}s/subject | ETA {rate * (len(held) - done) / 60:.0f} min",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stress_predictions.csv", index=False)

    # Site swap: mean change by (source site, target site).
    swap = []
    for src in SITES:
        for tgt in SITES:
            d = df[(df.site == src) & (df.condition == f"match {tgt}")].delta
            lo, hi = boot_ci(d, rng)
            swap.append({"source_site": src, "target_site": tgt, "n": len(d), "mean_delta": d.mean(),
                         "ci_low": lo, "ci_high": hi, "own_site_control": src == tgt})
    swap = pd.DataFrame(swap)
    swap.to_csv(out_dir / "site_swap_effects.csv", index=False)

    sweep = []
    for name in SWEEPS:
        d = df[df.condition == name].delta
        lo, hi = boot_ci(d, rng)
        sweep.append({"condition": name, "n": len(d), "mean_delta": d.mean(), "ci_low": lo, "ci_high": hi,
                      "mean_abs_delta": d.abs().mean()})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out_dir / "sweep_effects.csv", index=False)

    # Headline: does moving to/from IOP appearance shift age, beyond the own-site control?
    head = {}
    for src, tgt in [("Guys", "IOP"), ("HH", "IOP"), ("IOP", "Guys"), ("IOP", "HH")]:
        a = df[(df.site == src) & (df.condition == f"match {tgt}")].set_index("IXI_ID").predicted_age
        c = df[(df.site == src) & (df.condition == f"match {src}")].set_index("IXI_ID").predicted_age
        d = (a - c.loc[a.index]).to_numpy()
        lo, hi = boot_ci(d, rng)
        head[f"{src} -> {tgt} (vs own-site match)"] = {"n": len(d), "mean_delta": float(d.mean()),
                                                      "ci": [lo, hi], "share_negative": float((d < 0).mean())}
    summary = {"n_heldout": len(held), "n_sweep": len(sweep_ids), "forward_passes": predict.n,
               "minutes": (time.time() - start) / 60, "headline_vs_own_site_control": head,
               "note": "delta = predicted age (perturbed) - predicted age (original), raw years"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSite intensity swap: mean change in predicted age (years), rows = true site, cols = look of")
    print(swap.pivot(index="source_site", columns="target_site", values="mean_delta").round(2).to_string())
    print("\nAgainst the own-site control (same matching procedure, own site's look):")
    for k, h in head.items():
        print(f"  {k:32s} {h['mean_delta']:+.2f} yrs [{h['ci'][0]:+.2f}, {h['ci'][1]:+.2f}] "
              f"n={h['n']} | {100 * h['share_negative']:.0f}% of subjects younger")
    print("\nSingle-property changes (40-subject subset):")
    for _, s in sweep.iterrows():
        print(f"  {s.condition:24s} {s.mean_delta:+.2f} yrs [{s.ci_low:+.2f}, {s.ci_high:+.2f}] | "
              f"mean |change| {s.mean_abs_delta:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    M = swap.pivot(index="source_site", columns="target_site", values="mean_delta").loc[SITES, SITES]
    lim = max(1.0, np.abs(M.values).max())
    ax.imshow(M.values, cmap="RdBu", vmin=-lim, vmax=lim)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M.values[i, j]:+.1f}", ha="center", va="center", fontsize=11)
    ax.set_xticks(range(3), [f"looks like\n{s}" for s in SITES])
    ax.set_yticks(range(3), [f"{s} brains" for s in SITES])
    ax.set_title("Change in predicted age (years)\nafter matching image intensities to a site")
    ax = axes[1]
    y = np.arange(len(sweep))[::-1]
    ax.errorbar(sweep.mean_delta, y, xerr=[sweep.mean_delta - sweep.ci_low, sweep.ci_high - sweep.mean_delta],
                fmt="o", capsize=4)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_yticks(y, sweep.condition)
    ax.set(xlabel="Change in predicted age (years), mean and 95% CI",
           title=f"Single acquisition changes, same brains (n={len(sweep_ids)})")
    plt.tight_layout()
    fig.savefig(out_dir / "stress_test.png", dpi=130)
    print(f"\nSaved to {out_dir} ({summary['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
