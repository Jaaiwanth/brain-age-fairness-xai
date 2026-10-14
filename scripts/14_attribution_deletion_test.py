"""Phase 9g - Are the attribution maps faithful? Deletion test.

If an attribution map is faithful, removing the image content it ranks highest should change the
prediction more than removing random content, and much more than removing the lowest-ranked content
(Samek et al. 2017; Hooker et al. 2019 for the occlusion caveat).

For held-out subjects (val + test; a fixed random subset), using the Phase 7 maps
(results/phase7/gradcam_conv_2_lowres.npz, 20x24x20 cells of 8x8x8 voxels) and the frozen Phase 4
model:
    - brain cells: cells with at least half of their voxels inside the brain mask
    - remove the top-ranked / bottom-ranked / random 5%, 10%, 20% of brain cells
      (two random draws, averaged)
    - "remove" = replace the cell's voxels with a heavily smoothed copy of the image (Gaussian
      sigma 8 voxels, within the brain mask): local intensity is kept, anatomical detail is lost,
      so every condition applies the same kind of perturbation and only WHERE differs
Outcome: |change in predicted age| per subject; paired comparisons top vs random and
random vs bottom, 95% bootstrap CIs and Wilcoxon signed-rank tests.

Usage:
    python scripts/14_attribution_deletion_test.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo \
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
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

CELL = 8
FRACTIONS = [0.05, 0.10, 0.20]
N_SUBJECTS = 60
N_BOOT = 5000


def parse_args():
    p = argparse.ArgumentParser(description="Deletion test for the Phase 7 attribution maps.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--maps", type=Path, default=None)
    p.add_argument("--n-subjects", type=int, default=N_SUBJECTS)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def occlude(v, smooth, cells):
    out = v.copy()
    for i, j, k in cells:
        s = np.s_[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL, k * CELL:(k + 1) * CELL]
        out[s] = smooth[s]
    return (out / (out.sum() / ft.SFCN_NORMALISATION_VOXELS)).astype(np.float32)


def boot_ci(d, rng):
    d = np.asarray(d)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    args = parse_args()
    project_dir = args.project_dir
    checkpoint = args.checkpoint or project_dir / "checkpoints" / "sfcn_finetune_official" / "best.pt"
    volume_dir = args.volume_dir or project_dir / ft.DEFAULT_VOLUME_SUBDIR
    maps = np.load(args.maps or project_dir / "results" / "phase7" / f"gradcam_conv_2_lowres.npz")
    out_dir = args.out_dir or project_dir / "results" / "phase9g_deletion_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)

    held = pd.concat([ft.load_split(project_dir, s) for s in ["val", "test"]], ignore_index=True)
    held = held[~held.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)]
    held = held.sample(n=min(args.n_subjects, len(held)), random_state=0).reset_index(drop=True)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, centers = ft.make_bins(ckpt["config"]["bin_range"], ckpt["config"]["bin_step"])
    ft.ensure_sfcn_repo(args.sfcn_repo)
    model = ft.build_model(args.sfcn_repo, len(centers), torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    centers_t = torch.tensor(centers, dtype=torch.float32)
    print(f"Subjects: {len(held)} {held.site.value_counts().to_dict()} | fractions {FRACTIONS}")

    rows, start = [], time.time()
    for n, r in held.iterrows():
        v = np.load(ft.volume_path(volume_dir, r.IXI_ID)).astype(np.float32)
        v = v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)
        brain = v > 0
        smooth = gaussian_filter(v, 8) * brain
        frac_in = brain.reshape(20, CELL, 24, CELL, 20, CELL).mean(axis=(1, 3, 5))
        cells = np.argwhere(frac_in >= 0.5)
        score = maps[f"IXI{int(r.IXI_ID):03d}"][tuple(cells.T)]
        order = cells[np.argsort(-score)]
        conds = {"original": v}
        for f in FRACTIONS:
            k = max(1, int(round(f * len(cells))))
            conds[f"top {f:.2f}"] = occlude(v, smooth, order[:k])
            conds[f"bottom {f:.2f}"] = occlude(v, smooth, order[-k:])
            for d in range(2):
                conds[f"random{d} {f:.2f}"] = occlude(v, smooth, cells[rng.choice(len(cells), k, replace=False)])
        names = list(conds)
        preds = []
        with torch.no_grad():
            for i in range(0, len(names), 4):
                x = torch.from_numpy(np.stack([conds[m] for m in names[i:i + 4]])[:, None])
                preds += ft.predict_ages(model(x)[0].reshape(x.size(0), -1), centers_t).tolist()
        base = preds[0]
        for name, p in zip(names[1:], preds[1:]):
            kind, f = name.split()
            rows.append({"IXI_ID": r.IXI_ID, "site": r.site, "age": r.AGE, "n_cells": len(cells),
                         "kind": kind.rstrip("01"), "draw": kind, "fraction": float(f), "delta": p - base})
        if (n + 1) % 10 == 0 or n + 1 == len(held):
            rate = (time.time() - start) / (n + 1)
            print(f"  {n + 1}/{len(held)} | {rate:.1f}s/subject | ETA {rate * (len(held) - n - 1) / 60:.0f} min",
                  flush=True)

    df = pd.DataFrame(rows)
    df["abs_delta"] = df.delta.abs()
    df.to_csv(out_dir / "deletion_predictions.csv", index=False)
    per = df.groupby(["IXI_ID", "fraction", "kind"]).abs_delta.mean().unstack("kind")

    res = []
    print("\n|Change in predicted age| (years) after removing the top / random / bottom-ranked cells:")
    for f in FRACTIONS:
        x = per.xs(f, level="fraction")
        out = {"fraction": f}
        for kind in ["top", "random", "bottom"]:
            out[f"{kind}_mean"] = float(x[kind].mean())
            out[f"{kind}_ci"] = boot_ci(x[kind], rng)
        for a, b in [("top", "random"), ("random", "bottom")]:
            d = x[a] - x[b]
            out[f"{a}_minus_{b}"] = float(d.mean())
            out[f"{a}_minus_{b}_ci"] = boot_ci(d, rng)
            out[f"{a}_minus_{b}_wilcoxon_p"] = float(wilcoxon(x[a], x[b]).pvalue)
            out[f"{a}_gt_{b}_share"] = float((d > 0).mean())
        res.append(out)
        print(f"  {100 * f:4.0f}% of brain: top {out['top_mean']:.2f} | random {out['random_mean']:.2f} | "
              f"bottom {out['bottom_mean']:.2f}  -> top-random {out['top_minus_random']:+.2f} "
              f"[{out['top_minus_random_ci'][0]:+.2f}, {out['top_minus_random_ci'][1]:+.2f}] "
              f"(p={out['top_minus_random_wilcoxon_p']:.2g}, {100 * out['top_gt_random_share']:.0f}% of subjects)")
    pd.DataFrame(res).to_csv(out_dir / "deletion_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps({"n_subjects": len(held), "cell_voxels": CELL,
                                                      "results": res}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs = 100 * np.array(FRACTIONS)
    for kind, col, lab in [("top", "C3", "highest-attributed removed"), ("random", "C7", "random removed"),
                           ("bottom", "C0", "lowest-attributed removed")]:
        m = np.array([r[f"{kind}_mean"] for r in res])
        c = np.array([r[f"{kind}_ci"] for r in res])
        ax.plot(xs, m, "o-", color=col, label=lab)
        ax.fill_between(xs, c[:, 0], c[:, 1], color=col, alpha=0.15)
    ax.set(xlabel="Share of brain removed (%)", ylabel="|Change in predicted age| (yrs)",
           title=f"Deletion test of the attribution maps (n={len(held)})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "deletion_test.png", dpi=130)
    print(f"\nSaved to {out_dir} ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
