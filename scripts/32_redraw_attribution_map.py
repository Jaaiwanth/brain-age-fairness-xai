"""Redraws the paper's mean attribution map (all held-out subjects) at print resolution.

Uses the group-mean maps saved by Phase 7 (no model needed) and the same template, slices and colour
scale as the "all" row of mean_gradcam_by_group.png.

Usage: python scripts/32_redraw_attribution_map.py --project-dir D:/ML_Project --out paper/figures/attribution_mean_map.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TEMPLATE = "MNI152NLin6Asym"
KEYS = ["all", "site=Guys", "site=HH", "site=IOP", "sex=Female", "sex=Male"]


def center_crop(v, shape=(160, 192, 160)):
    s = [(a - b) // 2 for a, b in zip(v.shape, shape)]
    return v[s[0]:s[0] + shape[0], s[1]:s[1] + shape[1], s[2]:s[2] + shape[2]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    p.add_argument("--out", type=Path, default=Path("attribution_mean_map.png"))
    a = p.parse_args()
    import ants
    from templateflow import api as tflow
    tpl = center_crop(ants.image_read(str(tflow.get(TEMPLATE, resolution=1, suffix="T1w", desc=None,
                                                    extension=".nii.gz"))).numpy())
    means = np.load(a.project_dir / "results" / "phase7" / "group_mean_maps_heldout.npz")
    n = json.loads((a.project_dir / "results" / "phase7" / "summary.json").read_text())["n_heldout"]
    vmax = max(np.percentile(means[k][means[k] > 0], 99.5) for k in KEYS if k in means.files)
    m = means["all"]
    cx, cy, cz = 80, 100, 72
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9))
    for ax, (t_sl, m_sl) in zip(axes, [(tpl[cx], m[cx]), (tpl[:, cy], m[:, cy]), (tpl[:, :, cz], m[:, :, cz])]):
        ax.imshow(np.rot90(t_sl), cmap="gray")
        ax.imshow(np.rot90(np.ma.masked_less_equal(m_sl, 0)), cmap="hot", alpha=0.6, vmin=0, vmax=vmax)
        ax.axis("off")
    axes[0].set_title(f"all held-out subjects (n={n})", loc="left", fontsize=9)
    plt.tight_layout()
    fig.savefig(a.out, dpi=300)
    print("saved", a.out)


if __name__ == "__main__":
    main()
