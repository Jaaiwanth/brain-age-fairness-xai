"""Phase 9c - Image sharpness and resolution by acquisition site.

Phase 9b found that blurring the input lowers predicted age (about -3 yrs at sigma 1.5 voxels)
while global intensity does not explain the IOP offset. This checks whether IOP scans are
blurrier or of different resolution, and whether that accounts for the IOP brain-age gap.

Per subject (all splits):
    native voxel size and matrix   from the raw NIfTI header
    sharpness measures on the preprocessed model input (brain voxels, SFCN-normalised):
        grad_ratio    mean gradient magnitude / mean intensity        (higher = sharper)
        lap_ratio     std of the Laplacian / std of intensity          (higher = sharper)
        fwhm_mm       mean smoothness estimate over the three axes (Forman et al. 1995)
                      (higher = smoother)
Brain anatomy changes with age, so site differences are tested as metric ~ site + age + sex
(HC3). On held-out subjects (val + test, corrected BAG from Phase 9a) the IOP coefficient of
bag_corrected ~ site + age is compared with and without each sharpness measure as a covariate:
if sharpness explains the offset, the IOP coefficient should shrink towards 0.

Usage:
    python scripts/10_image_quality_by_site.py --project-dir D:/ML_Project
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.ndimage import binary_erosion, laplace, sobel

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

SITES = ["Guys", "HH", "IOP"]
METRICS = ["grad_ratio", "lap_ratio", "fwhm_mm"]


def parse_args():
    p = argparse.ArgumentParser(description="Image sharpness and resolution by site.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--raw-dir", type=Path, default=None, help="Defaults to <project-dir>/IXI_data")
    p.add_argument("--bag-csv", type=Path, default=None,
                   help="Defaults to <project-dir>/results/phase9a_site_probe/heldout_p_iop.csv")
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def native_header(raw_dir, ixi_id):
    hits = sorted(raw_dir.glob(f"IXI{ixi_id:03d}-*-T1.nii*"))
    if not hits:
        return {}
    path = hits[0]
    if path.is_dir():  # some entries are folders holding the scan
        inner = sorted(path.glob("*.nii*"))
        if not inner:
            return {}
        path = inner[0]
    h = nib.load(str(path)).header
    zooms = [float(z) for z in h.get_zooms()[:3]]
    return {"vox_x": zooms[0], "vox_y": zooms[1], "vox_z": zooms[2],
            "voxel_volume_mm3": float(np.prod(zooms)), "matrix": "x".join(str(int(s)) for s in h.get_data_shape()[:3])}


def sharpness(v):
    brain = v > 0
    inner = binary_erosion(brain, iterations=3)  # keep away from the skull-strip edge
    vals = v[inner]
    g = np.sqrt(sum(sobel(v, axis=a) ** 2 for a in range(3)))
    out = {"grad_ratio": float(g[inner].mean() / vals.mean()),
           "lap_ratio": float(laplace(v)[inner].std() / vals.std())}
    fw = []
    for a in range(3):  # Forman et al. 1995: FWHM from the variance of first differences (1 mm voxels)
        d = np.diff(v, axis=a)
        both = inner[tuple(slice(0, -1) if i == a else slice(None) for i in range(3))] & \
            inner[tuple(slice(1, None) if i == a else slice(None) for i in range(3))]
        ratio = np.var(d[both]) / (2 * np.var(vals))
        fw.append(np.sqrt(-2 * np.log(2) / np.log(1 - ratio)) if 0 < ratio < 1 else np.nan)
    out["fwhm_mm"] = float(np.nanmean(fw))
    return out


def main():
    args = parse_args()
    project_dir = args.project_dir
    volume_dir = args.volume_dir or project_dir / ft.DEFAULT_VOLUME_SUBDIR
    raw_dir = args.raw_dir or project_dir / "IXI_data"
    bag_csv = args.bag_csv or project_dir / "results" / "phase9a_site_probe" / "heldout_p_iop.csv"
    out_dir = args.out_dir or project_dir / "results" / "phase9c_image_quality"
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {s: ft.load_split(project_dir, s) for s in ["train", "val", "test"]}
    ft.check_labels_against_metadata(project_dir, splits)
    df = pd.concat([d.assign(split=s) for s, d in splits.items()], ignore_index=True)
    df = df[~df.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True)

    rows, start = [], time.time()
    for k, r in df.iterrows():
        v = np.load(ft.volume_path(volume_dir, r.IXI_ID)).astype(np.float32)
        v = v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)
        rows.append({"IXI_ID": r.IXI_ID, **native_header(raw_dir, r.IXI_ID), **sharpness(v)})
        if (k + 1) % 50 == 0 or k + 1 == len(df):
            rate = (time.time() - start) / (k + 1)
            print(f"  {k + 1}/{len(df)} | {rate:.1f}s/subject | ETA {rate * (len(df) - k - 1) / 60:.0f} min",
                  flush=True)
    q = df[["IXI_ID", "split", "site", "sex_label", "AGE"]].merge(pd.DataFrame(rows), on="IXI_ID")
    q.to_csv(out_dir / "image_quality.csv", index=False)

    print("\nNative voxel size (mm) by site:")
    print(q.groupby("site")[["vox_x", "vox_y", "vox_z", "voxel_volume_mm3"]].agg(["median", "min", "max"])
          .round(3).to_string())
    print("\nMost common matrix by site:", q.groupby("site").matrix.agg(lambda s: s.mode().iat[0]).to_dict())

    site_rows = []
    print("\nSharpness on the model input, site differences adjusted for age and sex (HC3):")
    for m in METRICS + ["voxel_volume_mm3"]:
        fit = smf.ols(f"{m} ~ C(site, Treatment('Guys')) + AGE + C(sex_label)", data=q).fit(cov_type="HC3")
        means = q.groupby("site")[m].mean()
        for s in ["HH", "IOP"]:
            t = f"C(site, Treatment('Guys'))[T.{s}]"
            lo, hi = fit.conf_int().loc[t]
            sd = q[m].std()
            site_rows.append({"metric": m, "site_vs_Guys": s, "estimate": fit.params[t], "ci_low": lo,
                              "ci_high": hi, "p": fit.pvalues[t], "effect_in_sd": fit.params[t] / sd})
            print(f"  {m:17s} {s:4s} vs Guys {fit.params[t]:+.4f} [{lo:+.4f}, {hi:+.4f}] "
                  f"({fit.params[t] / sd:+.2f} SD) p={fit.pvalues[t]:.3g}")
        print(f"  {'':17s} site means: " + ", ".join(f"{s} {means[s]:.4f}" for s in SITES))
    pd.DataFrame(site_rows).to_csv(out_dir / "site_differences.csv", index=False)

    # Does sharpness account for the IOP brain-age gap? (held-out subjects)
    held = pd.read_csv(bag_csv)[["IXI_ID", "bag_corrected"]].merge(q, on="IXI_ID")
    for m in METRICS:
        held[f"{m}_z"] = (held[m] - held[m].mean()) / held[m].std()
    base = smf.ols("bag_corrected ~ C(site, Treatment('Guys')) + AGE", data=held).fit(cov_type="HC3")
    t_iop = "C(site, Treatment('Guys'))[T.IOP]"
    med = [{"covariate": "none", "iop_effect": base.params[t_iop],
            "iop_ci": list(base.conf_int().loc[t_iop]), "covariate_slope_per_sd": None, "covariate_p": None}]
    print(f"\nHeld-out n={len(held)}: corrected BAG, IOP vs Guys (age-adjusted): {base.params[t_iop]:+.2f} yrs "
          f"[{base.conf_int().loc[t_iop][0]:+.2f}, {base.conf_int().loc[t_iop][1]:+.2f}]")
    for m in METRICS:
        f = smf.ols(f"bag_corrected ~ C(site, Treatment('Guys')) + AGE + {m}_z", data=held).fit(cov_type="HC3")
        med.append({"covariate": m, "iop_effect": f.params[t_iop], "iop_ci": list(f.conf_int().loc[t_iop]),
                    "covariate_slope_per_sd": f.params[f"{m}_z"], "covariate_p": f.pvalues[f"{m}_z"]})
        print(f"  + {m:11s}: IOP effect {f.params[t_iop]:+.2f} [{f.conf_int().loc[t_iop][0]:+.2f}, "
              f"{f.conf_int().loc[t_iop][1]:+.2f}] | BAG per 1 SD of {m}: {f.params[f'{m}_z']:+.2f} "
              f"(p={f.pvalues[f'{m}_z']:.3f})")
    (out_dir / "summary.json").write_text(json.dumps(
        {"n": len(q), "n_heldout": len(held), "site_differences": site_rows, "bag_adjustment": med},
        indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (m, label) in zip(axes, [("voxel_volume_mm3", "Native voxel volume (mm³)"),
                                     ("grad_ratio", "Gradient sharpness (higher = sharper)"),
                                     ("lap_ratio", "Laplacian sharpness (higher = sharper)"),
                                     ("fwhm_mm", "Smoothness FWHM, mm (higher = blurrier)")]):
        for i, s in enumerate(SITES):
            g = q[q.site == s]
            ax.scatter(g.AGE, g[m], s=8, alpha=0.6, label=s, color=f"C{i}")
        ax.set(xlabel="Age (years)", title=label)
    axes[0].legend()
    plt.tight_layout()
    fig.savefig(out_dir / "image_quality_by_site.png", dpi=130)
    print(f"\nSaved to {out_dir} ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
