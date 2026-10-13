"""Phase 9d - Blur sharp scans to IOP's measured smoothness: how much does predicted age move?

Phase 9c: IOP scans are much smoother than Guys/HH at the same native resolution
(smoothness FWHM +0.87 mm vs Guys, age/sex-adjusted). Phase 9b: blurring lowers predicted age.
Here, each held-out Guys and HH brain is blurred (Gaussian, then re-masked and re-normalised)
by the amount that raises ITS OWN smoothness FWHM by its site's measured gap to IOP. The
sigma is found per subject by interpolating FWHM over a grid of sigmas (no model involved).
The frozen Phase 4 model then predicts age on the original, half-dose and matched-blur volumes.

Comparison: the IOP vs Guys corrected-BAG offset (-5.06 yrs on test; -4.43 on val+test). The
change in raw predicted age is divided by the validation-fitted correction slope to put it on
the corrected-BAG scale.

Usage:
    python scripts/11_blur_to_iop.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo \
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ft = _load("ft", "04_finetune_sfcn.py")
iq = _load("iq", "10_image_quality_by_site.py")

SIGMA_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
N_BOOT = 5000


def parse_args():
    p = argparse.ArgumentParser(description="Blur Guys/HH scans to IOP smoothness.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def blur(v, sigma):
    brain = v > 0
    out = (gaussian_filter(v, sigma) * brain) if sigma > 0 else v
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
    out_dir = args.out_dir or project_dir / "results" / "phase9d_blur_to_iop"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    rng = np.random.default_rng(0)

    gaps = pd.read_csv(project_dir / "results" / "phase9c_image_quality" / "site_differences.csv")
    gaps = gaps[gaps.metric == "fwhm_mm"].set_index("site_vs_Guys").estimate
    target_gap = {"Guys": float(gaps["IOP"]), "HH": float(gaps["IOP"] - gaps["HH"])}
    slope = json.loads((project_dir / "results" / "phase5" / "summary.json").read_text())["bias_correction"]["slope"]
    print(f"FWHM gap to IOP: Guys +{target_gap['Guys']:.3f} mm, HH +{target_gap['HH']:.3f} mm | "
          f"correction slope {slope:.3f}")

    held = pd.concat([ft.load_split(project_dir, s).assign(split=s) for s in ["val", "test"]], ignore_index=True)
    held = held[held.site.isin(["Guys", "HH"]) & ~held.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True)
    if args.limit:
        held = held.head(args.limit)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, centers = ft.make_bins(ckpt["config"]["bin_range"], ckpt["config"]["bin_step"])
    ft.ensure_sfcn_repo(args.sfcn_repo)
    model = ft.build_model(args.sfcn_repo, len(centers), torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    centers_t = torch.tensor(centers, dtype=torch.float32)
    print(f"Subjects: {len(held)} {held.site.value_counts().to_dict()}")

    rows, start = [], time.time()
    for k, r in held.iterrows():
        v = np.load(ft.volume_path(volume_dir, r.IXI_ID)).astype(np.float32)
        v = blur(v, 0)
        fw = np.array([iq.sharpness(blur(v, s))["fwhm_mm"] for s in SIGMA_GRID])
        target = fw[0] + target_gap[r.site]
        sigma = float(np.interp(target, np.maximum.accumulate(fw), SIGMA_GRID))
        vols = {"original": v, "half": blur(v, sigma / 2), "matched": blur(v, sigma)}
        with torch.no_grad():
            x = torch.from_numpy(np.stack(list(vols.values()))[:, None])
            preds = ft.predict_ages(model(x)[0].reshape(len(vols), -1), centers_t).tolist()
        q_matched = iq.sharpness(vols["matched"])
        rows.append({"IXI_ID": r.IXI_ID, "site": r.site, "age": r.AGE, "sigma_matched": sigma,
                     "fwhm_original": fw[0], "fwhm_target": target, "fwhm_matched": q_matched["fwhm_mm"],
                     "grad_ratio_matched": q_matched["grad_ratio"], "lap_ratio_matched": q_matched["lap_ratio"],
                     **{f"pred_{n}": p for n, p in zip(vols, preds)}})
        if (k + 1) % 10 == 0 or k + 1 == len(held):
            rate = (time.time() - start) / (k + 1)
            print(f"  {k + 1}/{len(held)} | {rate:.1f}s/subject | ETA {rate * (len(held) - k - 1) / 60:.0f} min",
                  flush=True)

    df = pd.DataFrame(rows)
    for n in ["half", "matched"]:
        df[f"delta_{n}"] = df[f"pred_{n}"] - df.pred_original
        df[f"delta_{n}_corrected_scale"] = df[f"delta_{n}"] / slope
    df.to_csv(out_dir / "blur_to_iop_predictions.csv", index=False)

    iqdf = pd.read_csv(project_dir / "results" / "phase9c_image_quality" / "image_quality.csv")
    iop_ref = iqdf[iqdf.site == "IOP"][["fwhm_mm", "grad_ratio", "lap_ratio"]].mean()
    summary = {"n": len(df), "correction_slope": slope, "fwhm_gap_to_iop": target_gap,
               "iop_mean_sharpness": iop_ref.to_dict(), "groups": {}}
    print(f"\nMatched blur: median sigma {df.sigma_matched.median():.2f} voxels | FWHM {df.fwhm_original.mean():.2f}"
          f" -> {df.fwhm_matched.mean():.2f} mm (IOP mean {iop_ref.fwhm_mm:.2f})")
    print(f"  after matching: gradient sharpness {df.grad_ratio_matched.mean():.2f} (IOP {iop_ref.grad_ratio:.2f}), "
          f"Laplacian {df.lap_ratio_matched.mean():.3f} (IOP {iop_ref.lap_ratio:.3f})")
    for name, g in [("Guys+HH", df), ("Guys", df[df.site == "Guys"]), ("HH", df[df.site == "HH"])]:
        out = {"n": len(g)}
        for n in ["half", "matched"]:
            d = g[f"delta_{n}_corrected_scale"]
            out[n] = {"mean": float(d.mean()), "ci": boot_ci(d, rng), "share_younger": float((d < 0).mean())}
        summary["groups"][name] = out
        m = out["matched"]
        print(f"  {name:8s} n={len(g):3d} | half-dose {out['half']['mean']:+.2f} | matched-to-IOP blur "
              f"{m['mean']:+.2f} yrs [{m['ci'][0]:+.2f}, {m['ci'][1]:+.2f}] (corrected-BAG scale) | "
              f"{100 * m['share_younger']:.0f}% younger")
    off = {"test": -5.06, "heldout_val_test": -4.43}
    share = summary["groups"]["Guys"]["matched"]["mean"] / off["test"]
    summary["iop_offset_reference"] = off
    summary["share_of_test_offset_reproduced_by_blur_guys"] = share
    print(f"\nIOP vs Guys corrected-BAG offset: {off['test']} (test), {off['heldout_val_test']} (val+test)."
          f" Blurring Guys scans to IOP smoothness reproduces {100 * share:.0f}% of the test offset.")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for i, s in enumerate(["Guys", "HH"]):
        g = df[df.site == s]
        ax.scatter(g.fwhm_original, g.fwhm_matched, s=12, color=f"C{i}", label=s)
    ax.axhline(iop_ref.fwhm_mm, color="C2", ls="--", label="IOP mean")
    ax.set(xlabel="Smoothness FWHM before (mm)", ylabel="after matched blur (mm)", title="Blur matched to IOP")
    ax.legend()
    ax = axes[1]
    labels, means, cis = [], [], []
    for name in ["Guys", "HH"]:
        for n, lab in [("half", "half blur"), ("matched", "IOP-level blur")]:
            m = summary["groups"][name][n]
            labels.append(f"{name}\n{lab}")
            means.append(m["mean"])
            cis.append(m["ci"])
    x = np.arange(len(labels))
    means, cis = np.array(means), np.array(cis)
    ax.bar(x, means, color=["C0", "C0", "C1", "C1"], alpha=0.8)
    ax.errorbar(x, means, yerr=[means - cis[:, 0], cis[:, 1] - means], fmt="none", ecolor="k", capsize=4)
    ax.axhline(off["test"], color="C2", ls="--", label="IOP vs Guys offset (test)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x, labels)
    ax.set(ylabel="Change in brain-age gap (yrs)", title="Same brains, only blur added")
    ax.legend(loc="lower left")
    plt.tight_layout()
    fig.savefig(out_dir / "blur_to_iop.png", dpi=130)
    print(f"Saved to {out_dir} ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
