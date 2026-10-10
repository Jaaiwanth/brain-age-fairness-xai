"""Phase 7 - Grad-CAM attribution maps and atlas-based comparison across sex and site.

For every subject in the train/val/test splits, using the frozen Phase 4 model:
    1. Attribution map for the predicted age (expected value over the age bins):
       location-wise gradient x activation summed over channels (HiResCAM-style), as a
       magnitude, at feature-extractor block conv_2 (20x24x20 maps, 8 mm cells).
       Upsampled to 160x192x160, restricted to the subject's brain mask and normalised to
       sum to 1, so each map shows where the predicted age is most sensitive,
       whichever direction a location pushes it.
       (Standard Grad-CAM - channel-averaged gradients + ReLU - produced empty maps for
       20-80% of subjects at this layer in trial runs, so it was not used.)
    2. Atlas scoring: share of each subject's attribution falling in each region of the
       Harvard-Oxford atlases (TemplateFlow MNI152NLin6Asym, same grid as the volumes).
       Coarse level (7 regions, primary) and fine level (59 regions, secondary).
    3. Group comparison on subjects the model never trained on (val + test):
       share ~ sex + site + age (HC3), Benjamini-Hochberg FDR across regions per term.
    4. Sanity check (Adebayo et al. 2018): maps from a model whose top layers are
       re-initialised should look different from the trained model's maps.

Wording (Rule 3): attribution patterns "differ between" or "are associated with" groups;
never "the model thinks differently".

Usage:
    python scripts/07_gradcam.py --project-dir D:/ML_Project \
        --checkpoint "G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official/best.pt"
"""

import argparse
import copy
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

TEMPLATE = "MNI152NLin6Asym"
TARGET_LAYER = "conv_2"
COARSE = {
    "Cerebral cortex": None,               # all Harvard-Oxford cortical labels
    "Cerebral white matter": ["Cerebral White Matter"],
    "Lateral ventricles": ["Lateral Ventrical", "Lateral Ventricle"],
    "Deep grey matter": ["Thalamus", "Caudate", "Putamen", "Pallidum", "Accumbens"],
    "Hippocampus and amygdala": ["Hippocampus", "Amygdala"],
    "Brainstem": ["Brainstem"],
    "Cerebellum and unlabelled": [],        # in-brain voxels with no Harvard-Oxford label
}


def parse_args():
    p = argparse.ArgumentParser(description="Phase 7: Grad-CAM + atlas comparison.")
    p.add_argument("--project-dir", type=Path,
                   default=Path(os.environ.get("BRAINAGE_PROJECT_DIR", ft.DEFAULT_PROJECT_DIR)))
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Only the first N subjects (quick check).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Atlas
# ---------------------------------------------------------------------------

def center_crop(v, shape=(160, 192, 160)):
    for ax, n in enumerate(shape):
        s = (v.shape[ax] - n) // 2
        v = np.take(v, range(s, s + n), axis=ax)
    return v


def build_atlas():
    """Return (fine label volume, fine names, fine->coarse index map, coarse names)."""
    import ants
    from templateflow import api as tflow

    cort = ants.image_read(str(tflow.get(TEMPLATE, resolution=1, atlas="HOCPA", desc="th25",
                                         suffix="dseg", extension=".nii.gz"))).numpy().astype(int)
    sub = ants.image_read(str(tflow.get(TEMPLATE, resolution=1, atlas="HOSPA", desc="th25",
                                        suffix="dseg", extension=".nii.gz"))).numpy().astype(int)
    cort_names = pd.read_csv(tflow.get(TEMPLATE, atlas="HOCPA", suffix="dseg", extension=".tsv"),
                             sep="\t").set_index("index")["name"]
    sub_names = pd.read_csv(tflow.get(TEMPLATE, atlas="HOSPA", suffix="dseg", extension=".tsv"),
                            sep="\t").set_index("index")["name"].str.strip()

    names = ["Cerebellum and unlabelled"]          # fine label 0 = in-brain, no HO label
    fine = np.zeros(cort.shape, dtype=np.int16)
    # Subcortical structures, left and right merged; the HO "Cerebral Cortex" class is
    # replaced by the detailed cortical atlas below.
    merged = {}
    for idx, name in sub_names.items():
        if idx == 0 or "Cerebral Cortex" in name:
            continue
        base = name.replace("Left ", "").replace("Right ", "").replace("Ventrical", "Ventricle")
        if base not in merged:
            merged[base] = len(names)
            names.append(base)
        fine[sub == idx] = merged[base]
    for idx, name in cort_names.items():
        names.append(f"Cortex: {name}")
        fine[(cort == idx) & np.isin(fine, [0])] = len(names) - 1
    fine = center_crop(fine)

    coarse_names = list(COARSE)
    to_coarse = np.zeros(len(names), dtype=int)
    for i, n in enumerate(names):
        if n.startswith("Cortex: "):
            to_coarse[i] = coarse_names.index("Cerebral cortex")
        elif i == 0:
            to_coarse[i] = coarse_names.index("Cerebellum and unlabelled")
        else:
            to_coarse[i] = next(coarse_names.index(c) for c, parts in COARSE.items()
                                if parts and any(p.replace("Ventrical", "Ventricle") in n for p in parts))
    return fine, names, to_coarse, coarse_names


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, layer_name, bin_centers):
        self.model, self.centers, self.acts = model, bin_centers, {}
        getattr(model.feature_extractor, layer_name).register_forward_hook(
            lambda m, i, o: self.acts.__setitem__("a", o))

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        log_probs = self.model(x)[0].reshape(1, -1)
        age = (torch.exp(log_probs) @ self.centers).sum()
        a = self.acts["a"]
        g = torch.autograd.grad(age, a)[0]
        # Location-wise gradient x activation, summed over channels (HiResCAM-style),
        # magnitude only. Standard Grad-CAM (channel-averaged gradients + ReLU) left
        # 20-80% of maps empty at this layer in trial runs, so it was not usable here.
        cam = (g * a).sum(1, keepdim=True).abs()
        return cam.detach(), float(age.detach())


def upsample(cam_lowres, shape):
    return F.interpolate(cam_lowres, size=shape, mode="trilinear", align_corners=False)[0, 0].numpy()


def load_input(volume_dir, ixi_id):
    v = np.load(ft.volume_path(volume_dir, ixi_id)).astype(np.float32)
    x = v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)
    return v, torch.tensor(x[None, None])


def randomised_copy(model):
    """Cascading randomisation of the layers above the Grad-CAM target."""
    m = copy.deepcopy(model)
    getattr(m.feature_extractor, TARGET_LAYER)._forward_hooks.clear()   # drop the copied hook
    torch.manual_seed(0)
    for name in ["conv_3", "conv_4", "conv_5"]:
        for mod in getattr(m.feature_extractor, name).modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
    m.classifier.conv_6.reset_parameters()
    return m.eval()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def compare_groups(df, region_cols, level):
    import statsmodels.formula.api as smf
    rows = []
    for col in region_cols:
        d = df.assign(y=df[col] * 100)                 # percentage points of attribution
        m = smf.ols("y ~ C(sex, Treatment('Female')) + C(site, Treatment('Guys')) + age",
                    data=d).fit(cov_type="HC3")
        ci = m.conf_int()
        for term, label in [("C(sex, Treatment('Female'))[T.Male]", "Male vs Female"),
                            ("C(site, Treatment('Guys'))[T.HH]", "HH vs Guys"),
                            ("C(site, Treatment('Guys'))[T.IOP]", "IOP vs Guys"),
                            ("age", "Age (per year)")]:
            rows.append({"level": level, "region": col, "term": label,
                         "mean_share_pct": d.y.mean(), "estimate_pct_points": m.params[term],
                         "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1], "p": m.pvalues[term]})
    res = pd.DataFrame(rows)
    res["q_fdr"] = res.groupby("term")["p"].transform(bh_fdr)
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    project_dir = args.project_dir
    checkpoint = args.checkpoint or project_dir / "checkpoints" / "sfcn_finetune_official" / "best.pt"
    volume_dir = args.volume_dir or project_dir / ft.DEFAULT_VOLUME_SUBDIR
    out_dir = args.out_dir or project_dir / "results" / "phase7"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    subjects = pd.concat([ft.load_split(project_dir, s).assign(split=s) for s in ["train", "val", "test"]],
                         ignore_index=True)
    ft.check_labels_against_metadata(project_dir, {"all": subjects})
    subjects = subjects[~subjects.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True)
    if args.limit:
        subjects = subjects.groupby("split").head(args.limit).reset_index(drop=True)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, centers = ft.make_bins(ckpt["config"]["bin_range"], ckpt["config"]["bin_step"])
    centers_t = torch.tensor(centers, dtype=torch.float32)
    ft.ensure_sfcn_repo(args.sfcn_repo)
    model = ft.build_model(args.sfcn_repo, len(centers), torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    cam_fn = GradCAM(model, TARGET_LAYER, centers_t)

    fine, fine_names, to_coarse, coarse_names = build_atlas()
    print(f"Subjects: {len(subjects)} ({subjects.split.value_counts().to_dict()}) | layer {TARGET_LAYER} | "
          f"{len(fine_names)} fine / {len(coarse_names)} coarse regions")

    shape = ft.CONFIG["input_shape"]
    group_sums = {}
    fine_rows, lowres = [], {}
    start = time.time()
    for k, r in subjects.iterrows():
        vol, x = load_input(volume_dir, int(r.IXI_ID))
        cam_low, pred = cam_fn(x)
        cam = upsample(cam_low, shape)
        brain = vol > 0
        cam = np.where(brain, cam, 0)
        total = cam.sum()
        if total <= 0:
            print(f"  IXI{int(r.IXI_ID):03d}: empty map (all gradients negative) - recorded as missing")
            continue
        cam /= total
        shares = np.bincount(fine[brain], weights=cam[brain], minlength=len(fine_names))
        fine_rows.append({"IXI_ID": int(r.IXI_ID), "split": r.split, "site": r.site, "sex": r.sex_label,
                          "age": r.AGE, "predicted_age": pred, **dict(zip(fine_names, shares))})
        lowres[f"IXI{int(r.IXI_ID):03d}"] = cam_low[0, 0].numpy().astype(np.float32)
        if r.split in ("val", "test"):
            for key in ["all", f"site={r.site}", f"sex={r.sex_label}"]:
                group_sums[key] = group_sums.get(key, 0) + cam
                group_sums[key + "#n"] = group_sums.get(key + "#n", 0) + 1
        if (k + 1) % 25 == 0 or k + 1 == len(subjects):
            el = time.time() - start
            print(f"  {k + 1}/{len(subjects)} | {el / (k + 1):.1f}s/subject | ETA {el / (k + 1) * (len(subjects) - k - 1) / 60:.0f} min",
                  flush=True)

    fine_df = pd.DataFrame(fine_rows)
    fine_df.to_csv(out_dir / "roi_attribution_fine.csv", index=False)
    coarse_df = fine_df[["IXI_ID", "split", "site", "sex", "age", "predicted_age"]].copy()
    for ci, cname in enumerate(coarse_names):
        cols = [fine_names[i] for i in range(len(fine_names)) if to_coarse[i] == ci]
        coarse_df[cname] = fine_df[cols].sum(axis=1)
    coarse_df.to_csv(out_dir / "roi_attribution_coarse.csv", index=False)
    np.savez_compressed(out_dir / f"gradcam_{TARGET_LAYER}_lowres.npz", **lowres)
    means = {k: v / group_sums[k + "#n"] for k, v in group_sums.items() if not k.endswith("#n")}
    np.savez_compressed(out_dir / "group_mean_maps_heldout.npz", **means)

    # ---- Sanity check: randomised top layers ----
    rand_fn = GradCAM(randomised_copy(model), TARGET_LAYER, centers_t)
    corrs = []
    for i in subjects[subjects.split == "test"].IXI_ID.head(8):
        vol, x = load_input(volume_dir, int(i))
        brain = vol > 0
        a, _ = cam_fn(x)
        b, _ = rand_fn(x)
        a, b = upsample(a, shape)[brain], upsample(b, shape)[brain]
        if a.std() > 0 and b.std() > 0:
            corrs.append(float(pd.Series(a).corr(pd.Series(b), method="spearman")))
    sanity = {"randomised_layers": ["conv_3", "conv_4", "conv_5", "classifier.conv_6"],
              "spearman_trained_vs_randomised": corrs,
              "median": float(np.median(corrs)) if corrs else None}
    print(f"\nSanity check: trained vs randomised-top-layer maps, Spearman r median "
          f"{sanity['median']:.2f} (n={len(corrs)}; low = maps depend on what the model learned)")

    # ---- Group comparison on held-out subjects ----
    held = coarse_df[coarse_df.split.isin(["val", "test"])]
    held_fine = fine_df[fine_df.split.isin(["val", "test"])]
    coarse_res = compare_groups(held, coarse_names, "coarse")
    fine_cols = [c for c in fine_names if held_fine[c].mean() > 0.002]   # ignore regions with <0.2% share
    fine_res = compare_groups(held_fine, fine_cols, "fine")
    pd.concat([coarse_res, fine_res]).to_csv(out_dir / "attribution_group_effects.csv", index=False)

    print(f"\nHeld-out subjects (val+test): n={len(held)}; site {held.site.value_counts().to_dict()}; "
          f"sex {held.sex.value_counts().to_dict()}")
    print("\nCoarse regions - mean share of attribution (held-out):")
    print((held.groupby("site")[coarse_names].mean() * 100).round(1).T.to_string())
    print("\nCoarse-region group effects (percentage points, age-adjusted; FDR across 7 regions):")
    for _, row in coarse_res[coarse_res.term != "Age (per year)"].iterrows():
        flag = "  <- FDR q<0.05" if row.q_fdr < 0.05 else ""
        print(f"  {row.region:26s} {row.term:15s} {row.estimate_pct_points:+6.2f} "
              f"[{row.ci_low:+.2f}, {row.ci_high:+.2f}] q={row.q_fdr:.3f}{flag}")
    sig_fine = fine_res[(fine_res.q_fdr < 0.05) & (fine_res.term != "Age (per year)")]
    print(f"\nFine regions with FDR q<0.05 (excluding age): {len(sig_fine)}")
    if len(sig_fine):
        print(sig_fine[["region", "term", "estimate_pct_points", "ci_low", "ci_high", "q_fdr"]]
              .round(3).to_string(index=False))

    (out_dir / "summary.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "layer": TARGET_LAYER, "n_subjects": len(fine_df),
        "n_heldout": len(held), "sanity": sanity, "coarse_regions": coarse_names}, indent=2))

    # ---- Figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import ants
    from templateflow import api as tflow
    tpl = center_crop(ants.image_read(str(tflow.get(TEMPLATE, resolution=1, suffix="T1w", desc=None,
                                                    extension=".nii.gz"))).numpy())
    keys = [k for k in ["all", "site=Guys", "site=HH", "site=IOP", "sex=Female", "sex=Male"] if k in means]
    fig, axes = plt.subplots(len(keys), 3, figsize=(9, 2.9 * len(keys)))
    cx, cy, cz = 80, 100, 72
    vmax = max(np.percentile(means[k][means[k] > 0], 99.5) for k in keys)
    for row, key in zip(axes, keys):
        m = means[key]
        for ax, (t_sl, m_sl) in zip(row, [(tpl[cx], m[cx]), (tpl[:, cy], m[:, cy]), (tpl[:, :, cz], m[:, :, cz])]):
            ax.imshow(np.rot90(t_sl), cmap="gray")
            ax.imshow(np.rot90(np.ma.masked_less_equal(m_sl, 0)), cmap="hot", alpha=0.6, vmin=0, vmax=vmax)
            ax.axis("off")
        row[0].set_title(f"{key} (n={group_sums[key + '#n']})", loc="left", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_dir / "mean_gradcam_by_group.png", dpi=110)

    fig, ax = plt.subplots(figsize=(9, 4))
    site_means = held.groupby("site")[coarse_names].mean() * 100
    site_means.T.plot.bar(ax=ax)
    ax.set_ylabel("Share of attribution (%)")
    ax.set_title("Grad-CAM attribution by region and site (held-out subjects)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "coarse_share_by_site.png", dpi=130)
    print(f"\nSaved to {out_dir} ({(time.time() - start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
