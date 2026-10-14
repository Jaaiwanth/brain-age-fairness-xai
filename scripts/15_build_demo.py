"""Build the project website (demo/index.html): one self-contained, offline HTML file.

It tells the project's story chapter by chapter with interactive charts, then offers an
explorer of individual held-out brains. Every number is read from saved result files; no
model is run. Sources:
    phase3, phase5, phase6, phase6b, phase7, phase9a-9g result folders under <project-dir>/results
Explorer: a fixed sample of held-out subjects (val + test; 12 per site) with three slices of the
model input, the Phase 7 attribution overlay, prediction, corrected brain-age gap, the model's own
uncertainty, the change in predicted age under each site's intensity appearance, and P(IOP).

Usage:
    python scripts/15_build_demo.py --project-dir D:/ML_Project
"""

import argparse
import base64
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import zoom

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

SITES = ["Guys", "HH", "IOP"]
PER_SITE = 12
SLICES = {"sagittal": lambda v: v[80], "coronal": lambda v: v[:, 100], "axial": lambda v: v[:, :, 72]}


def parse_args():
    p = argparse.ArgumentParser(description="Build the project website.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=HERE.parent / "demo" / "index.html")
    return p.parse_args()


def hot(x):
    r = np.clip(x / 0.4, 0, 1)
    g = np.clip((x - 0.4) / 0.4, 0, 1)
    b = np.clip((x - 0.8) / 0.2, 0, 1)
    a = np.clip(x * 1.4, 0, 0.85)
    return (np.stack([r, g, b, a], -1) * 255).astype(np.uint8)


def png(arr, mode):
    buf = io.BytesIO()
    Image.fromarray(np.rot90(arr).copy(), mode).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def r2(x, d=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def collect_stats(R, bc):
    s = {}
    z = json.loads((R / "phase3" / "summary.json").read_text())
    p5 = json.loads((R / "phase5" / "summary.json").read_text())
    s["accuracy"] = {
        "zero_shot_mae": r2(z["val_raw"]["mae"]), "zero_shot_r": r2(z["val_raw"]["r"], 3),
        "zero_shot_corrected_mae": r2(z["val_corrected"]["mae"]),
        "finetuned_val_mae": r2(z["val_finetuned"]["mae"]), "finetuned_val_r": r2(z["val_finetuned"]["r"], 3),
        "test_mae": r2(p5["raw"]["metrics"]["mae"]), "test_mae_ci": [r2(v) for v in p5["raw"]["ci95"]["mae"]],
        "test_r": r2(p5["raw"]["metrics"]["r"], 3), "n_test": p5["n_test"]}

    e = pd.read_csv(R / "phase6" / "age_adjusted_effects.csv")
    s["fairness"] = [{"outcome": r.outcome, "term": r.term, "est": r2(r.estimate), "lo": r2(r.boot_ci_low),
                      "hi": r2(r.boot_ci_high), "p": r2(r.hc3_p, 3)}
                     for r in e.itertuples() if r.term != "Age (per year)"]

    lo = pd.read_csv(R / "phase6b" / "loso_effects.csv")
    s["loso"] = [{"site": r.heldout_site, "n": int(r.n_heldout), "mae_held": r2(r.heldout_mae),
                  "mae_ref": r2(r.reference_mae),
                  "err": [r2(r.abs_error_effect), r2(r.abs_error_ci_low), r2(r.abs_error_ci_high)],
                  "bag": [r2(r.bag_corrected_effect), r2(r.bag_corrected_ci_low), r2(r.bag_corrected_ci_high)]}
                 for r in lo.itertuples()]

    pr = json.loads((R / "phase9a_site_probe" / "summary.json").read_text())["probe"]
    names = {"demographics": "Age + sex only", "intensity": "Image intensities",
             "sfcn_pretrained": "Pretrained SFCN", "sfcn_finetuned": "Fine-tuned SFCN",
             "sfcn_pretrained_resid": "Pretrained, age/sex removed",
             "sfcn_finetuned_resid": "Fine-tuned, age/sex removed"}
    s["probe"] = [{"label": names[k], "key": k, "acc": r2(100 * v["balanced_accuracy"], 1),
                   "lo": r2(100 * v["ci_low"], 1), "hi": r2(100 * v["ci_high"], 1)} for k, v in pr.items()]
    s["probe_confusion"] = pr["sfcn_finetuned_resid"]["confusion"]

    sw = pd.read_csv(R / "phase9b_stress_test" / "site_swap_effects.csv")
    M = sw.pivot(index="source_site", columns="target_site", values="mean_delta").loc[SITES, SITES]
    rel = M.sub(np.diag(M.values), axis=0)
    s["swap"] = {"sites": SITES, "matrix": [[r2(v) for v in row] for row in rel.values],
                 "n": {r.source_site: int(r.n) for r in sw.itertuples() if r.own_site_control}}
    sweep = pd.read_csv(R / "phase9b_stress_test" / "sweep_effects.csv")
    s["sweep"] = [{"label": r.condition, "est": r2(r.mean_delta), "lo": r2(r.ci_low), "hi": r2(r.ci_high)}
                  for r in sweep.itertuples()]
    un = json.loads((R / "phase9b_stress_test_unadjusted" / "summary.json").read_text())
    ad = json.loads((R / "phase9b_stress_test" / "summary.json").read_text())
    s["swap_iop"] = {k: {"adjusted": r2(ad["headline_vs_own_site_control"][k]["mean_delta"]),
                         "unadjusted": r2(un["headline_vs_own_site_control"][k]["mean_delta"])}
                     for k in ad["headline_vs_own_site_control"]}

    sd = pd.read_csv(R / "phase9c_image_quality" / "site_differences.csv")
    s["sharpness"] = [{"metric": r.metric, "site": r.site_vs_Guys, "sd": r2(r.effect_in_sd)} for r in sd.itertuples()
                      if r.metric != "voxel_volume_mm3"]
    bl = json.loads((R / "phase9d_blur_to_iop" / "summary.json").read_text())
    s["blur"] = {g: {"est": r2(bl["groups"][g]["matched"]["mean"]), "ci": [r2(v) for v in bl["groups"][g]["matched"]["ci"]]}
                 for g in ["Guys", "HH"]}
    hm = json.loads((R / "phase9e_harmonisation" / "summary.json").read_text())["results"]
    s["harmonise"] = {v: {"probe": r2(100 * hm[v]["site_probe_balanced_accuracy"], 1), "mae": r2(hm[v]["heldout_mae_raw"]),
                          "iop": [r2(hm[v]["effects"]["IOP vs Guys"]["estimate"])] + [r2(c) for c in hm[v]["effects"]["IOP vs Guys"]["ci"]]}
                      for v in ["original", "harmonised"]}

    un9 = json.loads((R / "phase9f_uncertainty" / "summary.json").read_text())
    s["uncertainty"] = {"main": [{"site": r["site"], "sd": r2(r["mean_sd"]), "cov": r2(100 * r["coverage90"], 0),
                                  "mae": r2(r["mae"])} for r in un9["main_model_heldout"]["by_site"]],
                        "loso": {site: [{"group": r["group"], "sd": r2(r["mean_sd"]), "cov": r2(100 * r["coverage90"], 0),
                                         "mae": r2(r["mae"])} for r in v["by_group"]] for site, v in un9["loso"].items()},
                        "rho": r2(un9["main_model_heldout"]["spearman_sd_vs_abs_error"][0])}

    dl = R / "phase9g_deletion_test" / "deletion_summary.csv"
    if dl.exists():
        d = pd.read_csv(dl)
        s["deletion"] = [{"fraction": r2(100 * r.fraction, 0), "top": r2(r.top_mean), "random": r2(r.random_mean),
                          "bottom": r2(r.bottom_mean)} for r in d.itertuples()]
    co = pd.read_csv(R / "phase7" / "roi_attribution_coarse.csv")
    regions = [c for c in co.columns if c not in ("IXI_ID", "split", "site", "sex", "age", "predicted_age")]
    held = co[co.split != "train"]
    s["attribution"] = {"regions": regions,
                        "share": {site: [r2(100 * held[held.site == site][reg].mean(), 1) for reg in regions]
                                  for site in SITES}}
    s["sanity_rho"] = 0.13
    return s


def main():
    args = parse_args()
    P = args.project_dir
    volume_dir = args.volume_dir or P / ft.DEFAULT_VOLUME_SUBDIR
    R = P / "results"

    unc = pd.read_csv(R / "phase9f_uncertainty" / "main_model_uncertainty.csv")
    stress = pd.read_csv(R / "phase9b_stress_test" / "stress_predictions.csv")
    probe = pd.read_csv(R / "phase9a_site_probe" / "heldout_p_iop.csv")[["IXI_ID", "p_iop"]]
    bc = json.loads((R / "phase5" / "summary.json").read_text())["bias_correction"]
    maps = np.load(R / "phase7" / "gradcam_conv_2_lowres.npz")

    unc["bag"] = (unc.pred - bc["intercept"]) / bc["slope"] - unc.AGE
    points = [{"id": f"IXI{int(r.IXI_ID):03d}", "site": r.site, "sex": r.sex_label, "split": r.split,
               "age": r2(r.AGE, 1), "pred": r2(r.pred, 1), "bag": r2(r.bag, 1), "sd": r2(r.sd, 1)}
              for r in unc.itertuples()]

    match = stress[stress.condition.str.startswith("match ")].pivot(index="IXI_ID", columns="condition",
                                                                    values="predicted_age")
    df = unc.merge(probe, on="IXI_ID").merge(match, left_on="IXI_ID", right_index=True)
    df = df.groupby("site").sample(n=PER_SITE, random_state=0).sort_values(["site", "AGE"]).reset_index(drop=True)

    subjects = []
    for _, r in df.iterrows():
        v = np.load(ft.volume_path(volume_dir, r.IXI_ID)).astype(np.float32)
        brain = v > 0
        hi = np.percentile(v[brain], 99.5)
        gray = (np.clip(v / hi, 0, 1) * 255).astype(np.uint8)
        cam = zoom(maps[f"IXI{int(r.IXI_ID):03d}"], 8, order=1) * brain
        cam = np.clip(cam / max(np.percentile(cam[brain], 99.5), 1e-12), 0, 1)
        views = {k: {"img": png(f(gray), "L"), "cam": png(hot(f(cam)), "RGBA")} for k, f in SLICES.items()}
        own = r[f"match {r.site}"]
        subjects.append({
            "id": f"IXI{int(r.IXI_ID):03d}", "site": r.site, "sex": r.sex_label, "split": r.split,
            "age": r2(r.AGE, 1), "pred": r2(r.pred, 1), "bag": r2(r.bag, 1), "sd": r2(r.sd, 1),
            "lo": r2(r.lo90, 1), "hi": r2(r.hi90, 1), "p_iop": r2(r.p_iop, 3),
            "look": {s: r2(r[f"match {s}"] - own, 1) for s in SITES}, "views": views,
        })
        print(f"  {subjects[-1]['id']} {r.site}", flush=True)

    data = {"stats": collect_stats(R, bc), "points": points, "subjects": subjects}
    html = (HERE.parent / "demo" / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(subjects)} explorer subjects, "
          f"{len(points)} chart points)")


if __name__ == "__main__":
    main()
