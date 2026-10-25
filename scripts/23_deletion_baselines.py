"""Phase 10g - Deletion test with a second baseline, signed changes, and Integrated Gradients agreement.

Phase 9g removed attributed regions by replacing them with a heavily smoothed copy (sigma 8 voxels).
Blur itself makes predictions younger (Phase 9b), so the result could reflect the baseline. Here, on
the same 60 held-out subjects and Phase 7 maps:
    baselines  "smooth" (Phase 9g) and "mean" (cell voxels inside the brain set to the brain's mean
               intensity: no blur, no anatomy)
    removal    top-, bottom- and randomly-ranked 5% and 10% of brain cells
    outcome    signed and absolute change in predicted age
Then Integrated Gradients (zero baseline, 16 steps) is computed for 20 of these subjects, summed into
the same 8 mm cells, and compared with the gradient x activation maps (Spearman, brain cells).

Usage: python scripts/23_deletion_baselines.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)
CELL, FRACTIONS, N_IG, IG_STEPS = 8, [0.05, 0.10], 20, 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    p.add_argument("--sfcn-repo", type=Path, default=Path("D:/sfcn_repo"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official/best.pt"))
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    P = args.project_dir
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(1)
    vol_dir = P / ft.DEFAULT_VOLUME_SUBDIR
    maps = np.load(P / "results" / "phase7" / "gradcam_conv_2_lowres.npz")
    held = pd.concat([ft.load_split(P, s) for s in ["val", "test"]], ignore_index=True)
    held = held[~held.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].sample(n=60, random_state=0).reset_index(drop=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    _, centers = ft.make_bins((18, 90), 1)
    ft.ensure_sfcn_repo(args.sfcn_repo)
    model = ft.build_model(args.sfcn_repo, len(centers), torch.device("cpu"))
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    ct = torch.tensor(centers, dtype=torch.float32)
    out_dir = P / "results" / "phase10g_deletion_baselines"; out_dir.mkdir(parents=True, exist_ok=True)

    def predict(vols):
        with torch.no_grad():
            x = torch.from_numpy(np.stack(vols)[:, None])
            return ft.predict_ages(model(x)[0].reshape(len(vols), -1), ct).tolist()

    def norm(v):
        return (v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)).astype(np.float32)

    rows, ig_rows = [], []
    for n, r in held.iterrows():
        v = norm(np.load(ft.volume_path(vol_dir, r.IXI_ID)).astype(np.float32))
        brain = v > 0
        fills = {"smooth": gaussian_filter(v, 8) * brain, "mean": np.where(brain, v[brain].mean(), 0).astype(np.float32)}
        frac_in = brain.reshape(20, CELL, 24, CELL, 20, CELL).mean(axis=(1, 3, 5))
        cells = np.argwhere(frac_in >= 0.5)
        cam = maps[f"IXI{int(r.IXI_ID):03d}"]
        order = cells[np.argsort(-cam[tuple(cells.T)])]
        conds = {}
        for f in FRACTIONS:
            k = max(1, int(round(f * len(cells))))
            sel = {"top": order[:k], "bottom": order[-k:], "random": cells[rng.choice(len(cells), k, replace=False)]}
            for base, fill in fills.items():
                for kind, cs in sel.items():
                    out = v.copy()
                    for i, j, l in cs:
                        s = np.s_[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL, l * CELL:(l + 1) * CELL]
                        out[s] = np.where(brain[s], fill[s], 0)
                    conds[(base, kind, f)] = norm(out)
        keys = list(conds)
        preds = predict([v])
        for i in range(0, len(keys), 4):
            preds += predict([conds[k] for k in keys[i:i + 4]])
        for key, pr in zip(keys, preds[1:]):
            rows.append({"IXI_ID": r.IXI_ID, "site": r.site, "baseline": key[0], "kind": key[1], "fraction": key[2],
                         "delta": pr - preds[0]})

        if n < N_IG:
            x = torch.from_numpy(v[None, None])
            total = torch.zeros_like(x)
            for a in (np.arange(IG_STEPS) + 0.5) / IG_STEPS:
                xi = (a * x).requires_grad_(True)
                age = (torch.exp(model(xi)[0].reshape(1, -1)) @ ct).sum()
                total += torch.autograd.grad(age, xi)[0]
            ig = (x * total / IG_STEPS)[0, 0].abs().numpy()
            ig_cells = ig.reshape(20, CELL, 24, CELL, 20, CELL).sum(axis=(1, 3, 5))
            rho = spearmanr(ig_cells[tuple(cells.T)], cam[tuple(cells.T)]).correlation
            top_ig = set(map(tuple, cells[np.argsort(-ig_cells[tuple(cells.T)])][:max(1, len(cells) // 10)]))
            top_cam = set(map(tuple, order[:max(1, len(cells) // 10)]))
            ig_rows.append({"IXI_ID": r.IXI_ID, "spearman": float(rho), "top10_overlap": len(top_ig & top_cam) / len(top_cam)})
        print(f"{n + 1}/60", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "deletion_deltas.csv", index=False)
        pd.DataFrame(ig_rows).to_csv(out_dir / "ig_agreement.csv", index=False)

    d = pd.DataFrame(rows)
    d["abs"] = d.delta.abs()
    summ = d.groupby(["baseline", "fraction", "kind"]).agg(mean_signed=("delta", "mean"), mean_abs=("abs", "mean")).round(3)
    tests = {}
    for (base, f), g in d.groupby(["baseline", "fraction"]):
        w = g.pivot_table(index="IXI_ID", columns="kind", values="abs")
        tests[f"{base} {f:.2f}"] = {"top>random_share": float((w.top > w.random).mean()),
                                    "p_top_vs_random": float(wilcoxon(w.top, w.random).pvalue),
                                    "p_random_vs_bottom": float(wilcoxon(w.random, w.bottom).pvalue)}
    ig = pd.DataFrame(ig_rows)
    out = {"summary": summ.reset_index().to_dict("records"), "tests": tests,
           "ig_vs_gradxact": {"n": len(ig), "median_spearman": float(ig.spearman.median()),
                              "mean_top10_overlap": float(ig.top10_overlap.mean())}}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(summ.to_string()); print(json.dumps(tests, indent=1)); print(out["ig_vs_gradxact"])


if __name__ == "__main__":
    main()
