"""Phase 3 (redo) - Zero-shot pretrained SFCN on IXI, with the official IXI.xls labels.

The original Phase 3 numbers were computed with the wrong labels and are invalid. This
re-runs the untouched UK Biobank SFCN (40 bins, ages 42-82) on the v2 volumes:

    1. Predicts the train and validation subjects (the test split is not used).
    2. Reports validation MAE / r / mean BAG with 95% bootstrap CIs, overall and for the
       ages the pretrained model covers (42-82).
    3. Fits a linear age-bias correction on the TRAIN subjects (zero-shot never trained on
       them, so they are a clean calibration set) and reports corrected validation metrics.
    4. Puts the fine-tuned Phase 4 checkpoint's validation result alongside for comparison.

Usage:
    python scripts/03b_zero_shot.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo \
        --finetuned "G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official/best.pt"
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

UKB_RANGE = (42, 82)


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 redo: zero-shot SFCN on validation.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--finetuned", type=Path, default=None, help="Phase 4 checkpoint, for comparison")
    p.add_argument("--out-dir", type=Path, default=None, help="Defaults to <project-dir>/results/phase3")
    return p.parse_args()


def pretrained_sfcn(repo_dir):
    from dp_model.model_files.sfcn import SFCN
    model = SFCN()
    state = torch.load(repo_dir / ft.SFCN_WEIGHTS_RELPATH, map_location="cpu")
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
    return model.eval()


@torch.no_grad()
def predict(model, df, volume_dir, centers):
    ds = ft.IXIDataset(df, volume_dir, ft.CONFIG["input_shape"], augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    centers_t = torch.tensor(centers, dtype=torch.float32)
    ids, preds = [], []
    for volumes, _, batch_ids in loader:
        log_probs = model(volumes)[0].reshape(volumes.size(0), -1)
        preds += ft.predict_ages(log_probs, centers_t).tolist()
        ids += batch_ids.tolist()
    return pd.Series(preds, index=ids)


def metrics(age, pred):
    err = pred - age
    return {"n": int(len(age)), "mae": float(np.abs(err).mean()), "r": float(np.corrcoef(age, pred)[0, 1]),
            "mean_bag": float(err.mean())}


def with_ci(age, pred, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    draws = [metrics(age[i], pred[i]) for i in (rng.integers(0, len(age), len(age)) for _ in range(n_boot))]
    out = metrics(age, pred)
    for k in ["mae", "r", "mean_bag"]:
        out[f"{k}_ci95"] = [float(np.percentile([d[k] for d in draws], q)) for q in (2.5, 97.5)]
    return out


def main():
    args = parse_args()
    volume_dir = args.volume_dir or args.project_dir / ft.DEFAULT_VOLUME_SUBDIR
    out_dir = args.out_dir or args.project_dir / "results" / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)

    splits = {s: ft.load_split(args.project_dir, s) for s in ["train", "val"]}
    ft.check_labels_against_metadata(args.project_dir, splits)
    splits = {s: d[~d.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True) for s, d in splits.items()}

    ft.ensure_sfcn_repo(args.sfcn_repo)
    _, ukb_centers = ft.make_bins(UKB_RANGE, 1)
    model = pretrained_sfcn(args.sfcn_repo)
    print(f"Zero-shot SFCN (UK Biobank, bins {UKB_RANGE[0]}-{UKB_RANGE[1]}) | "
          f"train n={len(splits['train'])} | val n={len(splits['val'])}")

    rows = []
    for s, d in splits.items():
        pred = predict(model, d, volume_dir, ukb_centers)
        t = d.set_index("IXI_ID").loc[pred.index]
        rows.append(pd.DataFrame({"IXI_ID": pred.index, "split": s, "age": t.AGE.to_numpy(),
                                  "sex": t.sex_label.to_numpy(), "site": t.site.to_numpy(),
                                  "predicted_age": pred.to_numpy()}))
    df = pd.concat(rows, ignore_index=True)

    tr = df[df.split == "train"]
    a, b = np.polyfit(tr.age, tr.predicted_age, 1)
    df["predicted_age_corrected"] = (df.predicted_age - b) / a
    df.to_csv(out_dir / "zero_shot_predictions.csv", index=False)

    val = df[df.split == "val"]
    in_range = val[val.age.between(*UKB_RANGE)]
    summary = {
        "model": "pretrained SFCN (UK Biobank), no fine-tuning",
        "labels": "official IXI.xls",
        "bias_correction": {"slope": float(a), "intercept": float(b), "fitted_on": "train"},
        "val_raw": with_ci(val.age.to_numpy(), val.predicted_age.to_numpy()),
        "val_raw_ages_42_82": with_ci(in_range.age.to_numpy(), in_range.predicted_age.to_numpy()),
        "val_corrected": with_ci(val.age.to_numpy(), val.predicted_age_corrected.to_numpy()),
    }

    if args.finetuned:
        ckpt = torch.load(args.finetuned, map_location="cpu", weights_only=False)
        _, centers = ft.make_bins(ckpt["config"]["bin_range"], ckpt["config"]["bin_step"])
        fmodel = ft.build_model(args.sfcn_repo, len(centers), torch.device("cpu"))
        fmodel.load_state_dict(ckpt["model_state_dict"])
        fmodel.eval()
        fv = splits["val"]
        fpred = predict(fmodel, fv, volume_dir, centers)
        fage = fv.set_index("IXI_ID").AGE.loc[fpred.index].to_numpy()
        summary["val_finetuned"] = with_ci(fage, fpred.to_numpy())
        summary["val_finetuned"]["checkpoint_epoch"] = int(ckpt["epoch"])

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Bias correction (train): pred = {a:.3f}*age + {b:.2f}")
    for key, label in [("val_raw", "Zero-shot, raw"), ("val_raw_ages_42_82", "Zero-shot, raw, ages 42-82"),
                       ("val_corrected", "Zero-shot, train-fitted correction"),
                       ("val_finetuned", "Fine-tuned (Phase 4)")]:
        if key in summary:
            m = summary[key]
            print(f"  {label:36s} n={m['n']:3d} MAE {m['mae']:.2f} [{m['mae_ci95'][0]:.2f}, {m['mae_ci95'][1]:.2f}]"
                  f" | r {m['r']:.3f} [{m['r_ci95'][0]:.3f}, {m['r_ci95'][1]:.3f}] | mean BAG {m['mean_bag']:+.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.axhspan(*UKB_RANGE, color="0.92", label="UK Biobank training range")
    for site, g in val.groupby("site"):
        ax.scatter(g.age, g.predicted_age, s=18, label=site)
    ax.plot([15, 95], [15, 95], "k--", lw=1)
    m = summary["val_raw"]
    ax.set(xlim=(15, 95), ylim=(15, 95), xlabel="Age (years)", ylabel="Predicted age (years)",
           title=f"Zero-shot SFCN, validation: MAE {m['mae']:.2f}, r {m['r']:.3f}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(out_dir / "zero_shot_val_scatter.png", dpi=130)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
