"""Phase 5 - Evaluate the frozen fine-tuned SFCN on the held-out test split, once.

What it does:
    1. Loads the chosen checkpoint (selected on validation MAE in Phase 4).
    2. Predicts every validation and test subject.
    3. Fits an age-bias correction on the VALIDATION set only (Cole et al. 2018:
       regress predicted on true age, pred = a*age + b; corrected = (pred - b) / a).
       It uses no test labels, so applying it to the test set is clean.
    4. Reports test MAE, RMSE and Pearson r, raw and bias-corrected, with 95%
       bootstrap CIs, and saves per-subject predictions for the Phase 6 fairness audit.

Run-once guard: refuses to run if results already exist (use --force only to
re-create identical outputs, e.g. after a crash - never to try a different model).

Usage:
    python scripts/05_evaluate_test.py --project-dir D:/ML_Project \
        --checkpoint "G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official/best.pt"
"""

import argparse
import datetime
import hashlib
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


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5: one-time test-set evaluation.")
    p.add_argument("--project-dir", type=Path,
                   default=Path(os.environ.get("BRAINAGE_PROJECT_DIR", ft.DEFAULT_PROJECT_DIR)))
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Defaults to <project-dir>/checkpoints/sfcn_finetune_official/best.pt")
    p.add_argument("--volume-dir", type=Path, default=None,
                   help="Defaults to <project-dir>/IXI_preprocessed_v2")
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to <project-dir>/results/phase5")
    p.add_argument("--force", action="store_true", help="Overwrite existing results.")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def predict(model, df, volume_dir, bin_centers_t, device):
    ds = ft.IXIDataset(df, volume_dir, ft.CONFIG["input_shape"], augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    model.eval()
    ids, preds = [], []
    for volumes, _, batch_ids in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            log_probs = model(volumes.to(device))[0].reshape(volumes.size(0), -1)
        preds += ft.predict_ages(log_probs, bin_centers_t).cpu().tolist()
        ids += batch_ids.tolist()
    return pd.Series(preds, index=ids)


def metrics(age, pred):
    err = pred - age
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r": float(np.corrcoef(age, pred)[0, 1]), "mean_bag": float(err.mean()),
            "age_bag_r": float(np.corrcoef(age, err)[0, 1])}


def bootstrap(age, pred, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    draws = [metrics(age[i], pred[i]) for i in (rng.integers(0, len(age), len(age)) for _ in range(n_boot))]
    return {k: [float(np.percentile([d[k] for d in draws], 2.5)),
                float(np.percentile([d[k] for d in draws], 97.5))] for k in draws[0]}


def main():
    args = parse_args()
    project_dir = args.project_dir
    checkpoint = args.checkpoint or project_dir / "checkpoints" / "sfcn_finetune_official" / "best.pt"
    volume_dir = args.volume_dir or project_dir / ft.DEFAULT_VOLUME_SUBDIR
    out_dir = args.out_dir or project_dir / "results" / "phase5"
    preds_path = out_dir / "test_predictions.csv"
    if preds_path.exists() and not args.force:
        raise SystemExit(f"{preds_path} already exists. The test set is evaluated once; "
                         "use --force only to regenerate the same model's outputs.")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    # ---- Data ----
    val_df = ft.load_split(project_dir, "val")
    test_df = ft.load_split(project_dir, "test")
    train_ids = set(ft.load_split(project_dir, "train")["IXI_ID"])
    for name, df in [("val", val_df), ("test", test_df)]:
        if set(df.IXI_ID) & train_ids:
            raise ValueError(f"{name} overlaps the training split")
    if set(val_df.IXI_ID) & set(test_df.IXI_ID):
        raise ValueError("val overlaps test")
    ft.check_labels_against_metadata(project_dir, {"val": val_df, "test": test_df})
    val_df = val_df[~val_df.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True)
    test_df = test_df[~test_df.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True)

    # ---- Model ----
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    _, centers = ft.make_bins(cfg["bin_range"], cfg["bin_step"])
    bin_centers_t = torch.tensor(centers, dtype=torch.float32, device=device)
    ft.ensure_sfcn_repo(args.sfcn_repo)
    model = ft.build_model(args.sfcn_repo, len(centers), device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Checkpoint: {checkpoint} (epoch {ckpt['epoch']}, val MAE {ckpt['val_mae']:.2f})")
    print(f"Device: {device} | val n={len(val_df)} | test n={len(test_df)}")

    # ---- Predict ----
    val_pred = predict(model, val_df, volume_dir, bin_centers_t, device)
    val_age = val_df.set_index("IXI_ID").AGE.loc[val_pred.index].to_numpy()
    print(f"Val MAE re-computed: {np.abs(val_pred.to_numpy() - val_age).mean():.2f} "
          f"(Phase 4 reported {ckpt['val_mae']:.2f})")

    # Bias correction, fitted on validation only
    a, b = np.polyfit(val_age, val_pred.to_numpy(), 1)
    print(f"Bias correction (val): pred = {a:.3f} * age + {b:.2f}")

    test_pred = predict(model, test_df, volume_dir, bin_centers_t, device)
    t = test_df.set_index("IXI_ID").loc[test_pred.index]
    out = pd.DataFrame({
        "IXI_ID": test_pred.index, "age": t.AGE.to_numpy(), "sex": t.sex_label.to_numpy(),
        "site": t.site.to_numpy(), "predicted_age": test_pred.to_numpy(),
    })
    out["predicted_age_corrected"] = (out.predicted_age - b) / a
    out["bag"] = out.predicted_age - out.age
    out["bag_corrected"] = out.predicted_age_corrected - out.age
    out.to_csv(preds_path, index=False)

    age = out.age.to_numpy()
    summary = {
        "evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": int(ckpt["epoch"]), "val_mae_phase4": float(ckpt["val_mae"]),
        "n_test": len(out), "bias_correction": {"slope": float(a), "intercept": float(b), "fitted_on": "val"},
    }
    for label, col in [("raw", "predicted_age"), ("corrected", "predicted_age_corrected")]:
        pred = out[col].to_numpy()
        summary[label] = {"metrics": metrics(age, pred), "ci95": bootstrap(age, pred)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    lo, hi = 15, 95
    for ax, (label, col) in zip(axes, [("raw", "predicted_age"), ("bias-corrected", "predicted_age_corrected")]):
        for site, g in out.groupby("site"):
            ax.scatter(g.age, g[col], s=18, label=site)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        m = summary["raw" if label == "raw" else "corrected"]["metrics"]
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Age (years)", ylabel="Predicted age (years)",
               title=f"Test, {label}: MAE {m['mae']:.2f}, r {m['r']:.3f}")
        ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "test_scatter.png", dpi=130)

    print()
    for label in ["raw", "corrected"]:
        m, ci = summary[label]["metrics"], summary[label]["ci95"]
        print(f"TEST {label:9s} MAE {m['mae']:.2f} ({ci['mae'][0]:.2f}-{ci['mae'][1]:.2f}) | "
              f"RMSE {m['rmse']:.2f} ({ci['rmse'][0]:.2f}-{ci['rmse'][1]:.2f}) | "
              f"r {m['r']:.3f} ({ci['r'][0]:.3f}-{ci['r'][1]:.3f}) | "
              f"mean BAG {m['mean_bag']:+.2f} | age-BAG r {m['age_bag_r']:+.2f}")
    print(f"\nSaved: {preds_path}\n       {out_dir / 'summary.json'}\n       {out_dir / 'test_scatter.png'}")


if __name__ == "__main__":
    main()
