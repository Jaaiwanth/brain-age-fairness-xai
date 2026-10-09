"""Phase 4 - Fine-tune the pretrained SFCN brain-age model on IXI.

Uses the train and validation splits only. The test split is never read here.

Usage:
    python scripts/04_finetune_sfcn.py --smoke    # 1 quick epoch per stage on 8 subjects
    python scripts/04_finetune_sfcn.py --overfit  # can the model memorise 16 train subjects?
    python scripts/04_finetune_sfcn.py            # full run

Paths can be overridden with --project-dir / --sfcn-repo / --checkpoint-dir
or the BRAINAGE_PROJECT_DIR / BRAINAGE_SFCN_REPO / BRAINAGE_CHECKPOINT_DIR
environment variables.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# CONFIG - every path and setting lives here
# =============================================================================

IN_COLAB = Path("/content").exists()

# Cloud Drive when running on Colab; local Google Drive sync otherwise.
DEFAULT_PROJECT_DIR = (
    Path("/content/drive/MyDrive/ML_Project") if IN_COLAB
    else Path("G:/My Drive/ML_Project")
)
# Official SFCN code + weights (Peng et al., 2021). Cloned automatically if missing.
DEFAULT_SFCN_REPO = (
    Path("/content/UKBiobank_deep_pretrain") if IN_COLAB
    else Path(__file__).resolve().parent.parent / "external" / "UKBiobank_deep_pretrain"
)
SFCN_REPO_URL = "https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain.git"
SFCN_WEIGHTS_RELPATH = Path("brain_age") / "run_20190719_00_epoch_best_mae.p"

# Preprocessed volumes inside PROJECT_DIR. v2 = brain-to-brain registration to FSL MNI152
# (scripts/02b_preprocess_v2.py). The original IXI_preprocessed failed alignment QC.
DEFAULT_VOLUME_SUBDIR = "IXI_preprocessed_v2"

# Reading ~20 MB volumes from Drive every epoch is slow. On Colab they are copied
# once to the VM's local disk (one sub-folder per volume version, so v1 and v2
# files never mix). Set to None to always read straight from Drive.
DEFAULT_CACHE_DIR = Path("/content/ixi_cache") if IN_COLAB else None

CONFIG = {
    "seed": 42,
    "input_shape": (160, 192, 160),

    # Age bins: 20-90 years, 1-year bins -> 70 bins, centres 20.5 ... 89.5
    "bin_range": (20, 90),
    "bin_step": 1,
    "label_sigma": 2.0,          # width (years) of the Gaussian soft label

    "batch_size": 4,
    "num_workers": 2,
    "max_shift": 2,              # random shift of up to +/- this many voxels per axis

    # Stage 1: feature extractor frozen, only the new head trains
    "stage1_epochs": 5,
    "stage1_lr": 1e-3,

    # Stage 2: everything trains, lower learning rate for the backbone.
    # (Run 1 used 1e-5 / 1e-4 with patience 8 and stopped while still predicting the mean.)
    "stage2_max_epochs": 60,
    "stage2_lr_head": 1e-3,
    "stage2_lr_backbone": 1e-4,
    "early_stop_patience": 15,   # stage-2 epochs without val MAE improvement
    "plateau_patience": 5,       # halve the LR after this many epochs without improvement

    "weight_decay": 1e-3,
    "use_amp": True,             # mixed precision (only applied on GPU)
}

SMOKE_OVERRIDES = {
    "stage1_epochs": 1,
    "stage2_max_epochs": 1,
    "num_workers": 0,
}
SMOKE_N_SUBJECTS = 8

# Overfit test: train and "validate" on the same 16 train subjects with no augmentation.
# A working pipeline should drive MAE on them well below the ~14-year guess-the-mean level.
OVERFIT_OVERRIDES = {
    "stage1_epochs": 0,
    "stage2_max_epochs": 60,
    "early_stop_patience": 10**9,
    "max_shift": 0,
    "num_workers": 0,
}
OVERFIT_N_SUBJECTS = 16


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SFCN on IXI (train/val only).")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 1 epoch per stage on 8 train / 8 val subjects to check the pipeline.")
    parser.add_argument("--overfit", action="store_true",
                        help="Train and evaluate on the same 16 train subjects to check the model can learn at all.")
    parser.add_argument("--volume-subdir", default=DEFAULT_VOLUME_SUBDIR,
                        help="Folder of preprocessed .npy volumes inside --project-dir.")
    parser.add_argument("--volume-dir", type=Path, default=None,
                        help="Read the .npy volumes from this folder directly (e.g. unzipped on the Colab "
                             "VM's local disk), instead of <project-dir>/<volume-subdir>. Disables caching.")
    parser.add_argument("--project-dir", type=Path,
                        default=Path(os.environ.get("BRAINAGE_PROJECT_DIR", DEFAULT_PROJECT_DIR)))
    parser.add_argument("--sfcn-repo", type=Path,
                        default=Path(os.environ.get("BRAINAGE_SFCN_REPO", DEFAULT_SFCN_REPO)))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=os.environ.get("BRAINAGE_CHECKPOINT_DIR"),
                        help="Defaults to <project-dir>/checkpoints/sfcn_finetune[_smoke|_overfit].")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Local copy of the .npy volumes for faster reads.")
    parser.add_argument("--no-cache", action="store_true", help="Read volumes straight from --project-dir.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Some 3D cuDNN kernels are still non-deterministic; this keeps runs as close as practical.
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# Data
# =============================================================================

def load_split(project_dir, name):
    path = project_dir / f"ixi_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path)
    missing = {"IXI_ID", "AGE", "sex_label", "site"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return df


def volume_path(volume_dir, ixi_id):
    # Rebuilt from the ID: the CSVs' path columns hold machine-specific Windows paths.
    return volume_dir / f"IXI{int(ixi_id):03d}.npy"


def prepare_volume_dir(source_dir, cache_dir, ixi_ids):
    """Copy the needed volumes to a local cache (once) and return the dir to read from."""
    missing = [i for i in ixi_ids if not volume_path(source_dir, i).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} preprocessed volumes missing in {source_dir}, e.g. IXI{missing[0]:03d}.npy"
        )
    if cache_dir is None:
        return source_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    to_copy = [
        i for i in ixi_ids
        if not volume_path(cache_dir, i).exists()
        or volume_path(cache_dir, i).stat().st_size != volume_path(source_dir, i).stat().st_size
    ]
    if to_copy:
        print(f"Caching {len(to_copy)} volumes to {cache_dir} (one-off)...")
        start = time.time()
        for n, i in enumerate(to_copy, 1):
            shutil.copy2(volume_path(source_dir, i), volume_path(cache_dir, i))
            if n % 50 == 0 or n == len(to_copy):
                print(f"  {n}/{len(to_copy)} copied ({time.time() - start:.0f}s)")
    return cache_dir


def random_shift(volume, max_shift, rng):
    """Shift the volume by a random integer offset per axis, filling with zeros (no wrap-around)."""
    shifted = np.zeros_like(volume)
    src, dst = [], []
    for size in volume.shape:
        s = int(rng.integers(-max_shift, max_shift + 1))
        if s >= 0:
            src.append(slice(0, size - s))
            dst.append(slice(s, size))
        else:
            src.append(slice(-s, size))
            dst.append(slice(0, size + s))
    shifted[tuple(dst)] = volume[tuple(src)]
    return shifted


class IXIDataset(Dataset):
    def __init__(self, df, volume_dir, input_shape, augment=False, max_shift=0):
        self.ids = df["IXI_ID"].astype(int).to_numpy()
        self.ages = df["AGE"].astype(np.float32).to_numpy()
        self.volume_dir = volume_dir
        self.input_shape = tuple(input_shape)
        self.augment = augment
        self.max_shift = max_shift

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        volume = np.load(volume_path(self.volume_dir, self.ids[idx])).astype(np.float32)
        if volume.shape != self.input_shape:
            raise ValueError(f"IXI{self.ids[idx]:03d}: shape {volume.shape}, expected {self.input_shape}")

        if self.augment and self.max_shift > 0:
            volume = random_shift(volume, self.max_shift, np.random.default_rng(np.random.randint(2**31)))

        # SFCN convention: divide each volume by its own mean intensity.
        mean = volume.mean()
        if mean <= 0:
            raise ValueError(f"IXI{self.ids[idx]:03d}: non-positive mean intensity")
        volume = volume / mean

        return torch.from_numpy(volume[None]), torch.tensor(self.ages[idx]), int(self.ids[idx])


# =============================================================================
# Soft labels
# =============================================================================

def make_bins(bin_range, bin_step):
    edges = np.arange(bin_range[0], bin_range[1] + bin_step, bin_step, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def soft_labels(ages, edges, sigma):
    """Gaussian soft label: probability mass of N(age, sigma) falling in each bin, renormalised."""
    ages = np.asarray(ages, dtype=np.float64)[:, None]
    cdf = norm.cdf(edges[None, :], loc=ages, scale=sigma)
    probs = np.diff(cdf, axis=1)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


# =============================================================================
# Model
# =============================================================================

def ensure_sfcn_repo(repo_dir):
    if not (repo_dir / SFCN_WEIGHTS_RELPATH).exists():
        print(f"SFCN repo not found at {repo_dir}; cloning {SFCN_REPO_URL} ...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", SFCN_REPO_URL, str(repo_dir)], check=True)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def build_model(repo_dir, n_bins, device):
    from dp_model.model_files.sfcn import SFCN

    model = SFCN()  # original UK Biobank head: 40 bins covering ages 42-82
    state = torch.load(repo_dir / SFCN_WEIGHTS_RELPATH, map_location="cpu")
    # Weights were saved from a DataParallel wrapper, so keys start with "module.".
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    old_head = model.classifier.conv_6
    new_head = nn.Conv3d(old_head.in_channels, n_bins, kernel_size=1, padding=0)

    # Warm start: old bins covered ages 42-82, which are new bins 22-61. Copy those
    # weights so the head starts from the pretrained age knowledge; the remaining
    # bins (20-42 and 82-90) start from fresh random weights.
    old_start = 42 - CONFIG["bin_range"][0]
    with torch.no_grad():
        new_head.weight[old_start:old_start + old_head.out_channels] = old_head.weight
        new_head.bias[old_start:old_start + old_head.out_channels] = old_head.bias

    model.classifier.conv_6 = new_head
    return model.to(device)


def head_parameters(model):
    return list(model.classifier.parameters())


def backbone_parameters(model):
    return list(model.feature_extractor.parameters())


def set_backbone_trainable(model, trainable):
    for p in backbone_parameters(model):
        p.requires_grad = trainable


# =============================================================================
# Train / evaluate
# =============================================================================

def predict_ages(log_probs, bin_centers_t):
    """Expected value of the predicted age distribution."""
    return torch.exp(log_probs.float()) @ bin_centers_t


def kl_loss(log_probs, target_probs):
    return F.kl_div(log_probs.float(), target_probs, reduction="batchmean")


def train_one_epoch(model, loader, optimizer, scaler, edges, bin_centers_t, sigma, device, use_amp,
                    backbone_frozen):
    model.train()
    if backbone_frozen:
        # Keep the frozen BatchNorm running statistics fixed at their pretrained values.
        model.feature_extractor.eval()

    total_loss, total_abs_err, n_seen = 0.0, 0.0, 0
    for volumes, ages, _ in loader:
        volumes = volumes.to(device, non_blocking=True)
        targets = torch.from_numpy(soft_labels(ages.numpy(), edges, sigma)).to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            log_probs = model(volumes)[0].reshape(volumes.size(0), -1)
        loss = kl_loss(log_probs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * volumes.size(0)
        # Running MAE in train mode (dropout + augmentation on), so a bit pessimistic.
        preds = predict_ages(log_probs.detach(), bin_centers_t).cpu()
        total_abs_err += (preds - ages).abs().sum().item()
        n_seen += volumes.size(0)
    return total_loss / n_seen, total_abs_err / n_seen


@torch.no_grad()
def evaluate(model, loader, edges, bin_centers_t, sigma, device, use_amp):
    model.eval()
    ids, ages, preds = [], [], []
    total_loss, n_seen = 0.0, 0
    for volumes, batch_ages, batch_ids in loader:
        volumes = volumes.to(device, non_blocking=True)
        targets = torch.from_numpy(soft_labels(batch_ages.numpy(), edges, sigma)).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            log_probs = model(volumes)[0].reshape(volumes.size(0), -1)
        total_loss += kl_loss(log_probs, targets).item() * volumes.size(0)
        n_seen += volumes.size(0)

        ids.extend(batch_ids.tolist())
        ages.extend(batch_ages.tolist())
        preds.extend(predict_ages(log_probs, bin_centers_t).cpu().tolist())

    ages, preds = np.array(ages), np.array(preds)
    mae = float(np.mean(np.abs(preds - ages)))
    r = float(np.corrcoef(ages, preds)[0, 1]) if np.std(preds) > 0 else float("nan")
    return {"loss": total_loss / n_seen, "mae": mae, "r": r, "ids": ids, "ages": ages, "preds": preds}


def bootstrap_ci(ages, preds, n_boot=2000, seed=0):
    """95% percentile-bootstrap CIs for MAE and Pearson r."""
    rng = np.random.default_rng(seed)
    maes, rs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(ages), len(ages))
        maes.append(np.mean(np.abs(preds[idx] - ages[idx])))
        if np.std(preds[idx]) > 0 and np.std(ages[idx]) > 0:
            rs.append(np.corrcoef(ages[idx], preds[idx])[0, 1])
    return np.percentile(maes, [2.5, 97.5]), np.percentile(rs, [2.5, 97.5]) if rs else (np.nan, np.nan)


def save_checkpoint(path, model, epoch, stage, metrics, config):
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "stage": stage,
        "val_mae": metrics["mae"],
        "val_r": metrics["r"],
        "config": config,
    }, path)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.smoke and args.overfit:
        raise SystemExit("Use either --smoke or --overfit, not both.")
    mode = "smoke" if args.smoke else "overfit" if args.overfit else "full"
    config = dict(CONFIG)
    if args.smoke:
        config.update(SMOKE_OVERRIDES)
    if args.overfit:
        config.update(OVERFIT_OVERRIDES)
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config["use_amp"] and device.type == "cuda"

    project_dir = args.project_dir
    checkpoint_dir = args.checkpoint_dir or (
        project_dir / "checkpoints" / ("sfcn_finetune" if mode == "full" else f"sfcn_finetune_{mode}")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    history_path = checkpoint_dir / "history.csv"
    preds_path = checkpoint_dir / "val_predictions.csv"

    print(f"Device: {device} | AMP: {use_amp} | mode: {mode}")
    print(f"Project dir: {project_dir}")
    print(f"Volumes: {args.volume_subdir}")
    print(f"Checkpoints: {checkpoint_dir}")

    # ---- Data (train + val only; the test split is deliberately never loaded) ----
    train_df = load_split(project_dir, "train")
    val_df = load_split(project_dir, "val")
    overlap = set(train_df["IXI_ID"]) & set(val_df["IXI_ID"])
    if overlap:
        raise ValueError(f"{len(overlap)} subjects appear in both train and val")

    if args.smoke:
        train_df = train_df.sample(SMOKE_N_SUBJECTS, random_state=config["seed"])
        val_df = val_df.sample(SMOKE_N_SUBJECTS, random_state=config["seed"])
    if args.overfit:
        # Deliberately evaluate on the training subjects: this checks learning, not generalisation.
        train_df = train_df.sample(OVERFIT_N_SUBJECTS, random_state=config["seed"])
        val_df = train_df.copy()

    lo, hi = config["bin_range"]
    for name, df in [("train", train_df), ("val", val_df)]:
        out_of_range = df[(df["AGE"] < lo) | (df["AGE"] >= hi)]
        if len(out_of_range):
            raise ValueError(f"{len(out_of_range)} {name} subjects fall outside the {lo}-{hi} age bins")
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    all_ids = sorted(set(pd.concat([train_df["IXI_ID"], val_df["IXI_ID"]]).astype(int)))
    if args.volume_dir:
        volume_dir = prepare_volume_dir(args.volume_dir, None, all_ids)
    else:
        cache_dir = None if args.no_cache or args.cache_dir is None else args.cache_dir / args.volume_subdir
        volume_dir = prepare_volume_dir(project_dir / args.volume_subdir, cache_dir, all_ids)
    print(f"Volumes: {volume_dir}")

    generator = torch.Generator().manual_seed(config["seed"])
    loader_kwargs = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=config["num_workers"] > 0,
    )
    train_loader = DataLoader(
        IXIDataset(train_df, volume_dir, config["input_shape"], augment=True, max_shift=config["max_shift"]),
        shuffle=True, drop_last=len(train_df) > config["batch_size"], generator=generator, **loader_kwargs,
    )
    val_loader = DataLoader(
        IXIDataset(val_df, volume_dir, config["input_shape"], augment=False),
        shuffle=False, **loader_kwargs,
    )

    # ---- Model ----
    edges, centers = make_bins(config["bin_range"], config["bin_step"])
    bin_centers_t = torch.tensor(centers, dtype=torch.float32, device=device)
    ensure_sfcn_repo(args.sfcn_repo)
    model = build_model(args.sfcn_repo, len(centers), device)
    print(f"SFCN loaded; new head predicts {len(centers)} bins ({lo}-{hi} yrs).")

    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    sigma = config["label_sigma"]
    history = []
    best_mae = float("inf")

    def run_epoch(stage, epoch, optimizer, backbone_frozen):
        nonlocal best_mae
        start = time.time()
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, scaler, edges,
                                                bin_centers_t, sigma, device, use_amp, backbone_frozen)
        val = evaluate(model, val_loader, edges, bin_centers_t, sigma, device, use_amp)
        improved = val["mae"] < best_mae
        if improved:
            best_mae = val["mae"]
            save_checkpoint(best_path, model, epoch, stage, val, config)

        lrs = [g["lr"] for g in optimizer.param_groups]
        history.append({
            "stage": stage, "epoch": epoch, "lr": max(lrs), "train_loss": train_loss,
            "train_mae": train_mae, "val_loss": val["loss"], "val_mae": val["mae"], "val_r": val["r"],
            "val_pred_std": float(np.std(val["preds"])),
            "seconds": round(time.time() - start, 1),
        })
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"[stage {stage}] epoch {epoch:3d} | train loss {train_loss:.4f} MAE {train_mae:.2f} "
              f"| val loss {val['loss']:.4f} MAE {val['mae']:.2f} r {val['r']:.3f} "
              f"pred sd {np.std(val['preds']):.1f} | {time.time() - start:.0f}s"
              f"{'  <- best' if improved else ''}")
        return val, improved

    # ---- Stage 1: train the new head only ----
    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(head_parameters(model), lr=config["stage1_lr"],
                                  weight_decay=config["weight_decay"])
    epoch = 0
    for _ in range(config["stage1_epochs"]):
        epoch += 1
        run_epoch(1, epoch, optimizer, backbone_frozen=True)

    # ---- Stage 2: unfreeze everything, low learning rate, early stopping ----
    set_backbone_trainable(model, True)
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters(model), "lr": config["stage2_lr_backbone"]},
        {"params": head_parameters(model), "lr": config["stage2_lr_head"]},
    ], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config["plateau_patience"]
    )
    epochs_without_improvement = 0
    for _ in range(config["stage2_max_epochs"]):
        epoch += 1
        val, improved = run_epoch(2, epoch, optimizer, backbone_frozen=False)
        scheduler.step(val["mae"])
        epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
        if epochs_without_improvement >= config["early_stop_patience"]:
            print(f"Early stopping: no val MAE improvement for {epochs_without_improvement} epochs.")
            break

    # ---- Final validation predictions from the best checkpoint ----
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    val = evaluate(model, val_loader, edges, bin_centers_t, sigma, device, use_amp)

    meta = val_df.set_index(val_df["IXI_ID"].astype(int))
    preds_df = pd.DataFrame({
        "IXI_ID": val["ids"],
        "age": val["ages"],
        "predicted_age": val["preds"],
        "sex": meta.loc[val["ids"], "sex_label"].to_numpy(),
        "site": meta.loc[val["ids"], "site"].to_numpy(),
    })
    preds_df.to_csv(preds_path, index=False)

    mae_ci, r_ci = bootstrap_ci(val["ages"], val["preds"])
    summary = {
        "best_epoch": best["epoch"], "best_stage": best["stage"],
        "val_mae": val["mae"], "val_mae_95ci": list(map(float, mae_ci)),
        "val_r": val["r"], "val_r_95ci": list(map(float, r_ci)),
        "val_pred_std": float(np.std(val["preds"])), "val_age_std": float(np.std(val["ages"])),
        "n_val": len(val["ids"]), "mode": mode, "volume_subdir": args.volume_subdir, "config": config,
    }
    with open(checkpoint_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    if mode == "overfit":
        print("OVERFIT TEST - 'val' below is the same 16 TRAINING subjects (checks learning, not generalisation).")
    print(f"Best checkpoint: epoch {best['epoch']} (stage {best['stage']}) -> {best_path}")
    print(f"Val MAE {val['mae']:.2f} yrs (95% CI {mae_ci[0]:.2f}-{mae_ci[1]:.2f}) | "
          f"val r {val['r']:.3f} (95% CI {r_ci[0]:.3f}-{r_ci[1]:.3f}) | n={len(val['ids'])}")
    print(f"Prediction spread (sd) {np.std(val['preds']):.1f} yrs vs real age spread {np.std(val['ages']):.1f} yrs")
    print(f"Val predictions: {preds_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
