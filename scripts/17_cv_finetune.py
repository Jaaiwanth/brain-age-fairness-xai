"""Phase 10a - 5-fold cross-validated fine-tuning over train + val (the test split stays locked).

Why: the original split leaves 9 IOP test / 18 held-out subjects. Cross-validation over the 412
train+val subjects gives every one of them an out-of-sample prediction (about 61 IOP), and the fold
models double as independent training runs and as a deep ensemble.

Recipe (CPU-feasible partial fine-tuning). Full fine-tuning takes ~5 days per 5 folds on this
laptop's CPU, so the first three SFCN blocks (conv_0-conv_2) are kept at the UK Biobank weights,
which never saw IXI, and their output (128 x 20 x 24 x 20) is computed once per subject. Each fold
then trains blocks conv_3-conv_5 and a new 72-bin head with the Phase 4 objective and schedule
(Gaussian soft labels sigma 2, KL loss; stage 1 head only 5 epochs at 1e-3; stage 2 blocks 1e-4 and
head 1e-3, AdamW, weight decay 1e-3, early stopping on inner-validation MAE with patience 15,
LR halved on a 5-epoch plateau, at most 60 epochs). No random-shift augmentation (inputs are cached).

For fold k (StratifiedKFold by site, random_state 0):
    outer fold k            -> predicted only (never used for training or early stopping)
    remaining four folds    -> 85% training, 15% inner validation for early stopping (stratified by site)
Each epoch also logs predictions for the outer fold and the training IOP subjects (no influence on
training), to follow the IOP brain-age gap during training. The best checkpoint then predicts the
outer fold, the inner validation set and the locked test set (age, 72-bin distribution, 64-d features).

Usage:
    python scripts/17_cv_finetune.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo [--seeds 0 1]
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
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

N_FOLDS = 5
FROZEN = 3          # conv_0..conv_2 kept at UK Biobank weights
BATCH = 8


def parse_args():
    p = argparse.ArgumentParser(description="5-fold CV partial fine-tuning of SFCN.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--folds", type=int, nargs="*", default=list(range(N_FOLDS)))
    p.add_argument("--seeds", type=int, nargs="*", default=[0])
    p.add_argument("--fold-seed", type=int, default=0,
                   help="random state of the fold assignment (0 = the paper's folds; others = repeated CV)")
    p.add_argument("--cache-dir", type=Path, default=None, help="reuse an existing conv_2 cache")
    return p.parse_args()


def load_pool(project_dir, fold_seed=0):
    pool = pd.concat([ft.load_split(project_dir, "train"), ft.load_split(project_dir, "val")], ignore_index=True)
    pool = pool[~pool.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True)
    pool["fold"] = -1
    for k, (_, idx) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=fold_seed).split(pool, pool.site)):
        pool.loc[idx, "fold"] = k
    return pool


@torch.no_grad()
def build_cache(ids, volume_dir, cache_dir, sfcn_repo):
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (cache_dir / f"IXI{i:03d}.npy").exists()]
    if not todo:
        return
    model = ft.build_model(sfcn_repo, 72, torch.device("cpu")).eval()
    stem = model.feature_extractor[:FROZEN]
    start = time.time()
    for n, i in enumerate(todo, 1):
        v = np.load(ft.volume_path(volume_dir, i)).astype(np.float32)
        v = v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)
        out = stem(torch.from_numpy(v[None, None]))[0].numpy().astype(np.float16)
        np.save(cache_dir / f"IXI{i:03d}.npy", out)
        if n % 50 == 0 or n == len(todo):
            print(f"  cached {n}/{len(todo)} ({(time.time() - start) / n:.1f}s each)", flush=True)


class Tail(nn.Module):
    """conv_3..conv_5 + classifier of SFCN; also exposes the 64-d pooled features."""

    def __init__(self, full):
        super().__init__()
        self.blocks = full.feature_extractor[FROZEN:]
        self.classifier = full.classifier

    def forward(self, x, return_features=False):
        x = self.blocks(x)
        feat = self.classifier.average_pool(x)
        out = self.classifier.conv_6(self.classifier.dropout(feat))
        lp = torch.log_softmax(out.flatten(1), dim=1)
        return (lp, feat.flatten(1)) if return_features else lp


def batches(ids, cache, order=None):
    order = order if order is not None else np.arange(len(ids))
    for s in range(0, len(order), BATCH):
        sel = order[s:s + BATCH]
        yield sel, torch.from_numpy(np.stack([cache[ids[j]] for j in sel]).astype(np.float32))


@torch.no_grad()
def predict(model, df, cache, centers_t):
    model.eval()
    ids = df.IXI_ID.astype(int).to_numpy()
    lps, feats = [], []
    for _, x in batches(ids, cache):
        lp, f = model(x, return_features=True)
        lps.append(lp); feats.append(f)
    lp = torch.cat(lps)
    return ft.predict_ages(lp, centers_t).numpy(), torch.exp(lp).numpy(), torch.cat(feats).numpy(), ids


def train_fold(tr, iv, watch, cache, sfcn_repo, edges, centers, seed, out_dir):
    ft.set_seed(seed)
    centers_t = torch.tensor(centers, dtype=torch.float32)
    model = Tail(ft.build_model(sfcn_repo, len(centers), torch.device("cpu")))
    ids = tr.IXI_ID.astype(int).to_numpy()
    targets = torch.from_numpy(ft.soft_labels(tr.AGE.to_numpy(), edges, ft.CONFIG["label_sigma"]))
    rng = np.random.default_rng(seed)
    best, best_mae, history, log = None, np.inf, [], []
    cfg = ft.CONFIG

    def epoch_pass(optimizer, epoch, stage):
        nonlocal best, best_mae
        t0 = time.time()
        model.train()
        if stage == 1:
            model.blocks.eval()   # frozen blocks keep their BatchNorm statistics
        losses = []
        for sel, x in batches(ids, cache, rng.permutation(len(ids))):
            optimizer.zero_grad()
            loss = ft.kl_loss(model(x), targets[sel])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        pv = predict(model, iv, cache, centers_t)[0]
        mae = float(np.mean(np.abs(pv - iv.AGE.to_numpy())))
        pw, _, _, wid = predict(model, watch, cache, centers_t)
        log.extend({"epoch": epoch, "IXI_ID": int(i), "pred": float(p)} for i, p in zip(wid, pw))
        if mae < best_mae:
            best_mae, best = mae, (epoch, copy.deepcopy(model.state_dict()))
        history.append({"stage": stage, "epoch": epoch, "train_loss": float(np.mean(losses)), "val_mae": mae,
                        "seconds": round(time.time() - t0, 1)})
        print(f"  [stage {stage}] epoch {epoch:2d} | loss {np.mean(losses):.3f} | inner-val MAE {mae:.2f}"
              f" | {time.time() - t0:.0f}s{'  <- best' if best[0] == epoch else ''}", flush=True)
        return mae

    for p in model.blocks.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=cfg["stage1_lr"], weight_decay=cfg["weight_decay"])
    epoch = 0
    for _ in range(cfg["stage1_epochs"]):
        epoch += 1
        epoch_pass(opt, epoch, 1)
    for p in model.blocks.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.blocks.parameters(), "lr": cfg["stage2_lr_backbone"]},
                             {"params": model.classifier.parameters(), "lr": cfg["stage2_lr_head"]}],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=cfg["plateau_patience"])
    stale = 0
    for _ in range(cfg["stage2_max_epochs"]):
        epoch += 1
        mae = epoch_pass(opt, epoch, 2)
        sched.step(mae)
        stale = 0 if best[0] == epoch else stale + 1
        if stale >= cfg["early_stop_patience"]:
            break
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    pd.DataFrame(log).merge(watch[["IXI_ID", "site", "AGE", "role"]], on="IXI_ID").to_csv(
        out_dir / "epoch_predictions.csv", index=False)
    model.load_state_dict(best[1])
    torch.save({"epoch": best[0], "inner_val_mae": best_mae, "tail_state_dict": best[1]}, out_dir / "best_tail.pt")
    return model, best[0], best_mae


def main():
    args = parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))  # leave cores free for the user
    P = args.project_dir
    ft.ensure_sfcn_repo(args.sfcn_repo)
    root = args.out_dir or P / "results" / "phase10_cv"
    root.mkdir(parents=True, exist_ok=True)
    pool = load_pool(P, args.fold_seed)
    pool.to_csv(root / "cv_folds.csv", index=False)
    test = ft.load_split(P, "test")
    test = test[~test.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].reset_index(drop=True)
    print(pool.groupby(["fold", "site"]).size().unstack(), flush=True)

    cache_dir = args.cache_dir or root / "conv2_cache"
    all_ids = sorted(set(pool.IXI_ID.astype(int)) | set(test.IXI_ID.astype(int)))
    print(f"Caching conv_2 outputs of the UK Biobank SFCN for {len(all_ids)} subjects...", flush=True)
    build_cache(all_ids, P / ft.DEFAULT_VOLUME_SUBDIR, cache_dir, args.sfcn_repo)
    cache = {i: np.load(cache_dir / f"IXI{i:03d}.npy") for i in all_ids}
    edges, centers = ft.make_bins(ft.CONFIG["bin_range"], ft.CONFIG["bin_step"])
    centers_t = torch.tensor(centers, dtype=torch.float32)

    for seed in args.seeds:
        for k in args.folds:
            out = root / f"seed{seed}" / f"fold{k}"
            if (out / "outer_predictions.csv").exists():
                print(f"seed {seed} fold {k} done, skipping", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            outer = pool[pool.fold == k].reset_index(drop=True)
            rest = pool[pool.fold != k].reset_index(drop=True)
            tr, iv = train_test_split(rest, test_size=0.15, stratify=rest.site, random_state=k)
            tr, iv = tr.reset_index(drop=True), iv.reset_index(drop=True)
            watch = pd.concat([outer.assign(role="outer"), tr[tr.site == "IOP"].assign(role="train_IOP")],
                              ignore_index=True)
            print(f"\n===== seed {seed} fold {k}: train {len(tr)} | inner val {len(iv)} | outer {len(outer)} =====",
                  flush=True)
            t0 = time.time()
            model, best_epoch, best_mae = train_fold(tr, iv, watch, cache, args.sfcn_repo, edges, centers,
                                                     seed * 100 + k, out)
            for name, df in [("inner_val", iv), ("test", test), ("outer", outer)]:
                pred, prob, feat, ids = predict(model, df, cache, centers_t)
                np.savez_compressed(out / f"{name}_outputs.npz", ids=ids, pred=pred, prob=prob, feat=feat)
                meta = df.set_index(df.IXI_ID.astype(int)).loc[ids]
                pd.DataFrame({"IXI_ID": ids, "site": meta.site.values, "sex": meta.sex_label.values,
                              "age": meta.AGE.values, "pred": pred, "fold": k, "seed": seed}).to_csv(
                    out / f"{name}_predictions.csv", index=False)
            o = pd.read_csv(out / "outer_predictions.csv")
            summary = {"best_epoch": best_epoch, "inner_val_mae": best_mae,
                       "outer_mae": float((o.pred - o.age).abs().mean()), "minutes": round((time.time() - t0) / 60, 1)}
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"seed {seed} fold {k} done: {summary}", flush=True)


if __name__ == "__main__":
    main()
