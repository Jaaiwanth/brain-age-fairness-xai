"""Fine-tune the pretrained DeepBrainNet on the corrected IXI cohort (train split), selecting on the validation split only.

The TEST split is never read here. The original DBN_model.h5 is never modified (size and SHA256 are checked before and after every run);
checkpoints go to experiments/deepbrainnet/checkpoints/<run>/ (best.pt = best validation MAE, last.pt = final epoch).

Validation MAE is SUBJECT-level, exactly as the reference pipeline: median of the 80 slice predictions of each subject.

Stages (chain them with --init):
  python experiments/deepbrainnet/train.py --run headA  --stage head  --init pretrained
  python experiments/deepbrainnet/train.py --run last1B --stage last1 --init experiments/deepbrainnet/checkpoints/headA/best.pt
  python experiments/deepbrainnet/train.py --run last2C --stage last2 --init experiments/deepbrainnet/checkpoints/last1B/best.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dbn_ft as D

CKPT_ROOT = D.REPO / "experiments" / "deepbrainnet" / "checkpoints"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=list(D.STAGES), default="head")
    ap.add_argument("--init", default="pretrained", help="'pretrained' or a checkpoint saved by this script")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5, help="early stopping on validation (subject-level) MAE")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--lr-backbone", type=float, default=1e-5, help="conservative LR for pretrained convolutional layers")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["huber", "l1", "mse"], default="huber")
    ap.add_argument("--delta", type=float, default=3.0, help="Huber delta in years")
    ap.add_argument("--dropout", type=float, default=0.9, help="released model uses 0.9")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--min-delta", type=float, default=0.01)
    ap.add_argument("--max-train", type=int, default=0, help="limit train subjects (smoke tests)")
    ap.add_argument("--max-val", type=int, default=0, help="limit val subjects (smoke tests)")
    ap.add_argument("--slices-per-subject", type=int, default=0, help="train on this many evenly spaced slices per subject per epoch (0 = all 80)")
    ap.add_argument("--epoch-samples", type=int, default=0, help="each epoch uses this many RANDOM slice images (0 = all of them); more frequent validation per unit of compute")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args(argv)


def build_model(args, device):
    model = D.load_pretrained(dropout=args.dropout)
    if args.init != "pretrained":
        ck = torch.load(args.init, map_location="cpu")
        model.load_state_dict(ck["model"], strict=True)
    return model.to(device)


def worker_init(_):
    np.random.seed(torch.initial_seed() % 2 ** 32)


def run(args):
    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    out = CKPT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    assert D.WEIGHTS.stat().st_size == D.C.DBN_WEIGHTS_SIZE and D.sha256(D.WEIGHTS) == D.C.DBN_WEIGHTS_SHA256, "pretrained weights file is not the expected one"

    model = build_model(args, device)
    n_tr, n_tot = D.set_stage(model, args.stage)
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    back = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [dict(params=head, lr=args.lr_head)] + ([dict(params=back, lr=args.lr_backbone)] if back else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    tr = D.SliceData("train", train=True, limit=args.max_train or None, slices_per_subject=args.slices_per_subject or None)
    va = D.SubjectData("val", limit=args.max_val or None)
    test_ids = set(D.index().query("split == 'test'").IXI_ID)
    assert not (set(tr.idx.IXI_ID) & set(va.idx.IXI_ID)), "train/val overlap"
    assert not (set(tr.idx.IXI_ID) | set(va.idx.IXI_ID)) & test_ids, "test subjects leaked into train/val"
    sampler = torch.utils.data.RandomSampler(tr, replacement=False, num_samples=args.epoch_samples) if args.epoch_samples else None
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler, drop_last=True, num_workers=args.num_workers,
                                     persistent_workers=args.num_workers > 0, worker_init_fn=worker_init, pin_memory=True)
    vl = torch.utils.data.DataLoader(va, batch_size=2, shuffle=False, num_workers=min(2, args.num_workers))
    print(f"[{args.run}] stage={args.stage} trainable {n_tr:,}/{n_tot:,} | train {len(tr.idx)} subjects ({(args.epoch_samples or len(tr)):,} randomly drawn slice images/epoch of {len(tr):,}) val {len(va)} subjects "
          f"| loss={args.loss} | lr head {args.lr_head} backbone {args.lr_backbone} | batch {args.batch_size} | dropout {args.dropout} | amp={amp}", flush=True)

    base, _ = D.evaluate(model, vl, device)
    print(f"[{args.run}] start (before training) val: MAE {base['MAE']:.2f} bias {base['bias']:+.2f} r {base['r']:.3f}", flush=True)
    best, bad, hist = base["MAE"], 0, []
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    def save(name, epoch, m):
        torch.save(dict(model=model.state_dict(), epoch=epoch, val=m, config=cfg, pretrained_sha256=D.C.DBN_WEIGHTS_SHA256), out / name)

    save("best.pt", 0, base)
    m = base
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()                                   # VGG has no BatchNorm; dropout (0.9) is active as in the released model
        run_loss, run_abs, nb = 0.0, 0.0, 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                pred = model(x)
            loss = D.compute_loss(args.loss, pred.float(), y, args.delta)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.grad_clip)
            scaler.step(opt); scaler.update()
            run_loss += float(loss); run_abs += float((pred.float() - y).abs().mean()); nb += 1
        m, df = D.evaluate(model, vl, device)
        sched.step(m["MAE"])
        row = dict(epoch=epoch, train_loss=run_loss / nb, train_slice_MAE=run_abs / nb, val_MAE=m["MAE"], val_MAE_meanagg=m["MAE_meanagg"], val_RMSE=m["RMSE"],
                   val_bias=m["bias"], val_r=m["r"], val_slope=m["slope"], lr_head=opt.param_groups[0]["lr"], seconds=time.time() - t0)
        hist.append(row)
        improved = m["MAE"] < best - args.min_delta
        if improved:
            best, bad = m["MAE"], 0
            save("best.pt", epoch, m)
        else:
            bad += 1
        print(f"[{args.run}] ep {epoch:2d} loss {row['train_loss']:.2f} sliceMAE {row['train_slice_MAE']:.2f} | val MAE {m['MAE']:.2f} bias {m['bias']:+.2f} r {m['r']:.3f} "
              f"slope {m['slope']:.2f} | {'*best*' if improved else f'no gain {bad}/{args.patience}'} | {row['seconds']:.0f}s", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if bad >= args.patience:
            print(f"[{args.run}] early stop at epoch {epoch}", flush=True)
            break
    save("last.pt", epoch, m)

    ck = torch.load(out / "best.pt", map_location="cpu")
    net = D.DeepBrainNet(args.dropout); net.load_state_dict(ck["model"], strict=True)
    m2, _ = D.evaluate(net.to(device), vl, device)
    ok = abs(m2["MAE"] - ck["val"]["MAE"]) < 0.02
    assert D.sha256(D.WEIGHTS) == D.C.DBN_WEIGHTS_SHA256, "ORIGINAL PRETRAINED WEIGHTS CHANGED"
    summary = dict(run=args.run, stage=args.stage, start_val_MAE=base["MAE"], best_val_MAE=ck["val"]["MAE"], best_epoch=ck["epoch"], reload_val_MAE=m2["MAE"],
                   roundtrip_ok=bool(ok), epochs_run=len(hist), original_weights_unchanged=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{args.run}] best val MAE {ck['val']['MAE']:.2f} (epoch {ck['epoch']}) | reload check {m2['MAE']:.2f} -> {'OK' if ok else 'MISMATCH'} | original DBN_model.h5 unchanged", flush=True)
    return summary


if __name__ == "__main__":
    run(parse())
