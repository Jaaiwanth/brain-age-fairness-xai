"""Fine-tune the pretrained SFCN on the corrected IXI cohort (train split), selecting on the validation split only.

The TEST split is never read here. Original pretrained weights are never modified: checkpoints go to
phase4_finetune/checkpoints/<run>/ (best.pt = best validation MAE, last.pt = final epoch).

Stages (use --init to chain them):
  python phase4_finetune/train.py --run headA  --stage head    --init pretrained
  python phase4_finetune/train.py --run partB  --stage partial --init phase4_finetune/checkpoints/headA/best.pt
  python phase4_finetune/train.py --run fullC  --stage full    --init phase4_finetune/checkpoints/partB/best.pt
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sfcn_ft as ft

CKPT_ROOT = ft.REPO / "phase4_finetune" / "checkpoints"


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--stage", choices=list(ft.STAGES), default="head")
    ap.add_argument("--init", default="pretrained", help="'pretrained' or path to a checkpoint saved by this script")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6, help="early stopping on validation MAE")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps (effective batch = batch-size * accum)")
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=3e-5, help="conservative LR for pretrained layers")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--loss", choices=["huber", "l1", "kl"], default="huber")
    ap.add_argument("--delta", type=float, default=3.0, help="Huber delta in years")
    ap.add_argument("--kl-weight", type=float, default=0.0, help="optional auxiliary KL to Gaussian soft labels")
    ap.add_argument("--new-bin-penalty", type=float, default=6.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--min-delta", type=float, default=0.01, help="minimum val-MAE improvement that counts")
    ap.add_argument("--max-train", type=int, default=0, help="limit train subjects (smoke tests)")
    ap.add_argument("--max-val", type=int, default=0, help="limit val subjects (smoke tests)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    return ap.parse_args(argv)


def build_model(args, device):
    if args.init == "pretrained":
        model = ft.AgeModel(ft.extend_head(ft.load_pretrained40(), args.new_bin_penalty))
    else:
        ck = torch.load(args.init, map_location="cpu")
        net = ft.SFCN(output_dim=ft.NEW_BINS)
        net.load_state_dict(ck["model"], strict=True)
        model = ft.AgeModel(net)
    return model.to(device)


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def worker_init(_):
    np.random.seed(torch.initial_seed() % 2 ** 32)


def run(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    out = CKPT_ROOT / args.run
    out.mkdir(parents=True, exist_ok=True)
    pre_sha = ft.sha256(ft.PRETRAINED)

    model = build_model(args, device)
    n_tr, n_tot = ft.set_stage(model, args.stage)
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("net.classifier.conv_6")]
    back = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("net.classifier.conv_6")]
    groups = [dict(params=head, lr=args.lr_head)] + ([dict(params=back, lr=args.lr_backbone)] if back else [])
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    tr = ft.CachedData("train", train=True, limit=args.max_train or None)
    va = ft.CachedData("val", train=False, limit=args.max_val or None)
    assert not (set(tr.idx.IXI_ID) & set(va.idx.IXI_ID)), "train/val overlap"
    test_ids = set(pd.read_csv(ft.CACHE / "index.csv").query("split == 'test'").IXI_ID)
    assert not (set(tr.idx.IXI_ID) | set(va.idx.IXI_ID)) & test_ids, "test subjects leaked into train/val"
    mk = lambda ds, sh: torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=sh, drop_last=sh, num_workers=args.num_workers,
                                                    persistent_workers=args.num_workers > 0, worker_init_fn=worker_init, pin_memory=True)
    tl, vl = mk(tr, True), mk(va, False)
    print(f"[{args.run}] stage={args.stage} trainable {n_tr:,}/{n_tot:,} params | train {len(tr)} val {len(va)} | loss={args.loss} "
          f"| lr head {args.lr_head} backbone {args.lr_backbone} | eff. batch {args.batch_size * args.accum} | amp={amp}", flush=True)

    base, _ = ft.evaluate(model, vl, device)
    print(f"[{args.run}] start (before training) val: MAE {base['MAE']:.2f} bias {base['bias']:+.2f} r {base['r']:.3f}", flush=True)
    best, bad, hist = base["MAE"], 0, []
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    def save(name, epoch, m):
        torch.save(dict(model=model.net.state_dict(), epoch=epoch, val=m, config=cfg, bins=[ft.NEW_START, ft.NEW_END],
                        pretrained_sha256=pre_sha), out / name)

    save("best.pt", 0, base)                      # epoch 0 = the (extended-head) pretrained starting point
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        ft.train_mode(model)
        opt.zero_grad(set_to_none=True)
        run_loss, run_abs, nb = 0.0, 0.0, 0
        for step, (x, y, _) in enumerate(tl, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logp, pred = model(x)
            loss = ft.compute_loss(args.loss, logp, pred, y, args.delta, args.kl_weight)
            scaler.scale(loss / args.accum).backward()
            if step % args.accum == 0 or step == len(tl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run_loss += float(loss); run_abs += float((pred - y).abs().mean()); nb += 1
        m, df = ft.evaluate(model, vl, device)
        sched.step(m["MAE"])
        row = dict(epoch=epoch, train_loss=run_loss / nb, train_MAE=run_abs / nb, val_MAE=m["MAE"], val_RMSE=m["RMSE"], val_bias=m["bias"],
                   val_r=m["r"], val_slope=m["slope"], lr_head=opt.param_groups[0]["lr"], seconds=time.time() - t0)
        hist.append(row)
        improved = m["MAE"] < best - args.min_delta
        if improved:
            best, bad = m["MAE"], 0
            save("best.pt", epoch, m)
        else:
            bad += 1
        print(f"[{args.run}] ep {epoch:2d} loss {row['train_loss']:.3f} trainMAE {row['train_MAE']:.2f} | val MAE {m['MAE']:.2f} bias {m['bias']:+.2f} "
              f"r {m['r']:.3f} slope {m['slope']:.2f} | {'*best*' if improved else f'no gain {bad}/{args.patience}'} | {row['seconds']:.0f}s", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if bad >= args.patience:
            print(f"[{args.run}] early stop at epoch {epoch}", flush=True)
            break
    save("last.pt", epoch, m)

    # round trip: the saved best checkpoint must reproduce its recorded validation score
    ck = torch.load(out / "best.pt", map_location="cpu")
    net = ft.SFCN(output_dim=ft.NEW_BINS); net.load_state_dict(ck["model"], strict=True)
    m2, _ = ft.evaluate(ft.AgeModel(net).to(device), vl, device)
    ok = abs(m2["MAE"] - ck["val"]["MAE"]) < 0.05
    assert ft.sha256(ft.PRETRAINED) == pre_sha, "ORIGINAL PRETRAINED CHECKPOINT CHANGED"
    summary = dict(run=args.run, stage=args.stage, start_val_MAE=base["MAE"], best_val_MAE=ck["val"]["MAE"], best_epoch=ck["epoch"],
                   reload_val_MAE=m2["MAE"], roundtrip_ok=bool(ok), epochs_run=len(hist), original_checkpoint_unchanged=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{args.run}] best val MAE {ck['val']['MAE']:.2f} (epoch {ck['epoch']}) | reload check {m2['MAE']:.2f} -> {'OK' if ok else 'MISMATCH'} | "
          f"original pretrained file unchanged", flush=True)
    return summary


if __name__ == "__main__":
    run(parse())
