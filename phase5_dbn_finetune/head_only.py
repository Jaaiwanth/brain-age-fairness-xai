"""Stage A (head only), done efficiently and exactly: the VGG16 base is frozen, so its 512-d pooled feature for a given slice never changes.
Features are computed once (original and left-right-flipped slice, fp32, eval mode) and the 526k-parameter head (Dense 512->1024, ReLU, Dropout 0.9, Dense 1024->1)
is trained on them. Differences from end-to-end 'head' training: no shift/gain augmentation (flip only). Validation is subject level (median of 80 slices),
selection on validation MAE, early stopping, best/last checkpoints in the same format as train.py (full DeepBrainNet state dict) so stage B can start from it.

python phase5_dbn_finetune/head_only.py --run headA
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dbn_ft as D  # noqa: E402

FEATS = D.CACHE / "head_features.pt"


@torch.no_grad()
def compute_features(model, device):
    """(subject, 2, 80, 512): [orig, left-right flipped] x 80 slices x 512 pooled features. Splits come from the cache index."""
    idx = D.index()
    model.eval()
    out = {}
    for n, r in enumerate(idx.itertuples(), 1):
        a = torch.from_numpy(np.load(D.CACHE / f"IXI{int(r.IXI_ID):03d}.npy").astype(np.float32) / 255.0)[:, None].to(device)
        f = []
        for x in (a, a.flip(2)):                                  # rows = left-right axis
            f.append(torch.cat([model.features(x[i:i + 40].expand(-1, 3, -1, -1)).mean(dim=(2, 3)) for i in range(0, 80, 40)]).cpu())
        out[int(r.IXI_ID)] = torch.stack(f)
        if n % 50 == 0:
            print(f"features {n}/{len(idx)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="headA")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--dropout", type=float, default=0.9)
    ap.add_argument("--min-delta", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert D.sha256(D.WEIGHTS) == D.C.DBN_WEIGHTS_SHA256
    model = D.load_pretrained(dropout=a.dropout).to(dev)
    if FEATS.exists():
        feats = torch.load(FEATS)
    else:
        t0 = time.time()
        feats = compute_features(model, dev)
        torch.save(feats, FEATS)
        print(f"features computed in {time.time() - t0:.0f}s -> {FEATS}", flush=True)
    idx = D.index()
    assert not (set(idx[idx.split == "train"].IXI_ID) & set(idx[idx.split != "train"].IXI_ID))
    tr, va = idx[idx.split == "train"], idx[idx.split == "val"]          # the test split is not used here
    Xtr = torch.stack([feats[int(i)] for i in tr.IXI_ID]).to(dev)       # (S,2,80,512)
    ytr = torch.tensor(tr.AGE.to_numpy(), dtype=torch.float32, device=dev)
    Xva = torch.stack([feats[int(i)][0] for i in va.IXI_ID]).to(dev)     # (V,80,512) unflipped
    yva = va.AGE.to_numpy()
    S = Xtr.shape[0]
    head = model.head
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    out = HERE / "checkpoints" / a.run
    out.mkdir(parents=True, exist_ok=True)
    pre_sha = D.C.DBN_WEIGHTS_SHA256

    @torch.no_grad()
    def val():
        head.eval()
        p = head(Xva.flatten(0, 1)).view(len(va), 80)
        pred = torch.quantile(p, 0.5, dim=1).cpu().numpy()
        m = D.metrics(yva, pred)
        m["MAE_meanagg"] = float(np.abs(p.mean(1).cpu().numpy() - yva).mean())
        return m

    def save(name, epoch, m):
        torch.save(dict(model=model.state_dict(), epoch=epoch, val=m, config=dict(vars(a), stage="head", dropout=a.dropout, method="cached-features head-only"),
                        pretrained_sha256=pre_sha), out / name)

    base = val()
    print(f"[{a.run}] head-only on cached features | train {S} subjects x 80 slices x 2 (orig/flip) | start val: MAE {base['MAE']:.2f} bias {base['bias']:+.2f} r {base['r']:.3f}", flush=True)
    best, bad, hist, m = base["MAE"], 0, [], base
    save("best.pt", 0, base)
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        head.train()
        n = S * 80
        perm = torch.randperm(n, generator=g)
        run_loss, nb = 0.0, 0
        for i in range(0, n - a.batch_size + 1, a.batch_size):
            j = perm[i:i + a.batch_size]
            s, k = j // 80, j % 80
            f = torch.randint(0, 2, (len(j),))                                 # random left-right flip per sample
            x = Xtr[s.to(dev), f.to(dev), k.to(dev)]
            pred = head(x).squeeze(1)
            loss = D.compute_loss("huber", pred, ytr[s.to(dev)], a.delta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += float(loss); nb += 1
        m = val()
        sched.step(m["MAE"])
        hist.append(dict(epoch=epoch, train_loss=run_loss / nb, val_MAE=m["MAE"], val_MAE_meanagg=m["MAE_meanagg"], val_RMSE=m["RMSE"], val_bias=m["bias"], val_r=m["r"],
                         val_slope=m["slope"], lr_head=opt.param_groups[0]["lr"], seconds=time.time() - t0))
        imp = m["MAE"] < best - a.min_delta
        if imp:
            best, bad = m["MAE"], 0
            save("best.pt", epoch, m)
        else:
            bad += 1
        if epoch % 5 == 0 or imp:
            print(f"[{a.run}] ep {epoch:3d} loss {run_loss / nb:.2f} | val MAE {m['MAE']:.2f} bias {m['bias']:+.2f} r {m['r']:.3f} slope {m['slope']:.2f} | {'*best*' if imp else f'no gain {bad}/{a.patience}'}", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if bad >= a.patience:
            print(f"[{a.run}] early stop at epoch {epoch}", flush=True)
            break
    save("last.pt", epoch, m)
    # round trip: the saved best checkpoint (a full DeepBrainNet) must reproduce its validation score through the standard evaluation path
    ck = torch.load(out / "best.pt", map_location="cpu")
    net = D.DeepBrainNet(a.dropout)
    net.load_state_dict(ck["model"], strict=True)
    m2, _ = D.evaluate(net.to(dev), torch.utils.data.DataLoader(D.SubjectData("val"), batch_size=2), dev)
    ok = abs(m2["MAE"] - ck["val"]["MAE"]) < 0.05
    assert D.sha256(D.WEIGHTS) == D.C.DBN_WEIGHTS_SHA256, "ORIGINAL PRETRAINED WEIGHTS CHANGED"
    summary = dict(run=a.run, stage="head", method="cached-features head-only", start_val_MAE=base["MAE"], best_val_MAE=ck["val"]["MAE"], best_epoch=ck["epoch"],
                   reload_val_MAE=m2["MAE"], roundtrip_ok=bool(ok), epochs_run=len(hist), original_weights_unchanged=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{a.run}] best val MAE {ck['val']['MAE']:.2f} (epoch {ck['epoch']}) | reload through the full-network evaluation: {m2['MAE']:.2f} -> {'OK' if ok else 'MISMATCH'} | original DBN_model.h5 unchanged", flush=True)


if __name__ == "__main__":
    main()
