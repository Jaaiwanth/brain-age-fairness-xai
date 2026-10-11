"""Small, fast checks that training, loss, gradient updates and validation work BEFORE any long DeepBrainNet run.

python experiments/deepbrainnet/smoke_test.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dbn_ft as D  # noqa: E402
import train as T  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def batch(n=16):
    ds = D.SliceData("train", train=True, limit=4)
    xs = [ds[i * 7] for i in range(n)]
    return torch.stack([x[0] for x in xs]).to(device), torch.stack([x[1] for x in xs]).to(device)


def tiny(run, stage, epochs, extra=()):
    shutil.rmtree(T.CKPT_ROOT / run, ignore_errors=True)
    args = ["--run", run, "--stage", stage, "--epochs", str(epochs), "--max-train", "2", "--max-val", "4", "--slices-per-subject", "20", "--num-workers", "0",
            "--batch-size", "8", "--patience", "50", *extra]
    s = T.run(T.parse(args))
    return s, pd.read_csv(T.CKPT_ROOT / run / "history.csv")


def main():
    sha0 = D.sha256(D.WEIGHTS)
    print(f"device {device} | pretrained weights sha256 {sha0[:16]}...\n")

    # 1. the starting model is the pretrained network
    model = D.load_pretrained().to(device)
    ids = D.index().query("split == 'val'").IXI_ID.head(6).tolist()
    ds = D.SubjectData("val", ids=ids)
    m, df = D.evaluate(model, torch.utils.data.DataLoader(ds, batch_size=2), device)
    ref = pd.read_csv(D.C.OUT_ROOT / "inference" / "dbn_predictions_all.csv").set_index("IXI_ID")
    d = (df.set_index("IXI_ID").predicted_age - ref.loc[df.IXI_ID, "predicted_age"]).abs()
    check("1. starting model reproduces the reference (Keras) predictions", d.max() < 0.05, f"6 val subjects: max |diff| {d.max():.4f} y")
    del model

    # 2. freezing / gradients per stage
    xb, yb = batch(16)
    for stage in D.STAGES:
        model = D.load_pretrained().to(device)
        n_tr, n_tot = D.set_stage(model, stage)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            p = model(xb)
        loss = D.compute_loss("huber", p.float(), yb)
        loss.backward()
        frozen_ok = all(q.grad is None for n, q in model.named_parameters() if not q.requires_grad)
        gn = [float(q.grad.norm()) for n, q in model.named_parameters() if q.requires_grad]
        ok = frozen_ok and len(gn) == sum(q.requires_grad for q in model.parameters()) and all(np.isfinite(gn)) and max(gn) > 0
        check(f"2. stage '{stage}': freezing / gradients", ok, f"trainable {n_tr:,}/{n_tot:,}; frozen params have no grad: {frozen_ok}; trainable grads finite & non-zero: {max(gn) > 0 and all(np.isfinite(gn))}; loss {float(loss):.1f}")
        del model, loss, p
        torch.cuda.empty_cache()

    # 3. memory / speed
    for stage in D.STAGES:
        for bs in (32, 16):
            model = D.load_pretrained().to(device)
            D.set_stage(model, stage)
            model.train()
            x = torch.rand(bs, 1, 256, 256, device=device)
            y = torch.rand(bs, device=device) * 60 + 20
            try:
                torch.cuda.reset_peak_memory_stats()
                for k in range(4):
                    if k == 1:
                        torch.cuda.synchronize(); t0 = time.time()
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        p = model(x)
                    D.compute_loss("huber", p.float(), y).backward()
                torch.cuda.synchronize()
                ips = 3 * bs / (time.time() - t0)
                check(f"3. stage '{stage}' fits in GPU memory at batch {bs}", True, f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, {ips:.0f} images/s fwd+bwd")
                break
            except torch.cuda.OutOfMemoryError:
                check(f"3. stage '{stage}' at batch {bs}", False, "OUT OF MEMORY -> trying a smaller batch")
            finally:
                del model
                torch.cuda.empty_cache()

    # 4. which weights actually change
    xb, yb = batch(16)
    for stage, changed, same in (("head", ["head.fc1", "head.fc2"], ["features.block1_conv1", "features.block5_conv3"]),
                                 ("last1", ["head.fc1", "features.block5_conv3"], ["features.block4_conv3", "features.block1_conv1"]),
                                 ("last2", ["head.fc2", "features.block4_conv1", "features.block5_conv3"], ["features.block3_conv3", "features.block1_conv1"]),
                                 ("full", ["head.fc1", "features.block5_conv3", "features.block1_conv1"], [])):
        model = D.load_pretrained().to(device)
        D.set_stage(model, stage)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-3, weight_decay=0)
        for _ in range(3):
            model.train()
            opt.zero_grad()
            D.compute_loss("huber", model(xb).float(), yb).backward()
            opt.step()
        after = model.state_dict()
        chg = lambda pre: max(float((after[k].float() - before[k].float()).abs().max()) for k in after if k.startswith(pre))
        ok = all(chg(c) > 0 for c in changed) and all(chg(s) == 0 for s in same)
        check(f"4. stage '{stage}' updates exactly the intended layers", ok, "changed: " + ", ".join(f"{c} ({chg(c):.0e})" for c in changed) + ("; unchanged: " + ", ".join(same) if same else ""))
        del model, opt
        torch.cuda.empty_cache()

    # 5. tiny runs: learning, validation loop, checkpoint round trip, early stopping
    s, h = tiny("smoke_overfit", "full", 25, ["--dropout", "0.0", "--lr-backbone", "1e-5", "--lr-head", "1e-4"])
    drop = h.train_slice_MAE.iloc[:3].mean() / h.train_slice_MAE.iloc[-3:].mean()
    check("5. full network learns on 2 subjects (dropout off for this sanity check)", drop > 1.3, f"slice MAE {h.train_slice_MAE.iloc[:3].mean():.2f} -> {h.train_slice_MAE.iloc[-3:].mean():.2f} (x{drop:.1f} lower)")
    check("5b. validation loop + checkpoint round trip", s["roundtrip_ok"], f"reloaded best.pt gives val MAE {s['reload_val_MAE']:.3f} vs recorded {s['best_val_MAE']:.3f}")
    s2, _ = tiny("smoke_early", "head", 8, ["--lr-head", "0", "--lr-backbone", "0", "--wd", "0", "--patience", "2"])
    check("5c. early stopping", s2["epochs_run"] == 2 and s2["best_epoch"] == 0, f"stopped after {s2['epochs_run']} epochs (patience 2); best = epoch {s2['best_epoch']} (the starting point)")
    check("5d. original pretrained weights untouched", s["original_weights_unchanged"] and D.sha256(D.WEIGHTS) == sha0, "sha256 identical before/after")

    # 6. learning-rate stability probe (informational: picks safe defaults before any long run)
    print("\nLR stability probe (full network, 2 subjects, dropout off, 20 epochs): final slice MAE vs start")
    rows = []
    for lr in ("1e-5", "3e-5", "1e-4", "3e-4"):
        _, h = tiny(f"smoke_lr{lr}", "full", 20, ["--dropout", "0.0", "--lr-backbone", lr, "--lr-head", lr])
        rows.append(dict(lr=lr, start=h.train_slice_MAE.iloc[:3].mean(), end=h.train_slice_MAE.iloc[-3:].mean()))
        print(f"   lr {lr}: slice MAE {rows[-1]['start']:.2f} -> {rows[-1]['end']:.2f}", flush=True)
    dflt = rows[0]
    check("6. default backbone LR (1e-5) trains stably", dflt["end"] < 0.85 * dflt["start"], f"{dflt['start']:.2f} -> {dflt['end']:.2f}")
    for r in ("smoke_overfit", "smoke_early", "smoke_lr1e-5", "smoke_lr3e-5", "smoke_lr1e-4", "smoke_lr3e-4"):
        shutil.rmtree(T.CKPT_ROOT / r, ignore_errors=True)

    ok = all(r[1] for r in RESULTS)
    print("\n" + ("ALL SMOKE CHECKS PASSED" if ok else "SOME CHECKS FAILED: " + ", ".join(r[0] for r in RESULTS if not r[1])))
    (HERE / "smoke_results.json").write_text(json.dumps([dict(check=a, passed=b, detail=c) for a, b, c in RESULTS] + [dict(lr_probe=rows)], indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
