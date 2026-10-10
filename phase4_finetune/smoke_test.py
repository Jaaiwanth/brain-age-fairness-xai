"""Small, fast checks that training, loss, gradient updates and validation behave correctly BEFORE any long run.

python phase4_finetune/smoke_test.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sfcn_ft as ft  # noqa: E402
import train as T  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def batch(n=2, split="train"):
    ds = ft.CachedData(split, train=(split == "train"), limit=8)
    xs = [ds[i] for i in range(n)]
    return torch.stack([x[0] for x in xs]).to(device), torch.stack([x[1] for x in xs]).to(device)


def main():
    pre_sha = ft.sha256(ft.PRETRAINED)
    print(f"device {device} | original checkpoint sha256 {pre_sha[:16]}...\n")

    # ---- 1. head surgery: does the extended model start where the pretrained one was? ----
    m40 = ft.load_pretrained40().to(device).eval()
    m70 = ft.extend_head(ft.load_pretrained40(), 6.0)
    am = ft.AgeModel(m70).to(device).eval()
    ds = ft.CachedData("val", train=False, limit=6)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    c40 = torch.tensor(42.5 + np.arange(40), dtype=torch.float32, device=device)
    with torch.no_grad():
        a40 = (m40(x)[0].flatten(1).exp() * c40).sum(1)
        logp, a70 = am(x)
        out_mass = logp.exp()[:, :ft.OFFSET].sum(1) + logp.exp()[:, ft.OFFSET + ft.PRE_BINS:].sum(1)
    d = (a40 - a70).abs()
    check("1. extended head starts ~ pretrained model", d.max() < 0.5, f"max |expected-age change| {d.max():.3f} y, mean {d.mean():.3f}; probability mass on new bins {out_mass.mean():.4f}")
    check("1b. pretrained 40-bin weights copied exactly into the 70-bin head",
          torch.allclose(m70.classifier.conv_6.weight[ft.OFFSET:ft.OFFSET + 40].detach().cpu(), m40.classifier.conv_6.weight.detach().cpu()),
          "bins 42-82 identical")
    # how much of IXI is unreachable for the original head?
    import pandas as pd
    ages = pd.read_csv(ft.CACHE / "index.csv").AGE
    below = ages[ages < 42.5]
    check("1c. extension is needed", len(below) > 0, f"{len(below)}/{len(ages)} subjects are below the original floor of 42.5 y; best-case error for them would still be {np.mean(42.5 - below):.1f} y")

    # ---- 2. freezing + gradients per stage ----
    xb, yb = batch(2)
    for stage in ("head", "partial", "full"):
        model = ft.AgeModel(ft.extend_head(ft.load_pretrained40(), 6.0)).to(device)
        n_tr, n_tot = ft.set_stage(model, stage)
        bn_before = {k: v.clone() for k, v in model.state_dict().items() if "running_mean" in k}
        ft.train_mode(model)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lp, pr = model(xb)
        loss = ft.compute_loss("huber", lp, pr, yb)
        loss.backward()
        frozen_ok = all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
        gn = [float(p.grad.norm()) for n, p in model.named_parameters() if p.requires_grad and p.grad is not None]
        train_ok = len(gn) == sum(p.requires_grad for p in model.parameters()) and min(gn) >= 0 and max(gn) > 0 and all(np.isfinite(gn))
        bn_ok = all(torch.equal(bn_before[k], v) for k, v in model.state_dict().items() if "running_mean" in k)
        check(f"2. stage '{stage}': freezing / gradients / BN stats", frozen_ok and train_ok and bn_ok,
              f"trainable {n_tr:,}/{n_tot:,}; frozen params have no grad: {frozen_ok}; trainable grads finite & nonzero: {train_ok}; BN running stats unchanged: {bn_ok}; loss {float(loss):.2f}")
        del model, loss, lp, pr
        torch.cuda.empty_cache()

    # ---- 3. memory / speed per stage (batch 2, AMP) ----
    for stage in ("head", "partial", "full"):
        model = ft.AgeModel(ft.extend_head(ft.load_pretrained40(), 6.0)).to(device)
        ft.set_stage(model, stage)
        try:
            torch.cuda.reset_peak_memory_stats(); ft.train_mode(model)
            for k in range(3):
                if k == 1:
                    torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    lp, pr = model(xb)
                ft.compute_loss("huber", lp, pr, yb).backward()
            torch.cuda.synchronize(); dt = (time.time() - t0) / 2 / len(xb)
            check(f"3. stage '{stage}' fits in GPU memory at batch 2", True, f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, {dt:.2f} s/sample fwd+bwd")
        except torch.cuda.OutOfMemoryError:
            check(f"3. stage '{stage}' fits in GPU memory at batch 2", False, "OUT OF MEMORY")
        del model
        torch.cuda.empty_cache()

    # ---- 4. tiny overfit run: the loss must actually go down ----
    runs = HERE / "checkpoints"
    for r in ("smoke_overfit", "smoke_early", "smoke_partial", "smoke_full"):
        shutil.rmtree(runs / r, ignore_errors=True)
    s = T.run(T.parse(["--run", "smoke_overfit", "--stage", "head", "--epochs", "12", "--max-train", "8", "--max-val", "8", "--num-workers", "0",
                       "--patience", "20", "--accum", "1", "--lr-head", "3e-3"]))
    h = __import__("pandas").read_csv(runs / "smoke_overfit" / "history.csv")
    drop = h.train_loss.iloc[:3].mean() / h.train_loss.iloc[-3:].mean()
    improving = s["best_val_MAE"] < s["start_val_MAE"] - 0.3 and h.val_bias.abs().iloc[-1] < abs(h.val_bias.iloc[0]) and drop > 1.05
    check("4. head-only run on 8 subjects: loss and validation error improve", improving, f"train loss x{drop:.2f} lower; val MAE {s['start_val_MAE']:.2f} -> {s['best_val_MAE']:.2f}; |bias| {abs(h.val_bias.iloc[0]):.2f} -> {abs(h.val_bias.iloc[-1]):.2f}")
    check("4b. validation loop + checkpoint round trip", s["roundtrip_ok"], f"reloaded best.pt gives val MAE {s['reload_val_MAE']:.2f} vs recorded {s['best_val_MAE']:.2f}")
    check("4c. original pretrained checkpoint untouched", s["original_checkpoint_unchanged"] and ft.sha256(ft.PRETRAINED) == pre_sha, "sha256 identical before/after")

    # ---- 4d. weights change exactly where they should (3 optimiser steps with a visible learning rate) ----
    for stage, expect_changed, expect_same in (("head", ["classifier.conv_6"], ["feature_extractor.conv_0", "feature_extractor.conv_4", "feature_extractor.conv_5"]),
                                               ("partial", ["classifier.conv_6", "feature_extractor.conv_4", "feature_extractor.conv_5"], ["feature_extractor.conv_0", "feature_extractor.conv_3"]),
                                               ("full", ["classifier.conv_6", "feature_extractor.conv_5", "feature_extractor.conv_0"], [])):
        model = ft.AgeModel(ft.extend_head(ft.load_pretrained40(), 6.0)).to(device)
        ft.set_stage(model, stage)
        before = {k: v.detach().clone() for k, v in model.net.state_dict().items()}
        opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=1e-3, weight_decay=0)
        for _ in range(3):
            ft.train_mode(model)
            lp, pr = model(xb)
            opt.zero_grad(); ft.compute_loss("huber", lp, pr, yb).backward(); opt.step()
        after = model.net.state_dict()
        chg = lambda pre: max(float((after[k].float() - before[k].float()).abs().max()) for k in after if k.startswith(pre) and after[k].dtype.is_floating_point)
        ok = all(chg(pre) > 0 for pre in expect_changed) and all(chg(pre) == 0 for pre in expect_same)
        check(f"4d. stage '{stage}' updates exactly the intended layers", ok, "changed: " + ", ".join(f"{pre} (max |dw| {chg(pre):.1e})" for pre in expect_changed) + ("; unchanged: " + ", ".join(expect_same) if expect_same else ""))
        del model, opt
        torch.cuda.empty_cache()
    # ---- 4e. the whole network can overfit a tiny batch (standard sanity check that gradients drive learning end to end) ----
    shutil.rmtree(runs / "smoke_tiny", ignore_errors=True)
    s4 = T.run(T.parse(["--run", "smoke_tiny", "--stage", "full", "--epochs", "30", "--max-train", "4", "--max-val", "4", "--num-workers", "0", "--patience", "50",
                        "--accum", "1", "--lr-head", "1e-3", "--lr-backbone", "3e-5", "--batch-size", "2"]))
    h4 = __import__("pandas").read_csv(runs / "smoke_tiny" / "history.csv")
    check("4e. full network (default conservative LRs) learns on 4 subjects", h4.train_MAE.iloc[-3:].mean() < 0.75 * h4.train_MAE.iloc[:3].mean(), f"train MAE {h4.train_MAE.iloc[:3].mean():.2f} -> {h4.train_MAE.iloc[-3:].mean():.2f}")

    # ---- 5. early stopping logic: with lr = 0 nothing improves, so it must stop after `patience` epochs and keep the epoch-0 weights ----
    s2 = T.run(T.parse(["--run", "smoke_early", "--stage", "head", "--epochs", "8", "--patience", "2", "--max-train", "8", "--max-val", "8",
                        "--num-workers", "0", "--accum", "1", "--lr-head", "0", "--wd", "0"]))
    check("5. early stopping", s2["epochs_run"] == 2 and s2["best_epoch"] == 0, f"stopped after {s2['epochs_run']} epochs (patience 2); best checkpoint = epoch {s2['best_epoch']} (the starting point)")

    # ---- 6. partial and full stages train for real (2 short epochs each) ----
    for stage, name in (("partial", "smoke_partial"), ("full", "smoke_full")):
        try:
            s3 = T.run(T.parse(["--run", name, "--stage", stage, "--epochs", "2", "--max-train", "8", "--max-val", "8", "--num-workers", "0", "--patience", "5",
                                "--accum", "2", "--init", str(runs / "smoke_overfit" / "best.pt")]))
            h3 = __import__("pandas").read_csv(runs / name / "history.csv")
            check(f"6. stage '{stage}' runs and updates", np.isfinite(h3.train_loss).all() and s3["roundtrip_ok"], f"2 epochs, losses {h3.train_loss.round(2).tolist()}, val MAE {h3.val_MAE.round(2).tolist()}")
        except torch.cuda.OutOfMemoryError:
            check(f"6. stage '{stage}' runs and updates", False, "OUT OF MEMORY")
        torch.cuda.empty_cache()

    ok = all(r[1] for r in RESULTS)
    print("\n" + ("ALL SMOKE CHECKS PASSED" if ok else "SOME CHECKS FAILED: " + ", ".join(r[0] for r in RESULTS if not r[1])))
    (HERE / "smoke_results.json").write_text(json.dumps([dict(check=a, passed=b, detail=c) for a, b, c in RESULTS], indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
