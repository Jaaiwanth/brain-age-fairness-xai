"""One-shot evaluation on the TEST split.

Protocol (fixed before looking at any test number):
  * Final model = checkpoints/fullC/best.pt, chosen by lowest VALIDATION MAE among the three stages.
  * Stages A and B are reported only for context; they are not candidates.
  * Baselines on the same test subjects: zero-shot SFCN, zero-shot SFCN + linear calibration, zero-shot DeepBrainNet (+ calibration);
    every calibration is fitted on the TRAIN split only.
  * A marker file records that the test split was used; re-running requires --force and must not be used for tuning.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase2_corrected"))
import sfcn_ft as ft  # noqa: E402
import common as C  # noqa: E402

OUT = HERE / "reports"
MARK = OUT / "test_evaluated.json"
RNG = np.random.default_rng(0)
FINAL = "fullC"
BANDS = [20, 30, 40, 50, 60, 70, 100]
LABELS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def boot(x, fn, n=4000):
    x = np.asarray(x)
    return np.percentile([fn(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)], [2.5, 97.5])


def stats_row(name, age, pred):
    e = pred - age
    r = np.corrcoef(age, pred)[0, 1]
    lo, hi = boot(np.abs(e), np.mean)
    z, se = np.arctanh(r), 1 / np.sqrt(len(age) - 3)
    return dict(model=name, n=len(age), MAE=np.abs(e).mean(), MAE_lo=lo, MAE_hi=hi, RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(), r=r,
                r_lo=np.tanh(z - 1.96 * se), r_hi=np.tanh(z + 1.96 * se), slope=stats.linregress(age, pred).slope,
                within5=100 * (np.abs(e) <= 5).mean(), within10=100 * (np.abs(e) <= 10).mean())


def predict(ckpt, loader, dev):
    ck = torch.load(ckpt, map_location="cpu")
    net = ft.SFCN(output_dim=ft.NEW_BINS)
    net.load_state_dict(ck["model"], strict=True)
    return ft.evaluate(ft.AgeModel(net).to(dev), loader, dev)[1], ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if MARK.exists() and not a.force:
        print("The test split was already evaluated:", MARK.read_text())
        sys.exit(1)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = pd.read_csv(ft.CACHE / "index.csv")
    assert idx[idx.split == "test"].shape[0] == 73
    tl = torch.utils.data.DataLoader(ft.CachedData("test", train=False), batch_size=4, num_workers=0)
    final, ck = predict(HERE / "checkpoints" / FINAL / "best.pt", tl, dev)
    meta = idx.set_index("IXI_ID")
    final["site"] = final.IXI_ID.map(meta.site)
    final["sex"] = final.IXI_ID.map(meta.sex_label)
    final["split"] = "test"
    final.to_csv(OUT / "test_predictions_fullC.csv", index=False)
    age = final.actual_age.to_numpy()
    ids = final.IXI_ID.to_numpy()

    rows = [stats_row("FINAL: fine-tuned SFCN (stage C)", age, final.predicted_age.to_numpy())]
    for r, lab in (("headA", "context: stage A (head only)"), ("partB", "context: stage B (+2 blocks)")):
        d, _ = predict(HERE / "checkpoints" / r / "best.pt", tl, dev)
        rows.append(stats_row(lab, age, d.set_index("IXI_ID").loc[ids].predicted_age.to_numpy()))
    O = C.OUT_ROOT / "inference"
    base = {}
    for nm, f in (("SFCN", "sfcn_predictions_all.csv"), ("DeepBrainNet", "dbn_predictions_all.csv")):
        d = pd.read_csv(O / f)
        tr = d[d.split == "train"]
        te = d.set_index("IXI_ID").loc[ids]
        assert (te.split == "test").all() and np.allclose(te.actual_age, age)
        k, b = np.polyfit(tr.predicted_age, tr.actual_age, 1)          # calibration fitted on TRAIN only
        base[nm] = te.predicted_age.to_numpy()
        base[nm + "+cal"] = k * te.predicted_age.to_numpy() + b
        rows.append(stats_row(f"baseline: {nm} zero-shot", age, base[nm]))
        rows.append(stats_row(f"baseline: {nm} zero-shot + linear calibration (train-fitted)", age, base[nm + "+cal"]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "test_summary.csv", index=False)

    # paired bootstrap: FINAL vs baselines (MAE difference, negative = fine-tuned better)
    ef = np.abs(final.predicted_age.to_numpy() - age)
    paired = {}
    for k in ("SFCN+cal", "DeepBrainNet", "DeepBrainNet+cal", "SFCN"):
        eb = np.abs(base[k] - age)
        dm = []
        for _ in range(4000):
            i = RNG.integers(0, len(age), len(age))
            dm.append(ef[i].mean() - eb[i].mean())
        paired[k] = (ef.mean() - eb.mean(), *np.percentile(dm, [2.5, 97.5]))

    final["err"] = final.predicted_age - final.actual_age
    final["band"] = pd.cut(final.actual_age, BANDS, labels=LABELS, right=False)
    grp = {c: final.groupby(c, observed=True).err.agg(n="count", MAE=lambda e: e.abs().mean(), bias="mean").round(2) for c in ("band", "site", "sex")}

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    col = {"Guys": "#4C78A8", "HH": "#F58518", "IOP": "#54A24B"}
    mk = {"Guys": "o", "HH": "s", "IOP": "^"}
    for s, g in final.groupby("site"):
        ax[0, 0].scatter(g.actual_age, g.predicted_age, c=col[s], marker=mk[s], s=40, alpha=.8, label=f"{s} (n={len(g)})")
        ax[0, 1].scatter(g.actual_age, g.err, c=col[s], marker=mk[s], s=40, alpha=.8)
    xs = np.array([20, 88])
    ax[0, 0].plot(xs, xs, "k--", lw=1)
    lr = stats.linregress(age, final.predicted_age)
    ax[0, 0].plot(xs, lr.intercept + lr.slope * xs, c="#E45756", lw=2, label=f"fit (slope {lr.slope:.2f})")
    ax[0, 0].set(title=f"TEST: predicted vs actual (r = {rows[0]['r']:.2f}, MAE = {rows[0]['MAE']:.2f} y)", xlabel="Actual age (y)", ylabel="Predicted age (y)")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].axhline(0, c="k", lw=1)
    ax[0, 1].set(title="TEST: error vs age", xlabel="Actual age (y)", ylabel="Predicted - actual (y)")
    ax[0, 2].hist(final.err, bins=20, color="#4C78A8")
    ax[0, 2].axvline(0, c="k")
    ax[0, 2].axvline(final.err.mean(), c="#E45756", lw=2, label=f"mean {final.err.mean():+.2f}")
    ax[0, 2].set(title="TEST: error distribution", xlabel="Predicted - actual (y)")
    ax[0, 2].legend(fontsize=8)
    sel = [0, 3, 4, 5, 6]
    names = ["fine-tuned\nSFCN (C)", "SFCN\nzero-shot", "SFCN +\ncalibration", "DeepBrainNet\nzero-shot", "DeepBrainNet\n+ calibration"]
    ax[1, 0].bar(range(5), t.MAE.iloc[sel], yerr=[t.MAE.iloc[sel] - t.MAE_lo.iloc[sel], t.MAE_hi.iloc[sel] - t.MAE.iloc[sel]], capsize=4,
                 color=["#E45756"] + ["#9AA5B1"] * 4)
    ax[1, 0].set_xticks(range(5), names, fontsize=8)
    ax[1, 0].set(title="TEST MAE with 95% bootstrap CI", ylabel="MAE (y)")
    for axx, g_, ttl, c_ in ((ax[1, 1], grp["band"], "TEST MAE by age band", "#4C78A8"), (ax[1, 2], grp["site"], "TEST MAE by site", "#F58518")):
        b = axx.bar(g_.index.astype(str), g_.MAE, color=c_)
        for rect, n in zip(b, g_.n):
            axx.text(rect.get_x() + rect.get_width() / 2, 0.15, f"n={n}", ha="center", color="white", fontsize=8)
        axx.set(title=ttl, ylabel="MAE (y)")
    fig.suptitle(f"Fine-tuned SFCN (stage C best, chosen on validation) on the held-out TEST split, n = {len(age)}", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "test_evaluation.png", dpi=130, bbox_inches="tight")

    MARK.write_text(json.dumps(dict(evaluated=time.strftime("%Y-%m-%d %H:%M"), checkpoint=f"checkpoints/{FINAL}/best.pt", checkpoint_epoch=ck["epoch"],
                                    val_MAE_at_selection=ck["val"]["MAE"], pretrained_sha256=ft.sha256(ft.PRETRAINED)[:16]), indent=1))
    pd.set_option("display.width", 250)
    print(t[["model", "n", "MAE", "MAE_lo", "MAE_hi", "RMSE", "bias", "r", "slope", "within5", "within10"]].round(2).to_string(index=False))
    print("\npaired MAE difference, fine-tuned minus baseline (95% CI; negative = fine-tuned better):")
    for k, (m, lo, hi) in paired.items():
        print(f"  vs {k:18s} {m:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    for c, g in grp.items():
        print(f"\nTEST by {c}:")
        print(g.to_string())


if __name__ == "__main__":
    main()
