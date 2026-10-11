"""Validation-only summary of the DeepBrainNet fine-tuning stages (the TEST split is not touched).

python phase5_dbn_finetune/report_val.py --runs headA:A_head last1B:B_block5 last2C:C_blocks4-5
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dbn_ft as D  # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "reports"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run:label pairs in chain order")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    runs = [r.split(":") for r in a.runs]
    ref = pd.read_csv(D.C.OUT_ROOT / "inference" / "dbn_predictions_all.csv")
    tr, va = ref[ref.split == "train"], ref[ref.split == "val"]
    k, b = np.polyfit(tr.predicted_age, tr.actual_age, 1)
    base = dict(zero_shot=float(np.abs(va.predicted_age - va.actual_age).mean()), calibrated=float(np.abs(k * va.predicted_age + b - va.actual_age).mean()))

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    off, cols = 0, ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for (r, lab), c in zip(runs, cols):
        h = pd.read_csv(CK / r / "history.csv")
        x = h.epoch + off
        ax[0].plot(x, h.val_MAE, c=c, label=lab)
        ax[1].plot(x, h.val_bias, c=c)
        ax[2].plot(x, h.val_r, c=c)
        bi = h.val_MAE.idxmin()
        ax[0].scatter([x[bi]], [h.val_MAE[bi]], c=c, zorder=5)
        off += int(h.epoch.max())
    ax[0].axhline(base["zero_shot"], c="grey", ls=":", label=f"DeepBrainNet zero-shot ({base['zero_shot']:.2f})")
    ax[0].axhline(base["calibrated"], c="grey", ls="--", label=f"zero-shot + linear calibration ({base['calibrated']:.2f})")
    ax[0].set(title="Validation MAE (subject level, median of 80 slices)", xlabel="epoch (stages chained)", ylabel="MAE (y)")
    ax[0].legend(fontsize=7.5)
    ax[1].axhline(0, c="k", lw=1)
    ax[1].set(title="Validation bias (predicted - actual)", xlabel="epoch (stages chained)", ylabel="y")
    ax[2].set(title="Validation Pearson r", xlabel="epoch (stages chained)")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=130)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for r, lab in runs:
        ck = torch.load(CK / r / "best.pt", map_location="cpu")
        net = D.DeepBrainNet(ck["config"]["dropout"])
        net.load_state_dict(ck["model"])
        m, df = D.evaluate(net.to(dev), torch.utils.data.DataLoader(D.SubjectData("val"), batch_size=2), dev)
        rows.append(dict(model=lab, run=r, epoch=ck["epoch"], **{k_: round(v, 3) for k_, v in m.items()}))
        last = (r, df)
    t = pd.DataFrame(rows)
    t.loc[len(t)] = dict(model="zero-shot DeepBrainNet", run="-", epoch=0, n=len(va), MAE=round(base["zero_shot"], 3))
    t.loc[len(t)] = dict(model="zero-shot + linear calibration (train-fitted)", run="-", epoch=0, n=len(va), MAE=round(base["calibrated"], 3))
    t.to_csv(OUT / "stage_summary_val.csv", index=False)
    print(t.to_string(index=False))
    meta = D.index().set_index("IXI_ID")
    df = last[1]
    df["site"] = df.IXI_ID.map(meta.site)
    df["sex"] = df.IXI_ID.map(meta.sex_label)
    df["band"] = pd.cut(df.actual_age, [20, 30, 40, 50, 60, 70, 100], labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70+"], right=False)
    df["err"] = df.predicted_age - df.actual_age
    df.to_csv(OUT / f"{last[0]}_val_predictions.csv", index=False)
    for col in ("band", "site", "sex"):
        g = df.groupby(col, observed=True).err.agg(n="count", MAE=lambda e: e.abs().mean(), bias="mean").round(2)
        print(f"\nlast stage, validation, by {col}:")
        print(g.to_string())


if __name__ == "__main__":
    main()
