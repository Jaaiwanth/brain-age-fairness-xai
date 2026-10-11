"""Validation-only summary of the three fine-tuning stages (the test split is NOT touched here)."""
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
import sfcn_ft as ft  # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "reports"
OUT.mkdir(exist_ok=True)
runs = [("headA", "A: head only"), ("partB", "B: + last 2 blocks"), ("fullC", "C: full network")]
BASE = dict(zero_shot=10.85, calibrated=5.41, dbn=5.37)          # validation baselines computed earlier (same 73 subjects)

fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
off = 0
for (r, lab), c in zip(runs, ["#4C78A8", "#F58518", "#54A24B"]):
    h = pd.read_csv(CK / r / "history.csv"); x = h.epoch + off
    ax[0].plot(x, h.val_MAE, c=c, label=lab); ax[0].plot(x, h.train_MAE, c=c, ls=":", alpha=.8)
    ax[1].plot(x, h.val_bias, c=c); ax[2].plot(x, h.val_r, c=c)
    b = h.val_MAE.idxmin(); ax[0].scatter([x[b]], [h.val_MAE[b]], c=c, zorder=5)
    off += int(h.epoch.max())
for k, (v, t, ls) in enumerate([(BASE["calibrated"], "zero-shot + linear calibration (5.41)", "--"), (BASE["dbn"], "DeepBrainNet zero-shot (5.37)", "-."), (BASE["zero_shot"], "SFCN zero-shot (10.85)", ":")]):
    ax[0].axhline(v, c="grey", ls=ls, lw=1.2, label=t)
ax[0].set(title="Validation MAE (solid) and train MAE (dotted)", xlabel="epoch (stages chained)", ylabel="MAE (y)", ylim=(3, 11)); ax[0].legend(fontsize=7.5)
ax[1].axhline(0, c="k", lw=1); ax[1].set(title="Validation bias (predicted - actual)", xlabel="epoch (stages chained)", ylabel="y")
ax[2].set(title="Validation Pearson r", xlabel="epoch (stages chained)")
fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=130)

# best model, validation split only
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rows = []
for r, lab in runs:
    ck = torch.load(CK / r / "best.pt", map_location="cpu")
    net = ft.SFCN(output_dim=ft.NEW_BINS); net.load_state_dict(ck["model"]); model = ft.AgeModel(net).to(dev)
    vl = torch.utils.data.DataLoader(ft.CachedData("val", train=False), batch_size=4, num_workers=0)
    m, df = ft.evaluate(model, vl, dev)
    rows.append(dict(model=lab, epoch=ck["epoch"], **{k: round(v, 3) for k, v in m.items()}))
    if r == "fullC":
        meta = pd.read_csv(ft.CACHE / "index.csv").set_index("IXI_ID")
        df["site"] = df.IXI_ID.map(meta.site); df["sex"] = df.IXI_ID.map(meta.sex_label)
        df["band"] = pd.cut(df.actual_age, [20, 30, 40, 50, 60, 70, 100], labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70+"], right=False)
        df["err"] = df.predicted_age - df.actual_age
        df.to_csv(OUT / "fullC_val_predictions.csv", index=False)
t = pd.DataFrame(rows); t.to_csv(OUT / "stage_summary_val.csv", index=False)
print(t.to_string(index=False))
d = pd.read_csv(OUT / "fullC_val_predictions.csv"); d["band"] = pd.Categorical(d.band, ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"])
for col in ("band", "site", "sex"):
    g = d.groupby(col, observed=True).err.agg(n="count", MAE=lambda e: e.abs().mean(), bias="mean").round(2); print(f"\nstage C, validation, by {col}:"); print(g.to_string())
