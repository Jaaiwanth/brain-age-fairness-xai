"""Pilot report: prediction table + metrics (new vs old pipeline) + visual QC contact sheet.

python phase2_corrected/pilot_report.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

O = C.OUT_ROOT


def metrics(age, pred):
    age, pred = np.asarray(age, float), np.asarray(pred, float)
    if len(age) < 3:
        return {}
    return dict(n=len(age), MAE=float(np.abs(pred - age).mean()), pearson=float(np.corrcoef(age, pred)[0, 1]),
                spearman=float(stats.spearmanr(age, pred)[0]), slope=float(stats.linregress(age, pred).slope))


def main():
    pil = pd.read_csv(O / "qc" / "pilot_ids.csv")
    qc = pd.read_csv(O / "qc" / "qc_summary.csv").set_index("IXI_ID")
    t = pil[["IXI_ID", "site", "AGE"]].rename(columns={"AGE": "true_age"}).sort_values("true_age")
    for m, old_csv in [("sfcn", "phase3_sfcn_full_results.csv"), ("dbn", "phase3_deepbrainnet_full_results.csv")]:
        f = O / "inference" / f"{m}_predictions_pilot.csv"
        new = pd.read_csv(f).set_index("IXI_ID").predicted_age if f.exists() else pd.Series(dtype=float)
        old = pd.read_csv(C.PROJECT_DIR / old_csv).set_index("IXI_ID").predicted_age
        t[f"{m}_new"] = t.IXI_ID.map(new)
        t[f"{m}_old"] = t.IXI_ID.map(old)
    t["QC"] = t.IXI_ID.map(lambda i: qc.loc[i, "status"])
    t["dice"] = t.IXI_ID.map(lambda i: qc.loc[i, "dice_vs_template"])
    t["reg_r"] = t.IXI_ID.map(lambda i: qc.loc[i, "reg_pearson_r"])
    pd.set_option("display.width", 250)
    print(t.round(2).to_string(index=False))
    out = []
    for m in ["sfcn", "dbn"]:
        for v in ["new", "old"]:
            s = t.dropna(subset=[f"{m}_{v}"])
            if len(s) >= 3:
                mm = metrics(s.true_age, s[f"{m}_{v}"])
                out.append(dict(model=m, pipeline=v, **mm))
    r = pd.DataFrame(out)
    print("\n" + r.round(3).to_string(index=False))
    t.to_csv(O / "inference" / "pilot_table.csv", index=False)
    r.to_csv(O / "inference" / "pilot_metrics.csv", index=False)

    # ---- visual QC contact sheet ----
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import infer
    tm = nib.load(str(C.FSL_MASK)).get_fdata() > 0.5
    ids = t.IXI_ID.tolist()
    fig, ax = plt.subplots(len(ids), 6, figsize=(18, 2.9 * len(ids)))
    for r_, i in enumerate(ids):
        v = nib.load(str(O / "sfcn" / f"IXI{i:03d}.nii.gz")).get_fdata()
        vm = np.percentile(v[v > 0], 99.5)
        row = t[t.IXI_ID == i].iloc[0]
        for c, (sl, msk, ttl) in enumerate([(v[:, :, 80], tm[:, :, 80], "axial z=80"), (v[:, 109, :], tm[:, 109, :], "coronal y=109"),
                                            (v[91, :, :], tm[91, :, :], "sagittal x=91")]):
            ax[r_, c].imshow(sl.T, cmap="gray", origin="lower", vmin=0, vmax=vm)
            ax[r_, c].contour(msk.T, levels=[0.5], colors="r", linewidths=0.5)
            ax[r_, c].set_title(f"IXI{i} {row.site} age {row.true_age:.0f} {ttl}", fontsize=8)
            ax[r_, c].axis("off")
        d = nib.load(str(O / "deepbrainnet" / f"IXI{i:03d}.nii.gz")).get_fdata()
        b, _ = infer.dbn_input(d)
        for c, k in enumerate([0, 40, 79]):
            ax[r_, 3 + c].imshow(b[k][..., 0], cmap="gray", vmin=0, vmax=1)
            ax[r_, 3 + c].set_title(f"DBN input slice {C.DBN_SLICE_START + k} (256x256)", fontsize=8)
            ax[r_, 3 + c].axis("off")
    plt.tight_layout()
    (O / "qc").mkdir(exist_ok=True)
    plt.savefig(O / "qc" / "pilot_contact_sheet.png", dpi=45)
    print("saved", O / "qc" / "pilot_contact_sheet.png")


if __name__ == "__main__":
    main()
