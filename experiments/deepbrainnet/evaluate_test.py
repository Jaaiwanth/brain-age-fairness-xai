"""One-shot evaluation of the fine-tuned DeepBrainNet on the TEST split.

Protocol (the final run is passed explicitly and recorded in the marker file BEFORE any test number is computed):
  * Final model = checkpoints/<--final>/best.pt, chosen by lowest VALIDATION MAE among the stages.
  * Earlier stages are reported only for context; they are not candidates.
  * Baselines on the same test subjects: zero-shot DeepBrainNet (reference Keras predictions) and the same + linear calibration fitted on TRAIN only.
    Context only (not used for any choice): the fine-tuned SFCN test predictions, if present locally.
  * The marker file records that the test split was used; re-running requires --force and must not be used for tuning.

python experiments/deepbrainnet/evaluate_test.py --final last2C --context headA last1B
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
import dbn_ft as D  # noqa: E402

OUT = HERE / "reports"
MARK = OUT / "test_evaluated.json"
RNG = np.random.default_rng(0)
BANDS = [20, 30, 40, 50, 60, 70, 100]
LABELS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]


def boot(x, n=4000):
    x = np.asarray(x)
    return np.percentile([x[RNG.integers(0, len(x), len(x))].mean() for _ in range(n)], [2.5, 97.5])


def row(name, age, pred):
    e = pred - age
    r = np.corrcoef(age, pred)[0, 1]
    lo, hi = boot(np.abs(e))
    z, se = np.arctanh(r), 1 / np.sqrt(len(age) - 3)
    return dict(model=name, n=len(age), MAE=np.abs(e).mean(), MAE_lo=lo, MAE_hi=hi, RMSE=np.sqrt((e ** 2).mean()), bias=e.mean(), r=r,
                r_lo=np.tanh(z - 1.96 * se), r_hi=np.tanh(z + 1.96 * se), slope=stats.linregress(age, pred).slope,
                within5=100 * (np.abs(e) <= 5).mean(), within10=100 * (np.abs(e) <= 10).mean())


def predict(run, loader, dev):
    ck = torch.load(HERE / "checkpoints" / run / "best.pt", map_location="cpu")
    net = D.DeepBrainNet(ck["config"]["dropout"])
    net.load_state_dict(ck["model"], strict=True)
    return D.evaluate(net.to(dev), loader, dev)[1], ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--context", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if MARK.exists() and not a.force:
        print("The test split was already evaluated:", MARK.read_text())
        sys.exit(1)
    ck0 = torch.load(HERE / "checkpoints" / a.final / "best.pt", map_location="cpu")
    MARK.write_text(json.dumps(dict(status="started", started=time.strftime("%Y-%m-%d %H:%M"), final=a.final, final_epoch=ck0["epoch"],
                                    val_MAE_at_selection=ck0["val"]["MAE"], context=a.context, pretrained_sha256=D.C.DBN_WEIGHTS_SHA256[:16]), indent=1))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = D.index()
    assert (idx.split == "test").sum() == 73
    tl = torch.utils.data.DataLoader(D.SubjectData("test"), batch_size=2, num_workers=0)
    final, ck = predict(a.final, tl, dev)
    meta = idx.set_index("IXI_ID")
    final["site"] = final.IXI_ID.map(meta.site)
    final["sex"] = final.IXI_ID.map(meta.sex_label)
    final.to_csv(OUT / f"test_predictions_{a.final}.csv", index=False)
    age, ids = final.actual_age.to_numpy(), final.IXI_ID.to_numpy()
    rows = [row(f"FINAL: fine-tuned DeepBrainNet ({a.final})", age, final.predicted_age.to_numpy())]
    for r in a.context:
        d, _ = predict(r, tl, dev)
        rows.append(row(f"context: {r}", age, d.set_index("IXI_ID").loc[ids].predicted_age.to_numpy()))
    ref = pd.read_csv(D.C.OUT_ROOT / "inference" / "dbn_predictions_all.csv")
    tr, te = ref[ref.split == "train"], ref.set_index("IXI_ID").loc[ids]
    assert (te.split == "test").all() and np.allclose(te.actual_age, age)
    k, b = np.polyfit(tr.predicted_age, tr.actual_age, 1)
    base = {"zero": te.predicted_age.to_numpy(), "cal": k * te.predicted_age.to_numpy() + b}
    rows.append(row("baseline: DeepBrainNet zero-shot", age, base["zero"]))
    rows.append(row("baseline: DeepBrainNet zero-shot + linear calibration (train-fitted)", age, base["cal"]))
    sfcn = HERE.parent / "sfcn" / "reports" / "test_predictions_fullC.csv"
    if sfcn.exists():
        s = pd.read_csv(sfcn).set_index("IXI_ID").loc[ids]
        assert np.allclose(s.actual_age, age)
        base["sfcn"] = s.predicted_age.to_numpy()
        rows.append(row("context: fine-tuned SFCN (phase 4)", age, base["sfcn"]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "test_summary.csv", index=False)

    ef = np.abs(final.predicted_age.to_numpy() - age)
    paired = {}
    for k_, lab in (("zero", "DeepBrainNet zero-shot"), ("cal", "DeepBrainNet + calibration"), ("sfcn", "fine-tuned SFCN")):
        if k_ not in base:
            continue
        eb = np.abs(base[k_] - age)
        dm = []
        for _ in range(4000):
            i = RNG.integers(0, len(age), len(age))
            dm.append(ef[i].mean() - eb[i].mean())
        paired[lab] = (ef.mean() - eb.mean(), *np.percentile(dm, [2.5, 97.5]))

    final["err"] = final.predicted_age - final.actual_age
    final["band"] = pd.cut(final.actual_age, BANDS, labels=LABELS, right=False)
    grp = {c: final.groupby(c, observed=True).err.agg(n="count", MAE=lambda e: e.abs().mean(), bias="mean").round(2) for c in ("band", "site", "sex")}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    col = {"Guys": "#4C78A8", "HH": "#F58518", "IOP": "#54A24B"}
    mk = {"Guys": "o", "HH": "s", "IOP": "^"}
    for s_, g in final.groupby("site"):
        ax[0, 0].scatter(g.actual_age, g.predicted_age, c=col[s_], marker=mk[s_], s=40, alpha=.8, label=f"{s_} (n={len(g)})")
        ax[0, 1].scatter(g.actual_age, g.err, c=col[s_], marker=mk[s_], s=40, alpha=.8)
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
    pick = [0] + [i for i, m_ in enumerate(t.model) if m_.startswith("baseline") or "SFCN" in m_]
    ax[1, 0].bar(range(len(pick)), t.MAE.iloc[pick], yerr=[t.MAE.iloc[pick] - t.MAE_lo.iloc[pick], t.MAE_hi.iloc[pick] - t.MAE.iloc[pick]], capsize=4,
                 color=["#E45756"] + ["#9AA5B1"] * (len(pick) - 1))
    ax[1, 0].set_xticks(range(len(pick)), ["fine-tuned\nDeepBrainNet"] + [t.model.iloc[i].split(": ")[1].replace(" zero-shot", "\nzero-shot").replace(" + linear calibration (train-fitted)", "\n+ calibration").replace("fine-tuned SFCN (phase 4)", "fine-tuned\nSFCN") for i in pick[1:]], fontsize=7.5)
    ax[1, 0].set(title="TEST MAE with 95% bootstrap CI", ylabel="MAE (y)")
    for axx, g_, ttl, c_ in ((ax[1, 1], grp["band"], "TEST MAE by age band", "#4C78A8"), (ax[1, 2], grp["site"], "TEST MAE by site", "#F58518")):
        bb = axx.bar(g_.index.astype(str), g_.MAE, color=c_)
        for rect, n in zip(bb, g_.n):
            axx.text(rect.get_x() + rect.get_width() / 2, 0.15, f"n={n}", ha="center", color="white", fontsize=8)
        axx.set(title=ttl, ylabel="MAE (y)")
    fig.suptitle(f"Fine-tuned DeepBrainNet ({a.final}, chosen on validation) on the held-out TEST split, n = {len(age)}", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "test_evaluation.png", dpi=130, bbox_inches="tight")

    MARK.write_text(json.dumps(dict(status="completed", evaluated=time.strftime("%Y-%m-%d %H:%M"), final=a.final, final_epoch=ck["epoch"],
                                    val_MAE_at_selection=ck["val"]["MAE"], context=a.context, pretrained_sha256=D.C.DBN_WEIGHTS_SHA256[:16]), indent=1))
    pd.set_option("display.width", 250)
    print(t[["model", "n", "MAE", "MAE_lo", "MAE_hi", "RMSE", "bias", "r", "slope", "within5", "within10"]].round(2).to_string(index=False))
    print("\npaired MAE difference, fine-tuned DeepBrainNet minus baseline (95% CI; negative = fine-tuned better):")
    for k_, (m_, lo, hi) in paired.items():
        print(f"  vs {k_:28s} {m_:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    for c, g in grp.items():
        print(f"\nTEST by {c}:")
        print(g.to_string())


if __name__ == "__main__":
    main()
