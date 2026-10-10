"""Accuracy report for a pretrained brain-age model (DeepBrainNet or SFCN) on the corrected IXI cohort.

python phase2_corrected/accuracy_report.py --pred <predictions.csv> --tag all
Writes phase2_corrected/reports/accuracy_report_<tag>.md and accuracy_<tag>.png.
Descriptive accuracy only: no significance testing between sex/site groups (that is the separate fairness analysis).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RNG = np.random.default_rng(0)
BANDS = [20, 30, 40, 50, 60, 70, 100]
BAND_LABELS = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
SITE_COL = {"Guys": "#4C78A8", "HH": "#F58518", "IOP": "#54A24B"}
SITE_MK = {"Guys": "o", "HH": "s", "IOP": "^"}
IAUDIT = dict(n=560, mae=5.45, r=0.910)   # their README (antspynet brain_age on the same dataset)


def boot(x, fn, n=4000):
    x = np.asarray(x)
    v = [fn(x[RNG.integers(0, len(x), len(x))]) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def pearson_ci(r, n):
    if n < 4:
        return (np.nan, np.nan)
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def row(d, name):
    e = d.predicted_age - d.actual_age
    out = dict(group=name, n=len(d), MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()), ME=e.mean(),
               within5=(e.abs() <= 5).mean() * 100, within10=(e.abs() <= 10).mean() * 100)
    out["r"] = np.corrcoef(d.actual_age, d.predicted_age)[0, 1] if len(d) > 2 else np.nan
    return out


def md_table(df, fmt):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", default="all")
    ap.add_argument("--model", choices=["dbn", "sfcn"], default="dbn")
    ap.add_argument("--strict", action="store_true", help="evaluate only QC=PASS subjects (exclude QC=WARN)")
    ap.add_argument("--out", default=str(C.REPO / "phase2_corrected" / "reports"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sfcn = a.model == "sfcn"
    MODEL = "SFCN" if sfcn else "DeepBrainNet"
    LO, HI = 42.5, 81.5                       # SFCN output limits: 40 one-year bins from 42 to 82 (bin centres)

    d = pd.read_csv(a.pred)
    if "qc_status" not in d.columns:
        d["qc_status"] = "PASS"
    n_warn_all = int((d.qc_status == "WARN").sum())
    if a.strict:
        d = d[d.qc_status == "PASS"].reset_index(drop=True)
    meta = pd.read_csv(C.METADATA_PATH)
    d = d.drop(columns=[c for c in ("site", "sex", "split") if c in d.columns]).merge(
        meta[["IXI_ID", "site", "sex_label"]].rename(columns={"sex_label": "sex"}), on="IXI_ID", how="left")
    spl = {}
    for s in ("train", "val", "test"):
        for i in pd.read_csv(C.PROJECT_DIR / f"ixi_{s}.csv").IXI_ID:
            spl[int(i)] = s
    d["split"] = d.IXI_ID.map(spl)
    assert d.predicted_age.notna().all() and np.isfinite(d.predicted_age).all(), "non-finite predictions"
    d["error"] = d.predicted_age - d.actual_age
    d["abs_error"] = d.error.abs()
    d["band"] = pd.cut(d.actual_age, BANDS, labels=BAND_LABELS, right=False)

    # ---- cohort accounting ----
    qc = pd.read_csv(C.OUT_ROOT / "qc" / "qc_summary.csv")
    excl = pd.read_csv(C.PROJECT_DIR / "ixi_label_excluded.csv")
    qcm = qc[qc.IXI_ID.isin(meta.IXI_ID)]
    fails = qcm[qcm.status == "FAIL"]
    n_scans = len(meta) + len(excl)
    acct = [("Scans in the raw folder index", n_scans),
            ("Excluded: no usable label (see `ixi_label_excluded.csv`)", len(excl)),
            ("Labelled subjects", len(meta)),
            ("Failed hard preprocessing QC (not predicted)", len(fails)),
            ("Predicted with a QC warning (`reg_r` < 0.60 but mask Dice >= 0.80)" + (" - **excluded in this strict report**" if a.strict else " - included"), n_warn_all),
            ("Not yet preprocessed / no QC record", len(meta) - len(qcm)),
            ("**Predicted and evaluated**", len(d))]

    # ---- overall ----
    e = d.error
    ov = row(d, "all")
    mae_ci = boot(d.abs_error, np.mean)
    r_ci = pearson_ci(ov["r"], len(d))
    lr = stats.linregress(d.actual_age, d.predicted_age)
    rho = stats.spearmanr(d.actual_age, d.predicted_age)[0]
    r2 = ov["r"] ** 2
    # linear age-bias correction (fit on this same sample; standard but optimistic)
    gap = d.error
    bl = stats.linregress(d.actual_age, gap)
    corr_gap = gap - (bl.intercept + bl.slope * d.actual_age)

    by_band = pd.DataFrame([row(g, k) for k, g in d.groupby("band", observed=True)])
    by_site = pd.DataFrame([row(g, k) for k, g in d.groupby("site")])
    by_sex = pd.DataFrame([row(g, k) for k, g in d.groupby("sex")])
    by_split = pd.DataFrame([row(g, k) for k, g in d.groupby("split")])
    fmt = {"n": "{:d}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "ME": "{:+.2f}", "within5": "{:.0f}%", "within10": "{:.0f}%", "r": "{:.2f}"}
    rename = {"within5": "within 5 y", "within10": "within 10 y", "group": ""}

    # ---- figure ----
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    for s, g in d.groupby("site"):
        ax[0, 0].scatter(g.actual_age, g.predicted_age, s=22, alpha=.7, c=SITE_COL.get(s), marker=SITE_MK.get(s, "o"), label=f"{s} (n={len(g)})")
        ax[0, 1].scatter(g.actual_age, g.error, s=22, alpha=.7, c=SITE_COL.get(s), marker=SITE_MK.get(s, "o"))
    xs = np.array([20, 87])
    ax[0, 0].plot(xs, xs, "k--", lw=1, label="y = x"); ax[0, 0].plot(xs, lr.intercept + lr.slope * xs, c="#E45756", lw=2, label=f"fit (slope {lr.slope:.2f})")
    if sfcn:
        ax[0, 0].axvspan(18, 42, color="grey", alpha=.15, label="below SFCN's 42-82 range")
        ax[0, 0].axhline(LO, c="grey", ls=":", lw=1.2); ax[0, 0].axhline(HI, c="grey", ls=":", lw=1.2, label="SFCN output limits (42.5 / 81.5)")
    ax[0, 0].set(title=f"Predicted vs actual age (r = {ov['r']:.2f}, MAE = {ov['MAE']:.2f} y)", xlabel="Actual age (y)", ylabel="Predicted age (y)"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].axhline(0, c="k", lw=1); ax[0, 1].plot(xs, bl.intercept + bl.slope * xs, c="#E45756", lw=2)
    ax[0, 1].set(title=f"Error vs age (slope {bl.slope:+.2f} y/y: regression to the mean)", xlabel="Actual age (y)", ylabel="Predicted - actual (y)")
    ax[0, 2].hist(d.error, bins=30, color="#4C78A8"); ax[0, 2].axvline(0, c="k", lw=1); ax[0, 2].axvline(e.mean(), c="#E45756", lw=2, label=f"mean {e.mean():+.2f}")
    ax[0, 2].set(title="Error distribution", xlabel="Predicted - actual (y)", ylabel="Subjects"); ax[0, 2].legend(fontsize=8)

    def bars(axx, tab, title, col="#4C78A8"):
        ci = [1.96 * g.abs_error.std(ddof=1) / np.sqrt(len(g)) if len(g) > 1 else 0 for _, g in
              d.groupby(tab_key[title], observed=True)]
        b = axx.bar(tab.group.astype(str), tab.MAE, yerr=ci, capsize=4, color=col)
        for rect, n in zip(b, tab.n):
            axx.text(rect.get_x() + rect.get_width() / 2, 0.2, f"n={n}", ha="center", color="white", fontsize=8)
        axx.set(title=title, ylabel="MAE (y), 95% CI")
    tab_key = {"MAE by age band": "band", "MAE by site": "site", "MAE by sex": "sex"}
    bars(ax[1, 0], by_band, "MAE by age band"); bars(ax[1, 1], by_site, "MAE by site", "#F58518")
    bars(ax[1, 2], by_sex, "MAE by sex", "#54A24B")
    fig.suptitle(f"{MODEL} (pretrained, zero-shot) on corrected IXI, n={len(d)}", y=1.0)
    fig.tight_layout()
    fig.savefig(out / f"accuracy_{a.tag}.png", dpi=130, bbox_inches="tight")

    worst = d.reindex(d.abs_error.sort_values(ascending=False).index).head(10)[["IXI_ID", "site", "sex", "actual_age", "predicted_age", "error"]]
    young = d[d.actual_age < 45]; old = d[d.actual_age >= 60]

    def sub_metrics(x, name):
        ee = x.predicted_age - x.actual_age
        return (f"| {name} | {len(x)} | {ee.abs().mean():.2f} | {np.sqrt((ee ** 2).mean()):.2f} | {ee.mean():+.2f} | "
                f"{np.corrcoef(x.actual_age, x.predicted_age)[0, 1]:.3f} | {stats.spearmanr(x.actual_age, x.predicted_age)[0]:.3f} |")
    inrange_md = ""
    if sfcn:
        ir, orr = d[d.actual_age.between(42, 82)], d[d.actual_age < 42]
        head = "| Subset | n | MAE | RMSE | bias | Pearson r | Spearman |" + "\n" + "|---|---|---|---|---|---|---|"
        rows_md = "\n".join([sub_metrics(d, "all subjects"), sub_metrics(ir, "**age 42-82 (in range)**"), sub_metrics(orr, "age < 42 (out of range)")])
        inrange_md = ("### SFCN accuracy inside and outside its output range" + "\n\n"
                      + "SFCN can only output 42.5-81.5 y (40 one-year bins, `bin_range = [42, 82]` in the official example), so the fair test is subjects "
                      + "aged 42-82. Subjects under 42 cannot be predicted correctly by construction; their error is a property of the output design." + "\n\n"
                      + head + "\n" + rows_md + "\n\n"
                      + "For comparison, DeepBrainNet on the same in-range subjects: see `accuracy_report_all_inclusive.md`.")
    reference_md = "" if sfcn else (
        "**Reference:** iAudit reports r = " + f"{IAUDIT['r']:.3f}" + ", MAE = " + f"{IAUDIT['mae']:.2f}" + " y for antspynet's DeepBrainNet on "
        + str(IAUDIT['n']) + " IXI scans (`results/` of github.com/sudarsan2507-hue/iAudit). Same weights (byte-identical file), "
        + "different preprocessing wrapper and cohort filtering.")
    md = f"""# Accuracy report: pretrained {MODEL} on corrected IXI ({a.tag})

*Model:* {"SFCN `run_20190719_00_epoch_best_mae.p` (UK Biobank, Peng et al. 2021); output = expected value over 40 one-year bins, so predictions are limited to 42.5-81.5 y" if sfcn else "DeepBrainNet `DBN_model.h5` (SHA256 `9257e98e…41ea`)"}, **pretrained, no fine-tuning**, inference only.
*Input:* corrected preprocessing (FSL MNI152 grid, antspynet brain mask, reference slice chain; see `phase2_corrected/`).
*Labels:* rebuilt from the genuine `IXI.xls` joined on `IXI_ID` (verified identical to iAudit's XNAT-verified ages on 482/482 shared subjects).
*Predictions file:* `{Path(a.pred).name}` | *Scope:* {'strict (QC = PASS only)' if a.strict else 'all predicted subjects (QC = PASS or WARN)'}

## 1. Headline

| Metric | Value |
|---|---|
| Subjects evaluated | **{len(d)}** |
| **MAE** | **{ov['MAE']:.2f} y** (95% bootstrap CI {mae_ci[0]:.2f}-{mae_ci[1]:.2f}) |
| RMSE | {ov['RMSE']:.2f} y |
| Median absolute error | {d.abs_error.median():.2f} y |
| Mean error (bias, predicted - actual) | {e.mean():+.2f} y |
| Within 5 y / within 10 y | {ov['within5']:.0f}% / {ov['within10']:.0f}% |
| **Pearson r** | **{ov['r']:.3f}** (95% CI {r_ci[0]:.3f}-{r_ci[1]:.3f}) |
| Spearman rho | {rho:.3f} |
| R^2 | {r2:.3f} |
| Slope of predicted on actual (1.0 = ideal) | {lr.slope:.3f} (intercept {lr.intercept:.1f}) |
| MAE after linear age-bias correction* | {corr_gap.abs().mean():.2f} y |

\\*Correction `gap ~ age` fitted on this same sample (the standard approach, but optimistic; it removes the regression-to-the-mean trend).

{inrange_md}

{reference_md}

![accuracy](accuracy_{a.tag}.png)

## 2. Cohort accounting (nothing silently dropped)

*QC policy note:* `reg_r` (T1-vs-template correlation) was demoted from a hard failure to a warning on 2026-10-09 after two of the first 25 scans
failed on it alone while their mask Dice was 0.93-0.95; no predictions for those scans had been seen. A strict (PASS-only) report is provided alongside.

{chr(10).join(f"- {k}: **{v}**" if not str(k).startswith("**") else f"- {k}: {v}" for k, v in acct)}

{"**Hard-QC-failed subjects (not predicted):** " + ", ".join(f"IXI{int(i):03d} ({r})" for i, r in zip(fails.IXI_ID, fails.fail_reasons)) if len(fails) else "No subject failed hard preprocessing QC (every labelled subject was predicted)."}

## 3. Accuracy by age band

{md_table(by_band.rename(columns=rename), {rename.get(k, k): v for k, v in fmt.items()})}

Young subjects are over-predicted (mean error {young.error.mean():+.1f} y for < 45 y, n={len(young)}) and the oldest slightly under-predicted
({old.error.mean():+.1f} y for >= 60 y, n={len(old)}): the usual regression-to-the-mean pattern of age regressors, visible as slope {lr.slope:.2f}.

## 4. Accuracy by site and sex (descriptive)

{md_table(by_site.rename(columns=rename), {rename.get(k, k): v for k, v in fmt.items()})}

{md_table(by_sex.rename(columns=rename), {rename.get(k, k): v for k, v in fmt.items()})}

These are **unadjusted** group means. Sites and sexes differ in age mix, and error depends strongly on age, so group gaps here are not
evidence of bias. Significance testing and age/site adjustment belong to the separate fairness analysis.

## 5. Accuracy by split (all zero-shot, so splits are only a consistency check)

{md_table(by_split.rename(columns=rename), {rename.get(k, k): v for k, v in fmt.items()})}

## 6. Ten largest errors

{md_table(worst.round(1), {"IXI_ID": "{:d}"})}

## 7. Interpretation and limitations

- The pretrained model tracks age on corrected labels (r = {ov['r']:.2f}), confirming the earlier "failure" was a label bug, not a model problem.
{"- SFCN's overall MAE is dominated by subjects under 42, which lie outside its 42-82 output range; judge it on the in-range table above." if sfcn else ""}
- Errors are not uniform across age: see section 3. A flat MAE would hide this.
- Zero-shot: no domain adaptation to IXI scanners (Guys/HH Philips, IOP GE), which is expected to affect IOP most.
- Selection: subjects without usable labels or failing QC are excluded (section 2); results describe the evaluated subset.
- Age-bias correction is fitted in-sample, so the corrected MAE is optimistic.
- Single model; the preprocessing differs from antspynet's and from the original authors', so absolute numbers are not directly comparable.
"""
    (out / f"accuracy_report_{a.tag}.md").write_text(md, encoding="utf8")
    print(f"wrote {out / f'accuracy_report_{a.tag}.md'}")
    print(f"n={len(d)} MAE={ov['MAE']:.2f} r={ov['r']:.3f} slope={lr.slope:.3f}")


if __name__ == "__main__":
    main()
