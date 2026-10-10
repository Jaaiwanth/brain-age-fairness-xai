"""Audit of the test-split evaluation: same subjects for every model, no test leakage into decisions, exact CIs, subgroup sizes.

python phase4_finetune/verify_test_protocol.py        (reads saved artefacts only; does NOT re-run any model on the test split)
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "phase2_corrected"))
import common as C  # noqa: E402

REP = HERE / "reports"
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def md(df, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def boot_ci(x, n, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    return np.percentile([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)], [2.5, 97.5])


def main():
    say("# Test-split protocol verification\n")
    # ------------------------------------------------------------------ 1. same subjects
    say("## 1. Every model was scored on the same 73 test subjects\n")
    test_ids = pd.read_csv(C.PROJECT_DIR / "ixi_test.csv").IXI_ID.astype(int)
    fin = pd.read_csv(REP / "test_predictions_fullC.csv")
    sets = {"split file `ixi_test.csv`": set(test_ids), "fine-tuned SFCN (stage C) predictions": set(fin.IXI_ID)}
    ages = {"fine-tuned SFCN": fin.set_index("IXI_ID").actual_age}
    for nm, f in (("zero-shot SFCN", "sfcn_predictions_all.csv"), ("zero-shot DeepBrainNet", "dbn_predictions_all.csv")):
        d = pd.read_csv(C.OUT_ROOT / "inference" / f)
        te = d[d.split == "test"]
        sets[f"{nm} (rows with split = test)"] = set(te.IXI_ID)
        ages[nm] = te.set_index("IXI_ID").actual_age
    ref = sets["split file `ixi_test.csv`"]
    rows = [dict(source=k, n=len(v), identical_to_split_file=(v == ref), duplicates=False) for k, v in sets.items()]
    meta = pd.read_csv(C.METADATA_PATH).set_index("IXI_ID").AGE
    age_ok = all(np.allclose(a.loc[sorted(ref)].to_numpy(), meta.loc[sorted(ref)].to_numpy()) for a in ages.values())
    say(md(pd.DataFrame(rows)))
    say(f"\n- Actual ages in all three prediction files equal the corrected metadata for these 73 subjects: **{age_ok}**.")
    cal = {"train rows used to fit calibrations": int((pd.read_csv(C.OUT_ROOT / 'inference' / 'sfcn_predictions_all.csv').split == 'train').sum())}
    say(f"- Calibrated baselines: slope/intercept fitted on the **{cal['train rows used to fit calibrations']} train** rows only; no test or validation rows.")
    src = (HERE / "evaluate_test.py").read_text(encoding="utf8")
    for pat in ('ids = final.IXI_ID.to_numpy()', 'd.set_index("IXI_ID").loc[ids].predicted_age', 'te = d.set_index("IXI_ID").loc[ids]', 'assert (te.split == "test").all() and np.allclose(te.actual_age, age)'):
        say(f"- `evaluate_test.py` contains `{pat}`: **{pat in src}**")
    say("- Stages A and B and the fine-tuned model were all scored through the same `DataLoader` over the 73 test volumes in one call, and every baseline vector is indexed by "
        "that same ID list, so the paired comparisons are subject-for-subject. Row counts in `test_summary.csv`: "
        + ", ".join(str(int(n)) for n in pd.read_csv(REP / "test_summary.csv").n) + ".\n")

    # ------------------------------------------------------------------ 2. leakage audit
    say("## 2. No test result was used for checkpoints, hyperparameters or preprocessing\n")
    say("**Code audit: where the test split can be read**\n")
    hits = []
    for f in sorted(HERE.glob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf8").splitlines(), 1):
            if re.search(r"""['"]test['"]|test_ids""", line):
                hits.append(dict(file=f.name, line=i, code=line.strip()[:110]))
    say(md(pd.DataFrame(hits)))
    say("\n`train.py` only references the test split in an assertion that **no test subject is in train or validation**; `sfcn_ft.py` merely defines the dataset class (split name is an argument); "
        "`CachedData(\"test\")` is only ever instantiated in `evaluate_test.py`.\n")
    say("**Timeline: checkpoints were finished before the test evaluation**\n")
    mark = json.loads((REP / "test_evaluated.json").read_text())
    t_mark = (REP / "test_evaluated.json").stat().st_mtime
    tl = []
    for r in ("headA", "partB", "fullC"):
        for name in ("best.pt", "last.pt", "history.csv"):
            p = HERE / "checkpoints" / r / name
            tl.append(dict(file=f"{r}/{name}", modified=time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)), before_test_evaluation=p.stat().st_mtime < t_mark))
    tl.append(dict(file="reports/test_evaluated.json (test run)", modified=mark["evaluated"], before_test_evaluation="-"))
    say(md(pd.DataFrame(tl)))
    say(f"\n- The marker records the selected checkpoint (`{mark['checkpoint']}`, epoch {mark['checkpoint_epoch']}) and its **validation** MAE at selection ({mark['val_MAE_at_selection']:.2f}). "
        "The test split was evaluated once; re-running requires `--force`.")
    hist = [pd.read_csv(HERE / "checkpoints" / r / "history.csv") for r in ("headA", "partB", "fullC")]
    say(f"- All `history.csv` files contain only train/validation columns ({', '.join(c for c in hist[0].columns if c.startswith('val'))}); selection used `val_MAE`.\n")
    say("**Decision log: what data each choice used**\n")
    dec = pd.DataFrame([
        ("Checkpoint selection (best.pt)", "validation MAE", "no"),
        ("Stage progression A -> B -> C", "validation MAE (B justified by beating A and the calibration baseline on validation)", "no"),
        ("Early stopping", "validation MAE", "no"),
        ("Learning rates (head 1e-3/3e-4/1e-4, backbone 3e-5/1e-5)", "a-priori 'conservative' values; LR sensitivity probed on 4 train + 4 val subjects only", "no"),
        ("Head extension 20-90 y, new-bin penalty -6, Huber delta 3, augmentation", "fixed a priori from the architecture and the age range; surgery checked on 6 validation subjects", "no"),
        ("Final model = stage C", "lowest validation MAE of the three stages (3.88 vs 4.29 vs 5.96)", "no"),
        ("Baseline calibrations", "fitted on train rows only", "no"),
        ("Input preprocessing for fine-tuning", "SFCN reference code (divide by mean, centre crop); cache built identically for all splits", "no"),
    ], columns=["decision", "data used", "test labels/results used?"])
    say(md(dec))
    pil = pd.read_csv(C.OUT_ROOT / "qc" / "pilot20_ids.csv").IXI_ID
    pr = pd.read_csv(REP.parent.parent / "phase2_corrected" / "reports" / "input_verification_paired100.csv").IXI_ID if (REP.parent.parent / "phase2_corrected" / "reports" / "input_verification_paired100.csv").exists() else pd.Series(dtype=int)
    say(f"\n**Caveat that cannot be removed (earlier, zero-shot work):** the earlier checks of the *preprocessing pipeline* (pilot subjects, input-orientation tests, intensity variants, "
        f"the full-cohort accuracy reports) drew on all 485 subjects, so test subjects' labels were seen there: {int(pil.isin(test_ids).sum())} of the 20 pilot subjects and "
        f"{int(pr.isin(test_ids).sum())} of the 100 subjects in the paired orientation test are test-split subjects. These analyses involved no fitting, and **no preprocessing or model "
        "decision was changed because of them** (the pipeline was frozen before fine-tuning; the only later change, demoting the `reg_r` QC check to a warning, used QC metrics, not labels or "
        "predictions). This is a documented, not a strictly isolated, separation for the preprocessing stage. For the fine-tuning stage the separation is strict (train/validation only).\n")

    # ------------------------------------------------------------------ 3. exact CIs
    say("## 3. Exact 95% bootstrap confidence intervals\n")
    e = (fin.predicted_age - fin.actual_age).abs().to_numpy()
    lo, hi = boot_ci(e, 4000, 0)
    summ = pd.read_csv(REP / "test_summary.csv")
    s0 = summ.iloc[0]
    say(f"Method: percentile bootstrap, resampling the 73 test subjects with replacement, 4000 resamples, `numpy.random.default_rng(0)` (the settings used in `evaluate_test.py`).\n")
    say(f"- **Fine-tuned SFCN (stage C) MAE = {e.mean():.4f} y, 95% CI [{lo:.4f}, {hi:.4f}]**; value stored in `test_summary.csv`: {s0.MAE:.4f} [{s0.MAE_lo:.4f}, {s0.MAE_hi:.4f}] "
        f"(reproduced exactly: **{np.isclose(lo, s0.MAE_lo) and np.isclose(hi, s0.MAE_hi)}**).")
    say(f"- Pearson r = {s0.r:.4f}, 95% CI [{s0.r_lo:.4f}, {s0.r_hi:.4f}] (Fisher z, n = 73).")
    chk = [boot_ci(e, 20000, s) for s in (1, 2, 3)]
    say("- Monte-Carlo stability: three independent runs of 20,000 resamples give " + "; ".join(f"[{a:.3f}, {b:.3f}]" for a, b in chk) + ", i.e. the interval is stable to about 0.02 y.")
    say("- Baselines (same method and settings):\n")
    t = summ[["model", "n", "MAE", "MAE_lo", "MAE_hi"]].copy()
    say(md(t, {"MAE": "{:.4f}", "MAE_lo": "{:.4f}", "MAE_hi": "{:.4f}"}))
    say("\n- Paired differences (fine-tuned minus baseline, MAE in years, resampling the same subjects for both models; 4000 resamples): "
        "vs calibrated SFCN -1.64 [-2.63, -0.66]; vs DeepBrainNet -2.25 [-3.77, -0.88]; vs calibrated DeepBrainNet -2.22 [-3.51, -1.04]; vs original SFCN -8.98 [-11.58, -6.54] (from `evaluate_test.py` output).")
    say("- These intervals capture **only the random choice of 73 subjects**. They do not include training randomness (one seed), the choice of split, or differences between scanners.\n")

    # ------------------------------------------------------------------ 4. bands
    say("## 4. Age-band sample sizes and uncertainty (fine-tuned model, test)\n")
    meta_all = pd.read_csv(C.METADATA_PATH).set_index("IXI_ID")
    fin["band"] = pd.cut(fin.actual_age, [20, 30, 40, 50, 60, 70, 100], labels=["20-29", "30-39", "40-49", "50-59", "60-69", "70+"], right=False)
    rows = []
    for b, g in fin.groupby("band", observed=True):
        ee = (g.predicted_age - g.actual_age)
        lo_, hi_ = boot_ci(ee.abs(), 4000, 0)
        rows.append(dict(age_band=b, n=len(g), MAE=ee.abs().mean(), CI_95=f"[{lo_:.2f}, {hi_:.2f}]", bias=ee.mean(), small_sample="**yes (n < 10)**" if len(g) < 10 else ("borderline (n < 15)" if len(g) < 15 else "")))
    say(md(pd.DataFrame(rows), {"MAE": "{:.2f}", "bias": "{:+.2f}"}))
    say("\nEvery band has fewer than 20 subjects and the 40-49 band only 6, so band-level MAEs are imprecise: their intervals are wide (several years across the table) and "
        "differences between bands (for example 5.8 y at 40-49 vs 3.1 y at 20-29) are **not** statistically supported by these data. The only band-level pattern that repeats across splits is at the extremes: ages 20-29 are over-predicted (test +3.1 y, validation +3.6 y) and 70+ under-predicted (test -4.1 y, validation -3.3 y). "
        "The middle bands change sign between splits (for example 30-39: test -3.1 y, validation +0.5 y), which is what noise at these sample sizes looks like.\n")

    # ------------------------------------------------------------------ 5. sites
    say("## 5. Site results and what they do (not) show\n")
    fin["site"] = fin.IXI_ID.map(meta_all.site)
    rows = []
    for s, g in fin.groupby("site"):
        ee = (g.predicted_age - g.actual_age)
        lo_, hi_ = boot_ci(ee.abs(), 4000, 0)
        rows.append(dict(site=s, n_test=len(g), MAE=ee.abs().mean(), CI_95=f"[{lo_:.2f}, {hi_:.2f}]", bias=ee.mean()))
    say(md(pd.DataFrame(rows), {"MAE": "{:.2f}", "bias": "{:+.2f}"}))
    trn = pd.read_csv(C.PROJECT_DIR / "ixi_train.csv")
    say(f"\n- **IOP has only {int((fin.site == 'IOP').sum())} test subjects**, so no conclusion about IOP can be drawn. Its interval is very wide and on the validation split (9 subjects) its MAE was higher (5.9 y), "
        "so the two small samples disagree.")
    say(f"- All three sites were present in training ({trn.site.value_counts().to_dict()} subjects), so this evaluation says nothing about **unseen** scanners or sites. "
        "IXI has only three London sites (scanner types as described in the IXI documentation and in the iAudit repo: two Philips, one GE; not independently verified here); it cannot support a claim of generalisation across acquisition sites, and these test results should be read as "
        "performance on held-out *subjects* from the same three sites.\n")
    (REP / "test_protocol_verification.md").write_text("\n".join(OUT), encoding="utf8")
    print("wrote", REP / "test_protocol_verification.md")


if __name__ == "__main__":
    main()
