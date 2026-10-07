# Project Guide

## Project goal

Audit whether a brain-age model is fair beyond accuracy (working title: *"Right for the Wrong Reasons? Auditing Brain-Age Model Fairness Beyond Accuracy"*). We fine-tune a pretrained brain-age CNN on IXI T1 MRI, then check two things across sex and acquisition site: whether the brain-age gap (BAG = predicted − actual age) differs, and whether Grad-CAM attribution patterns differ. Full plan: `implementation.md`.

## Current status (as of 2026-10-07)

- **Phase 1 (data): done.** 581 IXI T1 scans; 499 matched to demographics. Ages 20.0–86.3. Sites: Guys 282, HH 150, IOP 67. `implementation.md` still says 525 subjects and 26 dropped; the notebook output says 499. Trust the notebook.
- **Phase 2 (preprocessing): v1 failed QC; v2 being run.**
  - Split 349 / 75 / 75 (train/val/test), stratified by site × sex. The split is unchanged by the re-run.
  - **v1 (`IXI_preprocessed/`, do not use):** N4, then whole-head affine registration to nilearn's skull-stripped MNI 2009 template (197×233×189), skull-strip, clip to the 1st/99th percentile, rescale to [0, 1], centre-crop to 160×192×160. An alignment check on 40 train scans found:
    - brain centres a median 5.3 mm (up to 18.3 mm) from the group median;
    - top-to-bottom brain height ranging 76–131 mm;
    - 2/40 brains cut off at the crop edge;
    - IOP worst (median offset 12.8 mm vs 4–5 mm for Guys/HH).

    Analyses on v1 would have mistaken this preprocessing artefact for an acquisition-site effect.
  - **v2 (`scripts/02b_preprocess_v2.py` → `IXI_preprocessed_v2/`):** N4, skull-strip in native space, brain-to-brain affine registration to FSL MNI152 1mm (TemplateFlow `MNI152NLin6Asym`, 182×218×182, the template SFCN used), centre-crop to 160×192×160, bias-corrected intensities kept (no clipping). Per-scan QC is written to `phase2_v2_qc.csv`.
- **Phase 3 (zero-shot validation): done. Decision: SFCN.** Same 8 test subjects for both models:
  - SFCN: MAE 16.29 yrs, Pearson r −0.01. This underestimates the model. The bin range was UK Biobank's `[42, 82]`, so it couldn't predict below 42, and the v1 input didn't match SFCN's preprocessing.
  - DeepBrainNet: MAE 13.53 yrs, Pearson r 0.36 (p ≈ 0.38).
  - Both compress predictions into ~50–68 yrs. n = 8 is too small to be conclusive.
  - SFCN is kept because its failure was largely caused by fixable mismatches (the 42–82 bin range, which fine-tuning replaces with 20–90, and preprocessing). It is also 3D, so Grad-CAM needs no per-slice aggregation.
- **Phase 4 (fine-tuning): first run failed, waiting on v2 data.** `scripts/04_finetune_sfcn.py` on v1 data:
  - Val MAE 14.73 yrs (95% CI 12.87–16.54), r 0.29.
  - Predictions spread only 2.7 yrs (sd) against a real-age sd of 17.4. In effect the model guessed the mean (guessing the train mean gives MAE 15.3).
  - Causes: v1 misalignment, plus too-cautious stage-2 settings, which have since been raised.
  - Next: `--overfit` check, then the full run on v2.
- **Phases 5–8 (evaluation, fairness audit, XAI, report): not started.**

## Data location

Nothing large is stored in this repo; data lives on Google Drive under `ML_Project/`:

- Colab: `/content/drive/MyDrive/ML_Project`
- Local Windows (Drive sync, used for Phase 2): `G:\My Drive\ML_Project`

Key contents:

- `IXI_data/`: raw `.nii` scans
- `IXI_preprocessed_v2/IXI###.npy`: preprocessed volumes (current), with per-scan QC JSON in `qc/`
- `IXI_preprocessed/IXI###.npy`: v1 volumes, failed alignment QC; kept only for comparison
- `phase2_v2_qc.csv`, `phase2_v2_qc_worst.png`: v2 QC table and the lowest-Dice scans
- `checkpoints/sfcn_finetune[_smoke|_overfit]/`: `best.pt`, `history.csv`, `val_predictions.csv`, `summary.json`
- `ixi_final_metadata.csv`: matched subjects
- `ixi_train.csv`, `ixi_val.csv`, `ixi_test.csv`: the splits
- `IXI_demographics.xls`
- `phase*_*.png` and `phase3_*_results.csv`: saved outputs

The `file_path` and `preprocessed_path` columns in the split CSVs hold Windows `G:\` paths. In Colab, rebuild paths from `IXI_ID` instead.

## Rules

1. **One model, trained on everyone.** Train a single model on the full training split. Never train separate models per sex or site. Split results by group only afterwards, at evaluation and analysis time.
   - Leave-one-site-out validation is a planned robustness check. It is not a per-group model.
2. **Wording for site differences.** Say "acquisition-site effects", never "hospital bias".
3. **Wording for attribution differences.** Never say "the model thinks differently" (or reasons, sees, or attends differently). Say that attribution patterns *differ between* groups or are *associated with* group membership.
4. **No clinical claims.** No diagnostic, prognostic, or patient-care conclusions. This is a methods and fairness audit on healthy volunteers.
5. **Control for age.** Any group comparison of BAG, error, or attribution must control for chronological age, for example by including age as a covariate or applying age-bias correction. Groups differ in age distribution, and BAG is age-dependent.
6. **Report confidence intervals.** Every reported metric and group difference needs a CI; bootstrap if needed. Subgroups are small (IOP test n = 10), so avoid overclaiming.
7. **Never tune on the test set.** All model selection, hyperparameters, early stopping, and bias-correction fitting use train/val only. The 75-subject test split is touched once, for final evaluation.
   - The Phase 3 zero-shot check drew 8 subjects from the test split. Don't use test data for any further tuning or preprocessing decisions.
   - Excluding scans for poor preprocessing quality is allowed in any split, but only on image-quality grounds (the QC metrics or a visual check). Decide without looking at age, sex, site or model predictions, and record every exclusion.
