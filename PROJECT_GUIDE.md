# Project Guide

## Project goal

Audit whether a brain-age model is fair beyond accuracy (working title: *"Right for the Wrong Reasons? Auditing Brain-Age Model Fairness Beyond Accuracy"*). We fine-tune a pretrained brain-age CNN on IXI T1 MRI, then check two things across sex and acquisition site: whether the brain-age gap (BAG = predicted − actual age) differs, and whether Grad-CAM attribution patterns differ. Full plan: `implementation.md`.

## Current status (as of 2026-10-07)

- **Phase 1 (data): done.** 581 IXI T1 scans; 499 matched to demographics. Ages 20.0–86.3. Sites: Guys 282, HH 150, IOP 67. `implementation.md` still says 525 subjects and 26 dropped; the notebook output says 499. Trust the notebook.
- **Phase 2 (preprocessing): done.** 499/499 scans processed: N4 bias correction, affine registration to MNI152 1mm, skull-strip, clip to the 1st/99th percentile, rescale to [0, 1], crop to 160×192×160. Split 349 / 75 / 75 (train/val/test), stratified by site × sex.
- **Phase 3 (zero-shot validation): run, no decision recorded yet.** Same 8 test subjects for both models:
  - SFCN: MAE 16.29 yrs, Pearson r −0.01. This underestimates the model. The bin range is still UK Biobank's `[42, 82]`, so it can't predict below 42, and our intensity normalisation differs from what SFCN expects.
  - DeepBrainNet: MAE 13.53 yrs, Pearson r 0.36 (p ≈ 0.38).
  - Both compress predictions into ~50–68 yrs. n = 8 is too small to be conclusive.
- **Phases 4–8 (fine-tuning, evaluation, fairness audit, XAI, report): not started.**

## Data location

Nothing large is stored in this repo; data lives on Google Drive under `ML_Project/`:

- Colab: `/content/drive/MyDrive/ML_Project`
- Local Windows (Drive sync, used for Phase 2): `G:\My Drive\ML_Project`

Key contents:

- `IXI_data/`: raw `.nii` scans
- `IXI_preprocessed/IXI###.npy`: preprocessed volumes
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
