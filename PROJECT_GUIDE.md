# Project Guide

## Project goal

Audit whether a brain-age model is fair beyond accuracy (working title: *"Right for the Wrong Reasons? Auditing Brain-Age Model Fairness Beyond Accuracy"*). We fine-tune a pretrained brain-age CNN on IXI T1 MRI, then check two things across sex and acquisition site: whether the brain-age gap (BAG = predicted − actual age) differs, and whether Grad-CAM attribution patterns differ. Full plan: `implementation.md`.

## Current status (as of 2026-10-09)

- **Phase 1 (data): done; labels corrected on 2026-10-09.** 581 IXI T1 scans; 502 subjects with a scan in the demographics index; **488 with clean official labels**. Ages 20.0–86.3. Sites × sex: Guys 167 F / 109 M, HH 83 / 68, IOP 39 / 22.
  - **The original Phase 1 labels were wrong.** The demographics file used in Phase 1 (`IXI_demographics.xls`, a third-party copy) has a `Subject_ID` column that is *not* the IXI ID, so every scan was paired with another person's age and sex. Against the official `IXI.xls`: 0/487 ages matched and sex matched at chance (242/490). The images agree with the official labels (CSF fraction vs age: r 0.64 official, 0.18 old).
  - `scripts/01b_official_labels.py` rebuilds `ixi_final_metadata.csv` from the official `IXI.xls` (http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls). The old file is kept as `ixi_final_metadata_OLD_wrong_labels.csv`.
  - Dropped from the 502: 10 with no row in `IXI.xls` (81, 88, 117, 228, 333, 337, 340, 345, 347, 457), 2 with no age (341, 435), and 2 with contradictory duplicate rows (**IXI219**: ages 53.1 and 58.1; **IXI328**: female and male). The last two are also listed in `EXCLUDED_SUBJECTS` in `scripts/04_finetune_sfcn.py` as a safety net.
  - Everything computed before 2026-10-09 with the old labels (Phase 3 zero-shot numbers, fine-tuning runs 1–3, the old "525 vs 499" explanations) is invalid.
- **Phase 2 (preprocessing): done (v2).** Preprocessing does not use labels, so the v2 volumes are unaffected by the label error.
  - **Split: 339 / 73 / 73 (train/val/test), 485 subjects**, from the shared team folder. Every age and sex matches the official `IXI.xls`; no overlap between splits; every subject has a v2 scan. It omits the 14 unusable subjects above plus IXI302, IXI386 and IXI550 (clean in `IXI.xls`, but not in this team split; kept out for consistency).
  - The earlier 349 / 75 / 75 split was stratified on the wrong sex labels; its files are kept as `ixi_*_OLD_wrong_labels.csv`. The training script refuses split files whose labels disagree with `ixi_final_metadata.csv`.
  - **v1 (`IXI_preprocessed/`, do not use):** N4, then whole-head affine registration to nilearn's skull-stripped MNI 2009 template (197×233×189), skull-strip, clip to the 1st/99th percentile, rescale to [0, 1], centre-crop to 160×192×160. An alignment check on 40 train scans found:
    - brain centres a median 5.3 mm (up to 18.3 mm) from the group median;
    - top-to-bottom brain height ranging 76–131 mm;
    - 2/40 brains cut off at the crop edge;
    - IOP worst (median offset 12.8 mm vs 4–5 mm for Guys/HH).

    Analyses on v1 would have mistaken this preprocessing artefact for an acquisition-site effect.
  - **v2 (`scripts/02b_preprocess_v2.py` → `IXI_preprocessed_v2/`):** N4, skull-strip in native space, brain-to-brain affine registration to FSL MNI152 1mm (TemplateFlow `MNI152NLin6Asym`, 182×218×182, the template SFCN used), centre-crop to 160×192×160, bias-corrected intensities kept (no clipping). Per-scan QC is written to `phase2_v2_qc.csv`.
    - Run on a laptop CPU (about 85 s per scan, 2 in parallel, ~7 h). All 502 scans, 0 failures.
    - Dice with the template brain mask: median 0.958 (min 0.941). Centre offset from the template: median 0.7 mm (max 1.9 mm), against 5.3 mm (max 18.3 mm) in v1.
    - IOP now matches the other sites (median offset 0.58 mm vs Guys 0.76, HH 0.54).
    - 36 scans flagged "cut off": a brain touches a crop face the template brain does not reach. Typically 0–1 voxels; worst IXI156, a 1 mm sliver at the back. Kept.
    - IXI457 flagged for native brain volume 2,024 mL, just over the 2,000 mL check; looks normal on inspection. Kept.
    - IXI292 (IOP) is visibly blurrier as acquired. That is an acquisition-site effect, not a preprocessing failure.
- **Phase 3 (zero-shot validation): decision SFCN stands; the original numbers are invalid** (old labels, v1 data). On v2 data with SFCN's input scaling and the official labels, the original un-fine-tuned SFCN on 30 validation subjects gives **r 0.93** (r 0.82 and MAE 4.6 yrs within its 45–80 training range). SFCN is also 3D, so Grad-CAM needs no per-slice aggregation.
- **Phase 4 (fine-tuning): done (2026-10-09).** `scripts/04_finetune_sfcn.py`, v2 volumes, official labels, 339 train / 73 val, age bins 18–90 (72 bins).
  - **Val MAE 3.90 yrs (95% CI 3.23–4.60), r 0.961 (95% CI 0.946–0.974)**, prediction sd 14.2 vs real-age sd 16.4. Best epoch 30 (stage 2); early stop at 45. Train MAE ~2.7 by the end, so a modest train/val gap.
  - Already in stage 1 (frozen backbone) val r was 0.84–0.87.
  - The checkpoint was selected on val MAE, so 3.90 is slightly optimistic; the test set gives the unbiased figure.
  - Results: `BrainAge_Project/checkpoints/sfcn_finetune_official/` (`best.pt`, `history.csv`, `val_predictions.csv`, `summary.json`).
  - Runs 1–3 (2026-10-08/09) all trained on the wrong labels and are discarded. They "memorised but did not generalise" (e.g. run 3: train MAE 6.6, val MAE 13.3 in eval mode), which is what mislabelled data produces.
  - Two genuine fixes found along the way are kept:
    - v2 preprocessing (above);
    - input scaling: SFCN's own example divides each scan by its mean over the full 182×218×182 box *before* cropping. We divided by the cropped volume's mean, so inputs were ~1.47× too small and the pretrained SFCN gave near-constant output (prediction spread 0.6 yrs vs 6.6 yrs with SFCN's scaling). The script now divides by the sum over the full-box voxel count (`SFCN_NORMALISATION_VOXELS`). Left-right mirroring (TemplateFlow stores the template RAS, FSL LAS) made no difference.
- **Phase 5 (test evaluation): done, once (2026-10-09).** `scripts/05_evaluate_test.py` on the frozen Phase 4 checkpoint (epoch 30); results in `results/phase5/`.
  - Test n = 73. Raw: MAE 3.71 yrs (95% CI 3.10–4.38), RMSE 4.66 (3.93–5.40), r 0.966 (0.952–0.978), mean BAG −0.87.
  - Age-bias correction (Cole et al. 2018) fitted on validation only: pred = 0.832·age + 8.30. Corrected: MAE 3.89 (3.20–4.61), r 0.966. The age–BAG correlation falls from −0.49 to +0.17; MAE rises slightly, as expected for this correction. Phase 6 uses the corrected BAG, with age also as a covariate.
  - Test-set cleanliness: every SFCN pipeline decision (preprocessing, input scaling, age bins, training settings, checkpoint choice, bias-correction fit) was made on train/val only. Image-quality QC covered all scans, which Rule 7 allows. This holds whatever any teammate did with the same test subjects for other models.
  - The model is now frozen. Any further change must be judged on validation only, and the test set must not be re-used to pick between versions.
- **Phase 6 (fairness audit): main analysis done (2026-10-09).** `scripts/06_fairness_audit.py` on the Phase 5 test predictions; results in `results/phase6/`. Test n = 73 (Guys 42, HH 22, IOP 9; 43 F / 30 M). Models control for age; 95% CIs from 5,000 bootstrap resamples (HC3 intervals agree).
  - **Sex: no detectable difference.** Corrected BAG, male vs female: −0.59 yrs [−2.67, +1.64]. Absolute error: −0.64 yrs [−1.90, +0.59].
  - **Acquisition-site effect at IOP.** Corrected BAG, IOP vs Guys: **−5.06 yrs [−8.66, −1.34]**, HC3 p = 0.012 (still below 0.05 after a Bonferroni correction for the three group terms). 8 of 9 IOP subjects have a negative BAG. Leaving out each IOP subject in turn gives −4.18 to −5.93, so no single subject drives it. HH vs Guys: +1.06 [−1.44, +3.53].
  - **Accuracy by group: no CI excludes 0.** Absolute error, IOP vs Guys: +1.92 yrs [−1.36, +5.14]. The model is not detectably less accurate for any group; IOP brains are predicted systematically younger.
  - Caveats: IOP n = 9, so the interval is wide. In IXI, site is confounded with the population each site recruited, so this design cannot separate scanner effects from cohort differences. It is an acquisition-site effect in the descriptive sense (Rule 2), not evidence about the people scanned (Rule 4).
  - **Leave-one-site-out (2026-10-10), `scripts/06b_loso_analysis.py`, results in `results/phase6b/`.** Three models (`--exclude-site`), each trained and early-stopped without one site. Each predicts that site's train+val subjects (IOP 52, HH 127, Guys 233; the test split is not used) and is compared with its own validation subjects from the seen sites. Bias correction is fitted on those seen-site validation subjects. Age-adjusted, 95% bootstrap CIs:
    - **IOP unseen: accuracy drops.** Absolute error +3.30 yrs [+1.62, +4.93] (MAE 6.80 vs 3.59). BAG −2.14 [−4.42, +0.20]: same direction as the main model, CI just includes 0.
    - **HH unseen: generalises.** Absolute error +0.25 [−0.88, +1.36]; BAG +0.51 [−1.34, +2.30].
    - **Guys unseen:** absolute error −0.71 [−1.86, +0.37]; BAG +3.08 [+0.95, +5.08] relative to an HH+IOP reference (n = 32, 9 of them IOP).
  - Reading across analyses: IOP is the one site a model cannot generalise to without seeing it. Its scans differ enough that accuracy falls by about 3 yrs when unseen. Including IOP in training restores accuracy (test MAE for IOP 5.19, CI overlapping the other sites) but not the offset. The IOP–Guys BAG difference has the same sign in every analysis (main model −5.06; IOP unseen −2.14; Guys unseen +3.08 against a partly-IOP reference). Consistent with an acquisition-site effect at IOP; sizes are uncertain (small reference sets) and site remains confounded with cohort.
  - Next: Phase 7 (Grad-CAM attribution by site and sex).
- **Phases 7–8 (XAI, report): not started.**

## Data location

Nothing large is stored in this repo; data lives on Google Drive under `ML_Project/`:

- Colab: `/content/drive/MyDrive/ML_Project`
- Local Windows (Drive sync, used for Phase 2): `G:\My Drive\ML_Project`

Key contents:

- `IXI_data/`: raw `.nii` scans
- `IXI_preprocessed_v2.zip`: the 502 v2 volumes (3.4 GB, lossless). Unzip on the Colab VM rather than reading 502 files from Drive.
- `IXI_preprocessed_v2/IXI###.npy`: preprocessed volumes (current), with per-scan QC JSON in `qc/`. The full folder currently exists only on the laptop that ran Phase 2 v2 (`D:\ML_Project`).
- `IXI_preprocessed/IXI###.npy`: v1 volumes, failed alignment QC; kept only for comparison
- `phase2_v2_qc.csv`, `phase2_v2_qc_worst.png`: v2 QC table and the lowest-Dice scans
- `checkpoints/sfcn_finetune[_smoke|_overfit]/`: `best.pt`, `history.csv`, `val_predictions.csv`, `summary.json`
- `ixi_final_metadata.csv`: subjects with official `IXI.xls` labels (built by `scripts/01b_official_labels.py`)
- `ixi_train.csv`, `ixi_val.csv`, `ixi_test.csv`: the splits (339 / 73 / 73)
- `IXI.xls`: official IXI demographics, the source of truth for age and sex
- `*_OLD_wrong_labels.csv`, `IXI_demographics.xls`: the mislabelled Phase 1 files; do not use
- `phase*_*.png` and `phase3_*_results.csv`: saved outputs

The `file_path` and `preprocessed_path` columns in the split CSVs hold Windows `G:\` paths. In Colab, rebuild paths from `IXI_ID` instead.

## Rules

1. **One model, trained on everyone.** Train a single model on the full training split. Never train separate models per sex or site. Split results by group only afterwards, at evaluation and analysis time.
   - Leave-one-site-out validation is a planned robustness check. It is not a per-group model.
2. **Wording for site differences.** Say "acquisition-site effects", never "hospital bias".
3. **Wording for attribution differences.** Never say "the model thinks differently" (or reasons, sees, or attends differently). Say that attribution patterns *differ between* groups or are *associated with* group membership.
4. **No clinical claims.** No diagnostic, prognostic, or patient-care conclusions. This is a methods and fairness audit on healthy volunteers.
5. **Control for age.** Any group comparison of BAG, error, or attribution must control for chronological age, for example by including age as a covariate or applying age-bias correction. Groups differ in age distribution, and BAG is age-dependent.
6. **Report confidence intervals.** Every reported metric and group difference needs a CI; bootstrap if needed. Subgroups are small (IOP test n = 9), so avoid overclaiming.
7. **Never tune on the test set.** All model selection, hyperparameters, early stopping, and bias-correction fitting use train/val only. The 73-subject test split is touched once, for final evaluation.
   - Ask the teammate who built this split whether its test subjects have been used for anything beyond zero-shot checks, and record the answer here.
   - Excluding scans for poor preprocessing quality is allowed in any split, but only on image-quality grounds (the QC metrics or a visual check). Decide without looking at age, sex, site or model predictions, and record every exclusion.
