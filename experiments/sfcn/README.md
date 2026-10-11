# SFCN fine-tuning experiment

Transfer learning with the pretrained UK Biobank **SFCN** on the corrected IXI cohort: head extension, staged fine-tuning, validation-based checkpoint selection, and one held-out test evaluation.

| | |
|---|---|
| **Final model** | stage C (full network), selected on validation MAE |
| **Validation (n = 73)** | best MAE 3.88 y, r = 0.956 |
| **Test (n = 73, evaluated once)** | **MAE 4.07 y, 95% CI [3.43, 4.76]**, r = 0.96 (exact interval in "Test-split result") |
| **Status** | complete; the test split is spent and must not be used for further tuning |

This is one of two SFCN fine-tuning lines in the repository. The other is the team's main pipeline in [`scripts/`](../../scripts/) (see [PROJECT_GUIDE.md](../../PROJECT_GUIDE.md)); the two differ in preprocessing and label bins and are **not** the same experiment (see "Relationship to the other experiments"). The DeepBrainNet experiment is documented in [`../deepbrainnet/README.md`](../deepbrainnet/README.md).

## Model and pretrained checkpoint

- **SFCN** (Peng et al., 2021, *Medical Image Analysis* 68:101871): a 3D fully convolutional network trained on UK Biobank (14,503-subject release). 2,952,936 parameters, 40 one-year age bins (42-82 y), output = probability-weighted mean of the bin centres.
- **Released weights:** `run_20190719_00_epoch_best_mae.p`, 11,827,312 bytes, SHA256 `f9197c92486a8dba7fd87bb9f2ecc4f42f7c9628b9150d1b95c8a5ac10721ad3` (MIT licence; clone instructions in the [root README](../../README.md)). Loaded with `strict=True`; the file is never modified (hash checked before and after every run). Fine-tuned checkpoints are written to separate files.

## Dataset and split

- **IXI** T1 MRI (CC BY-SA 3.0), 485 subjects with a usable label from the genuine `IXI.xls`; ages 20.0-86.3 y; sites Guys, HH and IOP.
- **Split (subject level):** 339 train / 73 validation / 73 test, stratified by site x sex (`random_state=42`); the assignment is in [`experiments/data_splits/ixi_splits.csv`](../data_splits/ixi_splits.csv) (IDs only) with the 14 excluded scans listed in [`experiments/data_splits/excluded_subjects.csv`](../data_splits/excluded_subjects.csv); provenance in [`experiments/data_splits/README.md`](../data_splits/README.md). Training reads the train split, selection uses validation, and the test split is read only by `evaluate_test.py`.

## Preprocessing and input handling

- **Volumes** come from [`experiments/preprocessing/`](../preprocessing/): N4 bias correction, affine registration (ANTs) to the FSL MNI152 linear template on SFCN's 182x218x182 LAS grid, antspynet brain mask (probability > 0.5), raw intensity scale preserved (no clipping or min-max). All 485 subjects are used (38 carry a registration-correlation QC warning; none failed hard QC).
- **SFCN input** follows the reference code: divide the volume by its mean over the full 182x218x182 box, then centre-crop to 160x192x160.
- **Training cache** (`build_cache.py`): the normalised volume cropped to 168x200x168, i.e. the SFCN crop plus a 4-voxel margin, so training can jitter the crop; its centre equals the reference crop exactly (verified against `dpu.crop_center`). Evaluation uses the centre crop. Augmentation: random +-4 voxel crop shift and random left-right flip.

## Fine-tuning stages and model selection

| Question | Decision | Why |
|---|---|---|
| Does the pretrained model already expose an age output? | **Yes, kept.** `classifier` = average pool to 64-d, dropout, 1x1x1 conv (64->40), log-softmax over 40 one-year bins; age = sum(p_i * bin_centre_i). | The expected value is differentiable, so it can be trained directly with a regression loss. No new regression head. |
| Is anything unsuitable? | **The label range.** Bins cover 42-82 y; IXI is 20-86 y. 199/485 subjects (41%) are below the floor of 42.5 y (best-case error for them 11.2 y). | A head that cannot output the target range cannot be trained to fit it. |
| Smallest change | Extend **only the last 1x1x1 conv** from 40 to **70 bins (20-90)**. Overlapping bins copy the pretrained weights exactly; new bins copy the nearest edge bin with a **-6 logit penalty**. | The starting model behaves like the pretrained one: expected age changes by at most **0.009 y**, probability on new bins 0.0002 (smoke test 1). |
| Loss | **Huber (delta = 3 y) on the expected age** (`--loss l1` and `--loss kl` available). | The metric is MAE; Huber is quadratic below 3 y (less gradient noise near convergence) and robust to outliers. Optional auxiliary KL to Gaussian soft labels (SFCN's original loss) via `--kl-weight`, off by default. |
| Strategy | **A: head only** (4,550 params) -> **B: + last two feature blocks** (1.79 M) -> **C: full network** (2.95 M). Each stage starts from the previous best checkpoint. | Controlled: each step must justify the next. |
| Learning rates | head 1e-3 / 3e-4 / 1e-4, backbone **3e-5** (B) and **1e-5** (C); AdamW, weight decay 1e-4, ReduceLROnPlateau, grad-clip 1.0 | Conservative for pretrained weights. **Measured:** with frozen BatchNorm a backbone LR of 1e-4 stalls and **3e-4 diverges** (predictions collapse to a constant); 1e-5 and 3e-5 train stably. |
| BatchNorm | running statistics **frozen**, affine parameters trainable | Batch size is 2 (x4 accumulation) on a 6 GB GPU; tiny-batch statistics would be noisy. |
| Selection / stopping | Best **validation MAE** checkpoint (`best.pt`), early stopping (patience 8 / 6 / 5 for A / B / C), `last.pt` also saved | No test data involved. The saved best checkpoint reproduces its recorded validation MAE exactly on reload. |

**Verification before the real runs** (`smoke_test.py`, all pass): head-surgery equivalence; pretrained weights copied exactly; freezing and gradients per stage (frozen parameters get no gradient, trainable ones finite non-zero gradients, BN statistics unchanged); weights change **only** in the intended layers per stage; memory and speed (peak 2.3 GB, 0.03-0.07 s/sample); loss decreases on a tiny run; the full network learns on 4 subjects (14.8 -> 8.7 MAE); validation loop and checkpoint round trip; early stopping logic; original checkpoint hash unchanged. The first version of the tiny-batch test used a backbone LR of 3e-4 and failed (training diverged), which exposed the LR sensitivity above; the test now uses the default LRs.

## Validation results (n = 73)

| Model | val MAE | bias | r | slope | best epoch |
|---|---|---|---|---|---|
| SFCN zero-shot (40 bins) | 10.85 | +8.92 | 0.891 | | |
| zero-shot + linear calibration fitted on train | 5.41 | +0.56 | 0.891 | | |
| DeepBrainNet zero-shot | 5.37 | +1.37 | 0.904 | | |
| **A** head only | 5.96 | +0.38 | 0.868 | 0.78 | 40 (hit the 40-epoch cap, still slowly improving) |
| **B** + last 2 blocks | 4.29 | -0.32 | 0.941 | 0.88 | 16 (early stop at 22) |
| **C** full network | **3.88** | -0.17 | **0.956** | 0.93 | 9 (early stop at 14) |

Stage C by subgroup (validation, small n): age 20-29 MAE 3.6 (n=8), 30-39 5.7 (8), 40-49 3.5 (10), 50-59 5.0 (9), 60-69 3.2 (26), 70+ 3.9 (12); sites Guys 3.5 (41), HH 3.8 (23), IOP 5.9 (9); sex F 3.8 (44), M 4.1 (29). Zero-shot SFCN had about 11 y error for under-30s.
Tables and curves: [`reports/stage_summary_val.csv`](reports/stage_summary_val.csv), [`reports/training_curves.png`](reports/training_curves.png).

## Test-split result (evaluated once; `evaluate_test.py`, 73 held-out subjects)

Protocol fixed beforehand: final model = stage C `best.pt` (lowest validation MAE); calibrations fitted on the train split only. The test split is now **used**; do not tune anything against it (a local marker file records the run, and re-running needs `--force`).

| Model | MAE (95% CI) | RMSE | bias | r | slope | within 5 y | within 10 y |
|---|---|---|---|---|---|---|---|
| **Fine-tuned SFCN (stage C)** | **4.07** (3.43-4.76) | 4.99 | -0.67 | **0.96** | 0.95 | 68% | 97% |
| context: stage B | 4.31 | 5.14 | -0.62 | 0.96 | 0.91 | 59% | 96% |
| context: stage A (head only) | 5.52 | 7.72 | -1.09 | 0.90 | 0.81 | 58% | 88% |
| SFCN zero-shot + linear calibration | 5.71 (4.64-6.91) | 7.51 | -1.08 | 0.90 | 0.83 | 52% | 86% |
| DeepBrainNet zero-shot | 6.32 (5.13-7.69) | 8.44 | +2.85 | 0.89 | 0.70 | 51% | 79% |
| DeepBrainNet + linear calibration | 6.29 | 7.79 | -1.28 | 0.89 | 0.79 | 45% | 81% |
| SFCN zero-shot (original) | 13.05 (10.73-15.40) | 16.73 | +11.57 | 0.90 | 0.31 | 33% | 53% |

Paired bootstrap, MAE difference of the fine-tuned model minus baseline (negative = better): vs SFCN+calibration **-1.64 y [-2.63, -0.66]**, vs DeepBrainNet **-2.25 [-3.77, -0.88]**, vs DeepBrainNet+calibration -2.22 [-3.51, -1.04], vs original SFCN -8.98 [-11.58, -6.54]. Table and figure: [`reports/test_summary.csv`](reports/test_summary.csv), [`reports/test_evaluation.png`](reports/test_evaluation.png); the full audit is in [`reports/test_protocol_verification.md`](reports/test_protocol_verification.md).

- **Exact interval:** fine-tuned MAE = **4.0722 y, 95% CI [3.4346, 4.7618]** (percentile bootstrap over the 73 test subjects, 4,000 resamples, `default_rng(0)`; three independent 20,000-resample runs give bounds within 0.02 y of these). Pearson r = 0.9584, 95% CI [0.9344, 0.9738]. These intervals reflect only which 73 subjects were drawn.
- **Same subjects:** all models (stages A-C, zero-shot and calibrated SFCN and DeepBrainNet) are scored on the identical 73 test IDs; calibrations were fitted on the 339 train rows only.
- **No test use in decisions:** checkpoints, stage progression, early stopping, learning rates and the final-model choice used train/validation data only; the test split is read only by `evaluate_test.py`, once, after all checkpoints were saved. One caveat: earlier zero-shot checks of the preprocessing pipeline used all 485 subjects (3 of 20 pilot subjects and 22 of 100 in a paired orientation test are test subjects); they involved no fitting and changed no decision, but they were not isolated from the test labels.
- **Age bands (n, MAE, 95% CI):** 20-29 (14) 3.10 [2.30, 3.90]; 30-39 (13) 4.08 [2.88, 5.40]; 40-49 (**6**) 5.79 [3.17, 8.32]; 50-59 (13) 5.26 [3.17, 7.59]; 60-69 (17) 3.33 [2.44, 4.36]; 70+ (10) 4.12 [2.34, 6.02]. Every band has fewer than 20 subjects and one has 6; the intervals overlap heavily, so differences between bands are **not** supported. Only the extremes repeat across splits (20-29 over-predicted, 70+ under-predicted).
- **Sites (test):** Guys 4.02 (n=42), HH 4.18 (n=22), IOP 4.05 (**n=9**). With nine IOP subjects (and a higher IOP MAE of 5.9 y on the nine validation subjects) nothing can be concluded about IOP. All three sites were in the training data, so these results say nothing about unseen scanners or sites.
- **Sex (test):** F 4.52 (n=43), M 3.43 (n=30), descriptive only. No age- or site-adjusted fairness analysis has been done on this model.

## Relationship to the other experiments

- **DeepBrainNet experiment** ([`../deepbrainnet/README.md`](../deepbrainnet/README.md)): a different network (2D VGG16 over 80 axial slices, scalar age output), fine-tuned and evaluated afterwards on the same test split. Its final model reached test MAE 4.53 y; the paired difference between that model and this SFCN (+0.45 y, 95% CI [-0.38, +1.26] for DeepBrainNet minus SFCN) includes zero, so the two **cannot be separated** on this test set. The SFCN numbers here are not a DeepBrainNet baseline and were not used to tune it.
- **Team pipeline in `scripts/`** ([PROJECT_GUIDE.md](../../PROJECT_GUIDE.md), Phases 4-7): a separate SFCN fine-tuning line (`scripts/04_finetune_sfcn.py`) on a different preprocessing ("v2", `scripts/02b_preprocess_v2.py`) with 72 age bins (18-90), which feeds the fairness audit, leave-one-site-out analysis and Grad-CAM attribution. The guide reports its test MAE as 3.71 y (95% CI 3.10-4.38), r = 0.966. This experiment is independent of it: different preprocessing, 70 bins (20-90), different code. Both describe the shared 339/73/73 split; I have not verified that the assignments are identical.
- **The test split has now been evaluated by several pipelines** (the team's SFCN, this SFCN, DeepBrainNet). Each result is reported separately; none of them should be used to choose between pipelines. Any such choice must be made on validation data only (PROJECT_GUIDE rule 7).

## Known limitations

- Validation MAE fluctuates by about +-0.3 y between epochs (n = 73) and the reported best is the minimum over epochs, so it is **optimistically biased**; a realistic level is about 4.3-4.5 for B and 3.9-4.2 for C. The B -> C gain (0.4 y validation, 0.24 y test) is of the same order as this noise. The less biased figure is the test result.
- Head-only (A) did not beat the 5.41 calibration baseline; the gain comes from unfreezing the last blocks (B), then the rest (C).
- Single seed, single split; variability over seeds or cross-validation was not assessed. Subgroups are small (6 to 17 subjects per age band; 9 from IOP) and descriptive only.
- All three IXI sites were in training, so the results do not show generalisation to unseen scanners or sites.
- The 40-bin to 70-bin extension and the preprocessing differ from the original UK Biobank pipeline (UK Biobank's exact FSL registration and masking cannot be reproduced here), so comparison with the published UK Biobank accuracy is not meaningful.

## Reproduce

Run from the repository root; large artefacts (cache, checkpoints, logs, per-subject predictions) are local-only and listed in `.gitignore`.

```
python experiments/sfcn/build_cache.py                 # local cache (SFCN /mean normalisation, 168x200x168 crops), ~6 GB
python experiments/sfcn/smoke_test.py                  # all checks must pass
python experiments/sfcn/train.py --run headA --stage head    --init pretrained --epochs 40 --patience 8
python experiments/sfcn/train.py --run partB --stage partial --init experiments/sfcn/checkpoints/headA/best.pt --epochs 25 --patience 6 --lr-head 3e-4 --lr-backbone 3e-5
python experiments/sfcn/train.py --run fullC --stage full    --init experiments/sfcn/checkpoints/partB/best.pt --epochs 20 --patience 5 --lr-head 1e-4 --lr-backbone 1e-5
python experiments/sfcn/report_val.py                  # curves + validation tables
python experiments/sfcn/evaluate_test.py               # one-shot; refuses to run again without --force
python experiments/sfcn/verify_test_protocol.py        # audit of the test protocol (reads saved artefacts only)
```

Checkpoints are written to `experiments/sfcn/checkpoints/<run>/best.pt` (also copied to `ML_Project/phase4_finetune/` on Drive; that folder name predates this reorganisation). They hold a 70-bin SFCN state dict (`sfcn_ft.SFCN(output_dim=70)`, bins 20-90) plus configuration and validation metrics.
