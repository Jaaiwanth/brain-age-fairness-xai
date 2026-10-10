# Transfer learning with the pretrained SFCN (corrected IXI cohort)

Fine-tunes the released UK Biobank SFCN (`run_20190719_00_epoch_best_mae.p`) on the corrected IXI data (train split, 339 subjects),
selects checkpoints on the validation split (73 subjects) only. **The test split (73) was never used for training, checkpoint selection or any tuning decision;
it was evaluated once, after all checkpoints were saved (see "Test-split result" at the end).**
The original checkpoint is never modified (sha256 checked before and after every run).

## Design decisions

| Question | Decision | Why |
|---|---|---|
| Does the pretrained model already expose an age output? | **Yes, keep it.** `classifier` = avg-pool to 64-d, dropout, 1x1x1 conv (64->40), log-softmax over 40 one-year bins; age = sum(p_i * bin_centre_i). | The expected value is differentiable, so it can be trained directly with a regression loss. No new regression head needed. |
| Is anything unsuitable? | **The label range.** Bins cover 42-82 y; IXI is 20-86 y. 199/485 subjects (41%) are below the floor of 42.5 y (best-case error for them 11.2 y). | A head that cannot output the target range cannot be trained to fit it. |
| Smallest change | Extend **only the last 1x1x1 conv** from 40 to **70 bins (20-90)**. Overlapping bins copy the pretrained weights exactly; new bins copy the nearest edge bin with a **-6 logit penalty**. | Starting model behaves like the pretrained one: expected age changes by at most **0.009 y**, probability on new bins 0.0002 (smoke test 1). Training then raises new bins where the data demand it. |
| Loss | **Huber (delta = 3 y) on the expected age** (`--loss l1` and `--loss kl` also available). | Evaluation metric is MAE, so an absolute-error loss matches it; Huber is quadratic below 3 y, which removes the gradient noise of pure L1 near convergence and is robust to outliers. Optional auxiliary KL to Gaussian soft labels (SFCN's original loss) via `--kl-weight`, off by default. |
| Strategy | **A: head only** (4,550 params) -> **B: + last two feature blocks** (1.79 M) -> **C: full network** (2.95 M). Each stage starts from the previous best checkpoint. | Controlled: each step must justify the next. |
| Learning rates | head 1e-3 / 3e-4 / 1e-4, backbone **3e-5** (B) and **1e-5** (C), AdamW, weight decay 1e-4, ReduceLROnPlateau, grad-clip 1.0 | Conservative for pretrained weights. **Measured:** with frozen BatchNorm a backbone LR of 1e-4 stalls and **3e-4 diverges** (predictions collapse to a constant); 1e-5 and 3e-5 train stably. |
| BatchNorm | running statistics **frozen**, affine parameters trainable | Batch size is 2 (x4 accumulation) on a 6 GB GPU; tiny-batch statistics would be noisy. |
| Augmentation | random +-4 voxel crop jitter, random left-right flip | Mirror flip changed predictions by ~0.2 y in earlier tests, so it is label-preserving; jitter reduces memorisation of 339 volumes. |
| Selection / stopping | Best **validation MAE** checkpoint (`best.pt`), early stopping with patience (6 for B, 5 for C, 8 for A), `last.pt` also saved | No test data involved. Reload check: the saved best checkpoint reproduces its recorded validation MAE exactly. |

## Verification before the real runs (`smoke_test.py`, all pass)

Head surgery equivalence; pretrained weights copied exactly; freezing and gradients per stage (frozen parameters get no gradient, trainable ones
finite non-zero gradients, BN statistics unchanged); weights change **only** in the intended layers per stage; memory/speed (peak 2.3 GB, 0.03-0.07 s/sample);
loss decreases on a tiny run; full network learns on 4 subjects (14.8 -> 8.7 MAE); validation loop and checkpoint round trip; early stopping logic
(zero learning rate must stop at `patience`); original checkpoint hash unchanged.
The first version of the tiny-batch test used a backbone LR of 3e-4 and failed (training diverged). That exposed the LR sensitivity above; the test now uses the default LRs.

## Results (validation split, n = 73)

| Model | val MAE | bias | r | slope | epoch |
|---|---|---|---|---|---|
| SFCN zero-shot (40 bins) | 10.85 | +8.92 | 0.891 | | |
| zero-shot + linear calibration fitted on train | 5.41 | +0.56 | 0.891 | | |
| DeepBrainNet zero-shot | 5.37 | +1.37 | 0.904 | | |
| **A** head only | 5.96 | +0.38 | 0.868 | 0.78 | 40 (hit the 40-epoch cap, still slowly improving) |
| **B** + last 2 blocks | 4.29 | -0.32 | 0.941 | 0.88 | 16 (early stop at 22) |
| **C** full network | **3.88** | -0.17 | **0.956** | 0.93 | 9 (early stop at 14) |

Stage C by subgroup (validation, small n): age 20-29 MAE 3.6 (n=8), 30-39 5.7 (8), 40-49 3.5 (10), 50-59 5.0 (9), 60-69 3.2 (26), 70+ 3.9 (12);
sites Guys 3.5 (41), HH 3.8 (23), IOP 5.9 (9); sex F 3.8 (44), M 4.1 (29). Zero-shot SFCN had ~11 y error for under-30s.

## Caveats

- Validation MAE fluctuates by about +-0.3 y between epochs (n = 73), and the reported best is the minimum over epochs, so it is **optimistically biased**.
  A realistic level for B is ~4.3-4.5 and for C ~3.9-4.2. The B -> C gain (0.4 y) is of the same order as this noise. The less biased number is the **test split**, which was evaluated once afterwards (see "Test-split result" at the end).
- Head-only (A) did not beat the 5.41 calibration baseline; the gain comes from unfreezing the last blocks (B), then the rest (C).
- IOP (9 validation subjects) is still the weakest site; subgroup numbers here are descriptive and too small for conclusions.
- Single seed, single split. Variability over seeds / cross-validation not assessed.

## Reproduce

```
python phase4_finetune/build_cache.py                 # local cache (SFCN /mean normalisation, 168x200x168 crops), ~6 GB
python phase4_finetune/smoke_test.py                  # all checks must pass
python phase4_finetune/train.py --run headA --stage head    --init pretrained --epochs 40 --patience 8
python phase4_finetune/train.py --run partB --stage partial --init phase4_finetune/checkpoints/headA/best.pt --epochs 25 --patience 6 --lr-head 3e-4 --lr-backbone 3e-5
python phase4_finetune/train.py --run fullC --stage full    --init phase4_finetune/checkpoints/partB/best.pt --epochs 20 --patience 5 --lr-head 1e-4 --lr-backbone 1e-5
python phase4_finetune/report_val.py                  # curves + validation tables
```

Checkpoints: `phase4_finetune/checkpoints/<run>/best.pt` (also copied to `ML_Project/phase4_finetune/` on Drive). They hold a 70-bin SFCN state dict
(`ft.SFCN(output_dim=70)`, bins 20-90) plus config and validation metrics.

## Test-split result (evaluated once; `evaluate_test.py`, 73 held-out subjects)

Protocol fixed beforehand: final model = stage C `best.pt` (lowest validation MAE); calibrations fitted on the train split only.
The test split is now **used**; do not tune anything against it (`reports/test_evaluated.json` records this, re-running needs `--force`).

| Model | MAE (95% CI) | RMSE | bias | r | slope | within 5 y | within 10 y |
|---|---|---|---|---|---|---|---|
| **Fine-tuned SFCN (stage C)** | **4.07** (3.43-4.76) | 4.99 | -0.67 | **0.96** | 0.95 | 68% | 97% |
| context: stage B | 4.31 | 5.14 | -0.62 | 0.96 | 0.91 | 59% | 96% |
| context: stage A (head only) | 5.52 | 7.72 | -1.09 | 0.90 | 0.81 | 58% | 88% |
| SFCN zero-shot + linear calibration | 5.71 (4.64-6.91) | 7.51 | -1.08 | 0.90 | 0.83 | 52% | 86% |
| DeepBrainNet zero-shot | 6.32 (5.13-7.69) | 8.44 | +2.85 | 0.89 | 0.70 | 51% | 79% |
| DeepBrainNet + linear calibration | 6.29 | 7.79 | -1.28 | 0.89 | 0.79 | 45% | 81% |
| SFCN zero-shot (original) | 13.05 (10.73-15.40) | 16.73 | +11.57 | 0.90 | 0.31 | 33% | 53% |

Paired bootstrap, MAE difference of the fine-tuned model minus baseline (negative = better): vs SFCN+calibration **-1.64 y [-2.63, -0.66]**,
vs DeepBrainNet **-2.25 [-3.77, -0.88]**, vs DeepBrainNet+calibration -2.22 [-3.51, -1.04], vs original SFCN -8.98 [-11.58, -6.54].

- Test MAE (4.07) is close to the validation estimate (3.88; "realistic level ~4.0"), so validation-based selection did not overfit the 73 validation subjects much.
- C vs B on test (4.07 vs 4.31) is within noise; the clear gains are zero-shot -> calibrated -> A -> B/C.
- **Exact interval:** fine-tuned MAE = **4.0722 y, 95% CI [3.4346, 4.7618]** (percentile bootstrap over the 73 test subjects, 4000 resamples, `default_rng(0)`; three
  independent 20,000-resample runs give lower/upper bounds within 0.02 y of these). Pearson r = 0.9584, 95% CI [0.9344, 0.9738]. These intervals reflect only which 73 subjects were drawn.
- **Same subjects:** all models (stages A-C, zero-shot and calibrated SFCN and DeepBrainNet) are scored on the identical 73 IDs of `ixi_test.csv`, verified in `reports/test_protocol_verification.md`.
  Calibrations were fitted on the 339 train rows only.
- **No test use in decisions:** checkpoints, stage progression, early stopping, learning rates and the final-model choice used train/validation data only; the test split is read only by
  `evaluate_test.py`, once, after all checkpoints were saved (timestamps in the verification report). One caveat: earlier zero-shot checks of the preprocessing pipeline used all 485 subjects (3 of 20 pilot
  subjects and 22 of 100 in a paired orientation test are test subjects); they involved no fitting and changed no decision, but they were not isolated from the test labels.
- **Age bands (n, MAE, 95% CI):** 20-29 (14) 3.10 [2.30, 3.90]; 30-39 (13) 4.08 [2.88, 5.40]; 40-49 (**6**) 5.79 [3.17, 8.32]; 50-59 (13) 5.26 [3.17, 7.59]; 60-69 (17) 3.33 [2.44, 4.36]; 70+ (10) 4.12 [2.34, 6.02].
  Every band has <20 subjects, one has 6, and the intervals overlap heavily, so differences between bands are **not** supported. Only the extremes repeat across splits (20-29 over-predicted, 70+ under-predicted).
- **Sites (test):** Guys 4.02 (n=42), HH 4.18 (n=22), IOP 4.05 (**n=9**). With nine IOP subjects (and a different, higher IOP MAE of 5.9 y on the nine validation subjects) nothing can be concluded about IOP.
  All three sites were in the training data, so these results say nothing about unseen scanners or sites and do **not** show generalisation across acquisition sites; they describe held-out subjects from the same three sites.
- Sex (test): F 4.52 (n=43), M 3.43 (n=30), descriptive only. No age/site-adjusted fairness analysis has been done on this model.
- Single seed, single split.
