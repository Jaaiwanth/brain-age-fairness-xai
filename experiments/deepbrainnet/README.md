# DeepBrainNet fine-tuning experiment

Transfer learning with the pretrained **DeepBrainNet** (2D VGG16 over axial slices) on the corrected IXI cohort.

| | |
|---|---|
| **Final model** | stage C (blocks 4-5 + head), selected on validation MAE |
| **Validation (n = 73)** | best MAE 4.12 y, r = 0.948 |
| **Test (n = 73, evaluated once)** | **MAE 4.53 y, 95% CI [3.86, 5.23]**, r = 0.95 |
| **Status** | complete; the test split is spent and **must not be used for further tuning** |

Related documentation: the earlier SFCN experiment ([`../sfcn/README.md`](../sfcn/README.md)), the project guide ([PROJECT_GUIDE.md](../../PROJECT_GUIDE.md)) and the [root README](../../README.md). (This folder was previously named `phase5_dbn_finetune`; that name is unrelated to "Phase 5" in the project guide.)

Fine-tunes the released DeepBrainNet (`DBN_model.h5`, Bashyam et al., 2020) on the corrected IXI data (train split, 339 subjects) and selects checkpoints on the
validation split (73 subjects). **The test split (73) was never used for training, checkpoint selection or tuning; it was evaluated once, after all checkpoints were saved
(see "Test-split result"). It must not be used to tune anything further: any new variant has to be judged on validation data only.** The original weights file is never modified (size and SHA256 are checked before and after every run).

## Design decisions

| Question | Decision | Why |
|---|---|---|
| Does the model already expose an age output? | **Yes, kept unchanged.** VGG16 conv base -> global average pooling -> Dense(1024, ReLU) -> Dropout(0.9) -> **Dense(1, linear)** = a scalar age in years | Nothing to replace. (Unlike SFCN, which needed its label range extended, this output has no range limit.) |
| Framework | Exact **PyTorch re-implementation**, loaded straight from the HDF5 file | TensorFlow on native Windows is CPU-only; fine-tuning ~27,000 slice images per epoch needs the GPU |
| Training signal | Every axial slice (80 per subject) labelled with the subject's age, loss = **Huber (delta 3 y)** on the slice prediction | As in the original DeepBrainNet training; Huber matches the MAE metric and is smooth near zero |
| Subject prediction / validation | **Median of the 80 slice predictions** (reference), subject-level MAE on the validation split, fp32 | Exactly the reference aggregation; selection uses the same quantity that is reported |
| Input | The exact reference chain (p97 x 185 scaling, slices 45-124, PIL float->RGB, JPEG round trip, 256x256 nearest), cached once as uint8 | R == G == B after the JPEG round trip, so one channel is stored losslessly; cached images equal the output of `dbn_input()` in `experiments/preprocessing/infer.py` |
| Strategy | **A** head only (526k params) -> **B** + block 5 (7.6 M) -> **C** + block 4 (13.5 M); full network (D) **not run** | Each step must justify the next; D was not needed for this evaluation |
| Learning rates | head 3e-4 (A) / 1e-4; backbone 1e-5 (B), 3e-5 (C); AdamW, weight decay 1e-4 | Conservative for pretrained weights. A probe on 2 subjects showed VGG16 (no BatchNorm) trains stably up to at least 3e-4, so 1e-5 / 3e-5 are safe, not borderline |
| Dropout | 0.9, as released | Very strong regularisation; evidence below suggests the model is under-fit rather than over-fit |
| Augmentation | left-right flip (rows), +-8 px shift, +-5% gain (A: flip only) | Label-preserving; mirror flip changed reference predictions by ~0.1 y |
| Selection / stopping | Best validation MAE (`best.pt`), early stopping (patience 15 / 6 / 8 for A / B / C); `last.pt` also saved | No test data involved |
| "Epoch" in B and C | 6,784 randomly drawn slice images (a quarter pass) | More frequent validation per unit of compute; evaluation always uses all 80 slices of all 73 subjects |

Stage A is done efficiently and exactly: the VGG base is frozen, so its 512-d pooled feature never changes; features are computed once (original and left-right flipped) and the head is
trained on them (`head_only.py`). The only difference from end-to-end head training is no shift/gain augmentation. Its checkpoint is a full DeepBrainNet and reloads to the same score
through the normal evaluation path (4.7585 vs 4.7584).

## Verification before and during training

- **PyTorch port** (`verify_port.py`): every converted tensor identical to the HDF5 file; Keras vs PyTorch outputs differ by at most 2e-5 y (fp32) on 240 real slices; subject-level
  medians match the saved reference (Keras) predictions to **0.005 y max (0.001 y mean)**; the starting validation score (MAE 5.37, bias +1.37, r 0.904) equals the zero-shot baseline computed earlier.
- **Smoke test** (`smoke_test.py`, 18/18 pass): freezing and gradients per stage, exactly the intended layers update, GPU memory and speed, tiny-run learning, validation loop,
  checkpoint round trip, early stopping, original weights unchanged, learning-rate stability probe.
- **Two bugs found and fixed by these checks**: (1) `torch.median` returns the *lower* of the two middle values for 80 slices, whereas the reference `np.median` averages them
  (differences up to 0.24 y; now `torch.quantile(..., 0.5)`); (2) validating 160 images per forward pass needed 7.6 GB and spilled out of the 6 GB GPU into shared memory (4x slower);
  evaluation now runs in chunks of 40 images (identical maths, no BatchNorm).
- Stage C was interrupted by a session end after `last.pt` was written but before its final reload check; the check was repeated manually (reload gives 4.1204, as recorded; original weights unchanged).
- Interrupted or superseded runs are kept for transparency in `checkpoints/` (`headA_endtoend_partial`, `last1B_aborted_slow_eval`, `last1B_interrupted_no_epochs`); none was used.

## Results

**Validation (n = 73, subject-level, median of 80 slices)**

| Model | MAE | bias | r | slope | best epoch |
|---|---|---|---|---|---|
| DeepBrainNet zero-shot | 5.37 | +1.37 | 0.904 | | |
| zero-shot + linear calibration (train-fitted) | 5.58 | | 0.904 | | |
| A head only | 4.76 | -2.34 | 0.935 | 0.82 | 25 |
| B + block 5 | 4.51 | -2.26 | 0.940 | 0.85 | 7 |
| **C + blocks 4-5** | **4.12** | -1.81 | **0.948** | 0.86 | 10 (early stop at 18) |

Final model = **C**, chosen as the lowest validation MAE before any test number existed.

**Test (n = 73; one-shot, `evaluate_test.py`)**

| Model | MAE (95% CI) | RMSE | bias | r | slope | within 5 y | within 10 y |
|---|---|---|---|---|---|---|---|
| **Fine-tuned DeepBrainNet (C)** | **4.5265** (3.8591-5.2319) | 5.45 | -1.28 | 0.9521 (0.9246-0.9698) | 0.87 | 62% | 96% |
| context: B / A | 4.86 / 5.20 | | | | | | |
| DeepBrainNet zero-shot | 6.3194 (5.1024-7.6744) | 8.44 | +2.85 | 0.8948 | 0.70 | 51% | 79% |
| zero-shot + linear calibration (train-fitted) | 6.2891 (5.2536-7.3919) | 7.79 | -1.28 | 0.8948 | 0.79 | 45% | 81% |
| context (not a candidate): fine-tuned SFCN, phase 4 | 4.0722 (3.4213-4.7287) | 4.99 | -0.67 | 0.9584 | 0.95 | 68% | 97% |

Percentile bootstrap over the 73 test subjects (4,000 resamples, `default_rng(0)`). Paired MAE differences, fine-tuned DeepBrainNet minus baseline (negative = better):
vs zero-shot **-1.79 y [-3.19, -0.52]**; vs calibrated zero-shot **-1.76 [-2.84, -0.82]**; vs fine-tuned SFCN **+0.45 [-0.38, +1.26]** (the interval includes zero:
**the two fine-tuned models cannot be separated on this test set**).

By age band (test, MAE; n): 20-29 3.56 (14), 30-39 3.20 (13), 40-49 4.45 (**6**), 50-59 6.48 (13), 60-69 4.01 (17), 70+ 5.99 (10; bias -6.0 y). By site: Guys 4.33 (42), HH 4.72 (22), IOP 4.98 (**9**).
By sex: F 5.06 (43), M 3.75 (30).

## Protocol evidence

The test split is read in `evaluate_test.py` only; `train.py` asserts that no test subject is in train or validation. All checkpoints were saved (10-10 18:49 to 22:19) before the test
marker was written (10-11 13:47); the marker records the final run, its epoch and its validation MAE at selection. Earlier zero-shot checks of the preprocessing pipeline used all 485 subjects
(see the root README and [`../sfcn/reports/test_protocol_verification.md`](../sfcn/reports/test_protocol_verification.md)); they involved no fitting and changed no decision.

## Relationship to the other experiments

- **SFCN experiment** ([`../sfcn/README.md`](../sfcn/README.md)): a different network (3D, 70-bin soft-label head) fine-tuned earlier. On the same test split its final model scored MAE 4.07 y; the paired difference for DeepBrainNet minus SFCN is +0.45 y (95% CI [-0.38, +1.26]), which includes zero, so the two **cannot be separated** here. The SFCN result was context only and was not used to tune or select this model.
- **Team pipeline in `scripts/`** ([PROJECT_GUIDE.md](../../PROJECT_GUIDE.md)): a separate SFCN pipeline (different preprocessing, 72 age bins) that feeds the fairness audit and Grad-CAM analysis. This experiment is independent of it and uses `experiments/preprocessing/` volumes and the shared split in [`experiments/data_splits/`](../data_splits/ixi_splits.csv).
- The test split has now been evaluated by several pipelines; none of those results should be used to choose between them (selection on validation only).

## Caveats

- Validation MAE (4.12) is the minimum over epochs and optimistically biased; the test MAE (4.53) is the less biased estimate. Validation fluctuates by about +-0.2 y between epochs.
- **The model is probably under-fit**: training-subject error (4.65 for stage B) was no better than validation error (4.51), and regression to the mean remains (slope 0.85-0.87; young over-predicted,
  70+ under-predicted). Full fine-tuning (stage D), less dropout, or longer training were not tried and might improve it; this is untested.
- The validation split is older (mean age 55.1 y) than train (47.1) and test (49.9), which explains most of the negative validation bias; it is not a training fault.
- Single seed, single split; intervals reflect only which 73 subjects were drawn. Band and site groups are small (6 to 17 subjects, 9 for IOP). IOP is not reliably better or worse: its MAE was 8.46 on the 9 validation
  subjects and 4.98 on the 9 test subjects.
- All three IXI sites were in the training data, so these results say nothing about unseen scanners or sites. No age/site-adjusted fairness analysis has been done on this model.

## Reproduce

```
python experiments/deepbrainnet/build_cache.py --workers 4            # local uint8 slice cache (~2.5 GB), verified against infer.dbn_input
python experiments/deepbrainnet/verify_port.py                        # PyTorch port vs the Keras original
python experiments/deepbrainnet/smoke_test.py                         # all checks must pass
python experiments/deepbrainnet/head_only.py --run headA               # stage A (cached features)
python experiments/deepbrainnet/train.py --run last1B --stage last1 --init experiments/deepbrainnet/checkpoints/headA/best.pt --epochs 30 --patience 6 --epoch-samples 6784
python experiments/deepbrainnet/train.py --run last2C --stage last2 --init experiments/deepbrainnet/checkpoints/last1B/best.pt --epochs 30 --patience 8 --epoch-samples 6784 --lr-backbone 3e-5
python experiments/deepbrainnet/report_val.py --runs headA:A_head_only last1B:B_+block5 last2C:C_+blocks4-5
python experiments/deepbrainnet/evaluate_test.py --final last2C --context headA last1B      # one-shot; refuses to run again without --force
```

Checkpoints are PyTorch state dicts of the 15.2 M-parameter network (`dbn_ft.DeepBrainNet`); they are not Keras files. Cache, checkpoints and logs are local-only (see `.gitignore`).
