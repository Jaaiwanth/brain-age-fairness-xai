# Test-split protocol verification

## 1. Every model was scored on the same 73 test subjects

| source | n | identical_to_split_file | duplicates |
|---|---|---|---|
| split file `ixi_test.csv` | 73 | True | False |
| fine-tuned SFCN (stage C) predictions | 73 | True | False |
| zero-shot SFCN (rows with split = test) | 73 | True | False |
| zero-shot DeepBrainNet (rows with split = test) | 73 | True | False |

- Actual ages in all three prediction files equal the corrected metadata for these 73 subjects: **True**.
- Calibrated baselines: slope/intercept fitted on the **339 train** rows only; no test or validation rows.
- `evaluate_test.py` contains `ids = final.IXI_ID.to_numpy()`: **True**
- `evaluate_test.py` contains `d.set_index("IXI_ID").loc[ids].predicted_age`: **True**
- `evaluate_test.py` contains `te = d.set_index("IXI_ID").loc[ids]`: **True**
- `evaluate_test.py` contains `assert (te.split == "test").all() and np.allclose(te.actual_age, age)`: **True**
- Stages A and B and the fine-tuned model were all scored through the same `DataLoader` over the 73 test volumes in one call, and every baseline vector is indexed by that same ID list, so the paired comparisons are subject-for-subject. Row counts in `test_summary.csv`: 73, 73, 73, 73, 73, 73, 73.

## 2. No test result was used for checkpoints, hyperparameters or preprocessing

**Code audit: where the test split can be read**

| file | line | code |
|---|---|---|
| build_cache.py | 28 | for s in ("train", "val", "test"): |
| evaluate_test.py | 70 | assert idx[idx.split == "test"].shape[0] == 73 |
| evaluate_test.py | 71 | tl = torch.utils.data.DataLoader(ft.CachedData("test", train=False), batch_size=4, num_workers=0) |
| evaluate_test.py | 76 | final["split"] = "test" |
| evaluate_test.py | 91 | assert (te.split == "test").all() and np.allclose(te.actual_age, age) |
| train.py | 90 | test_ids = set(pd.read_csv(ft.CACHE / "index.csv").query("split == 'test'").IXI_ID) |
| train.py | 91 | assert not (set(tr.idx.IXI_ID) | set(va.idx.IXI_ID)) & test_ids, "test subjects leaked into train/val" |
| verify_test_protocol.py | 45 | test_ids = pd.read_csv(C.PROJECT_DIR / "ixi_test.csv").IXI_ID.astype(int) |
| verify_test_protocol.py | 47 | sets = {"split file `ixi_test.csv`": set(test_ids), "fine-tuned SFCN (stage C) predictions": set(fin.IXI_ID)} |
| verify_test_protocol.py | 51 | te = d[d.split == "test"] |
| verify_test_protocol.py | 63 | for pat in ('ids = final.IXI_ID.to_numpy()', 'd.set_index("IXI_ID").loc[ids].predicted_age', 'te = d.set_index |
| verify_test_protocol.py | 75 | if re.search(r"""['"]test['"]|test_ids""", line): |
| verify_test_protocol.py | 109 | f"the full-cohort accuracy reports) drew on all 485 subjects, so test subjects' labels were seen there: {int(p |
| verify_test_protocol.py | 110 | f"{int(pr.isin(test_ids).sum())} of the 100 subjects in the paired orientation test are test-split subjects. T |

`train.py` only references the test split in an assertion that **no test subject is in train or validation**; `sfcn_ft.py` merely defines the dataset class (split name is an argument); `CachedData("test")` is only ever instantiated in `evaluate_test.py`.

**Timeline: checkpoints were finished before the test evaluation**

| file | modified | before_test_evaluation |
|---|---|---|
| headA/best.pt | 2026-10-10 01:32 | True |
| headA/last.pt | 2026-10-10 01:32 | True |
| headA/history.csv | 2026-10-10 01:32 | True |
| partB/best.pt | 2026-10-10 12:18 | True |
| partB/last.pt | 2026-10-10 12:21 | True |
| partB/history.csv | 2026-10-10 12:21 | True |
| fullC/best.pt | 2026-10-10 12:27 | True |
| fullC/last.pt | 2026-10-10 12:30 | True |
| fullC/history.csv | 2026-10-10 12:30 | True |
| reports/test_evaluated.json (test run) | 2026-10-10 13:30 | - |

- The marker records the selected checkpoint (`checkpoints/fullC/best.pt`, epoch 9) and its **validation** MAE at selection (3.88). The test split was evaluated once; re-running requires `--force`.
- All `history.csv` files contain only train/validation columns (val_MAE, val_RMSE, val_bias, val_r, val_slope); selection used `val_MAE`.

**Decision log: what data each choice used**

| decision | data used | test labels/results used? |
|---|---|---|
| Checkpoint selection (best.pt) | validation MAE | no |
| Stage progression A -> B -> C | validation MAE (B justified by beating A and the calibration baseline on validation) | no |
| Early stopping | validation MAE | no |
| Learning rates (head 1e-3/3e-4/1e-4, backbone 3e-5/1e-5) | a-priori 'conservative' values; LR sensitivity probed on 4 train + 4 val subjects only | no |
| Head extension 20-90 y, new-bin penalty -6, Huber delta 3, augmentation | fixed a priori from the architecture and the age range; surgery checked on 6 validation subjects | no |
| Final model = stage C | lowest validation MAE of the three stages (3.88 vs 4.29 vs 5.96) | no |
| Baseline calibrations | fitted on train rows only | no |
| Input preprocessing for fine-tuning | SFCN reference code (divide by mean, centre crop); cache built identically for all splits | no |

**Caveat that cannot be removed (earlier, zero-shot work):** the earlier checks of the *preprocessing pipeline* (pilot subjects, input-orientation tests, intensity variants, the full-cohort accuracy reports) drew on all 485 subjects, so test subjects' labels were seen there: 3 of the 20 pilot subjects and 22 of the 100 subjects in the paired orientation test are test-split subjects. These analyses involved no fitting, and **no preprocessing or model decision was changed because of them** (the pipeline was frozen before fine-tuning; the only later change, demoting the `reg_r` QC check to a warning, used QC metrics, not labels or predictions). This is a documented, not a strictly isolated, separation for the preprocessing stage. For the fine-tuning stage the separation is strict (train/validation only).

## 3. Exact 95% bootstrap confidence intervals

Method: percentile bootstrap, resampling the 73 test subjects with replacement, 4000 resamples, `numpy.random.default_rng(0)` (the settings used in `evaluate_test.py`).

- **Fine-tuned SFCN (stage C) MAE = 4.0722 y, 95% CI [3.4346, 4.7618]**; value stored in `test_summary.csv`: 4.0722 [3.4346, 4.7618] (reproduced exactly: **True**).
- Pearson r = 0.9584, 95% CI [0.9344, 0.9738] (Fisher z, n = 73).
- Monte-Carlo stability: three independent runs of 20,000 resamples give [3.421, 4.750]; [3.434, 4.757]; [3.438, 4.747], i.e. the interval is stable to about 0.02 y.
- Baselines (same method and settings):

| model | n | MAE | MAE_lo | MAE_hi |
|---|---|---|---|---|
| FINAL: fine-tuned SFCN (stage C) | 73 | 4.0722 | 3.4346 | 4.7618 |
| context: stage A (head only) | 73 | 5.5152 | 4.3281 | 6.7989 |
| context: stage B (+2 blocks) | 73 | 4.3145 | 3.6917 | 4.9860 |
| baseline: SFCN zero-shot | 73 | 13.0488 | 10.7298 | 15.4013 |
| baseline: SFCN zero-shot + linear calibration (train-fitted) | 73 | 5.7108 | 4.6432 | 6.9129 |
| baseline: DeepBrainNet zero-shot | 73 | 6.3194 | 5.1334 | 7.6921 |
| baseline: DeepBrainNet zero-shot + linear calibration (train-fitted) | 73 | 6.2891 | 5.2993 | 7.4187 |

- Paired differences (fine-tuned minus baseline, MAE in years, resampling the same subjects for both models; 4000 resamples): vs calibrated SFCN -1.64 [-2.63, -0.66]; vs DeepBrainNet -2.25 [-3.77, -0.88]; vs calibrated DeepBrainNet -2.22 [-3.51, -1.04]; vs original SFCN -8.98 [-11.58, -6.54] (from `evaluate_test.py` output).
- These intervals capture **only the random choice of 73 subjects**. They do not include training randomness (one seed), the choice of split, or differences between scanners.

## 4. Age-band sample sizes and uncertainty (fine-tuned model, test)

| age_band | n | MAE | CI_95 | bias | small_sample |
|---|---|---|---|---|---|
| 20-29 | 14 | 3.10 | [2.30, 3.90] | +3.10 | borderline (n < 15) |
| 30-39 | 13 | 4.08 | [2.88, 5.40] | -3.11 | borderline (n < 15) |
| 40-49 | 6 | 5.79 | [3.17, 8.32] | -4.58 | **yes (n < 10)** |
| 50-59 | 13 | 5.26 | [3.17, 7.59] | +0.22 | borderline (n < 15) |
| 60-69 | 17 | 3.33 | [2.44, 4.36] | +0.80 |  |
| 70+ | 10 | 4.12 | [2.34, 6.02] | -4.09 | borderline (n < 15) |

Every band has fewer than 20 subjects and the 40-49 band only 6, so band-level MAEs are imprecise: their intervals are wide (several years across the table) and differences between bands (for example 5.8 y at 40-49 vs 3.1 y at 20-29) are **not** statistically supported by these data. The only band-level pattern that repeats across splits is at the extremes: ages 20-29 are over-predicted (test +3.1 y, validation +3.6 y) and 70+ under-predicted (test -4.1 y, validation -3.3 y). The middle bands change sign between splits (for example 30-39: test -3.1 y, validation +0.5 y), which is what noise at these sample sizes looks like.

## 5. Site results and what they do (not) show

| site | n_test | MAE | CI_95 | bias |
|---|---|---|---|---|
| Guys | 42 | 4.02 | [3.15, 4.96] | -1.33 |
| HH | 22 | 4.18 | [2.98, 5.45] | +0.92 |
| IOP | 9 | 4.05 | [2.69, 5.25] | -1.51 |

- **IOP has only 9 test subjects**, so no conclusion about IOP can be drawn. Its interval is very wide and on the validation split (9 subjects) its MAE was higher (5.9 y), so the two small samples disagree.
- All three sites were present in training ({'Guys': 192, 'HH': 104, 'IOP': 43} subjects), so this evaluation says nothing about **unseen** scanners or sites. IXI has only three London sites (scanner types as described in the IXI documentation and in the iAudit repo: two Philips, one GE; not independently verified here); it cannot support a claim of generalisation across acquisition sites, and these test results should be read as performance on held-out *subjects* from the same three sites.
