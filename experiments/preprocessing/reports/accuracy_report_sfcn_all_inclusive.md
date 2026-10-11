# Accuracy report: pretrained SFCN on corrected IXI (sfcn_all_inclusive)

*Model:* SFCN `run_20190719_00_epoch_best_mae.p` (UK Biobank, Peng et al. 2021); output = expected value over 40 one-year bins, so predictions are limited to 42.5-81.5 y, **pretrained, no fine-tuning**, inference only.
*Input:* corrected preprocessing (FSL MNI152 grid, antspynet brain mask, reference slice chain; see `phase2_corrected/`).
*Labels:* rebuilt from the genuine `IXI.xls` joined on `IXI_ID` (verified identical to iAudit's XNAT-verified ages on 482/482 shared subjects).
*Predictions file:* `sfcn_predictions_all.csv` | *Scope:* all predicted subjects (QC = PASS or WARN)

## 1. Headline

| Metric | Value |
|---|---|
| Subjects evaluated | **485** |
| **MAE** | **13.86 y** (95% bootstrap CI 12.92-14.82) |
| RMSE | 17.35 y |
| Median absolute error | 11.36 y |
| Mean error (bias, predicted - actual) | +12.67 y |
| Within 5 y / within 10 y | 28% / 46% |
| **Pearson r** | **0.893** (95% CI 0.874-0.910) |
| Spearman rho | 0.896 |
| R^2 | 0.798 |
| Slope of predicted on actual (1.0 = ideal) | 0.308 (intercept 46.4) |
| MAE after linear age-bias correction* | 2.05 y |

\*Correction `gap ~ age` fitted on this same sample (the standard approach, but optimistic; it removes the regression-to-the-mean trend).

### SFCN accuracy inside and outside its output range

SFCN can only output 42.5-81.5 y (40 one-year bins, `bin_range = [42, 82]` in the official example), so the fair test is subjects aged 42-82. Subjects under 42 cannot be predicted correctly by construction; their error is a property of the output design.

| Subset | n | MAE | RMSE | bias | Pearson r | Spearman |
|---|---|---|---|---|---|---|
| all subjects | 485 | 13.86 | 17.35 | +12.67 | 0.893 | 0.896 |
| **age 42-82 (in range)** | 286 | 6.28 | 7.69 | +4.54 | 0.827 | 0.826 |
| age < 42 (out of range) | 196 | 24.93 | 25.58 | +24.93 | 0.333 | 0.331 |

For comparison, DeepBrainNet on the same in-range subjects: see `accuracy_report_all_inclusive.md`.



![accuracy](accuracy_sfcn_all_inclusive.png)

## 2. Cohort accounting (nothing silently dropped)

*QC policy note:* `reg_r` (T1-vs-template correlation) was demoted from a hard failure to a warning on 2026-10-09 after two of the first 25 scans
failed on it alone while their mask Dice was 0.93-0.95; no predictions for those scans had been seen. A strict (PASS-only) report is provided alongside.

- Scans in the raw folder index: **499**
- Excluded: no usable label (see `ixi_label_excluded.csv`): **14**
- Labelled subjects: **485**
- Failed hard preprocessing QC (not predicted): **0**
- Predicted with a QC warning (`reg_r` < 0.60 but mask Dice >= 0.80) - included: **38**
- Not yet preprocessed / no QC record: **0**
- **Predicted and evaluated**: 485

No subject failed hard preprocessing QC (every labelled subject was predicted).

## 3. Accuracy by age band

|  | n | MAE | RMSE | ME | within 5 y | within 10 y | r |
|---|---|---|---|---|---|---|---|
| 20-29 | 88 | 29.87 | 30.03 | +29.87 | 0% | 0% | 0.18 |
| 30-39 | 86 | 22.01 | 22.21 | +22.01 | 0% | 0% | 0.29 |
| 40-49 | 73 | 13.92 | 14.12 | +13.92 | 0% | 4% | 0.54 |
| 50-59 | 77 | 7.16 | 7.69 | +7.11 | 23% | 87% | 0.45 |
| 60-69 | 109 | 3.25 | 3.89 | +2.29 | 79% | 100% | 0.45 |
| 70+ | 51 | 4.75 | 6.37 | -4.48 | 63% | 90% | 0.13 |

Young subjects are over-predicted (mean error +24.0 y for < 45 y, n=215) and the oldest slightly under-predicted
(+0.1 y for >= 60 y, n=160): the usual regression-to-the-mean pattern of age regressors, visible as slope 0.31.

## 4. Accuracy by site and sex (descriptive)

|  | n | MAE | RMSE | ME | within 5 y | within 10 y | r |
|---|---|---|---|---|---|---|---|
| Guys | 275 | 12.05 | 15.31 | +11.19 | 32% | 53% | 0.93 |
| HH | 149 | 14.93 | 18.78 | +14.08 | 28% | 45% | 0.92 |
| IOP | 61 | 19.39 | 21.77 | +15.86 | 10% | 21% | 0.78 |

|  | n | MAE | RMSE | ME | within 5 y | within 10 y | r |
|---|---|---|---|---|---|---|---|
| Female | 287 | 12.98 | 16.61 | +11.50 | 30% | 51% | 0.88 |
| Male | 198 | 15.13 | 18.36 | +14.36 | 25% | 40% | 0.92 |

These are **unadjusted** group means. Sites and sexes differ in age mix, and error depends strongly on age, so group gaps here are not
evidence of bias. Significance testing and age/site adjustment belong to the separate fairness analysis.

## 5. Accuracy by split (all zero-shot, so splits are only a consistency check)

|  | n | MAE | RMSE | ME | within 5 y | within 10 y | r |
|---|---|---|---|---|---|---|---|
| test | 73 | 13.05 | 16.73 | +11.57 | 33% | 53% | 0.90 |
| train | 339 | 14.68 | 18.02 | +13.71 | 24% | 42% | 0.89 |
| val | 73 | 10.85 | 14.51 | +8.92 | 44% | 62% | 0.89 |

## 6. Ten largest errors

| IXI_ID | site | sex | actual_age | predicted_age | error |
|---|---|---|---|---|---|
| 276 | HH | Male | 20.9 | 57.8 | 36.9 |
| 230 | IOP | Female | 21.2 | 57.4 | 36.3 |
| 425 | IOP | Female | 20.0 | 56.2 | 36.2 |
| 70 | Guys | Female | 20.7 | 55.7 | 35.0 |
| 21 | Guys | Female | 21.6 | 56.4 | 34.9 |
| 202 | HH | Male | 20.2 | 54.5 | 34.3 |
| 80 | HH | Female | 21.2 | 55.5 | 34.3 |
| 278 | HH | Male | 20.2 | 54.4 | 34.2 |
| 426 | IOP | Female | 23.1 | 57.2 | 34.1 |
| 201 | HH | Female | 22.4 | 56.2 | 33.8 |

## 7. Interpretation and limitations

- The pretrained model tracks age on corrected labels (r = 0.89), confirming the earlier "failure" was a label bug, not a model problem.
- SFCN's overall MAE is dominated by subjects under 42, which lie outside its 42-82 output range; judge it on the in-range table above.
- Errors are not uniform across age: see section 3. A flat MAE would hide this.
- Zero-shot: no domain adaptation to IXI scanners (Guys/HH Philips, IOP GE), which is expected to affect IOP most.
- Selection: subjects without usable labels or failing QC are excluded (section 2); results describe the evaluated subset.
- Age-bias correction is fitted in-sample, so the corrected MAE is optimistic.
- Single model; the preprocessing differs from antspynet's and from the original authors', so absolute numbers are not directly comparable.
