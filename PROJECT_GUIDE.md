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
- **Phase 3 (zero-shot validation): redone 2026-10-11 with the official labels, `scripts/03b_zero_shot.py`, results in `results/phase3/`.** The original UK Biobank SFCN (40 bins, ages 42–82), un-fine-tuned, on v2 data and the full validation split (n = 73), 95% bootstrap CIs:
  - Raw: MAE 10.21 [8.19, 12.29], r 0.910 [0.869, 0.943], mean BAG +8.33. Its outputs cannot leave 42–82, so everyone under ~45 is predicted 51–61.
  - Ages 42–82 only (n = 51): MAE 5.32 [4.25, 6.41], r 0.842.
  - With a linear correction fitted on the train split (pred = 0.340·age + 44.19): MAE 6.85 [5.80, 7.94].
  - Fine-tuned Phase 4 model, same subjects: MAE 3.91 [3.25, 4.60], r 0.961 [0.945, 0.974]. Fine-tuning roughly halves the error compared with the best zero-shot option.
  - Decision unchanged: SFCN. It is already strongly age-related zero-shot, and it is 3D, so Grad-CAM needs no per-slice aggregation. The old Phase 3 numbers (wrong labels, v1 data) must not be reported.
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
- **Phase 7 (attribution): done (2026-10-10).** `scripts/07_gradcam.py` on the frozen Phase 4 checkpoint; results in `results/phase7/` (also on Drive). It took 15 min on a laptop CPU for all 485 subjects.
  - Method: gradient × activation at `conv_2` (20×24×20), |Σ_c g·A| (HiResCAM-style), target = predicted age (expected value over bins). Maps are upsampled to 160×192×160, masked to the brain, and normalised to sum 1 so each map is a share of attribution. Standard Grad-CAM (channel-averaged gradients with ReLU) gave empty maps for 20–80% of subjects. The switch was made for that reason alone, before any group result was seen.
  - Sanity check (Adebayo et al. 2018): re-initialising conv_3 to conv_6 changes the maps (median Spearman r 0.13 vs trained, n = 8). The maps therefore depend on the learned weights.
  - Regions: Harvard-Oxford (TemplateFlow MNI152NLin6Asym, th25) on the same grid as the scans. 48 cortical regions plus subcortical structures (left/right merged) give 59 fine regions in 7 coarse groups. Brain voxels with no atlas label are counted as "Cerebellum and unlabelled".
  - Group analysis: held-out subjects only (val + test, n = 146; Guys 83, HH 45, IOP 18; 87 F / 59 M). Model: share ~ sex + site + age, HC3, Benjamini-Hochberg FDR per term. Fine regions are tested only if their mean share is above 0.2%.
  - Across all groups, the most attribution falls on deep grey matter around the lateral ventricles (thalamus, caudate, putamen), the periventricular white matter and the cerebellum.
  - **Sex: no region differs** (coarse or fine, all FDR q > 0.05).
  - **Site: attribution patterns differ between acquisition sites.** Compared with Guys, HH and IOP scans place less attribution on deep grey matter (HH −1.14, IOP −1.29 percentage points) and on the hippocampus/amygdala (−0.25, −0.28), and more on the cerebellum/unlabelled class (HH +2.66; IOP +1.98, q = 0.07). IOP also places less on the brainstem (−0.90). 43 fine region × site terms pass FDR; most are small (< 0.5 pp).
  - Reading: attribution patterns are associated with acquisition site, not with sex. This matches Phase 6, where BAG and accuracy differed by site but not by sex. The design cannot say whether the differences come from scanner/protocol differences in the images or from the cohorts recruited at each site. The finding is not evidence about the people scanned (Rules 3 and 4).
- **Phase 9a (site probe, "right for the wrong reasons?"): done (2026-10-13).** `scripts/08_site_probe.py`, results in `results/phase9a_site_probe/` (also on Drive). It took 43 min on a laptop CPU.
  - Question: the model was trained only on age. Does its 64-d internal summary (global-average-pooled features, which the age head reads) also carry the acquisition site?
  - Method:
    - A linear probe (multinomial logistic regression, class-balanced; C chosen by 5-fold CV on train) is fitted on the 339 train subjects and scored on the 146 held-out subjects (val + test).
    - Metric: balanced accuracy, chance 33.3%; 95% bootstrap CIs.
    - "Age/sex removed" means age, age² and sex are regressed out of the features first (fit on train).
    - The age model is not changed, and test subjects are used only to score the probe.
  - Results:
    - Age + sex only: 40.8% [31.2, 50.9]. Who was recruited explains almost none of the site.
    - Brain-intensity quantiles of the preprocessed input: 99.6% [98.7, 100]. Site remains visible in the images after N4, skull-stripping and registration.
    - Pretrained SFCN, never trained on IXI: 92.8% [87.2, 96.9].
    - Fine-tuned SFCN: 93.5% [88.1, 97.7].
    - Fine-tuned SFCN, age/sex removed: 91.2% [86.6, 95.2]. All 18 held-out IOP subjects are identified, and no Guys or HH subject is called IOP; Guys and HH are partly confused.
  - Reading:
    - Site information is strongly present in the representation the age prediction is read from, even after removing age and sex.
    - It is already there in the pretrained model, so it comes from image appearance, not from fine-tuning on IXI.
    - IOP is the most distinct site, the same site that shows the brain-age-gap offset and the leave-one-site-out accuracy drop.
    - This shows the information is **present**. It does not show the age prediction **uses** it. That needs an intervention (acquisition perturbation or harmonisation; next steps).
  - Exploratory link to BAG: uninformative. The probe's P(IOP) is ~0 or ~1 for nearly everyone, so it adds nothing beyond the site label (overall Spearman ρ −0.16, within-site p = 0.96).
- **Phase 9b (acquisition stress test): done (2026-10-13).** `scripts/09_acquisition_stress_test.py`, results in `results/phase9b_stress_test/` (also on Drive). It took 25 min on a laptop CPU.
  - Question: Phase 9a showed site is readable from the features. Does the age prediction respond to acquisition appearance when the anatomy is unchanged?
  - Method:
    - The frozen Phase 4 model was run on the 146 held-out subjects. Nothing was fitted to them.
    - Site intensity swap: each brain's intensity distribution was quantile-matched to a target site's reference (101 quantiles).
    - The references were built from train subjects and standardised to a common age and sex mix (per-site regression of quantiles on age, age² and sex). This is needed because IOP's train subjects are younger (mean 40.6 vs 48.4 Guys, 47.3 HH).
    - Effects are compared with the own-site match, which uses the same procedure.
    - Single-property sweeps on 40 subjects: contrast, noise, blur and smooth non-uniformity.
    - Outcome: paired change in raw predicted age, with 95% bootstrap CIs.
  - Results (age-standardised references; change vs own-site match):
    - HH → IOP look: −3.92 yrs [−4.47, −3.38]; 45 of 45 younger.
    - IOP → HH look: +3.14 [+2.44, +3.97]; 18 of 18 older.
    - Guys → HH look: +3.31 (matrix); HH → Guys look: −2.87.
    - Guys → IOP look: +0.11 [−0.18, +0.39]. IOP → Guys look: +0.90 [+0.17, +1.70].
    - Sweeps: blur σ 1.5 voxels −3.17 [−5.11, −1.24]; contrast γ 0.8 −0.92 [−1.27, −0.58]; 5% noise +0.36; ±10% non-uniformity −0.22 / +0.35.
  - Reading:
    - **The age prediction responds to acquisition appearance.** Changing only the intensity distribution to another site's moves predicted age by 3–4 years, in the same direction for every subject. Image smoothness also shifts it by about 3 years.
    - **Global intensity does not explain the IOP offset.** Between Guys and IOP the appearance swap moves predictions by about 1 year or less, far below the −5.06-year corrected-BAG offset. The offset must come from other image properties (e.g. spatial resolution or smoothness) or from cohort differences.
    - The HH appearance is the one the model is most sensitive to, but HH subjects themselves show no BAG offset (Phase 6). This is consistent with the model having learned HH's appearance together with its anatomy.
  - Without age standardisation (`results/phase9b_stress_test_unadjusted/`), Guys → IOP was −0.60 and IOP → Guys +1.60. Part of the apparent IOP look effect was IOP's younger recruitment, which shows why the standardisation was needed.
- **Phase 9c (image sharpness by site): done (2026-10-13).** `scripts/10_image_quality_by_site.py`, results in `results/phase9c_image_quality/` (also on Drive). It took 5 min on CPU for all 485 subjects.
  - Native resolution is the **same** at every site: 0.94 × 0.94 × 1.2 mm (matrix 256×256×150; 146 slices at IOP). Resolution therefore does not explain any site effect.
  - Effective sharpness of the model input differs strongly. Age- and sex-adjusted, HC3, compared with Guys:
    - IOP is much blurrier: gradient sharpness −2.09 SD, Laplacian sharpness −2.21 SD, smoothness FWHM +0.87 mm (+2.38 SD); all p < 1e-100. IOP barely overlaps the other sites on the scatter plots.
    - HH is slightly softer than Guys on gradient sharpness (−0.89 SD) and nearly identical on the other two measures.
    - So, with the same nominal voxel size, IOP images are smoother, presumably from scanner reconstruction or filtering. Phase 9b showed that blurring lowers predicted age (σ 1.5 voxels: −3.17 yrs).
  - Regression adjustment cannot separate sharpness from site on held-out subjects. IOP and sharpness are almost collinear, so adding a sharpness covariate makes the IOP effect swing between +1.9 and −10 yrs with very wide CIs. The sharpness measures also change with age, in opposite directions, because they partly measure anatomy. This adjustment is therefore inconclusive and is not reported as mediation.
  - Reading: IOP is the blurriest site by a wide margin, and blur lowers predicted age. That makes image smoothness a plausible acquisition-related contributor to the IOP offset, but not yet shown to be one. The direct test is to blur Guys scans to IOP's measured smoothness and measure the age shift.
- **Phase 9d (blur to IOP smoothness): done (2026-10-13).** `scripts/11_blur_to_iop.py`, results in `results/phase9d_blur_to_iop/` (also on Drive). It took 38 min on CPU.
  - Method:
    - Each held-out Guys (83) and HH (45) brain was Gaussian-blurred, with σ chosen per subject so that its smoothness FWHM rises by its site's age/sex-adjusted gap to IOP (+0.87 mm Guys, +0.82 mm HH). The median σ was 0.61 voxels.
    - The frozen model predicted age on the original, half-dose and matched volumes. Changes are expressed on the corrected-BAG scale (÷ 0.832).
  - The blur matched IOP's FWHM (5.55 vs IOP 5.47 mm). It did **not** reproduce IOP's other sharpness measures: gradient 2.53 vs IOP 2.00, Laplacian 0.49 vs IOP 0.59. IOP's softer appearance is therefore not simple Gaussian smoothing.
  - Result: IOP-level blur does **not** lower predicted age.
    - Guys: +0.93 yrs [+0.60, +1.28], slightly older.
    - HH: +0.01 [−0.32, +0.36].
    - Half dose: ~0 for both.
    - The test offset is −5.06, so this smoothness reproduces none of it. The −3.17-yr shift in Phase 9b came from much stronger blur (σ 1.5).
  - Reading across Phase 9:
    - Site is readable from the model's features (9a).
    - Predicted age does respond to acquisition appearance, by 3–4 yrs for intensity-distribution swaps, mostly towards HH's appearance (9b).
    - Yet the two most obvious acquisition explanations for the IOP offset are ruled out at IOP's measured levels: global intensity distribution (≤ ~1 yr, 9b) and Gaussian smoothness (0 or opposite sign, 9d).
    - The IOP offset therefore comes from acquisition properties not tested here (e.g. the spatial pattern of tissue contrast, scanner-specific reconstruction or distortion), from cohort differences, or from both. IXI cannot separate these.
- **Phase 9e (feature harmonisation): done (2026-10-14).** `scripts/12_harmonise_features.py`, results in `results/phase9e_harmonisation/` (also on Drive). It takes about 1 min: it uses the cached Phase 9a features and needs no model run.
  - Method:
    - The 64-d pooled features were harmonised with ComBat (location/scale, no empirical Bayes), preserving age, age² and sex. Site parameters came from train subjects only.
    - Age was re-predicted through the model's own, unchanged age head; re-applying the head to the original features reproduces the model's predictions exactly.
    - Bias correction was refitted on val for each version. Held-out subjects (val + test, n = 146); bootstrap CIs.
  - Results, original → harmonised:
    - Site probe (age/sex removed): **91.2% → 36.3%** (chance 33%). The fingerprint is removed.
    - MAE: 3.81 → 3.90 held-out, 3.71 → 3.79 test-only; r 0.965 → 0.962. Accuracy is essentially unchanged.
    - Corrected BAG, IOP vs Guys: **−4.53 [−7.04, −2.01] → −3.95 [−6.40, −1.37]**, so the offset largely persists. HH vs Guys: −0.69 → −1.02. Male vs female: −1.28 → −1.48 (both CIs include 0).
  - The offset appears only in subjects the model did not train on. Using the Phase 5 correction, IOP vs Guys BAG (age- and sex-adjusted) is:
    - train −0.97 [−3.09, +1.15], with train MAE 1.5–2.4, so the model fits its training subjects closely;
    - val −4.08 [−7.31, −0.85];
    - test −5.06 [−8.99, −1.13].

    So the offset **replicates independently in val and test**. It is a failure to generalise to new IOP individuals, not something present in the fitted data. ComBat estimated on train therefore has no offset to remove.
  - Reading: removing the site-identifying shifts from the features costs almost no accuracy and leaves the IOP offset mostly intact. So the offset is not carried by the features' site fingerprint. It lies along the age-relevant feature directions, for unseen IOP subjects. Together with Phases 9b and 9d, the IOP offset is not explained by global intensity, by Gaussian smoothness, or by the linear site signature in the features. Cohort differences or acquisition properties not tested remain possible; IXI cannot separate them.
  - Note: a first implementation had a sign/intercept error (double-subtracted grand mean) that produced nonsense (MAE 14). It was caught by checking that harmonised features keep their means, and fixed before any result was used.
- **Phase 9f (predictive uncertainty): done (2026-10-14).** `scripts/13_uncertainty.py`, results in `results/phase9f_uncertainty/` (also on Drive). It took about 10 min on CPU.
  - Measure: the spread of SFCN's own predicted age distribution (predictive SD), plus whether the true age lies inside its 5th–95th percentile range (90% coverage). Raw predictions; 5,000 bootstrap resamples.
  - Main model, held-out (n = 146):
    - Overall coverage is 88%. The spread is wide relative to the error (mean |error|/SD ≈ 0.6–0.7), so the ranges are conservative rather than over-confident.
    - IOP: predictive SD 8.06 vs Guys 5.66 and HH 6.39; IOP vs Guys, age/sex-adjusted, **+2.48 yrs [+1.30, +3.65]**. IOP coverage is 83% [67, 100]; Guys 87%, HH 91%.
    - Per subject, uncertainty only weakly tracks error (Spearman +0.16, p = 0.057).
  - Leave-one-site-out models (same subjects as Phase 6b; re-runs reproduce the saved predictions):
    - **IOP unseen:** SD 12.46 vs 6.13 for seen sites, **+5.32 yrs [+4.65, +5.99]**. Coverage is 92% [85, 98] despite MAE 6.80. The model is less accurate but says so: it is **not confidently wrong**.
    - **HH unseen:** SD 9.36 vs 8.70, +0.79 [−0.08, +1.67]; coverage 93%. HH generalises (Phase 6b), and uncertainty barely rises.
  - Reading: the model's own uncertainty rises at the site where it is least reliable (IOP), most of all when IOP was never seen, and its ranges keep roughly nominal coverage there. At group level it flags the acquisition-site problem. At the individual level the signal is weak, so it is not a per-scan error detector.
- **Phase 9g (attribution deletion test): done (2026-10-14).** `scripts/14_attribution_deletion_test.py`, results in `results/phase9g_deletion_test/` (also on Drive). It took 21 min on CPU.
  - Method:
    - 60 held-out subjects (fixed random sample: Guys 32, HH 18, IOP 10).
    - Brain cells of the Phase 7 maps (8 mm) were ranked by attribution.
    - The top, bottom or random 5 / 10 / 20% of brain cells were replaced by a heavily smoothed copy of the image (σ = 8 voxels), so every condition applies the same perturbation and only its location differs. Random removal used two draws.
    - Outcome: |change in predicted age|.
  - Results, |change| in yrs (top / random / bottom):
    - 5%: **4.83 / 1.04 / 0.46**
    - 10%: 6.05 / 1.89 / 0.70
    - 20%: 8.71 / 4.17 / 1.21
  - Top minus random: +3.79 [+3.03, +4.66] at 5%, Wilcoxon p = 5e-11, larger for 93% of subjects. At 10% and 20%: +4.16 and +4.54, p ≤ 3e-9.
  - Random minus bottom is also positive at every fraction (p ≤ 5e-10).
  - Reading: the maps pass the deletion test. The regions they rank highest matter most to the prediction and the lowest-ranked matter least. Together with the randomisation sanity check (Phase 7), the attribution maps are faithful to the model at their 8 mm resolution.
- **Phase 9h (website): built (2026-10-14).** `scripts/15_build_demo.py` writes `demo/index.html`, a self-contained offline page with story chapters, interactive charts and a brain explorer. All numbers are read from the saved results.
- **Phase 9i (live demo): built (2026-10-21).** `demo/live_demo.py` (Gradio; launcher `demo/run_live_demo.bat`) runs the fine-tuned model live on any val/test brain: prediction with its age distribution, attribution map with a live deletion test, scanner-look swap (Phase 9b references, own-site look as control) with optional blur, and the site probe (refitted on train features from Phase 9a). Checked against saved results: features match the Phase 9a cache, and IXI559 gives 59.1 → 55.2 with an IOP look, as in Phase 9b.
- **Phase 8 (report): draft written (2026-10-14)**, `report/report_draft.md`. Still to do: team contributions, and the team's review and edits.

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
