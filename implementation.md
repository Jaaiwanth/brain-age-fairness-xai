# Implementation Plan — Brain Age Fairness Project (Updated)

**Working title:** *"Right for the Wrong Reasons? Auditing Brain-Age Model Fairness Beyond Accuracy"*
**Suggested repo name:** `brain-age-fairness`

---

## 1. Primary Pretrained Model — SFCN (Simple Fully Convolutional Network)

**Repository:** https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain
**Paper:** Peng, H. et al. (2021). "Accurate brain age prediction with lightweight deep neural networks." *Medical Image Analysis*, 68, 101871.

### Why this model
- Standard baseline architecture in brain-age literature — cited and re-implemented across many published papers (SFCNR follow-up work, PAC 2019 challenge winner).
- Pretrained on **14,503 UK Biobank T1-weighted MRI scans**.
- Lightweight — fits comfortably on Colab's T4 GPU (16GB).
- Reported validation MAE ≈ 2.2–2.9 years on UK Biobank / PAC 2019.
- Weight file available directly in the repo: `run_20190719_00_epoch_best_mae.p`

### Note on provenance
This repo is a small personal GitHub account and can look unpolished, but it is the **official code release tied to a peer-reviewed paper** in a respected journal. This is common for academic code — provenance was verified via the paper citation, not just the repo's appearance.

### Architecture summary
- Fully convolutional 3D CNN
- Predicts age as a **soft label** (probability distribution over discretized age bins), then takes the expected value as the final predicted age
- Expects input registered to **MNI152 1mm standard space**, center-cropped to **160 × 192 × 160**

---

## 2. Fallback Pretrained Model — DeepBrainNet

**Repository:** https://github.com/vishnubashyam/DeepBrainNet
**Paper:** Bashyam, V.M. et al. (2020). "MRI signatures of brain age and disease over the lifespan based on a deep brain network and 14,468 individuals worldwide." *Brain*, 143(7), 2312–2324.

### Why this is a strong fallback
- Trained on an even larger and more diverse cohort — **11,729 to 14,468 individuals** across multiple studies, scanners, ages, ethnicities, and geographic locations worldwide (sources vary slightly by paper version, but both figures come from the same official release).
- Published in *Brain* — a top-tier, highly-cited neurology journal, giving it strong provenance credibility.
- Multiple architecture variants provided as ready-to-use `.h5` weight files: **InceptionResNetV2, DenseNet169, ResNet50, VGG16** — all hosted directly in the repo via Git LFS.
- Explicitly validated for **multi-site, multi-scanner generalization** — directly relevant to our site-effect analysis, since the authors specifically designed it to generalize across different scanners.
- Independently cited and reused across many follow-up studies (e.g., pain/function studies, Alzheimer's studies), showing real-world community trust beyond the original authors.
- The original paper itself shows the pretrained age-prediction weights work better as a transfer-learning starting point than ImageNet weights, particularly when the target dataset is small — directly supporting our use case with IXI's 525 subjects.

### Trade-off to be aware of
- DeepBrainNet is a **2D CNN**, not 3D — it processes each scan as a stack of 80 axial slices and takes the median prediction across slices, rather than operating on the full 3D volume at once like SFCN does.
- This means our Grad-CAM analysis would need to be adapted to work per-slice and then aggregated, rather than producing a single 3D heatmap directly. This is a well-established approach in the literature (used in the DeepBrainNet papers themselves) but adds an extra aggregation step to Phase 7.
- Preprocessing requirement: skull-stripping + bias correction + linear registration to MNI152 (via the ANTs pipeline, as used by the original authors).

### Decision rule
We use **SFCN by default**. We switch to **DeepBrainNet** only if the SFCN validation step (Section 3) shows the model is broken, unreasonably inaccurate, or otherwise untrustworthy in practice.

---

## 3. Validation Step — Test Before Committing *(current stage)*

Before fine-tuning or building the rest of the pipeline, we run a **zero-shot sanity check**:

1. Load SFCN pretrained weights (no fine-tuning yet)
2. Preprocess a handful of our real IXI scans to match SFCN's expected input (MNI152 registration, skull-strip, normalize, crop to 160×192×160)
3. Run inference — get predicted age for each test scan
4. Compare predicted age vs. real chronological age (from our metadata)
5. Compute MAE on this small sample

**Interpreting the result:**

| Outcome | Action |
|---|---|
| Model loads correctly, predictions are in a plausible range and roughly track real age | Proceed with SFCN, move to full fine-tuning pipeline |
| Model fails to load, crashes, or produces nonsensical predictions | Switch to DeepBrainNet as the pretrained backbone instead |

---

## 4. Dataset — IXI (confirmed appropriate, already in use)

**Source:** https://brain-development.org/ixi-dataset/ (official site blocks Colab downloads; using Kaggle mirror `kbacon/ixi-t1` for scans + a Google Drive–hosted demographics file for real age/sex data)

| Property | Value |
|---|---|
| Subjects (scans + real demographics merged) | 525 |
| Modalities | T1-weighted structural MRI |
| Sites | Guy's Hospital (295), Hammersmith (156), Institute of Psychiatry (74) |
| Age range | 20.0 – 86.3 years |
| Sex | 291 Female, 234 Male |
| License | CC BY-SA 3.0 — must cite source |
| Registry status | RRID: SCR_005839 — actively cited in 2024–2026 literature, no alerts |

Hospital/site information is embedded directly in each scan's filename (e.g., `IXI002-Guys-0828-T1.nii`), so no extra lookup is needed — this is already indexed as part of our Phase 1 data pipeline.

---

## 5. Full Pipeline (Phases)

```
PHASE 1 — Setup & Data                         [in progress]
  → IXI scans + real demographics (525 subjects)
  → Resolving file-corruption issue in scan downloads

PHASE 2 — Preprocessing (pretrained-model-compatible)
  → Register to MNI152, skull-strip, normalize, crop
  → Train/Val/Test split, stratified by sex + site

PHASE 3 — Pretrained Model Validation            [current stage]
  → Zero-shot test of SFCN on a handful of real IXI scans
  → Decide: proceed with SFCN, or switch to DeepBrainNet

PHASE 4 — Model Setup & Fine-Tuning
  → Load validated pretrained weights
  → Freeze early layers, fine-tune later layers + head on IXI
  → Track MAE on validation set

PHASE 5 — Evaluation
  → Test set MAE / RMSE (overall)

PHASE 6 — Bias / Fairness Audit
  → BAG = Predicted − Actual age
  → Compare across sex, site; age-controlled statistical comparison
  → Leave-one-site-out validation

PHASE 7 — Explainability (XAI)
  → Grad-CAM on fine-tuned model (3D if SFCN, per-slice
    aggregated if DeepBrainNet)
  → Atlas-based ROI attribution scoring
  → Statistical comparison of attribution patterns across subgroups

PHASE 8 — Results & Report
```

---

## 6. Citations to Include in Your Report

- Peng, H., Gong, W., Beckmann, C.F., Vedaldi, A., Smith, S.M. (2021). Accurate brain age prediction with lightweight deep neural networks. *Medical Image Analysis*, 68, 101871.
- (If DeepBrainNet used) Bashyam, V.M. et al. (2020). MRI signatures of brain age and disease over the lifespan based on a deep brain network and 14,468 individuals worldwide. *Brain*, 143(7), 2312–2324.
- IXI Dataset: https://brain-development.org/ixi-dataset/ (CC BY-SA 3.0)
- UK Biobank: Sudlow, C. et al. (2015). UK Biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age. *PLoS Medicine*, 12(3), e1001779.

---

## 7. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Pretrained model fails zero-shot validation | Switch to DeepBrainNet (Section 2), documented decision |
| Registration/preprocessing mismatch hurts fine-tuning | Follow exact preprocessing steps from the chosen model's example code |
| DeepBrainNet's 2D-slice architecture complicates 3D Grad-CAM | Aggregate per-slice Grad-CAM maps into a 3D volume, following precedent set in the DeepBrainNet literature |
| Small IXI subgroup sizes (esp. IOP, 74 subjects) limit statistical power | Report confidence intervals, use bootstrapping, avoid overclaiming |
| Domain gap between pretraining cohort and IXI | Treated as a feature — informs the cross-site generalization analysis, not just a limitation |
