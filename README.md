# Brain-age prediction and fairness audit on IXI

Working title: *"Right for the Wrong Reasons? Auditing Brain-Age Model Fairness Beyond Accuracy"* (see `implementation.md` for the original plan).

This repository applies two pretrained brain-age networks to T1-weighted MRI from the [IXI dataset](https://brain-development.org/ixi-dataset/):
**SFCN** (3D CNN, UK Biobank; Peng et al., 2021) and **DeepBrainNet** (2D CNN over axial slices; Bashyam et al., 2020).
It rebuilds the preprocessing to match each model's reference input, corrects an earlier label error, and fine-tunes SFCN on IXI.

## Status and results

| Stage | Result |
|---|---|
| Zero-shot DeepBrainNet, all 485 labelled subjects | MAE 6.50 y, r = 0.877 (`phase2_corrected/reports/accuracy_report_all_inclusive.md`) |
| Zero-shot SFCN | MAE 13.86 y overall, 6.28 y on ages 42-82 (its output range); see `phase2_corrected/reports/accuracy_report_sfcn_all_inclusive.md` |
| Fine-tuned SFCN, stage B (last two blocks), validation | best MAE 4.29 y |
| Fine-tuned SFCN, stage C (full network), validation | best MAE 3.88 y |
| **Fine-tuned SFCN, stage C, held-out test (73 subjects)** | **MAE 4.07 y, 95% CI [3.43, 4.76]**, r = 0.96 |

Details, baselines and confidence intervals are in `phase4_finetune/README.md` and `phase4_finetune/reports/test_protocol_verification.md`.

**Read these caveats with the numbers**
- Checkpoints were selected on validation MAE; the test split was evaluated once, afterwards.
- Earlier zero-shot checks of the *preprocessing pipeline* used all 485 subjects (3 of 20 pilot subjects and 22 of 100 in an orientation test are test-split subjects). They involved no fitting and changed no decision, but they were not isolated from test labels.
- Single seed and single split; the intervals reflect only which 73 subjects were drawn. Age bands and sites have small groups (for example 6 test subjects aged 40-49, 9 from IOP), so subgroup numbers are descriptive only.
- IXI has three sites and all appear in training: the results do **not** show generalisation to unseen scanners or sites.
- The fairness analysis (age/site-adjusted tests) has **not** been repeated on the corrected labels and fine-tuned model. Earlier fairness outputs were removed because they came from mis-aligned labels (`analysis/README.md`).

## An important correction

An early version of this project attached age and sex to the wrong subjects (a row-index column had been used as the IXI ID), which made correct models look broken.
`CASE_STUDY_iAudit_vs_ml_brain.md` documents how it was found (by comparison with the public iAudit repository), the evidence, and the fix.
Labels are now rebuilt from the genuine `IXI.xls` joined on `IXI_ID` (`phase2_corrected/label_fix/build_labels.py`).

## Repository layout

| Path | Purpose |
|---|---|
| `notebooks/` | Phase 1-3 notebooks (data indexing, original preprocessing, zero-shot validation; Colab oriented) |
| `phase2_corrected/` | Corrected preprocessing, inference, accuracy reports, input verification, label rebuild |
| `phase4_finetune/` | SFCN transfer learning: cache, smoke tests, training, validation and one-shot test evaluation |
| `splits/` | Train/validation/test assignment (IDs only) and the list of excluded scans |
| `analysis/` | Sex-disparity analysis script (no valid results published yet) |
| `external/` | **Not tracked.** Local clones of the two pretrained-model repositories (below) |
| `implementation.md`, `CASE_STUDY_iAudit_vs_ml_brain.md` | Plan and the label-error case study |

## Setup

Tested with Python 3.11, `antspyx 0.6.3`, `antspynet 0.3.2`, `torch 2.6.0+cu124`, `tensorflow 2.20.0`, `tf_keras 2.20.1`, `nibabel 5.4.2`, `nilearn 0.14.1`,
`numpy 2.3.5`, `pandas 3.0.6`, `scipy 1.15.3`, `statsmodels 0.15.0`, `scikit-learn 1.9.1`, `matplotlib 3.11.2`, `xlrd 2.0.2`, `pillow 12.3.0`.
DeepBrainNet is a legacy Keras-2 model, so it is loaded through `tf_keras`. Fine-tuning ran on a 6 GB laptop GPU.

### Pretrained-model repositories (clone into `external/`)

| Directory | Source | Pinned commit | Licence | Weights (verify with SHA256) |
|---|---|---|---|---|
| `external/UKBiobank_deep_pretrain` | <https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain> | `4e56dc1522ac89caadc5c6fc779799ee58c23adc` | MIT | `brain_age/run_20190719_00_epoch_best_mae.p`, 11,827,312 bytes, `f9197c92486a8dba7fd87bb9f2ecc4f42f7c9628b9150d1b95c8a5ac10721ad3` (stored in the repo) |
| `external/DeepBrainNet` | <https://github.com/vishnubashyam/DeepBrainNet> | `505f4e4fdda6a2f5773bec0c98ef567a7a4ae36e` | no licence file in the repository | `Models/DBN_model.h5` (Git LFS), 182,983,936 bytes, `9257e98e9aff88bfcd83acdb1e74ffde762967503ef6df16c9270aa39d9941ea` |

Run these from the repository root in a POSIX shell (Git Bash, WSL, macOS/Linux). On Windows PowerShell, use `New-Item -ItemType Directory external` and
`Get-FileHash DeepBrainNet\Models\DBN_model.h5 -Algorithm SHA256` instead of `mkdir -p` and `sha256sum`. If `external/` already exists, do not re-run the clone commands over it.

```
mkdir -p external && cd external
git clone https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain.git
git -C UKBiobank_deep_pretrain checkout 4e56dc1522ac89caadc5c6fc779799ee58c23adc
git lfs install
git clone https://github.com/vishnubashyam/DeepBrainNet.git
git -C DeepBrainNet checkout 505f4e4fdda6a2f5773bec0c98ef567a7a4ae36e
git -C DeepBrainNet lfs pull                       # if this is blocked, download Models/DBN_model.h5 in a browser
sha256sum DeepBrainNet/Models/DBN_model.h5          # must match the table
```

`phase2_corrected/infer.py` refuses to run DeepBrainNet if the weights' size or SHA256 differ, and loads SFCN with `strict=True`.
(antspynet's `brainAgeDeepBrainNet` weights are a byte-identical copy of `DBN_model.h5`.) Weights are never committed or modified by this project; fine-tuned checkpoints are written to separate files.

### Data (not redistributed)

- IXI T1 scans and the demographics spreadsheet `IXI.xls` (CC BY-SA 3.0; cite the source) from <https://brain-development.org/ixi-dataset/>.
- Registration templates: `phase2_corrected/assets/README.md`.
- Paths are configured at the top of `phase2_corrected/common.py` (`PROJECT_DIR`, currently `G:\My Drive\ML_Project`); edit it for your machine. Large outputs (volumes, predictions, QC records) are written under that folder, not into Git.

## Reproducing the pipeline

```
# 1. labels: build the scan index with the Phase 1 notebook, then rebuild labels from the genuine IXI.xls
python phase2_corrected/label_fix/build_labels.py --xls <path/to/IXI.xls> --out <PROJECT_DIR>
# 2. preprocessing (resumable) and zero-shot inference
python phase2_corrected/preprocess.py --pilot            # small check first
python phase2_corrected/preprocess.py --all --retry-failed --workers 3
python phase2_corrected/infer.py --model dbn  --subset all
python phase2_corrected/infer.py --model sfcn --subset all
python phase2_corrected/accuracy_report.py --pred <dbn_predictions_all.csv> --tag all_inclusive
python phase2_corrected/verify_inputs.py                 # orientation / tensor / split checks
python phase2_corrected/to_npy.py                        # optional .npy copies
# 3. fine-tuning (see phase4_finetune/README.md for the reasoning)
python phase4_finetune/build_cache.py
python phase4_finetune/smoke_test.py
python phase4_finetune/train.py --run headA --stage head    --init pretrained --epochs 40 --patience 8
python phase4_finetune/train.py --run partB --stage partial --init phase4_finetune/checkpoints/headA/best.pt --epochs 25 --patience 6 --lr-head 3e-4 --lr-backbone 3e-5
python phase4_finetune/train.py --run fullC --stage full    --init phase4_finetune/checkpoints/partB/best.pt --epochs 20 --patience 5 --lr-head 1e-4 --lr-backbone 1e-5
python phase4_finetune/report_val.py
python phase4_finetune/evaluate_test.py                  # one-shot; refuses to run again without --force
python phase4_finetune/verify_test_protocol.py
```

`phase2_corrected/run_rest_after_preprocess.sh` is an optional wrapper that chains preprocessing, DeepBrainNet inference and the reports; it contains hard-coded `G:` paths and must be adapted.

## What is not in Git

Logs, per-subject prediction and label tables, cached MRI volumes and `.npy` arrays, model checkpoints and pretrained weights, the registration templates, and the local `external/` clones
are generated or third-party and are listed in `.gitignore`. Reports refer to some of these files by name; they are produced by the scripts above.

## Licences and attribution

IXI data: CC BY-SA 3.0 (cite the IXI project). SFCN code and weights: MIT (Han Peng). DeepBrainNet: no licence file; the repository is therefore not redistributed here, and only a figure showing
one of its sample slices is included in `phase2_corrected/reports/input_verification.png`. The comparison repository iAudit (<https://github.com/sudarsan2507-hue/iAudit>) is MIT licensed. The registration template (TemplateFlow `MNI152Lin`, ICBM152) is under the permissive MNI/McGill licence (keep the copyright notice; see `phase2_corrected/assets/README.md`).
This repository does not yet declare its own licence.

References: Peng et al. (2021), *Medical Image Analysis* 68:101871; Bashyam et al. (2020), *Brain* 143(7):2312-2324.
