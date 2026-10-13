# Brain-age prediction and fairness audit on IXI

Working title: *"Right for the Wrong Reasons? Auditing Brain-Age Model Fairness Beyond Accuracy"*.

## Overview and objectives

A brain-age model predicts a person's age from a T1-weighted MRI; the gap between predicted and chronological age (the brain-age gap) is a candidate biomarker, so it matters whether the model is **fair beyond accuracy**.
This project fine-tunes pretrained brain-age CNNs on the public [IXI dataset](https://brain-development.org/ixi-dataset/) (T1 MRI, three acquisition sites, ages 20-86) and asks:

1. How accurate are the models, and what does fine-tuning add over zero-shot use?
2. Do prediction error or the brain-age gap differ by sex or acquisition site once age is controlled for?
3. Where in the brain do the models' attributions fall, and do those patterns differ between groups?

The project is a methods and fairness audit on healthy volunteers; it makes no clinical claims.
The step-by-step plan is in [`implementation.md`](implementation.md); the living status, data layout and analysis rules are in [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md).

## Workflow

| Phase | Content | Where |
|---|---|---|
| 1-2 | Data indexing, labels from the official `IXI.xls`, preprocessing to each model's reference input | [`notebooks/`](notebooks/), [`scripts/`](scripts/) (team pipeline, "v2"), [`experiments/preprocessing/`](experiments/preprocessing/) (alternative pipeline used by the experiments) |
| 3-5 | Zero-shot validation, fine-tuning, one-shot test evaluation | [`scripts/`](scripts/) (SFCN, team pipeline); [`experiments/sfcn/`](experiments/sfcn/README.md) and [`experiments/deepbrainnet/`](experiments/deepbrainnet/README.md) (additional fine-tuning experiments) |
| 6 | Fairness audit (sex, acquisition site; age-controlled) and leave-one-site-out analysis | [`scripts/06_fairness_audit.py`](scripts/06_fairness_audit.py), [`scripts/06b_loso_analysis.py`](scripts/06b_loso_analysis.py) |
| 7 | Grad-CAM attribution and atlas-based comparison across sex and site | [`scripts/07_gradcam.py`](scripts/07_gradcam.py) |
| 8 | Report | not started |

## Results summary

All numbers are held-out **test** MAE on the same 73-subject test split (339 / 73 / 73 train / validation / test), evaluated once per pipeline; 95% bootstrap intervals in brackets.

| Pipeline | Model | Test MAE (years) | r | Documentation |
|---|---|---|---|---|
| Team pipeline (`scripts/`) | SFCN, fine-tuned, "v2" preprocessing | 3.71 [3.10, 4.38] | 0.966 | [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md) (Phase 5) |
| `experiments/sfcn` | SFCN, fine-tuned (70 bins, `experiments/preprocessing` volumes) | 4.07 [3.43, 4.76] | 0.96 | [`experiments/sfcn/README.md`](experiments/sfcn/README.md) |
| `experiments/deepbrainnet` | DeepBrainNet, fine-tuned (blocks 4-5 + head) | 4.53 [3.86, 5.23] | 0.95 | [`experiments/deepbrainnet/README.md`](experiments/deepbrainnet/README.md) |
| Reference | DeepBrainNet zero-shot / + linear calibration | 6.32 / 6.29 | 0.89 | same |
| Reference | SFCN zero-shot / + linear calibration (`experiments/preprocessing` volumes) | 13.05 / 5.71 | 0.90 | [`experiments/sfcn/README.md`](experiments/sfcn/README.md) |

On **validation** (n = 73), where comparisons between pipelines are allowed, the two fine-tuned SFCNs are equivalent (MAE 3.90 team pipeline, 3.88 `experiments/sfcn`); fine-tuned DeepBrainNet reaches 4.12. Zero-shot SFCN on the team pipeline's validation volumes gives MAE 10.21 [8.19, 12.29], r 0.910 (6.85 with a train-fitted linear correction), because its UK Biobank head cannot predict outside ages 42-82 (`scripts/03b_zero_shot.py`, `PROJECT_GUIDE.md` Phase 3).

Fairness and attribution findings from the team pipeline (details, intervals and caveats in [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md), Phases 6-7): no detectable sex difference in the age-corrected brain-age gap; an **acquisition-site effect at IOP** (age-corrected gap lower than Guys by about 5 years, 95% CI -8.66 to -1.34, with only 9 IOP test subjects); accuracy falls by about 3 years when IOP is left out of training; and attribution patterns differ between acquisition sites but not between sexes.

**Read these caveats with the numbers**
- The three fine-tuned models were each selected on validation data and then evaluated once on the **same** test split. These test results must not be used to choose between pipelines; any such choice has to be made on validation data only. Within the two `experiments/` evaluations the SFCN-versus-DeepBrainNet difference (+0.45 y, 95% CI [-0.38, +1.26]) is not distinguishable from zero.
- The SFCN and DeepBrainNet experiments in `experiments/` use the [`experiments/preprocessing/`](experiments/preprocessing/) volumes and have **not** had a fairness or attribution analysis; those analyses apply to the team's SFCN pipeline.
- Single seed and single split everywhere; the intervals reflect only which 73 subjects were drawn. Subgroups are small (for example 6 test subjects aged 40-49, 9 from IOP), so subgroup numbers are descriptive.
- All three IXI sites appear in the main model's training data. Generalisation to an unseen site was tested only by the leave-one-site-out analysis (`scripts/06b_loso_analysis.py`, `PROJECT_GUIDE.md` Phase 6). In that analysis HH generalises, but a model that never saw IOP is about 3 years less accurate there. With only three sites, this says nothing about scanners outside IXI. In IXI, site is confounded with the population each site recruited.
- Earlier zero-shot checks of the `experiments/preprocessing` pipeline used all 485 subjects (3 of 20 pilot subjects and 22 of 100 in an orientation test are test-split subjects); they involved no fitting and changed no decision, but they were not isolated from the test labels.
- Analyses done on the original, mis-labelled data are invalid (see below); the earlier sex-disparity outputs were removed (`experiments/analysis/README.md`).

## Repository structure

| Path | Purpose |
|---|---|
| [`README.md`](README.md), [`PROJECT_GUIDE.md`](PROJECT_GUIDE.md), [`implementation.md`](implementation.md) | Overview, living project guide (status, data layout, analysis rules), original plan |
| `notebooks/` | Phase 1-3 notebooks (data indexing, original preprocessing, zero-shot validation; Colab oriented) |
| `scripts/` | Team pipeline: official labels, v2 preprocessing, SFCN fine-tuning and test evaluation, fairness audit, leave-one-site-out, Grad-CAM |
| [`experiments/preprocessing/`](experiments/preprocessing/README.md) | Alternative preprocessing, zero-shot inference, accuracy reports, input verification, label rebuild; `experiments/preprocessing/legacy/` holds the original, superseded preprocessing script |
| `experiments/sfcn/` | SFCN fine-tuning experiment: training and evaluation code, reports, [README](experiments/sfcn/README.md) |
| `experiments/deepbrainnet/` | DeepBrainNet fine-tuning experiment (PyTorch port of the Keras model): code, reports, [README](experiments/deepbrainnet/README.md) |
| [`experiments/data_splits/`](experiments/data_splits/README.md) | Train/validation/test assignment (IDs only) and the list of excluded scans |
| [`experiments/analysis/`](experiments/analysis/README.md) | Sex-disparity analysis script (no valid results published yet) |
| `external/` | **Not tracked.** Local clones of the two pretrained-model repositories (below) |

## An important correction

An early version of this project attached age and sex to the wrong subjects (a row-index column had been used as the IXI ID), which made correct models look broken.
It was found by comparing against the public iAudit repository, whose per-subject ages agree with the corrected labels.
Labels are now rebuilt from the genuine `IXI.xls` joined on `IXI_ID` (`experiments/preprocessing/label_fix/build_labels.py`).

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

`experiments/preprocessing/infer.py` refuses to run DeepBrainNet if the weights' size or SHA256 differ, and loads SFCN with `strict=True`.
(antspynet's `brainAgeDeepBrainNet` weights are a byte-identical copy of `DBN_model.h5`.) Weights are never committed or modified by this project; fine-tuned checkpoints are written to separate files.

### Data (not redistributed)

- IXI T1 scans and the demographics spreadsheet `IXI.xls` (CC BY-SA 3.0; cite the source) from <https://brain-development.org/ixi-dataset/>.
- Registration templates: `experiments/preprocessing/assets/README.md`.
- Paths are configured at the top of `experiments/preprocessing/common.py` (`PROJECT_DIR`, currently `G:\My Drive\ML_Project`); edit it for your machine. Large outputs (volumes, predictions, QC records) are written under that folder, not into Git.

## Reproducing preprocessing and zero-shot inference

```
# 1. labels: build the scan index with the Phase 1 notebook, then rebuild labels from the genuine IXI.xls
python experiments/preprocessing/label_fix/build_labels.py --xls <path/to/IXI.xls> --out <PROJECT_DIR>
# 2. preprocessing (resumable) and zero-shot inference
python experiments/preprocessing/preprocess.py --pilot            # small check first
python experiments/preprocessing/preprocess.py --all --retry-failed --workers 3
python experiments/preprocessing/infer.py --model dbn  --subset all
python experiments/preprocessing/infer.py --model sfcn --subset all
python experiments/preprocessing/accuracy_report.py --pred <dbn_predictions_all.csv> --tag all_inclusive
python experiments/preprocessing/verify_inputs.py                 # orientation / tensor / split checks
python experiments/preprocessing/to_npy.py                        # optional .npy copies
# 3. fine-tuning experiments: see experiments/sfcn/README.md and experiments/deepbrainnet/README.md
```

The two fine-tuning experiments document their own commands. `experiments/preprocessing/run_rest_after_preprocess.sh` is an optional wrapper that chains preprocessing, DeepBrainNet inference and the reports; it contains hard-coded `G:` paths and must be adapted.

## What is not in Git

Logs, per-subject prediction and label tables, cached MRI volumes and `.npy` arrays, model checkpoints and pretrained weights, the registration templates, and the local `external/` clones
are generated or third-party and are listed in `.gitignore`. Reports refer to some of these files by name; they are produced by the scripts above.

## Licences and attribution

IXI data: CC BY-SA 3.0 (cite the IXI project). SFCN code and weights: MIT (Han Peng). DeepBrainNet: no licence file; the repository is therefore not redistributed here, and only a figure showing
one of its sample slices is included in `experiments/preprocessing/reports/input_verification.png`. The comparison repository iAudit (<https://github.com/sudarsan2507-hue/iAudit>) is MIT licensed. The registration template (TemplateFlow `MNI152Lin`, ICBM152) is under the permissive MNI/McGill licence (keep the copyright notice; see `experiments/preprocessing/assets/README.md`).
This repository does not yet declare its own licence.

References: Peng et al. (2021), *Medical Image Analysis* 68:101871; Bashyam et al. (2020), *Brain* 143(7):2312-2324.
