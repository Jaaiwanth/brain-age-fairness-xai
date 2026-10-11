# experiments/preprocessing/

Corrected preprocessing, zero-shot inference and input verification for the two pretrained models. This is the source of the volumes used by [`../sfcn/`](../sfcn/README.md) and [`../deepbrainnet/`](../deepbrainnet/README.md). It was previously a top-level folder called `phase2_corrected/`.

## What it does

N4 bias correction, ANTs affine registration to the FSL MNI152 grid (182x218x182), an antspynet brain mask (p > 0.5), raw intensities kept. Registration QC records a warning (not a failure) when the registration correlation is low. Each model then gets its own reference input (SFCN: divide by the mean, centre crop 160x192x160; DeepBrainNet: LPS orientation, 80 axial slices 45-124, p97 scaling).
Labels come from the genuine `IXI.xls` (see [`label_fix/build_labels.py`](label_fix/build_labels.py); it replaces an earlier join that used a row counter as the IXI ID).

## Contents

| Path | Purpose |
|---|---|
| `common.py` | Paths and constants shared by all scripts and by the `sfcn` / `deepbrainnet` experiments (`REPO` is the repository root) |
| `preprocess.py`, `antspynet_reference_run.py` | Preprocessing and the independent antspynet reference run |
| `infer.py`, `accuracy_report.py`, `pilot_report.py` | Zero-shot inference and reports |
| `verify_inputs.py`, `to_npy.py` | Input verification checks, optional `.npy` copies |
| `label_fix/` | Rebuild of the age/sex labels |
| `assets/` | Registration template instructions (the `.nii.gz` templates are re-downloadable and not tracked) |
| `reports/`, `case_study/` | Written reports and the label-error evidence figure |
| `legacy/` | The original, superseded `parallel_preprocess.py`, kept unchanged for reference |

Run the scripts from the repository root, for example `python experiments/preprocessing/infer.py --model dbn --subset all`. Large outputs are written under `PROJECT_DIR` in `common.py`; the folder name `phase2_corrected` that appears there is a data folder on that drive, not a path in this repository.

Reports generated before the folder was renamed (`reports/accuracy_report_*.md`) still say `phase2_corrected/`; they are kept as generated.
