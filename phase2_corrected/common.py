"""Shared constants/helpers for the corrected preprocessing + inference pipeline.

Reference spaces (verified from the repos / sample data, see audit):
  * SFCN  : FSL MNI152 1 mm, 182x218x182, stored LAS (affine diag(-1,1,1), origin 90,-126,-72).
            Reference code (examples.ipynb): data/data.mean() -> dpu.crop_center -> (160,192,160).
  * DBN   : the 4 shipped sample volumes (external/DeepBrainNet/Data) are 182x218x182 stored LPS,
            i.e. the FSL grid with the y axis flipped (affine diag(-1,-1,1), origin 90,91,-72),
            raw integer-scale intensities (max ~700-1100, p97 ~485).
"""
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"
EXTERNAL = REPO / "external"

PROJECT_DIR = Path(r"G:\My Drive\ML_Project")
RAW_DIR = PROJECT_DIR / "IXI_data"
METADATA_PATH = PROJECT_DIR / "ixi_final_metadata.csv"
OUT_ROOT = PROJECT_DIR / "phase2_corrected"   # sfcn/ deepbrainnet/ qc/ logs/

FSL_SHAPE = (182, 218, 182)
FSL_AFFINE = np.array([[-1, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]], dtype=float)  # LAS
DBN_AFFINE = np.array([[-1, 0, 0, 90], [0, -1, 0, 91], [0, 0, 1, -72], [0, 0, 0, 1]], dtype=float)   # LPS

# TemplateFlow MNI152Lin 1 mm (= FSL MNI152 6th-gen linear) is the FSL grid minus one voxel
# layer on the high-index side of each axis, so it embeds exactly (no interpolation).
TF_T1 = ASSETS / "tpl-MNI152Lin_res-01_T1w.nii.gz"
TF_MASK = ASSETS / "tpl-MNI152Lin_res-01_desc-brain_mask.nii.gz"
FSL_T1 = ASSETS / "MNI152Lin_T1_1mm_182x218x182_LAS.nii.gz"
FSL_MASK = ASSETS / "MNI152Lin_brain_mask_1mm_182x218x182_LAS.nii.gz"

# SFCN reference constants (examples.ipynb)
SFCN_CROP = (160, 192, 160)
SFCN_BIN_RANGE, SFCN_BIN_STEP, SFCN_SIGMA = [42, 82], 1, 1
# DeepBrainNet reference constants (Slicer.py / Model_Test.py)
DBN_P97_TARGET = 185.0
DBN_SLICE_START, DBN_N_SLICES = 45, 80
DBN_IMG_SIZE = (256, 256)   # keras flow_from_directory default target_size
DBN_WEIGHTS = EXTERNAL / "DeepBrainNet" / "Models" / "DBN_model.h5"
DBN_WEIGHTS_SHA256 = "9257e98e9aff88bfcd83acdb1e74ffde762967503ef6df16c9270aa39d9941ea"  # from the repo's Git-LFS pointer file
DBN_WEIGHTS_SIZE = 182983936

# QC thresholds. Fixed BEFORE looking at pilot results; any later change is reported explicitly.
QC = dict(
    mask_ratio_range=(0.6, 1.4),   # our brain-mask voxels / template brain-mask voxels
    dice_min=0.80,                 # Dice(our mask, template brain mask)
    reg_r_min=0.60,                # Pearson r(registered head, template) inside template brain mask
    nonzero_frac_range=(0.15, 0.40),
)


def build_fsl_grid_assets(force=False):
    """Embed the TemplateFlow 181x217x181 RAS template/mask into the 182x218x182 LAS FSL grid."""
    for src, dst in [(TF_T1, FSL_T1), (TF_MASK, FSL_MASK)]:
        if dst.exists() and not force:
            continue
        img = nib.load(str(src))
        a = np.asarray(img.dataobj, dtype=np.float32)
        assert a.shape == (181, 217, 181), a.shape
        exp = np.array([[1, 0, 0, -90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]], float)
        assert np.allclose(img.affine, exp), img.affine
        out = np.zeros(FSL_SHAPE, np.float32)
        out[:181, :217, :181] = a[::-1]          # fsl[i,j,k] = tf[180-i, j, k]
        nib.save(nib.Nifti1Image(out, FSL_AFFINE), str(dst))
    return FSL_T1, FSL_MASK


def load_dir_volume(path):
    """nibabel load -> float64 array, as Slicer.py/examples do (get_data/get_fdata)."""
    img = nib.load(str(path))
    return img, np.asarray(img.dataobj).astype(np.float64)
