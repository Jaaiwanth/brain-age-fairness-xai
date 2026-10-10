# Template assets (not committed)

The registration template and brain mask are large binary files and are **not** stored in Git (`*.nii.gz` is ignored here).
They are re-created in two steps.

1. Download the TemplateFlow `MNI152Lin` 1 mm template and brain mask into this folder:

```
curl -L -o tpl-MNI152Lin_res-01_T1w.nii.gz            https://templateflow.s3.amazonaws.com/tpl-MNI152Lin/tpl-MNI152Lin_res-01_T1w.nii.gz
curl -L -o tpl-MNI152Lin_res-01_desc-brain_mask.nii.gz https://templateflow.s3.amazonaws.com/tpl-MNI152Lin/tpl-MNI152Lin_res-01_desc-brain_mask.nii.gz
```

(181 x 217 x 181, RAS, ICBM152 linear average brain as distributed by TemplateFlow. Licence: the MNI/McGill permissive licence, copyright 1993-2009 Louis Collins,
McConnell Brain Imaging Centre, MNI, McGill University: use, copying, modification and distribution are allowed provided the copyright notice is kept; full text at
<https://templateflow.s3.amazonaws.com/tpl-MNI152Lin/LICENSE>.)

2. Embed them into SFCN's 182 x 218 x 182 LAS grid (this is exact: the two grids share the same voxel lattice):

```
cd phase2_corrected
python -c "import common; common.build_fsl_grid_assets()"
```

This writes `MNI152Lin_T1_1mm_182x218x182_LAS.nii.gz` and `MNI152Lin_brain_mask_1mm_182x218x182_LAS.nii.gz`, which `preprocess.py` uses.
