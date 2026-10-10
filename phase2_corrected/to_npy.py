"""Convert the corrected preprocessed volumes to .npy (one file per subject).

Source of truth stays the NIfTI files; this is a convenience copy.
  Source : <OUT_ROOT>/sfcn/IXI###.nii.gz   (FSL MNI152 1 mm grid, LAS, raw intensity scale, brain-masked)
  Output : <OUT_ROOT>/npy/IXI###.npy       float32, shape (182, 218, 182), identical array
           <OUT_ROOT>/npy/index.csv         per-subject metadata + QC status + array stats
           <OUT_ROOT>/npy/README.txt        orientation / usage notes

Orientation: array axes are (x, y, z) with index increasing toward Left, Anterior, Superior (LAS, the FSL convention).
DeepBrainNet's reference convention (LPS) is the same array with the y axis flipped:  arr[:, ::-1, :].
Only QC PASS/WARN subjects are converted. Resumable: existing, valid files are skipped.

python phase2_corrected/to_npy.py
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

README = """Corrected IXI volumes as .npy  (one file per subject: IXI###.npy)

shape      : (182, 218, 182), dtype float32
space      : FSL MNI152 1 mm, affine-registered, N4 bias-corrected, brain-masked (antspynet), background = 0
intensity  : raw scanner scale (NOT normalised). Models apply their own scaling:
               SFCN        : x / x.mean(), then centre-crop to (160, 192, 160)
               DeepBrainNet: x * (185 / percentile(x, 97)), slices 45..124, ...
orientation: axes (x, y, z) increase toward Left, Anterior, Superior  (LAS, FSL convention)
             DeepBrainNet's reference data is LPS: use  arr[:, ::-1, :]  (flip the y axis) before slicing.
labels     : see index.csv (age, sex, site from the corrected ixi_final_metadata.csv) and qc_status (PASS / WARN).
source     : ../sfcn/IXI###.nii.gz (NIfTI) is the source of truth; these arrays are identical copies.
"""


def main():
    import preprocess
    preprocess.aggregate(C.OUT_ROOT)                      # refresh qc_summary.csv from the per-subject records
    out = C.OUT_ROOT / "npy"
    out.mkdir(parents=True, exist_ok=True)
    qc = pd.read_csv(C.OUT_ROOT / "qc" / "qc_summary.csv")
    meta = pd.read_csv(C.METADATA_PATH).set_index("IXI_ID")
    ok = qc[qc.status.isin(["PASS", "WARN"]) & qc.IXI_ID.isin(meta.index)].sort_values("IXI_ID")
    rows, converted, skipped, bad = [], 0, 0, []
    for ixi in ok.IXI_ID.astype(int):
        src = C.OUT_ROOT / "sfcn" / f"IXI{ixi:03d}.nii.gz"
        dst = out / f"IXI{ixi:03d}.npy"
        try:
            if dst.exists():
                a = np.load(dst, mmap_mode="r")
                assert a.shape == C.FSL_SHAPE and a.dtype == np.float32
                skipped += 1
                arr = np.asarray(a)
            else:
                arr = np.asarray(nib.load(str(src)).dataobj, dtype=np.float32)
                assert arr.shape == C.FSL_SHAPE, arr.shape
                assert np.isfinite(arr).all(), "NaN/Inf"
                tmp = dst.with_suffix(".tmp.npy")
                np.save(tmp, arr)
                chk = np.load(tmp, mmap_mode="r")           # read back and compare before publishing the file
                assert chk.shape == arr.shape and np.array_equal(np.asarray(chk), arr)
                del chk
                tmp.replace(dst)
                converted += 1
            m = meta.loc[ixi]
            rows.append(dict(IXI_ID=ixi, file=dst.name, age=m.AGE, sex=m.sex_label, site=m.site,
                             qc_status=ok.set_index("IXI_ID").loc[ixi, "status"], shape="x".join(map(str, arr.shape)),
                             dtype=str(arr.dtype), min=float(arr.min()), max=float(arr.max()), mean=float(arr.mean()),
                             nonzero_frac=float((arr > 0).mean())))
        except Exception as e:
            bad.append((ixi, f"{type(e).__name__}: {e}"))
            print(f"!!! IXI{ixi:03d} failed: {e}", flush=True)
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    (out / "README.txt").write_text(README, encoding="utf8")
    print(f"npy: converted {converted}, already present {skipped}, failed {len(bad)}, total indexed {len(rows)} -> {out}")


if __name__ == "__main__":
    main()
