"""Build a local training cache: SFCN-normalised, slightly oversized crops of every corrected volume.

For each subject: x = npy / npy.mean()   (SFCN's own normalisation, mean over the whole 182x218x182 volume)
                  crop (7:175, 9:209, 7:175) -> (168, 200, 168), float32.
The SFCN input crop (160,192,160) is the *centre* of this crop (offset 4,4,4 == crop_center of the full volume);
training can jitter the offset by +-4 voxels. Local SSD, so epochs do not stream from Google Drive.

python phase4_finetune/build_cache.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "phase2_corrected"))
import common as C  # noqa: E402

CACHE = REPO / "phase4_finetune" / "cache"
NPY = C.OUT_ROOT / "npy"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(C.METADATA_PATH)
    split = {}
    for s in ("train", "val", "test"):
        for i in pd.read_csv(C.PROJECT_DIR / f"ixi_{s}.csv").IXI_ID:
            split[int(i)] = s
    meta["split"] = meta.IXI_ID.map(split)
    assert meta.split.notna().all()
    done = skipped = 0
    for r in meta.itertuples():
        dst = CACHE / f"IXI{r.IXI_ID:03d}.npy"
        if dst.exists():
            skipped += 1
            continue
        a = np.load(NPY / f"IXI{r.IXI_ID:03d}.npy")
        assert a.shape == (182, 218, 182) and np.isfinite(a).all()
        x = (a / a.mean())[7:175, 9:209, 7:175].astype(np.float32)
        np.save(dst.with_suffix(".tmp.npy"), x)
        dst.with_suffix(".tmp.npy").replace(dst)
        done += 1
        if done % 50 == 0:
            print(f"{done} cached", flush=True)
    meta[["IXI_ID", "AGE", "sex_label", "site", "split"]].to_csv(CACHE / "index.csv", index=False)
    # the centre of the cache must equal SFCN's own preprocessing exactly
    import torch  # noqa: F401
    sys.path.insert(0, str(REPO / "external" / "UKBiobank_deep_pretrain"))
    from dp_model import dp_utils as dpu
    for i in (2, 100, 300):
        if i not in set(meta.IXI_ID):
            continue
        a = np.load(NPY / f"IXI{i:03d}.npy").astype(np.float64)
        ref = dpu.crop_center(a / a.mean(), (160, 192, 160)).astype(np.float32)
        got = np.load(CACHE / f"IXI{i:03d}.npy")[4:164, 4:196, 4:164]
        assert np.allclose(ref, got, atol=1e-5), i
    print(f"cache built: {done} new, {skipped} existing; centre crop == dpu.crop_center verified; -> {CACHE}")


if __name__ == "__main__":
    main()
