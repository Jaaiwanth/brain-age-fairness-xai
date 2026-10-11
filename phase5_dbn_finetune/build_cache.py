"""Cache the exact DeepBrainNet reference-chain slice images (uint8, one channel) for every labelled subject.

Per subject: deepbrainnet/IXI###.nii.gz -> infer.dbn_input() (p97*185, slices 45-124, PIL F->RGB, JPEG round trip, 256x256 nearest, /255)
             -> uint8 (80, 256, 256). R == G == B after the JPEG round trip (verified), so one channel is stored losslessly.
Local SSD, ~2.5 GB for 485 subjects. Resumable.

python phase5_dbn_finetune/build_cache.py --workers 4
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "phase2_corrected"))
import common as C  # noqa: E402

CACHE = REPO / "phase5_dbn_finetune" / "cache"


def one(ixi):
    import infer
    dst = CACHE / f"IXI{ixi:03d}.npy"
    if dst.exists():
        return ixi, "skipped"
    _, v = C.load_dir_volume(C.OUT_ROOT / "deepbrainnet" / f"IXI{ixi:03d}.nii.gz")
    b, _ = infer.dbn_input(v)                                    # (80,256,256,3) float32 == uint8/255
    u = np.rint(b * 255).astype(np.uint8)
    assert np.abs(b * 255 - u).max() < 1e-3 and (u[..., 0] == u[..., 1]).all() and (u[..., 1] == u[..., 2]).all()
    tmp = dst.with_suffix(".tmp.npy")
    np.save(tmp, u[..., 0])
    tmp.replace(dst)
    return ixi, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(REPO / "splits" / "ixi_splits.csv")
    meta = pd.read_csv(C.METADATA_PATH)[["IXI_ID", "AGE", "sex_label", "site"]]
    idx = sp.merge(meta, on="IXI_ID", how="left")
    assert idx.AGE.notna().all() and len(idx) == 485
    idx[["IXI_ID", "AGE", "sex_label", "site", "split"]].to_csv(CACHE / "index.csv", index=False)
    ids = [int(i) for i in idx.IXI_ID]
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for ixi, st in ex.map(one, ids):
            done += 1
            if done % 25 == 0:
                print(f"{done}/{len(ids)}", flush=True)
    # consistency: cached images equal a fresh run of the reference chain for a few subjects
    import infer
    for ixi in (2, 130, 480):
        if ixi in ids:
            _, v = C.load_dir_volume(C.OUT_ROOT / "deepbrainnet" / f"IXI{ixi:03d}.nii.gz")
            b, _ = infer.dbn_input(v)
            assert np.allclose(np.load(CACHE / f"IXI{ixi:03d}.npy").astype(np.float32) / 255.0, b[..., 0], atol=1e-6), ixi
    print(f"cache complete: {len(list(CACHE.glob('IXI*.npy')))} subjects; cached images identical to infer.dbn_input for 3 test subjects -> {CACHE}")


if __name__ == "__main__":
    main()
