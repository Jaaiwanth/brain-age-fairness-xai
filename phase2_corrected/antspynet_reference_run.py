"""DIAGNOSTIC: run antspynet's own DeepBrainNet pipeline (the one iAudit uses) on the pilot raw scans.

Identical to antspynet.brain_age(raw, do_preprocessing=True): we call preprocess_brain_image() with
the same arguments, save its output, then call brain_age(..., do_preprocessing=False) on it.
Saving the intermediate lets us diff antspynet's input against ours.

python phase2_corrected/antspynet_reference_run.py --workers 3
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.OUT_ROOT / "antspynet_reference"


def run_one(task):
    ixi, raw = task
    import ants
    from antspynet.utilities import brain_age, preprocess_brain_image
    f = OUT / "per_subject" / f"IXI{ixi:03d}.json"
    if f.exists():
        return json.loads(f.read_text())
    t0 = time.time()
    rec = dict(IXI_ID=ixi)
    try:
        t1 = ants.image_read(raw)
        pre = preprocess_brain_image(t1, truncate_intensity=(0.01, 0.99), brain_extraction_modality="t1",
                                     template="croppedMni152",
                                     template_transform_type="antsRegistrationSyNQuickRepro[a]",
                                     do_bias_correction=True, do_denoising=True, verbose=False)
        img = pre["preprocessed_image"] * pre["brain_mask"]
        OUT.mkdir(parents=True, exist_ok=True)
        ants.image_write(img, str(OUT / f"IXI{ixi:03d}_preprocessed.nii.gz"))
        res = brain_age(img, do_preprocessing=False, verbose=False)
        rec.update(predicted_age=float(res["predicted_age"]), shape="x".join(map(str, img.shape)),
                   slice_min=float(res["brain_age_per_slice"].min()), slice_max=float(res["brain_age_per_slice"].max()))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.time() - t0, 1)
    (OUT / "per_subject").mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rec))
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    pil = pd.read_csv(C.OUT_ROOT / "qc" / "pilot_ids.csv")
    tasks = [(int(r.IXI_ID), str(C.RAW_DIR / r.filename)) for r in pil.itertuples()]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(run_one, t) for t in tasks]):
            r = fut.result()
            rows.append(r)
            print(r, flush=True)
    d = pd.DataFrame(rows).merge(pil[["IXI_ID", "site", "AGE"]], on="IXI_ID").rename(columns={"AGE": "actual_age"})
    d.to_csv(C.OUT_ROOT / "inference" / "dbn_antspynet_brain_age_pilot.csv", index=False)
    ok = d.dropna(subset=["predicted_age"])
    if len(ok) > 2:
        import numpy as np
        from scipy import stats
        print(f"n={len(ok)} MAE={np.abs(ok.predicted_age - ok.actual_age).mean():.2f} "
              f"r={np.corrcoef(ok.actual_age, ok.predicted_age)[0, 1]:.3f} "
              f"rho={stats.spearmanr(ok.actual_age, ok.predicted_age)[0]:.3f}")
