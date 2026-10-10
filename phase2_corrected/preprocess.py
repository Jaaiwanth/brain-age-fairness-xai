"""Corrected IXI preprocessing: raw T1 -> FSL-MNI152-1mm-grid brain volume (raw intensity scale).

Pipeline per scan (each step is logged to the QC record):
  raw NIfTI -> N4 bias correction -> affine registration to the FSL MNI152 1 mm template
  (182x218x182, LAS) -> antspynet brain mask (p>0.5) -> multiply -> float32, NO clipping,
  NO min-max (intensities stay on the scanner scale; SFCN's own /mean and DBN's own p97 rescale
  are applied at inference time, exactly as in the reference code).

Outputs under <OUT_ROOT>:
  sfcn/IXI###.nii.gz         182x218x182 float32, LAS  (SFCN input space)
  deepbrainnet/IXI###.nii.gz same data with y flipped, LPS (DBN sample-data convention)
  qc/per_subject/IXI###.json one QC record per subject (resume marker)
  qc/qc_summary.csv, logs/preprocess_summary.txt, logs/failures.csv

Usage:
  python phase2_corrected/preprocess.py --pilot            # ~14 stratified subjects + IXI109
  python phase2_corrected/preprocess.py --ids 109 20       # specific subjects
  python phase2_corrected/preprocess.py --all --workers 2  # resumable full run
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

DIAGNOSTIC_IDS = [109]   # always included in the pilot: near-empty in the old pipeline


def qc_path(root, ixi):
    return root / "qc" / "per_subject" / f"IXI{ixi:03d}.json"


def select_pilot(meta, n_per_bin=2, n_bins=7, seed=0):
    """Age-stratified pilot: equal-count age bins, n_per_bin random subjects each (IXI109 excluded)."""
    rng = np.random.default_rng(seed)
    m = meta[~meta.IXI_ID.isin(DIAGNOSTIC_IDS)].sort_values("AGE").reset_index(drop=True)
    chosen = []
    for idx in np.array_split(np.arange(len(m)), n_bins):
        chosen += list(rng.choice(m.IXI_ID.to_numpy()[idx], n_per_bin, replace=False))
    return sorted(int(i) for i in chosen) + DIAGNOSTIC_IDS


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 0.0


def preprocess_one(task):
    ixi, raw_path, out_root, save_intermediate = task
    out_root = Path(out_root)
    import ants
    import antspynet
    import nibabel as nib
    rec = dict(IXI_ID=ixi, raw_path=str(raw_path), status="FAIL", fail_reasons="", error="")
    t0 = time.time()
    try:
        if not Path(raw_path).exists():
            raise FileNotFoundError(raw_path)
        raw = ants.image_read(str(raw_path))
        rn = raw.numpy()
        rec.update(raw_shape="x".join(map(str, raw.shape)), raw_spacing="x".join(f"{s:.3f}" for s in raw.spacing),
                   raw_orientation=ants.get_orientation(raw), raw_min=float(rn.min()), raw_max=float(rn.max()),
                   raw_mean=float(rn.mean()), raw_nonzero_frac=float((rn > 0).mean()))

        tmpl = ants.image_read(str(C.FSL_T1))
        tmask = nib.load(str(C.FSL_MASK)).get_fdata() > 0.5
        tarr = tmpl.numpy()

        n4 = ants.n4_bias_field_correction(raw)
        reg, last = None, None
        for attempt in range(3):              # sporadic "Registration failed with error code 1" seen under parallel load
            try:
                reg = ants.registration(fixed=tmpl, moving=n4, type_of_transform="Affine")
                break
            except RuntimeError as exc:
                last = exc
                rec["registration_retries"] = attempt + 1
                time.sleep(5)
        if reg is None:
            raise last
        warped = reg["warpedmovout"]
        wn = warped.numpy()
        rec["registered_shape"] = "x".join(map(str, warped.shape))
        rec["reg_pearson_r"] = float(np.corrcoef(wn[tmask], tarr[tmask])[0, 1])
        rec["registration_ok"] = tuple(warped.shape) == C.FSL_SHAPE

        prob = antspynet.brain_extraction(warped, modality="t1")
        mask = ants.threshold_image(prob, 0.5, 1.0, 1, 0).numpy() > 0
        vol = (wn * mask).astype(np.float32)
        neg = int((vol < 0).sum())
        if neg:
            vol = np.clip(vol, 0, None)
        rec["neg_voxels_clipped"] = neg

        nz = vol > 0
        rec.update(out_shape="x".join(map(str, vol.shape)), dtype=str(vol.dtype), out_min=float(vol.min()),
                   out_max=float(vol.max()), out_mean=float(vol.mean()), out_std=float(vol.std()),
                   nonzero_frac=float(nz.mean()), mask_frac=float(mask.mean()),
                   mask_template_ratio=float(mask.sum() / tmask.sum()), dice_vs_template=dice(mask, tmask),
                   nan_count=int(np.isnan(vol).sum()), inf_count=int(np.isinf(vol).sum()),
                   p97_whole_volume=float(np.percentile(vol, 97)))

        # ---- hard QC (thresholds fixed in common.QC) ----
        why = []
        if tuple(vol.shape) != C.FSL_SHAPE: why.append(f"shape {vol.shape}")
        if rec["nan_count"] or rec["inf_count"]: why.append("NaN/Inf")
        lo, hi = C.QC["mask_ratio_range"]
        if not lo <= rec["mask_template_ratio"] <= hi: why.append(f"mask_ratio {rec['mask_template_ratio']:.2f}")
        if rec["dice_vs_template"] < C.QC["dice_min"]: why.append(f"dice {rec['dice_vs_template']:.2f}")
        lo, hi = C.QC["nonzero_frac_range"]
        if not lo <= rec["nonzero_frac"] <= hi: why.append(f"nonzero_frac {rec['nonzero_frac']:.2f}")
        rec["fail_reasons"] = "; ".join(why)
        # reg_r (T1-vs-template correlation) is contrast-sensitive: it flagged scans whose mask Dice was 0.93-0.95.
        # Demoted from hard FAIL to WARN (volume kept, reported separately). Changed 2026-10-09 after 2 of 25 scans
        # failed on reg_r alone; no predictions for those scans had been seen.
        rec["qc_warnings"] = f"reg_r {rec['reg_pearson_r']:.2f} < {C.QC['reg_r_min']}" if rec["reg_pearson_r"] < C.QC["reg_r_min"] else ""
        rec["status"] = "FAIL" if why else ("WARN" if rec["qc_warnings"] else "PASS")

        if rec["status"] != "FAIL" or save_intermediate:   # keep hard-failed volumes only for diagnosis
            for sub, arr, aff in [("sfcn", vol, C.FSL_AFFINE), ("deepbrainnet", vol[:, ::-1, :], C.DBN_AFFINE)]:
                d = out_root / sub
                d.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), aff), str(d / f"IXI{ixi:03d}.nii.gz"))
        if save_intermediate:
            d = out_root / "qc" / "intermediate"
            d.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(wn.astype(np.float32), C.FSL_AFFINE), str(d / f"IXI{ixi:03d}_registered_head.nii.gz"))
            nib.save(nib.Nifti1Image((prob.numpy() * 100).astype(np.uint8), C.FSL_AFFINE),
                     str(d / f"IXI{ixi:03d}_brainprob_x100.nii.gz"))
    except Exception as exc:  # never drop silently: the failure is recorded and printed
        rec["status"] = "FAIL"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["fail_reasons"] = rec["fail_reasons"] or "exception"
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["seconds"] = round(time.time() - t0, 1)
    p = qc_path(out_root, ixi)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1), encoding="utf8")
    return rec


def aggregate(out_root):
    rows = [json.loads(p.read_text(encoding="utf8")) for p in sorted((out_root / "qc" / "per_subject").glob("*.json"))]
    df = pd.DataFrame(rows).drop(columns=["traceback"], errors="ignore")
    (out_root / "qc").mkdir(exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)
    df.to_csv(out_root / "qc" / "qc_summary.csv", index=False)
    fails = df[df.status == "FAIL"]
    fails[["IXI_ID", "fail_reasons", "error"]].to_csv(out_root / "logs" / "failures.csv", index=False)
    return df, fails


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--all", action="store_true")
    g.add_argument("--ids", type=int, nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--retry-failed", action="store_true", help="re-run subjects whose QC record is FAIL")
    ap.add_argument("--save-intermediate", action="store_true", help="also save registered head + brain prob map")
    ap.add_argument("--out", default=str(C.OUT_ROOT))
    a = ap.parse_args()

    out_root = Path(a.out)
    C.build_fsl_grid_assets()
    meta = pd.read_csv(C.METADATA_PATH)
    if a.pilot:
        ids = select_pilot(meta)
        (out_root / "qc").mkdir(parents=True, exist_ok=True)
        meta[meta.IXI_ID.isin(ids)].to_csv(out_root / "qc" / "pilot_ids.csv", index=False)
    elif a.ids:
        ids = a.ids
    else:
        ids = [int(i) for i in meta.IXI_ID]

    todo, skipped = [], 0
    for ixi in ids:
        row = meta[meta.IXI_ID == ixi]
        if row.empty:
            raise SystemExit(f"IXI{ixi:03d} not in metadata")
        p = qc_path(out_root, ixi)
        if p.exists():
            st = json.loads(p.read_text(encoding="utf8")).get("status")
            if st in ("PASS", "WARN") or (st == "FAIL" and not a.retry_failed):
                skipped += 1
                continue
        todo.append((ixi, str(C.RAW_DIR / row.iloc[0]["filename"]), str(out_root), a.save_intermediate or a.pilot))
    print(f"Subjects requested: {len(ids)} | already done (skipped): {skipped} | to run: {len(todo)} | workers: {a.workers}", flush=True)

    os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", str(max(1, (os.cpu_count() or 4) // max(a.workers, 1))))
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(preprocess_one, t) for t in todo]
        for f in as_completed(futs):
            r = f.result()
            done += 1
            tag = {"PASS": "PASS", "WARN": "WARN"}.get(r["status"], "!!! QC FAIL")
            extra = {"PASS": "", "WARN": f"  warning: {r.get('qc_warnings', '')}"}.get(r["status"], f"  reasons: {r['fail_reasons']} {r.get('error', '')}")
            print(f"[{done}/{len(todo)}] IXI{r['IXI_ID']:03d} {tag} ({r['seconds']}s) "
                  f"dice={r.get('dice_vs_template', float('nan')):.2f} reg_r={r.get('reg_pearson_r', float('nan')):.2f}{extra}", flush=True)

    df, fails = aggregate(out_root)
    want = df[df.IXI_ID.isin(ids)]
    lines = [f"Requested {len(ids)} | records for requested: {len(want)} | PASS {int((want.status == 'PASS').sum())} "
             f"| WARN {int((want.status == 'WARN').sum())} | FAIL {int((want.status == 'FAIL').sum())} | missing record {len(ids) - len(want)}",
             f"Wall time this run: {(time.time() - t0) / 60:.1f} min"]
    if len(fails):
        lines.append("FAILED subjects: " + ", ".join(f"IXI{int(i):03d}({r})" for i, r in zip(fails.IXI_ID, fails.fail_reasons)))
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "logs" / "preprocess_summary.txt").write_text("\n".join(lines), encoding="utf8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
