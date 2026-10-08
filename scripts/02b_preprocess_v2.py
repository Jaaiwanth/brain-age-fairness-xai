"""Phase 2 (v2) - Re-preprocess IXI T1 scans so every brain is aligned to SFCN's template.

Why v2: the first pipeline registered whole heads (with skull) to nilearn's
brain-only MNI template. QC showed brains up to 18 mm out of place, inconsistent
sizes, some brains cut off, and IOP scans worst of all.

v2 pipeline, per scan:
    1. N4 bias-field correction
    2. Skull-strip in the scan's own (native) space (ANTsPyNet)
    3. Affine-register the brain-only image to the brain-only FSL MNI152 1mm
       template (MNI152NLin6Asym from TemplateFlow, 182x218x182) - the template
       the pretrained SFCN was built on
    4. Centre-crop to 160x192x160 (same crop as SFCN's own code)
    5. Keep bias-corrected intensities; the training script divides by the mean.
    6. Quality check: Dice overlap with the template brain mask, centre offset,
       brain size, edge cut-off, native brain volume.

Writes to a NEW folder (IXI_preprocessed_v2); the old IXI_preprocessed folder is
never touched. Safe to stop and re-run: finished scans are skipped.

Usage:
    python scripts/02b_preprocess_v2.py --smoke    # 3 scans (one per site), separate output folder
    python scripts/02b_preprocess_v2.py            # all subjects
    python scripts/02b_preprocess_v2.py --workers 1   # if the laptop runs out of memory
"""

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

IN_COLAB = Path("/content").exists()
DEFAULT_PROJECT_DIR = (
    Path("/content/drive/MyDrive/ML_Project") if IN_COLAB
    else Path("G:/My Drive/ML_Project")
)

TEMPLATE_NAME = "MNI152NLin6Asym"            # FSL MNI152 1mm (what SFCN used)
TEMPLATE_SHAPE = (182, 218, 182)
TARGET_SHAPE = (160, 192, 160)
REGISTRATION_TYPE = "antsRegistrationSyNQuick[a]"   # rigid + affine, multi-resolution
BRAIN_PROB_THRESHOLD = 0.5

# QC thresholds. Flagged scans are reviewed by eye, not dropped automatically.
QC_MIN_DICE = 0.85               # overlap of subject brain mask with template brain mask
QC_MAX_OFFSET_MM = 5.0           # distance between subject and template brain centres
QC_NATIVE_VOLUME_ML = (900, 2000)  # plausible adult brain volume before registration

SMOKE_SUFFIX = "_smoke"


def parse_args():
    parser = argparse.ArgumentParser(description="Re-preprocess IXI T1 scans (v2).")
    parser.add_argument("--smoke", action="store_true",
                        help="Process 3 scans (one per site) into a separate *_smoke folder.")
    parser.add_argument("--project-dir", type=Path,
                        default=Path(os.environ.get("BRAINAGE_PROJECT_DIR", DEFAULT_PROJECT_DIR)))
    parser.add_argument("--workers", type=int, default=2,
                        help="Scans processed in parallel. Each needs ~3-4 GB RAM; use 1 if memory runs out.")
    return parser.parse_args()


# =============================================================================
# Per-scan processing (runs inside worker processes)
# =============================================================================

_TEMPLATE = {}


def init_worker(template_t1_path, template_mask_path, itk_threads):
    # Set before ANTs is imported so each worker gets its share of CPU threads.
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(itk_threads)
    import ants

    t1 = ants.image_read(str(template_t1_path))
    mask = ants.image_read(str(template_mask_path))
    if tuple(t1.shape) != TEMPLATE_SHAPE:
        raise ValueError(f"Template shape {t1.shape}, expected {TEMPLATE_SHAPE}")
    _TEMPLATE["mask"] = ants.threshold_image(mask, 0.5, 1e9, 1, 0)
    _TEMPLATE["brain"] = t1 * _TEMPLATE["mask"]
    mask_np = center_crop(_TEMPLATE["mask"].numpy() > 0, TARGET_SHAPE)
    _TEMPLATE["mask_cropped"] = mask_np
    _TEMPLATE["centre"], _TEMPLATE["size"] = centre_and_size(mask_np)
    # The template brain itself reaches the bottom face after the standard crop (the
    # lower brainstem extends below it), so only faces it does NOT touch count as cut-off.
    _TEMPLATE["faces_touched"] = {face for face, n in face_voxels(mask_np).items() if n}


def center_crop(volume, target_shape):
    for axis, target in enumerate(target_shape):
        current = volume.shape[axis]
        if current < target:
            raise ValueError(f"Axis {axis} is {current}, smaller than crop size {target}")
        start = (current - target) // 2
        volume = np.take(volume, range(start, start + target), axis=axis)
    return volume


def centre_and_size(mask):
    idx = np.argwhere(mask)
    return idx.mean(axis=0), idx.max(axis=0) - idx.min(axis=0) + 1


def face_voxels(mask):
    """Number of mask voxels on each of the six faces of the volume."""
    return {
        "x_first": int(mask[0].sum()), "x_last": int(mask[-1].sum()),
        "y_first": int(mask[:, 0].sum()), "y_last": int(mask[:, -1].sum()),
        "z_first": int(mask[:, :, 0].sum()), "z_last": int(mask[:, :, -1].sum()),
    }


def save_npy_atomic(path, array):
    tmp = path.with_name(path.stem + ".partial")
    with open(tmp, "wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


def preprocess_one(ixi_id, site, scan_path, out_path, qc_path):
    import ants
    import antspynet

    start = time.time()
    raw = ants.image_read(str(scan_path))

    # 1. Bias correction
    n4 = ants.n4_bias_field_correction(raw)

    # 2. Skull-strip in native space
    prob = antspynet.brain_extraction(n4, modality="t1")
    mask = ants.threshold_image(prob, BRAIN_PROB_THRESHOLD, 1.0, 1, 0)
    mask = ants.iMath(mask, "GetLargestComponent")
    mask = ants.iMath(mask, "FillHoles")
    native_volume_ml = float(mask.numpy().sum() * np.prod(mask.spacing) / 1000.0)
    brain = n4 * mask

    # 3. Brain-to-brain affine registration to the template
    reg = ants.registration(fixed=_TEMPLATE["brain"], moving=brain,
                            type_of_transform=REGISTRATION_TYPE)
    warped_brain = ants.apply_transforms(fixed=_TEMPLATE["brain"], moving=brain,
                                         transformlist=reg["fwdtransforms"], interpolator="linear")
    warped_mask = ants.apply_transforms(fixed=_TEMPLATE["brain"], moving=mask,
                                        transformlist=reg["fwdtransforms"], interpolator="nearestNeighbor")

    # 4-5. Mask, crop, keep intensities (no clipping/rescaling; linear interpolation
    # can produce tiny negatives at the edge, so those are set to 0)
    mask_np = center_crop(warped_mask.numpy() > 0.5, TARGET_SHAPE)
    volume = center_crop(warped_brain.numpy(), TARGET_SHAPE)
    volume = np.where(mask_np, np.clip(volume, 0, None), 0).astype(np.float32)
    if not mask_np.any() or volume.mean() <= 0:
        raise ValueError("Empty brain after registration")

    # 6. QC
    template_mask = _TEMPLATE["mask_cropped"]
    dice = float(2 * (mask_np & template_mask).sum() / (mask_np.sum() + template_mask.sum()))
    centre, size = centre_and_size(mask_np)
    offset_mm = float(np.linalg.norm(centre - _TEMPLATE["centre"]))
    faces = face_voxels(mask_np)
    cut_faces = sorted(f for f, n in faces.items() if n and f not in _TEMPLATE["faces_touched"])
    touches_edge = bool(cut_faces)
    reasons = []
    if dice < QC_MIN_DICE:
        reasons.append(f"dice<{QC_MIN_DICE}")
    if offset_mm > QC_MAX_OFFSET_MM:
        reasons.append(f"offset>{QC_MAX_OFFSET_MM}mm")
    if touches_edge:
        reasons.append("cut_off_at_edge")
    if not QC_NATIVE_VOLUME_ML[0] <= native_volume_ml <= QC_NATIVE_VOLUME_ML[1]:
        reasons.append("native_volume_out_of_range")

    save_npy_atomic(out_path, volume)
    qc = {
        "IXI_ID": int(ixi_id), "site": site,
        "dice": round(dice, 4), "offset_mm": round(offset_mm, 2),
        "size_x": int(size[0]), "size_y": int(size[1]), "size_z": int(size[2]),
        "template_size_x": int(_TEMPLATE["size"][0]), "template_size_y": int(_TEMPLATE["size"][1]),
        "template_size_z": int(_TEMPLATE["size"][2]),
        "touches_edge": touches_edge, "cut_faces": ";".join(cut_faces),
        "bottom_face_voxels": faces["z_first"], "native_brain_volume_ml": round(native_volume_ml, 1),
        "qc_flag": bool(reasons), "qc_reasons": ";".join(reasons),
        "seconds": round(time.time() - start, 1),
    }
    qc_path.write_text(json.dumps(qc))
    return qc


def process_task(task):
    try:
        return "ok", preprocess_one(**task)
    except Exception as exc:
        return "failed", {"IXI_ID": task["ixi_id"], "site": task["site"],
                          "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(limit=3)}


# =============================================================================
# Main
# =============================================================================

def get_template_paths():
    from templateflow import api as tflow

    t1 = tflow.get(TEMPLATE_NAME, resolution=1, suffix="T1w", desc=None, extension=".nii.gz")
    mask = tflow.get(TEMPLATE_NAME, resolution=1, suffix="mask", desc="brain", extension=".nii.gz")
    for name, path in [("T1w", t1), ("brain mask", mask)]:
        if not path or isinstance(path, list):
            raise RuntimeError(f"Could not get a unique {TEMPLATE_NAME} {name} from TemplateFlow: {path}")
    return Path(t1), Path(mask)


def resolve_scan_path(path):
    """Return the .nii file for an IXI_data entry.

    The entry is either the scan file itself, or (as unpacked from the Kaggle mirror)
    a folder named like the scan, e.g. IXI002-Guys-0828-T1.nii/, holding one .nii file.
    """
    if not path.is_dir():
        return path
    inner = sorted(p for p in path.iterdir()
                   if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz")))
    if len(inner) != 1:
        raise FileNotFoundError(f"{path} is a folder with {len(inner)} .nii files; expected exactly 1")
    return inner[0]


def save_worst_montage(qc_df, volume_dir, out_path, n=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    worst = qc_df.sort_values("dice").head(n)
    fig, axes = plt.subplots(len(worst), 3, figsize=(7.5, 2.5 * len(worst)), squeeze=False)
    for row, (_, r) in zip(axes, worst.iterrows()):
        v = np.load(volume_dir / f"IXI{int(r.IXI_ID):03d}.npy")
        cx, cy, cz = (s // 2 for s in v.shape)
        for ax, sl in zip(row, [v[cx], v[:, cy], v[:, :, cz]]):
            ax.imshow(np.rot90(sl), cmap="gray")
            ax.axis("off")
        row[0].set_title(f"IXI{int(r.IXI_ID):03d} {r.site} dice={r.dice:.3f} offset={r.offset_mm:.1f}mm",
                         loc="left", fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    args = parse_args()
    project_dir = args.project_dir
    suffix = SMOKE_SUFFIX if args.smoke else ""
    data_dir = project_dir / "IXI_data"
    out_dir = project_dir / f"IXI_preprocessed_v2{suffix}"
    qc_dir = out_dir / "qc"
    qc_csv = project_dir / f"phase2_v2_qc{suffix}.csv"
    failures_csv = project_dir / f"phase2_v2_failures{suffix}.csv"
    montage_png = project_dir / f"phase2_v2_qc_worst{suffix}.png"

    metadata_path = project_dir / "ixi_final_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    meta = pd.read_csv(metadata_path)
    if args.smoke:
        meta = meta.groupby("site").sample(1, random_state=42)

    # The CSV's file_path column holds machine-specific paths, so rebuild from the filename.
    meta["scan_path"] = meta["filename"].map(lambda f: data_dir / f)
    missing = meta[~meta["scan_path"].map(Path.exists)]
    if len(missing):
        raise FileNotFoundError(f"{len(missing)} raw scans missing in {data_dir}, e.g. {missing.iloc[0]['filename']}")
    meta["scan_path"] = meta["scan_path"].map(resolve_scan_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    tasks, already_done = [], 0
    for _, r in meta.iterrows():
        ixi_id = int(r["IXI_ID"])
        out_path = out_dir / f"IXI{ixi_id:03d}.npy"
        qc_path = qc_dir / f"IXI{ixi_id:03d}.json"
        if out_path.exists() and qc_path.exists():
            already_done += 1
            continue
        tasks.append({"ixi_id": ixi_id, "site": r["site"], "scan_path": r["scan_path"],
                      "out_path": out_path, "qc_path": qc_path})

    print("=" * 60)
    print(f"IXI PREPROCESSING v2{' (SMOKE TEST)' if args.smoke else ''}")
    print("=" * 60)
    print(f"Project dir:      {project_dir}")
    print(f"Output dir:       {out_dir}")
    print(f"Subjects:         {len(meta)}")
    print(f"Already done:     {already_done}")
    print(f"To process:       {len(tasks)}")

    failures = []
    if tasks:
        print("Fetching template from TemplateFlow (downloads once)...")
        t1_path, mask_path = get_template_paths()
        # Smoke test runs single-process so first-time model downloads can't collide.
        workers = 1 if args.smoke else max(1, min(args.workers, len(tasks)))
        itk_threads = max(1, (os.cpu_count() or 2) // workers)
        print(f"Template:         {t1_path.name}")
        print(f"Workers:          {workers} x {itk_threads} threads")
        print()

        start_all, done = time.time(), 0
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=(t1_path, mask_path, itk_threads)) as pool:
            futures = [pool.submit(process_task, t) for t in tasks]
            for future in as_completed(futures):
                status, info = future.result()
                done += 1
                elapsed = time.time() - start_all
                eta_min = elapsed / done * (len(tasks) - done) / 60
                if status == "ok":
                    flag = f"  FLAG: {info['qc_reasons']}" if info["qc_flag"] else ""
                    print(f"[{done}/{len(tasks)}] IXI{info['IXI_ID']:03d} {info['site']:<4} "
                          f"dice={info['dice']:.3f} offset={info['offset_mm']:.1f}mm "
                          f"{info['seconds']:.0f}s | ETA {eta_min:.0f} min{flag}")
                else:
                    failures.append(info)
                    print(f"[{done}/{len(tasks)}] IXI{info['IXI_ID']:03d} FAILED: {info['error']}")

    # Combine every per-scan QC file (including scans finished in earlier runs).
    qc_rows = [json.loads(p.read_text()) for p in sorted(qc_dir.glob("IXI*.json"))]
    qc_ids = {row["IXI_ID"] for row in qc_rows}
    qc_df = pd.DataFrame(qc_rows)
    if len(qc_df):
        qc_df = qc_df.merge(meta[["IXI_ID", "AGE", "sex_label"]], on="IXI_ID", how="left")
        qc_df.to_csv(qc_csv, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(failures_csv, index=False)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Preprocessed:  {len(qc_ids & set(meta['IXI_ID'].astype(int)))}/{len(meta)}")
    print(f"Failed:        {len(failures)}" + (f"  -> {failures_csv}" if failures else ""))
    if len(qc_df):
        print(f"Dice with template brain mask: median {qc_df.dice.median():.3f}, "
              f"min {qc_df.dice.min():.3f}")
        print(f"Centre offset (mm):            median {qc_df.offset_mm.median():.1f}, "
              f"max {qc_df.offset_mm.max():.1f}")
        print(f"Cut off at edge:               {int(qc_df.touches_edge.sum())}")
        print(f"Flagged for review:            {int(qc_df.qc_flag.sum())}")
        print("\nBy site (median dice / median offset mm / flagged):")
        by_site = qc_df.groupby("site").agg(n=("IXI_ID", "size"), dice=("dice", "median"),
                                            offset_mm=("offset_mm", "median"), flagged=("qc_flag", "sum"))
        print(by_site.round(3).to_string())
        save_worst_montage(qc_df, out_dir, montage_png)
        print(f"\nQC table:      {qc_csv}")
        print(f"Worst scans:   {montage_png}")
    print("=" * 60)


if __name__ == "__main__":
    main()
