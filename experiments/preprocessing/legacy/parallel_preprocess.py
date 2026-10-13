from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import ants
import antspynet
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import load_mni152_template


TARGET_SHAPE = (160, 192, 160)

PROJECT_DIR = Path(r"G:\My Drive\ML_Project")

DATA_DIR = PROJECT_DIR / "IXI_data"
PREPROCESSED_DIR = PROJECT_DIR / "IXI_preprocessed"
METADATA_PATH = PROJECT_DIR / "ixi_final_metadata.csv"


MAX_WORKERS = 1

def center_crop_or_pad(volume, target_shape):

    result = volume

    for axis, target in enumerate(target_shape):

        current = result.shape[axis]

        if current > target:

            start = (current - target) // 2

            result = np.take(
                result,
                range(start, start + target),
                axis=axis
            )

        elif current < target:

            pad_before = (target - current) // 2
            pad_after = target - current - pad_before

            pad_width = [(0, 0)] * result.ndim

            pad_width[axis] = (
                pad_before,
                pad_after
            )

            result = np.pad(
                result,
                pad_width,
                mode="constant",
                constant_values=0
            )

    return result



def preprocess_scan(scan_path, mni_template):

    raw = ants.image_read(str(scan_path))


    bias_corrected = ants.n4_bias_field_correction(
        raw
    )

    registration = ants.registration(
        fixed=mni_template,
        moving=bias_corrected,
        type_of_transform="Affine"
    )

    registered = registration["warpedmovout"]

    brain_mask_prob = antspynet.brain_extraction(
        registered,
        modality="t1"
    )

    brain_mask = ants.threshold_image(
        brain_mask_prob,
        0.5,
        1.0,
        1,
        0
    )

    skull_stripped = registered * brain_mask


    volume = skull_stripped.numpy()

    brain_voxels = volume[volume > 0]

    if brain_voxels.size == 0:

        raise ValueError(
            "Skull-stripping produced an empty brain mask."
        )


    low, high = np.percentile(
        brain_voxels,
        [1, 99]
    )

    volume = np.clip(
        volume,
        low,
        high
    )

    volume = (
        volume - low
    ) / max(
        high - low,
        1e-6
    )



    volume = center_crop_or_pad(
        volume,
        TARGET_SHAPE
    )

    return volume.astype(
        np.float32
    )


def process_one(row):

    ixi_id = int(
        row["IXI_ID"]
    )

    out_path = (
        PREPROCESSED_DIR
        / f"IXI{ixi_id:03d}.npy"
    )


    if out_path.exists():

        return (
            ixi_id,
            "skipped",
            0
        )


    try:

        mni_nib = load_mni152_template(
            resolution=1
        )

        mni_path = (
            Path.cwd()
            / "MNI152_T1_1mm.nii.gz"
        )

        if not mni_path.exists():

            nib.save(
                mni_nib,
                mni_path
            )

        mni_template = ants.image_read(
            str(mni_path)
        )



        start = time.time()


        volume = preprocess_scan(
            row["file_path"],
            mni_template
        )


        np.save(
            out_path,
            volume
        )


        elapsed = (
            time.time()
            - start
        )


        return (
            ixi_id,
            "success",
            elapsed
        )


    except Exception as exc:

        return (
            ixi_id,
            "failed",
            str(exc)
        )


if __name__ == "__main__":

    print("=" * 60)
    print("IXI BRAIN MRI PREPROCESSING")
    print("=" * 60)


    PREPROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )



    final_df = pd.read_csv(
        METADATA_PATH
    )



    final_df["file_path"] = final_df[
        "file_path"
    ].apply(
        lambda p: str(
            DATA_DIR
            / Path(str(p)).name
        )
    )



    remaining_rows = []

    already_processed = 0

    for _, row in final_df.iterrows():

        out_path = (
            PREPROCESSED_DIR
            / f"IXI{int(row['IXI_ID']):03d}.npy"
        )

        if out_path.exists():

            already_processed += 1

        else:

            remaining_rows.append(
                row.to_dict()
            )



    print(
        f"Total subjects in metadata: "
        f"{len(final_df)}"
    )

    print(
        f"Already processed: "
        f"{already_processed}"
    )

    print(
        f"Remaining to process: "
        f"{len(remaining_rows)}"
    )

    print(
        f"Parallel workers: "
        f"{MAX_WORKERS}"
    )

    print()


    if not remaining_rows:

        print(
            "🎉 All scans are already "
            "preprocessed!"
        )

        raise SystemExit

    start_total = time.time()

    success_count = 0
    failed_count = 0

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [

            executor.submit(
                process_one,
                row
            )

            for row in remaining_rows

        ]


        for future in as_completed(
            futures
        ):

            result = future.result()

            ixi_id = result[0]
            status = result[1]


            if status == "success":

                success_count += 1

                print(
                    f" IXI{ixi_id:03d} "
                    f"finished in "
                    f"{result[2]:.1f}s "
                    f""
                    f"({success_count}/"
                    f"{len(remaining_rows)})"
                )


            elif status == "skipped":

                print(
                    f" IXI{ixi_id:03d} "
                    f"skipped"
                )

            else:

                failed_count += 1

                print(
                    f" IXI{ixi_id:03d}: "
                    f"{result[2]}"
                )


    total_time = (
        time.time()
        - start_total
    )


    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        f"Successful: "
        f"{success_count}"
    )

    print(
        f"Failed: "
        f"{failed_count}"
    )

    print(
        f"Already processed before run: "
        f"{already_processed}"
    )

    print(
        f"Total available after run: "
        f"{already_processed + success_count}"
    )

    print(
        f"Total time: "
        f"{total_time / 60:.1f} minutes"
    )

    print("=" * 60)