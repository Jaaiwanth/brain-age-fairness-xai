"""Rebuild ixi_final_metadata.csv with age/sex from the official IXI.xls.

Why: Phase 1 took age/sex from a third-party demographics file whose `Subject_ID`
column is not the IXI ID, so every scan was paired with another person's age and sex
(0/487 ages matched IXI.xls; sex matched at chance). This script keeps the scan index
(IXI_ID, site, filename - taken from the scan file names, which are correct) and
replaces the labels with the official ones.

Subjects are dropped when IXI.xls has no row for them, no age, or contradictory
duplicate rows (different ages or sexes for the same IXI_ID).

The old file is kept as ixi_final_metadata_OLD_wrong_labels.csv.

Usage:
    python scripts/01b_official_labels.py --project-dir D:/ML_Project
"""

import argparse
import os
import urllib.request
from pathlib import Path

import pandas as pd

IXI_XLS_URL = "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI.xls"
IN_COLAB = Path("/content").exists()
DEFAULT_PROJECT_DIR = (
    Path("/content/drive/MyDrive/ML_Project") if IN_COLAB else Path("G:/My Drive/ML_Project")
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-dir", type=Path,
                        default=Path(os.environ.get("BRAINAGE_PROJECT_DIR", DEFAULT_PROJECT_DIR)))
    args = parser.parse_args()
    project_dir = args.project_dir

    xls = project_dir / "IXI.xls"
    if not xls.exists():
        print(f"Downloading {IXI_XLS_URL} ...")
        urllib.request.urlretrieve(IXI_XLS_URL, xls)

    meta_path = project_dir / "ixi_final_metadata.csv"
    old_path = project_dir / "ixi_final_metadata_OLD_wrong_labels.csv"
    if not old_path.exists():
        os.replace(meta_path, old_path)
        print(f"Kept the old file as {old_path.name}")
    scans = (pd.read_csv(old_path)[["IXI_ID", "site", "filename"]]
             .drop_duplicates("IXI_ID").sort_values("IXI_ID"))

    off = pd.read_excel(xls).rename(columns={"SEX_ID (1=m, 2=f)": "SEX"})
    off = off[off.IXI_ID.isin(scans.IXI_ID)]
    groups = off.groupby("IXI_ID")
    n_ages = groups.AGE.apply(lambda s: s.dropna().round(4).nunique())
    n_sexes = groups.SEX.nunique()

    no_row = sorted(set(scans.IXI_ID) - set(off.IXI_ID))
    no_age = sorted(n_ages[n_ages == 0].index)
    conflict = sorted(set(n_ages[n_ages > 1].index) | set(n_sexes[n_sexes > 1].index))
    clean_ids = set(scans.IXI_ID) - set(no_row) - set(no_age) - set(conflict)

    labels = (off[off.IXI_ID.isin(clean_ids)].dropna(subset=["AGE"])
              .drop_duplicates("IXI_ID")[["IXI_ID", "SEX", "AGE"]])
    out = scans.merge(labels, on="IXI_ID", how="inner")
    out["SEX"] = out.SEX.astype(int)
    out["sex_label"] = out.SEX.map({1: "Male", 2: "Female"})
    out.to_csv(meta_path, index=False)

    print(f"Scanned subjects:        {len(scans)}")
    print(f"No row in IXI.xls:       {len(no_row)} {no_row}")
    print(f"No age in IXI.xls:       {len(no_age)} {no_age}")
    print(f"Contradictory rows:      {len(conflict)} {conflict}")
    print(f"Written with labels:     {len(out)} -> {meta_path}")
    print(out.groupby(["site", "sex_label"]).size().unstack().to_string())


if __name__ == "__main__":
    main()
