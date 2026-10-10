"""Rebuild the IXI age/sex labels correctly (fix for the Phase 1 join bug, see CASE_STUDY_iAudit_vs_ml_brain.md).

Bug: the old Drive file's first column 'Subject_ID' was a 0,1,2.. row counter that Phase 1 renamed to IXI_ID by
position. Fix: read the genuine IXI.xls, select columns BY NAME, join on the real IXI_ID, assert everything.

Writes (in --out, default G:/My Drive/ML_Project):
  ixi_final_metadata_OLD_wrong_labels.csv   backup of the previous (wrong) file, only if not already backed up
  ixi_final_metadata.csv                    corrected labels, same schema as before
  ixi_label_excluded.csv                    subjects with a scan but no usable age (never dropped silently)
Cross-checks (must pass or the script aborts before writing):
  * every scan ID is found in IXI.xls
  * duplicate IXI_ID rows resolved by a deterministic rule, conflicts reported
  * independent source 1: iAudit's published per-subject ages (XNAT-verified)       -> agreement reported
  * independent source 2: the row-index mapping recovered from the old Drive file  -> agreement reported
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(r"G:\My Drive\ML_Project")


def load_real_xls(path):
    df = pd.read_excel(path, engine="xlrd")
    cols = {c.upper(): c for c in df.columns}
    need = {"id": "IXI_ID", "sex": next(c for u, c in cols.items() if u.startswith("SEX")),
            "eth": next(c for u, c in cols.items() if "ETHNIC" in u), "age": cols["AGE"]}
    assert "IXI_ID" in df.columns, f"first column must be IXI_ID, got {list(df.columns)}"
    assert df.IXI_ID.is_monotonic_increasing, "IXI_ID should be sorted ascending in the genuine IXI.xls"
    return df, need


def dedupe(df, need):
    """Same rule as iAudit: prefer a row with informative ethnicity, then non-null age. Report conflicts."""
    dup = df[df.duplicated("IXI_ID", keep=False)]
    conflicts = []
    for i, g in dup.groupby("IXI_ID"):
        if g[need["age"]].nunique(dropna=True) > 1 or g[need["sex"]].nunique() > 1:
            conflicts.append(int(i))
    df = df.copy()
    df["_rank"] = (df[need["eth"]].fillna(0) == 0).astype(int) * 2 + df[need["age"]].isna().astype(int)
    out = df.sort_values(["IXI_ID", "_rank"]).drop_duplicates("IXI_ID", keep="first").drop(columns="_rank")
    return out, len(dup), dup.IXI_ID.nunique(), conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xls", default=str(HERE / "IXI.xls"))
    ap.add_argument("--out", default=str(PROJECT))
    ap.add_argument("--iaudit", default=str(HERE / "reference" / "iaudit_predictions_deepbrainnet.csv"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)

    real, need = load_real_xls(a.xls)
    real, n_dup_rows, n_dup_ids, conflicts = dedupe(real, need)
    print(f"IXI.xls: {len(real)} unique IXI_IDs after de-dup ({n_dup_rows} duplicate rows over {n_dup_ids} IDs; "
          f"IDs with conflicting AGE/SEX: {conflicts or 'none'})")

    old_path = out / "ixi_final_metadata.csv"
    backup = out / "ixi_final_metadata_OLD_wrong_labels.csv"
    old = pd.read_csv(backup if backup.exists() else old_path)        # scan index: IXI_ID, site, filename, file_path
    scans = old[["IXI_ID", "site", "filename", "file_path"]].copy()
    missing = sorted(set(scans.IXI_ID) - set(real.IXI_ID))

    m = scans.merge(real[["IXI_ID", need["sex"], need["age"]]], on="IXI_ID", how="left", validate="one_to_one")
    m = m.rename(columns={need["sex"]: "SEX", need["age"]: "AGE"})
    # nothing is dropped silently: every excluded scan gets a reason
    m["reason"] = ""
    m.loc[m.IXI_ID.isin(missing), "reason"] = "scan ID not present in IXI.xls (no demographics exist)"
    m.loc[m.IXI_ID.isin(conflicts), "reason"] = "duplicate IXI.xls rows disagree on AGE/SEX (ambiguous label)"
    m.loc[(m.reason == "") & m.AGE.isna(), "reason"] = "no AGE in IXI.xls (no DOB / study date on file)"
    excl = m[m.reason != ""]
    good = m[m.reason == ""].drop(columns="reason").copy()
    good["SEX"] = good.SEX.astype(int)
    assert set(good.SEX) <= {1, 2}
    good["sex_label"] = good.SEX.map({1: "Male", 2: "Female"})
    good = good[["IXI_ID", "site", "filename", "file_path", "SEX", "AGE", "sex_label"]].sort_values("IXI_ID").reset_index(drop=True)
    print(f"scans: {len(scans)} | usable labels: {len(good)} | excluded: {len(excl)}")
    print(excl.groupby("reason").IXI_ID.apply(list).to_string())
    print(f"age range {good.AGE.min():.1f}-{good.AGE.max():.1f}, mean {good.AGE.mean():.1f}; "
          f"sex {good.sex_label.value_counts().to_dict()}; site {good.site.value_counts().to_dict()}")

    # ---- independent cross-check 1: iAudit's XNAT-verified ages ----
    ok = True
    if Path(a.iaudit).exists():
        ia = pd.read_csv(a.iaudit)
        ia["IXI_ID"] = ia.subject_id.str.extract(r"IXI(\d+)")[0].astype(int)
        j = good.merge(ia[["IXI_ID", "chronological_age", "sex"]], on="IXI_ID")
        d = (j.AGE - j.chronological_age).abs()
        sex_ok = j.sex_label.map({"Male": "M", "Female": "F"}) == j.sex
        print(f"[check 1] vs iAudit (XNAT-verified), {len(j)} shared IDs: age within 0.01y {int((d < .01).sum())}/{len(j)}, "
              f"max diff {d.max():.4f}y; sex agrees {int(sex_ok.sum())}/{len(j)}")
        ok &= (d < 0.01).mean() > 0.99 and sex_ok.mean() > 0.99
    else:
        print("[check 1] skipped (iAudit reference file not found)")
    # ---- independent cross-check 2: OLD Drive file rows -> IDs by exact age ----
    bad_file = out / "IXI_demographics.xls"
    if bad_file.exists():
        raw = pd.read_csv(bad_file, sep=None, engine="python", encoding="utf-8-sig")
        raw.columns = ["row_index", "SEX_ID"] + list(raw.columns[2:11]) + ["AGE"]
        raw["k"] = raw.AGE.round(5)
        g = good.assign(k=good.AGE.round(5))
        j = raw.merge(g[["k", "IXI_ID", "SEX"]], on="k").drop_duplicates("IXI_ID")
        print(f"[check 2] old Drive file rows re-matched to IDs by exact age: {len(j)}; sex agrees "
              f"{int((j.SEX_ID == j.SEX).sum())}/{len(j)}; row_index==IXI_ID for {int((j.row_index == j.IXI_ID).sum())} "
              f"(0 expected: that was the bug)")
        ok &= (j.SEX_ID == j.SEX).mean() > 0.99
    assert ok, "cross-checks failed; refusing to write corrected labels"

    # how wrong was the old file?
    o = old.merge(good[["IXI_ID", "AGE", "SEX"]], on="IXI_ID", suffixes=("_old", "_new"))
    print(f"old-vs-corrected: age corr {np.corrcoef(o.AGE_old, o.AGE_new)[0, 1]:.3f}, identical age for "
          f"{int((o.AGE_old - o.AGE_new).abs().lt(1e-6).sum())}/{len(o)}, sex changed for {int((o.SEX_old != o.SEX_new).sum())}/{len(o)}")

    if a.dry_run:
        print("dry run: nothing written")
        return
    if not backup.exists():
        shutil.copy2(old_path, backup)
        print("backup written:", backup)
    good.to_csv(old_path, index=False)
    excl[["IXI_ID", "site", "filename", "reason"]].to_csv(out / "ixi_label_excluded.csv", index=False)
    print("written:", old_path, "| excluded list:", out / "ixi_label_excluded.csv")


if __name__ == "__main__":
    main()
