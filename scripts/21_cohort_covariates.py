"""Phase 10e - Who did each site recruit? Cohort covariates, ethnicity, and the IOP offset.

IXI.xls records ethnicity, highest qualification, occupation, marital status, height and weight.
1. Describe each site's cohort and test site differences (age/sex adjusted).
2. Ethnicity fairness: brain-age gap of White vs non-White participants (descriptive; small n).
3. Does adjusting for cohort covariates (university degree, BMI, ethnicity) shrink the IOP offset?
   Done on three prediction sources: the zero-shot UK Biobank model (all subjects 42-82, never trained
   on IXI), the main model's held-out subjects, and the 5-fold CV out-of-sample predictions if present.

Usage: python scripts/21_cohort_covariates.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ETHNIC = {1: "White", 4: "Black", 3: "Asian", 5: "Chinese", 6: "Other"}
T = "C(site, Treatment('Guys'))[T.IOP]"


def load_cohort(P):
    x = pd.read_excel(P / "IXI.xls")
    x = x.rename(columns={"ETHNIC_ID": "eth", "QUALIFICATION_ID": "qual", "OCCUPATION_ID": "occ",
                          "MARITAL_ID": "mar", "HEIGHT": "h", "WEIGHT": "w"})[["IXI_ID", "eth", "qual", "occ", "mar", "h", "w"]]
    x = x.drop_duplicates("IXI_ID")
    for c in ["eth", "qual", "occ", "mar"]:
        x.loc[~x[c].between(1, 9), c] = np.nan
    x.loc[(x.h < 120) | (x.h > 220), "h"] = np.nan
    x.loc[(x.w < 30) | (x.w > 200), "w"] = np.nan
    x["bmi"] = x.w / (x.h / 100) ** 2
    x["degree"] = (x.qual == 5).astype(float).where(x.qual.notna())
    x["white"] = (x.eth == 1).astype(float).where(x.eth.notna())
    x["ethnicity"] = x.eth.map(ETHNIC)
    return x


def iop(df, outcome, extra=""):
    m = smf.ols(f"{outcome} ~ C(site, Treatment('Guys')) + C(sex) + age{extra}", data=df).fit(cov_type="HC3")
    return {"n": int(m.nobs), "iop_vs_guys": float(m.params[T]), "ci": [float(x) for x in m.conf_int().loc[T]],
            "p": float(m.pvalues[T])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    P = p.parse_args().project_dir
    out_dir = P / "results" / "phase10e_cohort"; out_dir.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(P)
    meta = pd.concat([pd.read_csv(P / f"ixi_{s}.csv").assign(split=s) for s in ["train", "val", "test"]])
    meta = meta.rename(columns={"AGE": "age", "sex_label": "sex"})[["IXI_ID", "site", "age", "sex", "split"]]
    d = meta.merge(cohort, on="IXI_ID", how="left")
    res = {}

    # 1. Cohort description and site differences.
    desc = d.groupby("site").agg(n=("age", "size"), age=("age", "mean"), degree=("degree", "mean"),
                                 white=("white", "mean"), bmi=("bmi", "mean"),
                                 missing_eth=("eth", lambda s: s.isna().mean()))
    res["cohort_by_site"] = desc.round(3).reset_index().to_dict("records")
    res["ethnicity_counts"] = pd.crosstab(d.site, d.ethnicity).to_dict()
    res["qualification_counts"] = pd.crosstab(d.site, d.qual).to_dict()
    res["site_differences_age_sex_adjusted"] = {}
    for c in ["degree", "white", "bmi"]:
        m = smf.ols(f"{c} ~ C(site, Treatment('Guys')) + C(sex) + age", data=d).fit(cov_type="HC3")
        res["site_differences_age_sex_adjusted"][c] = {"IOP_vs_Guys": float(m.params[T]), "p": float(m.pvalues[T]),
                                                       "n": int(m.nobs)}

    # 2/3. Prediction sources.
    sources = {}
    z = pd.read_csv(P / "results" / "phase3" / "zero_shot_predictions.csv").rename(columns={"predicted_age": "pred"})
    z = z[(z.age >= 42) & (z.age <= 82)][["IXI_ID", "pred"]]
    sources["zero-shot UK Biobank SFCN, all subjects 42-82 (outcome: raw prediction, age-adjusted)"] = (
        d.merge(z, on="IXI_ID"), "pred")
    bc = json.loads((P / "results" / "phase5" / "summary.json").read_text())["bias_correction"]
    f = np.load(P / "results" / "phase9a_site_probe" / "features.npz", allow_pickle=True)
    mm = d.merge(pd.DataFrame({"IXI_ID": f["ids"].astype(int), "pred": f["pred"]}), on="IXI_ID")
    mm["bag"] = (mm.pred - bc["intercept"]) / bc["slope"] - mm.age
    sources["main model, held-out val+test (outcome: corrected gap)"] = (mm[mm.split != "train"], "bag")
    folds = sorted((P / "results" / "phase10_cv" / "seed0").glob("fold*/outer_predictions.csv"))
    if len(folds) == 5:
        parts = []
        for fp in folds:
            o, iv = pd.read_csv(fp), pd.read_csv(fp.parent / "inner_val_predictions.csv")
            a, b = np.polyfit(iv.age, iv.pred, 1)
            parts.append(o.assign(bag=(o.pred - b) / a - o.age)[["IXI_ID", "bag"]])
        sources["5-fold CV out-of-sample (outcome: corrected gap)"] = (d.merge(pd.concat(parts), on="IXI_ID"), "bag")

    res["iop_offset_with_cohort_covariates"] = {}
    for name, (df, y) in sources.items():
        cc = df.dropna(subset=["degree", "bmi", "white"])
        res["iop_offset_with_cohort_covariates"][name] = {
            "baseline (all subjects)": iop(df, y),
            "baseline (complete covariates)": iop(cc, y),
            "+ degree": iop(cc, y, " + degree"),
            "+ BMI": iop(cc, y, " + bmi"),
            "+ ethnicity (White)": iop(cc, y, " + white"),
            "+ all three": iop(cc, y, " + degree + bmi + white"),
        }
        e = df.dropna(subset=["white"])
        m = smf.ols(f"{y} ~ white + C(site) + C(sex) + age", data=e).fit(cov_type="HC3")
        res.setdefault("ethnicity_effect_nonwhite_minus_white", {})[name] = {
            "est": float(-m.params["white"]), "ci": [float(-m.conf_int().loc["white"][1]), float(-m.conf_int().loc["white"][0])],
            "p": float(m.pvalues["white"]), "n_nonwhite": int((e.white == 0).sum()), "n": int(m.nobs)}

    (out_dir / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    print(desc.round(2).to_string())
    print(json.dumps({k: res[k] for k in ["site_differences_age_sex_adjusted", "iop_offset_with_cohort_covariates",
                                          "ethnicity_effect_nonwhite_minus_white"]}, indent=1))


if __name__ == "__main__":
    main()
