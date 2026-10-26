"""Phase 10f - From decodability to use: erase site from the features and re-predict age.

Phase 9a showed the site is decodable from the 64-d features the age head reads. To test whether the
prediction (and the IOP offset) USES that information, site is removed from the features in three
ways, each fitted on train subjects only, and age is re-predicted through the model's own unchanged
head (which reproduces the model exactly on the original features):
    ComBat      location/scale, preserving age, age^2, sex (Phase 9e)
    ComBat-GAM  the same with a B-spline of age (4 df), so nonlinear ageing is preserved
    LEACE       least-squares concept erasure (Belrose et al. 2023): the closed-form affine map that
                makes site linearly unpredictable from the features with the smallest change
For each: site-probe balanced accuracy (age/sex removed), held-out MAE and the age-adjusted IOP-vs-Guys
gap. A permutation null for the probe (site labels shuffled within train) is also reported.

Usage: python scripts/22_site_erasure.py --project-dir D:/ML_Project
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from patsy import dmatrix

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hm = _load("hm", "12_harmonise_features.py")
sp = hm.sp
ft = hm.ft
SITES = hm.SITES
T = "C(site, Treatment('Guys'))[T.IOP]"


def leace_fit(X, Z):
    mu = X.mean(0)
    Xc, Zc = X - mu, Z - Z.mean(0)
    cov = Xc.T @ Xc / len(X)
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > vals.max() * 1e-8
    W = vecs[:, keep] @ np.diag(vals[keep] ** -0.5) @ vecs[:, keep].T
    W_pinv = vecs[:, keep] @ np.diag(vals[keep] ** 0.5) @ vecs[:, keep].T
    u, s, _ = np.linalg.svd(W @ (Xc.T @ Zc / len(X)), full_matrices=False)
    u = u[:, s > s.max() * 1e-8]
    P = u @ u.T
    return lambda Y: Y - (Y - mu) @ (W_pinv @ P @ W).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official/best.pt"))
    args = p.parse_args()
    P = args.project_dir
    rng = np.random.default_rng(0)
    splits = {s: ft.load_split(P, s) for s in ["train", "val", "test"]}
    df = pd.concat([d[~d.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)].assign(split=s) for s, d in splits.items()], ignore_index=True)
    feats = np.load(P / "results" / "phase9a_site_probe" / "features.npz")
    assert np.array_equal(feats["ids"], df.IXI_ID.to_numpy())
    E = feats["sfcn_finetuned"].astype(np.float64)
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    W = sd["classifier.conv_6.weight"].reshape(72, -1).double().numpy()
    b = sd["classifier.conv_6.bias"].double().numpy()
    _, centers = ft.make_bins((18, 90), 1)
    assert np.abs(hm.head_predict(E, W, b, centers) - feats["pred"]).max() < 0.05

    train = (df.split == "train").to_numpy()
    site = df.site.to_numpy()
    male = (df.sex_label == "Male").astype(float).to_numpy()
    lin = np.column_stack([df.AGE, df.AGE ** 2, male])
    spline = np.column_stack([np.asarray(dmatrix("bs(a, df=4) - 1", {"a": df.AGE})), male])
    Z = pd.get_dummies(pd.Categorical(site, categories=SITES)).to_numpy(float)
    versions = {
        "original": E,
        "ComBat (linear age)": hm.combat_apply(E, site, lin, hm.combat_fit(E, site, lin, train)),
        "ComBat-GAM (spline age)": hm.combat_apply(E, site, spline, hm.combat_fit(E, site, spline, train)),
        "LEACE site erasure (plain)": leace_fit(E[train], Z[train])(E),
    }
    # Review check: does a smaller or larger spline change the ComBat-GAM result?
    for k in [3, 5, 6]:
        sk = np.column_stack([np.asarray(dmatrix(f"bs(a, df={k}) - 1", {"a": df.AGE})), male])
        versions[f"ComBat-GAM (spline age, {k} df)"] = hm.combat_apply(E, site, sk, hm.combat_fit(E, site, sk, train))
    # Conditional LEACE: IOP's train subjects are younger, so plain erasure also removes age
    # information that is correlated with site. Erase site only from the part of the features that
    # age, age^2 and sex do not explain, then add that part back.
    Cov = np.column_stack([np.ones(len(df)), lin])
    beta, *_ = np.linalg.lstsq(Cov[train], E[train], rcond=None)
    R = E - Cov @ beta
    versions["LEACE site erasure (age/sex-conditional)"] = leace_fit(R[train], Z[train])(R) + Cov @ beta
    rows = []
    for name, X in versions.items():
        pred = hm.head_predict(X, W, b, centers)
        d = df.assign(pred=pred)
        v = d[d.split == "val"]
        a_, b_ = np.polyfit(v.AGE, v.pred, 1)
        d["bag"] = (d.pred - b_) / a_ - d.AGE
        held = d[d.split != "train"]
        m = smf.ols("bag ~ C(site, Treatment('Guys')) + C(sex_label) + AGE", data=held).fit(cov_type="HC3")
        probe = sp.probe(sp.residualise(X, lin, train), site, train, rng)
        probe_raw = sp.probe(X, site, train, rng)
        rows.append({"features": name, "probe_bal_acc": float(probe["balanced_accuracy"]),
                     "probe_bal_acc_no_residualisation": float(probe_raw["balanced_accuracy"]),
                     "heldout_mae": float((held.pred - held.AGE).abs().mean()),
                     "heldout_r_age_pred": float(np.corrcoef(held.AGE, held.pred)[0, 1]),
                     "iop_heldout_mae": float((held[held.site == "IOP"].pred - held[held.site == "IOP"].AGE).abs().mean()),
                     "iop_vs_guys": float(m.params[T]), "ci": [float(x) for x in m.conf_int().loc[T]],
                     "p": float(m.pvalues[T]),
                     "max_pred_change_yrs": float(np.abs(pred - feats["pred"]).max()),
                     "mean_abs_pred_change_yrs": float(np.abs(pred - feats["pred"]).mean())})
        print(rows[-1], flush=True)

    Xr = sp.residualise(E, lin, train)
    null = []
    for _ in range(30):
        perm = site.copy()
        perm[train] = rng.permutation(site[train])
        null.append(sp.probe(Xr, perm, train, rng)["balanced_accuracy"])
    out = {"versions": rows, "probe_permutation_null": {"n": len(null), "mean": float(np.mean(null)),
                                                         "p95": float(np.percentile(null, 95)), "max": float(np.max(null))}}
    o = P / "results" / "phase10f_site_erasure"; o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["probe_permutation_null"], indent=2))


if __name__ == "__main__":
    main()
