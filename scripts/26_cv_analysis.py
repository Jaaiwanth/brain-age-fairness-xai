"""Phase 10j - Cross-validation results: larger-sample IOP offset, seed robustness, training dynamics.

From the Phase 10a outputs:
1. Out-of-sample accuracy and the age-adjusted IOP/HH/sex effects on all CV subjects (each fold's
   bias correction fitted on its own inner-validation subjects), per seed and pooled over seeds.
2. Training dynamics: for every fold, the mean brain-age gap of training IOP subjects and of
   out-of-sample IOP subjects at each epoch (raw gap minus the fold's Guys/HH gap at that epoch).
3. Site decodability in each fold model's out-of-sample features (5-fold CV probe within the fold).
4. Locked test set: the fold ensemble's MAE and IOP offset (a second look at the test set with
   new models; the original single-model test result is unchanged).

Usage: python scripts/26_cv_analysis.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TERMS = {"IOP vs Guys": "C(site, Treatment('Guys'))[T.IOP]", "HH vs Guys": "C(site, Treatment('Guys'))[T.HH]",
         "Male vs Female": "C(sex, Treatment('Female'))[T.Male]"}
F = "bag ~ C(site, Treatment('Guys')) + C(sex, Treatment('Female')) + age"


def corrected(seed_dir, name):
    parts = []
    for f in sorted(seed_dir.glob(f"fold*/{name}_predictions.csv")):
        d, iv = pd.read_csv(f), pd.read_csv(f.parent / "inner_val_predictions.csv")
        a, b = np.polyfit(iv.age, iv.pred, 1)
        parts.append(d.assign(bag=(d.pred - b) / a - d.age))
    return pd.concat(parts, ignore_index=True)


def effects(d, rng, n_boot=2000):
    m = smf.ols(F, data=d).fit(cov_type="HC3")
    out = {}
    groups = [g for _, g in d.groupby("site")]
    boots = {k: [] for k in TERMS}
    for _ in range(n_boot):
        s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for g in groups])
        fb = smf.ols(F, data=s).fit()
        for k, t in TERMS.items():
            boots[k].append(fb.params[t])
    for k, t in TERMS.items():
        out[k] = {"est": float(m.params[t]), "ci": [float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))],
                  "p": float(m.pvalues[t])}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    P = p.parse_args().project_dir
    root = P / "results" / "phase10_cv"
    seeds = sorted(d for d in root.glob("seed*") if len(list(d.glob("fold*/outer_predictions.csv"))) == 5)
    out_dir = P / "results" / "phase10j_cv_analysis"; out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    res = {"seeds": [s.name for s in seeds]}

    per_seed = {}
    for s in seeds:
        d = corrected(s, "outer")
        per_seed[s.name] = {"n": len(d), "n_iop": int((d.site == "IOP").sum()), "mae": float((d.pred - d.age).abs().mean()),
                            "r": float(np.corrcoef(d.pred, d.age)[0, 1]), "effects": effects(d, rng)}
        print(s.name, json.dumps(per_seed[s.name]), flush=True)
    res["per_seed"] = per_seed
    if len(seeds) > 1:
        allp = pd.concat([corrected(s, "outer") for s in seeds]).groupby(
            ["IXI_ID", "site", "sex", "age"], as_index=False).bag.mean()
        res["seed_averaged_gap_effects"] = effects(allp, rng)

    # 2. Training dynamics (seed 0).
    traj = []
    for f in sorted(seeds[0].glob("fold*/epoch_predictions.csv")):
        e = pd.read_csv(f)
        e["gap"] = e.pred - e.AGE
        for ep, g in e.groupby("epoch"):
            # Age-adjusted: IOP indicator in a gap ~ age + group model, against out-of-sample Guys/HH.
            ref = g[(g.role == "outer") & (g.site != "IOP")].assign(grp=0)
            def adj(target):
                d = pd.concat([ref, target.assign(grp=1)])
                return float(smf.ols("gap ~ age_ + grp", data=d.assign(age_=d.AGE)).fit().params["grp"])
            traj.append({"fold": f.parent.name, "epoch": ep,
                         "train_IOP_minus_ref": adj(g[g.role == "train_IOP"]),
                         "outer_IOP_minus_ref": adj(g[(g.role == "outer") & (g.site == "IOP")])})
    traj = pd.DataFrame(traj)
    traj.to_csv(out_dir / "training_dynamics.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for fold, g in traj.groupby("fold"):
        ax.plot(g.epoch, g.train_IOP_minus_ref, color="#946200", alpha=.45, lw=1)
        ax.plot(g.epoch, g.outer_IOP_minus_ref, color="#1baf7a", alpha=.45, lw=1)
    m = traj.groupby("epoch")[["train_IOP_minus_ref", "outer_IOP_minus_ref"]].mean()
    ax.plot(m.index, m.train_IOP_minus_ref, color="#946200", lw=2.5, label="IOP subjects in training")
    ax.plot(m.index, m.outer_IOP_minus_ref, color="#1baf7a", lw=2.5, label="IOP subjects held out")
    ax.axhline(0, color="k", lw=.8); ax.axvline(5.5, color="grey", ls=":", lw=1)
    ax.set(xlabel="epoch (stage 2 from epoch 6)", ylabel="age-adjusted IOP gap vs Guys/HH (yrs)",
           title="IOP offset during training (5 CV folds)")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out_dir / "training_dynamics.png", dpi=300)
    first = traj[traj.epoch == 1][["train_IOP_minus_ref", "outer_IOP_minus_ref"]].mean()
    last = traj.groupby("fold").tail(1)[["train_IOP_minus_ref", "outer_IOP_minus_ref"]].mean()
    res["training_dynamics"] = {"epoch1_mean": first.round(3).to_dict(), "final_epoch_mean": last.round(3).to_dict()}

    # 3. Site probe within each fold's out-of-sample features.
    accs = []
    for f in sorted(seeds[0].glob("fold*/outer_outputs.npz")):
        z = np.load(f)
        d = pd.read_csv(f.parent / "outer_predictions.csv")
        X = z["feat"]
        C = np.column_stack([np.ones(len(d)), d.age, d.age ** 2, (d.sex == "Male").astype(float)])
        Xr = X - C @ np.linalg.lstsq(C, X, rcond=None)[0]
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 2, 11), cv=3, class_weight="balanced",
                                                                   max_iter=5000))
        pr = cross_val_predict(clf, Xr, d.site, cv=StratifiedKFold(5, shuffle=True, random_state=0))
        accs.append(balanced_accuracy_score(d.site, pr))
    res["site_probe_within_fold_bal_acc"] = {"per_fold": [round(a, 3) for a in accs], "mean": float(np.mean(accs))}

    # 4. Locked test, fold ensemble.
    t = pd.concat([corrected(s, "test") for s in seeds]).groupby(["IXI_ID", "site", "sex", "age"], as_index=False).agg(
        pred=("pred", "mean"), bag=("bag", "mean"))
    res["locked_test_fold_ensemble"] = {"n_models": 5 * len(seeds), "mae": float((t.pred - t.age).abs().mean()),
                                        "r": float(np.corrcoef(t.pred, t.age)[0, 1]), "effects": effects(t, rng)}
    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "per_seed"}, indent=1))


if __name__ == "__main__":
    main()
