"""Phase 10i - Uncertainty baselines: reliability, conformal prediction (pooled and per site), ensembles.

Uses the Phase 10a cross-validation outputs (every train+val subject predicted out-of-sample).
1. Reliability of the model's own predictive distribution: coverage of its central 50/80/90/95%
   ranges and their width, per site.
2. Split conformal intervals from absolute residuals, calibrated out-of-fold: for each CV fold k the
   quantile is taken from the other folds' out-of-sample residuals and applied to fold k.
   Pooled conformal vs group-conditional (Mondrian) conformal with one quantile per site.
3. Locked test set: fold-ensemble prediction, with conformal quantiles from all out-of-fold residuals.
4. Deep ensemble (if several seeds exist): spread of the seed predictions, per site.
Note: soft labels with sigma = 2 years put a floor of about 2 years on the model's own spread.

Usage: python scripts/25_uncertainty_conformal.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = [0.5, 0.8, 0.9, 0.95]
EDGES = np.arange(18, 91, 1.0)


def central(prob, level):
    cdf = np.cumsum(prob, 1)
    def q(a):
        k = np.minimum((cdf < a).sum(1), prob.shape[1] - 1)
        prev = np.where(k > 0, cdf[np.arange(len(k)), np.maximum(k - 1, 0)], 0.0)
        return EDGES[k] + (a - prev) / np.maximum(prob[np.arange(len(k)), k], 1e-12)
    return q((1 - level) / 2), q((1 + level) / 2)


def conformal_q(res, level):
    n = len(res)
    return np.quantile(res, min(1.0, np.ceil((n + 1) * level) / n), method="higher")


def load(seed_dir, name):
    parts = []
    for f in sorted(seed_dir.glob(f"fold*/{name}_predictions.csv")):
        d = pd.read_csv(f)
        prob = np.load(f.parent / f"{name}_outputs.npz")["prob"]
        d["prob_idx"] = range(len(d))
        parts.append((d, prob))
    return parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    P = p.parse_args().project_dir
    root = P / "results" / "phase10_cv"
    seeds = sorted(d for d in root.glob("seed*") if len(list(d.glob("fold*/outer_predictions.csv"))) == 5)
    out_dir = P / "results" / "phase10i_uncertainty"; out_dir.mkdir(parents=True, exist_ok=True)
    res = {"seeds_used": [s.name for s in seeds]}

    # Out-of-fold predictions of seed 0 (primary).
    oof, probs = [], []
    for d, prob in load(seeds[0], "outer"):
        oof.append(d); probs.append(prob)
    oof = pd.concat(oof, ignore_index=True); prob = np.concatenate(probs)
    oof["res"] = (oof.pred - oof.age).abs()

    rel = []
    for lv in LEVELS:
        lo, hi = central(prob, lv)
        cov = (oof.age >= lo) & (oof.age <= hi)
        for site, g in oof.assign(cov=cov, w=hi - lo).groupby("site"):
            rel.append({"method": "model's own distribution", "level": lv, "site": site, "n": len(g),
                        "coverage": float(g.cov.mean()), "mean_width": float(g.w.mean())})
        for method in ["conformal (pooled)", "conformal (per site, Mondrian)"]:
            cov_all, width_all = np.zeros(len(oof), bool), np.zeros(len(oof))
            for k in oof.fold.unique():
                cal, ev = oof[oof.fold != k], oof.fold == k
                if method.startswith("conformal (pooled)"):
                    q = conformal_q(cal.res.to_numpy(), lv)
                    qs = np.full(ev.sum(), q)
                else:
                    qmap = {s: conformal_q(g.res.to_numpy(), lv) for s, g in cal.groupby("site")}
                    qs = oof.loc[ev, "site"].map(qmap).to_numpy()
                cov_all[ev.to_numpy()] = oof.loc[ev, "res"].to_numpy() <= qs
                width_all[ev.to_numpy()] = 2 * qs
            for site, g in oof.assign(cov=cov_all, w=width_all).groupby("site"):
                rel.append({"method": method, "level": lv, "site": site, "n": len(g),
                            "coverage": float(g.cov.mean()), "mean_width": float(g.w.mean())})
    rel = pd.DataFrame(rel)
    rel.to_csv(out_dir / "reliability_by_site.csv", index=False)
    res["reliability_90"] = rel[rel.level == 0.9].round(3).to_dict("records")

    # Locked test: fold ensemble (all available seeds x folds).
    tests = []
    for s in seeds:
        for d, _ in load(s, "test"):
            tests.append(d)
    t = pd.concat(tests).groupby(["IXI_ID", "site", "age"], as_index=False).agg(pred=("pred", "mean"), spread=("pred", "std"),
                                                                                  n_models=("pred", "size"))
    t["res"] = (t.pred - t.age).abs()
    q_pool = conformal_q(oof.res.to_numpy(), 0.9)
    q_site = {s: conformal_q(g.res.to_numpy(), 0.9) for s, g in oof.groupby("site")}
    t["cov_pool"] = t.res <= q_pool
    t["cov_site"] = t.res <= t.site.map(q_site)
    res["locked_test_ensemble"] = {
        "n_models": int(t.n_models.iloc[0]), "mae": float(t.res.mean()),
        "conformal90_pooled_halfwidth": float(q_pool), "conformal90_site_halfwidth": {k: float(v) for k, v in q_site.items()},
        "coverage_by_site": t.groupby("site")[["cov_pool", "cov_site"]].mean().round(3).to_dict(),
        "ensemble_spread_by_site": t.groupby("site").spread.mean().round(3).to_dict()}

    if len(seeds) > 1:
        ens = []
        for s in seeds:
            for d, _ in load(s, "outer"):
                ens.append(d)
        e = pd.concat(ens).groupby(["IXI_ID", "site"], as_index=False).agg(spread=("pred", "std"))
        res["oof_seed_ensemble_spread_by_site"] = e.groupby("site").spread.mean().round(3).to_dict()

    (out_dir / "summary.json").write_text(json.dumps(res, indent=2))
    print(rel[rel.level == 0.9].round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in res.items() if k != "reliability_90"}, indent=1))


if __name__ == "__main__":
    main()
