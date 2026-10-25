"""Phase 10k - Does the acquisition stress test replicate across independently trained models?

For each CV fold model (seed 0), every out-of-sample subject is quantile-matched to each site's
age/sex-standardised reference (Phase 9b) and re-predicted; the change is measured relative to
matching to the subject's own site, exactly as in Phase 9b. This gives the site-swap matrix for
all 412 train+val subjects and five separately trained models.

Usage: python scripts/27_cv_stress_test.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st = _load("st", "09_acquisition_stress_test.py")
cv = _load("cv", "17_cv_finetune.py")
ft = st.ft
SITES = ["Guys", "HH", "IOP"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    p.add_argument("--sfcn-repo", type=Path, default=Path("D:/sfcn_repo"))
    p.add_argument("--threads", type=int, default=10)
    args = p.parse_args()
    P = args.project_dir
    torch.set_num_threads(args.threads)
    ft.ensure_sfcn_repo(args.sfcn_repo)
    ref = np.load(P / "results" / "phase9b_stress_test" / "site_reference_quantiles.npz")
    root = P / "results" / "phase10_cv" / "seed0"
    _, centers = ft.make_bins((18, 90), 1)
    ct = torch.tensor(centers, dtype=torch.float32)
    full = ft.build_model(args.sfcn_repo, 72, torch.device("cpu")).eval()
    stem = full.feature_extractor[:cv.FROZEN]
    out_dir = P / "results" / "phase10k_cv_stress_test"; out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for k in range(5):
        tail = cv.Tail(ft.build_model(args.sfcn_repo, 72, torch.device("cpu")))
        tail.load_state_dict(torch.load(root / f"fold{k}" / "best_tail.pt", weights_only=False)["tail_state_dict"])
        tail.eval()
        outer = pd.read_csv(root / f"fold{k}" / "outer_predictions.csv")
        for n, r in outer.iterrows():
            v = st.load_normalised(P / ft.DEFAULT_VOLUME_SUBDIR, int(r.IXI_ID))
            x = torch.from_numpy(np.stack([st.match_to(v, ref[s]) for s in SITES])[:, None])
            with torch.no_grad():
                pr = ft.predict_ages(tail(stem(x)), ct).numpy()
            own = pr[SITES.index(r.site)]
            for s, val in zip(SITES, pr):
                rows.append({"fold": k, "IXI_ID": r.IXI_ID, "site": r.site, "age": r.age, "look": s, "delta": float(val - own)})
            if n % 20 == 0:
                print(f"fold {k}: {n + 1}/{len(outer)}", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "cv_stress_deltas.csv", index=False)
    d = pd.DataFrame(rows)
    mat = d.groupby(["site", "look"]).delta.agg(["mean", "count", lambda s: float((s < 0).mean())]).round(3)
    mat.columns = ["mean_delta", "n", "share_negative"]
    per_fold = d[(d.site == "HH") & (d.look == "IOP")].groupby("fold").delta.mean().round(2).to_dict()
    per_fold_iop = d[(d.site == "IOP") & (d.look == "HH")].groupby("fold").delta.mean().round(2).to_dict()
    out = {"matrix": mat.reset_index().to_dict("records"), "HH_to_IOP_by_fold": per_fold, "IOP_to_HH_by_fold": per_fold_iop}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2))
    print(mat.to_string()); print(per_fold, per_fold_iop)


if __name__ == "__main__":
    main()
