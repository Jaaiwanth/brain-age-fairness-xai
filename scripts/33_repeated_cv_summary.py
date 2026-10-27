"""Phase 11c - Repeated cross-validation: IOP offset across new fold assignments.

For every completed repetition (5 folds of scripts/17_cv_finetune.py --fold-seed r) the out-of-fold
brain-age gap is bias-corrected per fold on its inner validation set, exactly as in Phase 10d, and
the IOP-vs-Guys offset is estimated with bag ~ sex + site + age (OLS, HC3). The paper's original
fold assignment (fold seed 0, model seed 0) is included as the reference.

Usage: python scripts/33_repeated_cv_summary.py --project-dir D:/ML_Project
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

T = "C(site, Treatment('Guys'))[T.IOP]"
F = "bag ~ C(sex, Treatment('Female')) + C(site, Treatment('Guys')) + age"


def oof(seed_dir):
    folds = sorted(seed_dir.glob("fold*/outer_predictions.csv"))
    if len(folds) != 5:
        return None
    parts = []
    for f in folds:
        o, iv = pd.read_csv(f), pd.read_csv(f.parent / "inner_val_predictions.csv")
        a, b = np.polyfit(iv.age, iv.pred, 1)
        parts.append(o.assign(bag=(o.pred - b) / a - o.age))
    return pd.concat(parts, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
    R = p.parse_args().project_dir / "results"
    runs = {"fold seed 0 (paper)": R / "phase10_cv" / "seed0"}
    for d in sorted((R / "phase11c_repeated_cv").glob("rep*")):
        runs[f"fold seed {d.name[3:]}"] = d / "seed0"
    rows = []
    for name, d in runs.items():
        o = oof(d)
        if o is None:
            print(f"{name}: incomplete, skipped")
            continue
        m = smf.ols(F, data=o).fit(cov_type="HC3")
        lo, hi = m.conf_int().loc[T]
        rows.append({"run": name, "n": len(o), "n_iop": int((o.site == "IOP").sum()),
                     "mae": float((o.pred - o.age).abs().mean()), "iop_vs_guys": float(m.params[T]),
                     "ci": [float(lo), float(hi)], "p": float(m.pvalues[T])})
    df = pd.DataFrame(rows)
    new = df[df.run != "fold seed 0 (paper)"]
    out = {"runs": rows,
           "new_fold_assignments": {"n_repetitions": len(new),
                                    "mean_iop_offset": float(new.iop_vs_guys.mean()) if len(new) else None,
                                    "range": [float(new.iop_vs_guys.min()), float(new.iop_vs_guys.max())] if len(new) else None}}
    o = R / "phase11c_repeated_cv"; o.mkdir(parents=True, exist_ok=True)
    (o / "summary.json").write_text(json.dumps(out, indent=2))
    print(df.round(3).to_string(index=False))
    print(json.dumps(out["new_fold_assignments"], indent=1))


if __name__ == "__main__":
    main()
