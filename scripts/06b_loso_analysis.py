"""Phase 6b - Leave-one-site-out (LOSO) analysis.

For each site S, the model in checkpoints/loso_without_S/ never saw S during training or
early stopping. We compare its predictions on S's train+val subjects (heldout_predictions.csv)
with its predictions on the other sites' validation subjects (val_predictions.csv):

    1. Age-bias correction (Cole et al. 2018) fitted on the in-distribution validation
       subjects only, applied to both groups.
    2. bag_corrected ~ heldout + age   -> is the unseen site shifted older/younger?
    3. abs_error     ~ heldout + age   -> is the unseen site predicted less accurately?

95% CIs: 5,000 bootstrap resamples (stratified by group); HC3 intervals alongside.
The in-distribution reference subjects were used for early stopping, so their error is
slightly optimistic; the accuracy comparison is therefore conservative in the model's favour
only for the reference group, and the BAG comparison is unaffected after correction.

Usage:
    python scripts/06b_loso_analysis.py --project-dir "G:/My Drive/BrainAge_Project" --out-dir D:/ML_Project/results/phase6b
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SITES = ["IOP", "HH", "Guys"]
N_BOOT = 5000


def fit(df, formula):
    return smf.ols(formula, data=df).fit(cov_type="HC3")


def boot_term(df, formula, term, rng):
    held, ref = df[df.heldout == 1], df[df.heldout == 0]
    vals = []
    for _ in range(N_BOOT):
        s = pd.concat([held.iloc[rng.integers(0, len(held), len(held))],
                       ref.iloc[rng.integers(0, len(ref), len(ref))]])
        vals.append(fit(s, formula).params[term])
    return np.percentile(vals, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    rows, all_preds = [], []
    for site in SITES:
        d = args.project_dir / "checkpoints" / f"loso_without_{site}"
        ref = pd.read_csv(d / "val_predictions.csv").assign(heldout=0)
        held = pd.read_csv(d / "heldout_predictions.csv").assign(heldout=1)
        assert set(held.site) == {site} and site not in set(ref.site)

        a, b = np.polyfit(ref.age, ref.predicted_age, 1)
        df = pd.concat([ref, held], ignore_index=True)
        df["predicted_age_corrected"] = (df.predicted_age - b) / a
        df["bag_corrected"] = df.predicted_age_corrected - df.age
        df["abs_error"] = (df.predicted_age - df.age).abs()
        df["model"] = f"without_{site}"
        all_preds.append(df)

        out = {"heldout_site": site, "n_heldout": len(held), "n_reference": len(ref),
               "reference_sites": "+".join(sorted(ref.site.unique())),
               "correction_slope": a, "correction_intercept": b,
               "heldout_mae": held.eval("abs(predicted_age - age)").mean(),
               "reference_mae": ref.eval("abs(predicted_age - age)").mean(),
               "heldout_mean_bag_corrected": df[df.heldout == 1].bag_corrected.mean()}
        for outcome in ["bag_corrected", "abs_error"]:
            formula = f"{outcome} ~ heldout + age"
            m = fit(df, formula)
            lo, hi = boot_term(df, formula, "heldout", rng)
            out[f"{outcome}_effect"] = m.params["heldout"]
            out[f"{outcome}_ci_low"], out[f"{outcome}_ci_high"] = lo, hi
            out[f"{outcome}_hc3_p"] = m.pvalues["heldout"]
        rows.append(out)

        print(f"\n=== Model trained WITHOUT {site} (held out n={len(held)}, "
              f"reference {out['reference_sites']} val n={len(ref)}) ===")
        print(f"  Bias correction (reference val): pred = {a:.3f}*age + {b:.2f}")
        print(f"  MAE: held-out {site} {out['heldout_mae']:.2f} | reference {out['reference_mae']:.2f}")
        for outcome, label in [("bag_corrected", "Brain-age gap"), ("abs_error", "Absolute error")]:
            e, lo, hi, pv = (out[f"{outcome}_effect"], out[f"{outcome}_ci_low"],
                             out[f"{outcome}_ci_high"], out[f"{outcome}_hc3_p"])
            flag = "  <- CI excludes 0" if lo > 0 or hi < 0 else ""
            print(f"  {label:15s} {site} vs reference, age-adjusted: {e:+.2f} yrs "
                  f"[{lo:+.2f}, {hi:+.2f}] (HC3 p={pv:.3f}){flag}")

    res = pd.DataFrame(rows)
    res.to_csv(args.out_dir / "loso_effects.csv", index=False)
    pd.concat(all_preds).to_csv(args.out_dir / "loso_predictions_corrected.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps({"n_boot": N_BOOT, "sites": SITES}, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    y = np.arange(len(res))
    for ax, outcome, title in [(axes[0], "bag_corrected", "Brain-age gap (corrected)"),
                               (axes[1], "abs_error", "Absolute error")]:
        e = res[f"{outcome}_effect"]
        ax.errorbar(e, y, xerr=[e - res[f"{outcome}_ci_low"], res[f"{outcome}_ci_high"] - e],
                    fmt="o", capsize=4)
        ax.axvline(0, color="k", lw=1, ls="--")
        ax.set_yticks(y, [f"{s} unseen" for s in res.heldout_site])
        ax.set_xlabel("Held-out site vs seen sites (years), age-adjusted, 95% CI")
        ax.set_title(title)
    plt.tight_layout()
    fig.savefig(args.out_dir / "loso_effects.png", dpi=130)
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
