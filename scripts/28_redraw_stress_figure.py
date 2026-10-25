"""Redraws the Phase 9b stress-test figure with the site-swap matrix expressed relative to matching
each brain to its OWN site (the control used for every number in the text), so the diagonal is 0.
The original figure showed changes relative to the unmatched input.

Usage: python scripts/28_redraw_stress_figure.py --project-dir D:/ML_Project --out figures/stress_test.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SITES = ["Guys", "HH", "IOP"]
p = argparse.ArgumentParser()
p.add_argument("--project-dir", type=Path, default=Path("D:/ML_Project"))
p.add_argument("--out", type=Path, default=Path("figures/stress_test.png"))
a = p.parse_args()
d = a.project_dir / "results" / "phase9b_stress_test"
pred = pd.read_csv(d / "stress_predictions.csv")
sweep = pd.read_csv(d / "sweep_effects.csv")
wide = pred[pred.condition.str.startswith("match")].pivot_table(index=["IXI_ID", "site"], columns="condition",
                                                                 values="predicted_age").reset_index()
M = np.zeros((3, 3)); share = np.zeros((3, 3))
for i, s in enumerate(SITES):
    g = wide[wide.site == s]
    for j, t in enumerate(SITES):
        diff = g[f"match {t}"] - g[f"match {s}"]
        M[i, j] = diff.mean()
        share[i, j] = max((diff < 0).mean(), (diff > 0).mean())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.4]})
ax = axes[0]
lim = np.abs(M).max()
ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim)
for i in range(3):
    for j in range(3):
        ax.text(j, i, "0 (own)" if i == j else f"{M[i, j]:+.1f}\n{100 * share[i, j]:.0f}% same sign",
                ha="center", va="center", fontsize=10, color="white" if abs(M[i, j]) > 0.55 * lim else "black")
ax.set_xticks(range(3), [f"given {s}'s\nappearance" for s in SITES])
ax.set_yticks(range(3), [f"{s} brains\n(n={(wide.site == s).sum()})" for s in SITES])
ax.set_title("Change in predicted age (years) relative to\nmatching to the brain's own site")
ax = axes[1]
y = np.arange(len(sweep))[::-1]
ax.errorbar(sweep.mean_delta, y, xerr=[sweep.mean_delta - sweep.ci_low, sweep.ci_high - sweep.mean_delta],
            fmt="o", capsize=4)
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_yticks(y, sweep.condition)
ax.set(xlabel="Change in predicted age (years), mean and 95% CI",
       title="Single acquisition changes, same brains (n=40)")
plt.tight_layout()
fig.savefig(a.out, dpi=130)
print(np.round(M, 2)); print(np.round(share, 2))
