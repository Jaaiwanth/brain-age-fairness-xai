"""Draws Fig. 1 of the paper: the audit pipeline (data -> model -> four audit layers)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(7.16, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(-1, 33); ax.axis("off")


def box(x, y, w, h, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=8, fontweight="bold", color=ec)
    ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.0, color="#222", linespacing=1.4)


def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#555", lw=1))


box(0.5, 6, 16, 25, "IXI T1 MRI", "485 healthy adults\nages 20-86\n3 London sites:\nGuys  ·  HH  ·  IOP\n\nsplit 339/73/73\n(test used once)", "#eef3fb", "#1f4f99")
box(20, 6, 17, 25, "Preprocessing", "N4 bias correction\nskull-stripping\naffine to MNI152\ncrop 160×192×160\nSFCN normalisation", "#eef3fb", "#1f4f99")
box(40.5, 6, 18, 25, "SFCN brain age", "UK Biobank\npretrained, fine-tuned\n72 soft-label bins\n18-90 yrs\n64-d features\n+ bias correction", "#eef3fb", "#1f4f99")
arrow(17, 18.5, 19.6, 18.5); arrow(37.5, 18.5, 40.1, 18.5)

layers = [("A  Accuracy", "MAE, r, bootstrap CIs", "#0f7a3a", "#e7f5ec"),
          ("B  Group fairness", "sex & site, age-adjusted\nleave-one-site-out", "#b42318", "#fdebe9"),
          ("C  Shortcut tests", "site probe · look swap\nblur · ComBat", "#b42318", "#fdebe9"),
          ("D  Trust checks", "attribution sanity +\ndeletion · uncertainty", "#0f7a3a", "#e7f5ec")]
for i, (t, b, ec, fc) in enumerate(layers):
    y = 25.4 - i * 6.45
    ax.add_patch(FancyBboxPatch((62, y), 22.5, 5.5, boxstyle="round,pad=0.2,rounding_size=1", fc=fc, ec=ec, lw=1))
    ax.text(63, y + 4.0, t, fontsize=7, fontweight="bold", color=ec, va="center")
    ax.text(63, y + 1.55, b, fontsize=5.6, color="#222", va="center", linespacing=1.2)
    arrow(59, 18.5, 61.7, y + 2.75)
box(87.5, 6, 12, 25, "Fairness\npassport", "\n6 verdicts\nPASS / FAIL\nwith evidence\nand usage\nguidance", "#fdf4dd", "#946200")
for i in range(4):
    arrow(84.8, 28.15 - i * 6.45, 87.2, 18.5)
ax.text(50, 0.6, "Audit question: is the model accurate, fair, and right for the right reasons?",
        ha="center", fontsize=7.4, style="italic", color="#444")
fig.savefig("paper/figures/fig1_overview.pdf", bbox_inches="tight")
fig.savefig("paper/figures/fig1_overview.png", dpi=300, bbox_inches="tight")
