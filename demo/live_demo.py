"""Live demo - the whole audit, running the real fine-tuned model in the browser.

Pick a held-out brain and walk through the project live:
    1. Predict    - brain age, the model's own age distribution and its uncertainty
    2. Explain    - where the model looks (gradient x activation map) and the deletion test
    3. Swap look  - change only the scanner "look" (intensity profile of another site, or blur)
    4. Detective  - can the model's internal features tell which site took the scan?
    5. Verdict    - the audit scorecard

Only val + test subjects are offered; nothing is fitted to them. The bias correction comes from
Phase 5 (fitted on val), the site looks from Phase 9b (train subjects, age/sex standardised) and the
site probe is refitted on train-subject features cached by Phase 9a.

Usage:
    python demo/live_demo.py --project-dir D:/ML_Project --sfcn-repo D:/sfcn_repo \
        --checkpoint "G:/My Drive/BrainAge_Project/checkpoints/sfcn_finetune_official/best.pt"
Then open http://127.0.0.1:7860 (add --share for a temporary public link).
"""

import argparse
import functools
import importlib.util
import json
import os
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("ft", HERE.parent / "scripts" / "04_finetune_sfcn.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

SITES = ["Guys", "HH", "IOP"]
COLORS = {"Guys": "#2a78d6", "HH": "#eb6834", "IOP": "#1baf7a"}
QS = np.linspace(0, 100, 101)
CELL = 8
DEFAULT_SUBJECT = 559  # the median HH brain used in the README animation


def parse_args():
    p = argparse.ArgumentParser(description="Live brain-age audit demo.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--volume-dir", type=Path, default=None)
    p.add_argument("--sfcn-repo", type=Path, default=ft.DEFAULT_SFCN_REPO)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create a temporary public link.")
    return p.parse_args()


ARGS = parse_args()
PROJECT = ARGS.project_dir
RESULTS = PROJECT / "results"
VOLUMES = ARGS.volume_dir or PROJECT / ft.DEFAULT_VOLUME_SUBDIR
torch.set_num_threads(os.cpu_count() or 4)

# --- model ------------------------------------------------------------------------------------
_ckpt = torch.load(ARGS.checkpoint or PROJECT / "checkpoints" / "sfcn_finetune_official" / "best.pt",
                   map_location="cpu", weights_only=False)
EDGES, CENTERS = ft.make_bins(_ckpt["config"]["bin_range"], _ckpt["config"]["bin_step"])
CENTERS_T = torch.tensor(CENTERS, dtype=torch.float32)
ft.ensure_sfcn_repo(ARGS.sfcn_repo)
MODEL = ft.build_model(ARGS.sfcn_repo, len(CENTERS), torch.device("cpu"))
MODEL.load_state_dict(_ckpt["model_state_dict"])
MODEL.eval()
for p_ in MODEL.parameters():
    p_.requires_grad_(False)
_store = {}
MODEL.feature_extractor.conv_2.register_forward_hook(lambda m, i, o: _store.__setitem__("act", o))
MODEL.classifier.average_pool.register_forward_hook(lambda m, i, o: _store.__setitem__("pool", o.flatten(1)))

_bc = json.loads((RESULTS / "phase5" / "summary.json").read_text())["bias_correction"]
SLOPE, INTERCEPT = _bc["slope"], _bc["intercept"]

# --- subjects ---------------------------------------------------------------------------------
_splits = {s: ft.load_split(PROJECT, s) for s in ["train", "val", "test"]}
_splits = {s: d[~d.IXI_ID.isin(ft.EXCLUDED_SUBJECTS)] for s, d in _splits.items()}
HELD = {int(r.IXI_ID): r for d in (_splits["val"], _splits["test"]) for r in d.itertuples()}
SUBJECT_CHOICES = [(f"IXI{i:03d} · {r.site} · {r.sex_label} · real age {r.AGE:.0f}", i)
                   for i, r in sorted(HELD.items(), key=lambda kv: (kv[1].site, kv[1].AGE))]

# --- site looks (Phase 9b) and site probe (Phase 9a features, train subjects only) ------------
_ref = np.load(RESULTS / "phase9b_stress_test" / "site_reference_quantiles.npz")
SITE_LOOK = {s: _ref[s] for s in SITES}
_feat = np.load(RESULTS / "phase9a_site_probe" / "features.npz", allow_pickle=True)
_site_of = {int(r.IXI_ID): r.site for r in _splits["train"].itertuples()}
_tr = np.array([int(i) in _site_of for i in _feat["ids"]])
PROBE = make_pipeline(StandardScaler(), LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 11), cv=5, class_weight="balanced", scoring="balanced_accuracy", max_iter=10000))
PROBE.fit(_feat["sfcn_finetuned"][_tr], [_site_of[int(i)] for i in _feat["ids"][_tr]])


# =============================================================================================
# Model helpers
# =============================================================================================

def renormalise(v):
    return (v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)).astype(np.float32)


@functools.lru_cache(maxsize=8)
def load(ixi_id):
    return renormalise(np.load(ft.volume_path(VOLUMES, ixi_id)).astype(np.float32))


@torch.no_grad()
def run(v):
    """Predicted age, age distribution and 64-d pooled features for one volume."""
    log_probs = MODEL(torch.from_numpy(v[None, None]))[0].reshape(1, -1)
    prob = torch.exp(log_probs)[0].numpy()
    return float(prob @ CENTERS), prob, _store["pool"][0].numpy()


@functools.lru_cache(maxsize=8)
def base(ixi_id):
    return run(load(ixi_id))


@functools.lru_cache(maxsize=8)
def attribution(ixi_id):
    x = torch.from_numpy(load(ixi_id)[None, None]).requires_grad_(True)  # weights are frozen
    with torch.enable_grad():
        log_probs = MODEL(x)[0].reshape(1, -1)
        age = (torch.exp(log_probs) @ CENTERS_T).sum()
        act = _store["act"]
        grad = torch.autograd.grad(age, act)[0]
    return (grad * act).sum(1)[0].abs().detach().numpy()  # 20 x 24 x 20 cells of 8 mm


def corrected(pred):
    return (pred - INTERCEPT) / SLOPE


def quantile(prob, q):
    cdf = np.cumsum(prob)
    k = min(int((cdf < q).sum()), len(prob) - 1)
    prev = cdf[k - 1] if k > 0 else 0.0
    return EDGES[k] + (q - prev) / max(prob[k], 1e-12) * (EDGES[k + 1] - EDGES[k])


def match_to(v, target_q):
    out = np.zeros_like(v)
    brain = v > 0
    out[brain] = np.interp(v[brain], np.percentile(v[brain], QS), target_q)
    return renormalise(np.clip(out, 0, None))


# =============================================================================================
# Drawing
# =============================================================================================

def _slices(v):
    return [np.rot90(v[80]), np.rot90(v[:, 100]), np.rot90(v[:, :, 72])]


def brain_figure(v, title=None, cam=None):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), facecolor="black")
    hi = np.percentile(v[v > 0], 99.5)
    cams = _slices(cam) if cam is not None else [None] * 3
    for ax, sl, c, name in zip(axes, _slices(v), cams, ["side", "front", "top"]):
        ax.imshow(sl, cmap="gray", vmin=0, vmax=hi)
        if c is not None:
            ax.imshow(np.ma.masked_less(c, 0.25), cmap="inferno", vmin=0, vmax=1, alpha=0.6)
        ax.set_title(name, color="#bbb", fontsize=10)
        ax.axis("off")
    if title:
        fig.suptitle(title, color="white", fontsize=12)
    fig.tight_layout()
    return fig


def distribution_figure(prob, true_age, pred, color):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    lo, hi = quantile(prob, 0.05), quantile(prob, 0.95)
    ax.axvspan(lo, hi, color=color, alpha=0.12, label=f"model's 90% range {lo:.0f}–{hi:.0f}")
    ax.bar(CENTERS, prob, width=0.9, color=color, alpha=0.85)
    ax.axvline(true_age, color="#14161a", lw=2, ls="--", label=f"real age {true_age:.1f}")
    ax.axvline(pred, color=color, lw=2.5, label=f"model's raw guess {pred:.1f} (before bias correction)")
    ax.set_xlim(15, 92)
    ax.set_xlabel("age (years)")
    ax.set_ylabel("probability")
    ax.set_title("How sure is the model? Its probability for every age 18–90", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def bars_figure(labels, values, colors, title, xlabel, ref=None):
    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.62 * len(labels)))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.6)
    for yi, val in zip(y, values):
        label = f"{val:.0f}%" if ref is not None else f"{val:+.1f}"
        ax.text(val, yi, f"  {label}  ", va="center", ha="right" if val < 0 else "left", fontsize=11)
    if ref is not None:
        ax.axvline(ref, color="#8a8f97", ls=":", lw=1.5)
        ax.text(ref, len(labels) - 0.45, " pure guessing", color="#8a8f97", fontsize=9)
    else:
        ax.axvline(0, color="#14161a", lw=1)
    ax.set_yticks(y, labels, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    lim = max(abs(min(values, default=0)), abs(max(values, default=0))) * 1.35 + 0.5
    ax.set_xlim((-lim, lim) if ref is None else (0, 105))
    fig.tight_layout()
    return fig


def card(items, note=""):
    cells = "".join(f'<div class="kpi"><span>{k}</span><b style="color:{c}">{v}</b></div>' for k, v, c in items)
    return f'<div class="kpis">{cells}</div>' + (f'<p class="note">{note}</p>' if note else "")


# =============================================================================================
# Tab actions
# =============================================================================================

def info(ixi_id):
    r = HELD[int(ixi_id)]
    return r, COLORS[r.site]


def do_predict(ixi_id):
    r, color = info(ixi_id)
    v = load(int(ixi_id))
    pred, prob, _ = base(int(ixi_id))
    corr = corrected(pred)
    gap = corr - r.AGE
    sd = float(np.sqrt(prob @ (CENTERS - pred) ** 2))
    gap_color = "#b42318" if abs(gap) > 5 else "#0f7a3a"
    html = card([("Real age", f"{r.AGE:.1f}", "#14161a"),
                 ("Model's brain age", f"{corr:.1f}", color),
                 ("Brain-age gap", f"{gap:+.1f} yrs", gap_color),
                 ("Uncertainty (±1 SD)", f"±{sd:.1f} yrs", "#4a4f57")],
                note=f"Scanned at <b>{r.site}</b>. The brain age is bias-corrected (slope {SLOPE:.2f}, "
                     f"fitted on validation only). The model never trained on this person.")
    return (brain_figure(v, f"IXI{int(ixi_id):03d}  ·  {r.site}  ·  {r.sex_label}, {r.AGE:.0f}"),
            html, distribution_figure(prob, r.AGE, pred, color))


def do_explain(ixi_id):
    v = load(int(ixi_id))
    cam_low = attribution(int(ixi_id))
    cam = F.interpolate(torch.tensor(cam_low)[None, None], size=v.shape, mode="trilinear",
                        align_corners=False)[0, 0].numpy() * (v > 0)
    cam /= np.percentile(cam[v > 0], 99.5) + 1e-12
    return brain_figure(v, "Bright = regions that move the age prediction most", np.clip(cam, 0, 1))


def do_deletion(ixi_id, percent):
    v = load(int(ixi_id))
    pred0 = base(int(ixi_id))[0]
    brain = v > 0
    smooth = gaussian_filter(v, 8) * brain
    frac_in = brain.reshape(20, CELL, 24, CELL, 20, CELL).mean(axis=(1, 3, 5))
    cells = np.argwhere(frac_in >= 0.5)
    order = cells[np.argsort(-attribution(int(ixi_id))[tuple(cells.T)])]
    k = max(1, int(round(percent / 100 * len(cells))))
    rng = np.random.default_rng(int(ixi_id))

    def remove(chosen):
        out = v.copy()
        for i, j, l in chosen:
            s = np.s_[i * CELL:(i + 1) * CELL, j * CELL:(j + 1) * CELL, l * CELL:(l + 1) * CELL]
            out[s] = smooth[s]
        return run(renormalise(out))[0] - pred0

    changes = [remove(order[:k]), remove(cells[rng.choice(len(cells), k, replace=False)]), remove(order[-k:])]
    fig = bars_figure(["most important (map)", "random", "least important (map)"], changes,
                      ["#b42318", "#8a8f97", "#0f7a3a"],
                      f"Blur away {percent}% of the brain ({k} blocks of 8 mm): how much does the age change?",
                      "change in predicted age (years)")
    verdict = ("The map is <b>faithful</b>: removing what it highlights moves the prediction most."
               if abs(changes[0]) > abs(changes[1]) else
               "For this brain the map's top regions did not beat random ones; across 60 brains they did "
               "(4.8 vs 1.0 years at 5%).")
    return fig, f'<p class="note">{verdict}</p>'


@functools.lru_cache(maxsize=8)
def own_look(ixi_id):
    """The control: the brain matched to its OWN site's look (same processing, no change of site)."""
    w = match_to(load(ixi_id), SITE_LOOK[HELD[ixi_id].site])
    return w, run(w)[0]


def do_swap(ixi_id, look, blur):
    r, color = info(ixi_id)
    ctrl, pred0 = own_look(int(ixi_id))
    w = match_to(load(int(ixi_id)), SITE_LOOK[look])
    if blur > 0:
        w = renormalise(gaussian_filter(w, blur) * (w > 0))
    pred1 = run(w)[0]
    hi = np.percentile(ctrl[ctrl > 0], 99.5)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2), facecolor="black")
    for ax, img, t in [(axes[0], ctrl, f"{r.site} look (its own site)"),
                       (axes[1], w, f"{look} look" + (f" + blur {blur:.1f}" if blur else ""))]:
        ax.imshow(np.rot90(img[:, :, 72]), cmap="gray", vmin=0, vmax=hi)
        ax.set_title(t, color="white", fontsize=11)
        ax.axis("off")
    fig.tight_layout()
    delta = pred1 - pred0
    html = card([(f"Model's guess, {r.site} look", f"{pred0:.1f}", color),
                 (f"Model's guess, {look} look", f"{pred1:.1f}", COLORS[look]),
                 ("Change from look alone", f"{delta:+.1f} yrs", "#b42318" if abs(delta) >= 2 else "#0f7a3a")],
                note="The anatomy is identical; only the brightness/contrast profile (and blur) changed. "
                     "Across 146 held-out brains: an IOP look made HH brains 3.9 years younger, "
                     "an HH look made IOP brains 3.1 years older, in 100% of cases the same direction.")
    return fig, html


def do_all_looks(ixi_id):
    r, _ = info(ixi_id)
    v = load(int(ixi_id))
    pred0 = own_look(int(ixi_id))[1]
    deltas = [0.0 if s == r.site else run(match_to(v, SITE_LOOK[s]))[0] - pred0 for s in SITES]
    return bars_figure([f"{s} look" + (" (own site)" if s == r.site else "") for s in SITES], deltas,
                       [COLORS[s] for s in SITES], f"IXI{int(ixi_id):03d} ({r.site}): one brain, three scanner looks",
                       "change in the model's guess (years)")


def do_detective(ixi_id):
    r, _ = info(ixi_id)
    features = base(int(ixi_id))[2]
    proba = PROBE.predict_proba(features[None])[0]
    classes = list(PROBE.classes_)
    p = [100 * proba[classes.index(s)] for s in SITES]
    guess = SITES[int(np.argmax(p))]
    fig = bars_figure(SITES, p, [COLORS[s] for s in SITES],
                      "Reading only the model's 64 internal numbers: which site took this scan?",
                      "probability (%)", ref=100 / 3)
    ok = guess == r.site
    html = card([("Truly scanned at", r.site, COLORS[r.site]),
                 ("Guess from the model's features", f"{guess} ({max(p):.0f}%)", COLORS[guess]),
                 ("Correct?", "yes" if ok else "no", "#b42318" if ok else "#0f7a3a")],
                note="Nobody told the model the site, and it was trained only to predict age. If the site "
                     "can still be read off its features, a scanner fingerprint is inside the model. Across "
                     "all held-out brains: 91% correct, against 33% for pure guessing.")
    return fig, html


VERDICT = """
<div class="verdict">
  <div class="stamp pass"><b>PASS</b>Accuracy<small>3.7 yrs mean error, r 0.97</small></div>
  <div class="stamp pass"><b>PASS</b>Fair by sex<small>gap −0.6 yrs, CI spans 0</small></div>
  <div class="stamp fail"><b>FAIL</b>Fair by site<small>IOP brains −5.1 yrs</small></div>
  <div class="stamp fail"><b>FAIL</b>Intensity-swap invariance<small>site readable 91%; look alone shifts 3–4 yrs</small></div>
  <div class="stamp pass"><b>PASS</b>Faithful explanations<small>top regions 4.8 vs random 1.0 yrs</small></div>
  <div class="stamp pass"><b>PASS</b>Honest uncertainty<small>doubles on an unseen site, 92% coverage</small></div>
</div>
<p class="big">Right on average, fair by sex, but <b>partly right for the wrong reasons</b>: acquisition-site
effects are inside the model, so brain-age gaps should not be compared across scanners without site-aware
validation. Research use only; no clinical claims.</p>
"""

CSS = """
.gradio-container{max-width:1180px!important}
#hero{padding:22px 26px;border-radius:14px;background:linear-gradient(135deg,#0d1b33,#1f4f99 60%,#1baf7a);color:#fff}
#hero h1{margin:0;font-size:30px;letter-spacing:-.02em;color:#fff}#hero p{margin:6px 0 0;opacity:.88;font-size:15px;color:#fff}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}
.kpi{border:1px solid #dcdfe4;border-radius:10px;padding:12px 14px;background:#fff}
.kpi span{display:block;font-size:11px;letter-spacing:.12em;text-transform:uppercase;color:#8a8f97}
.kpi b{font-size:26px;font-variant-numeric:tabular-nums}
.note{color:#4a4f57;font-size:14px;margin:10px 2px 0}
.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.stamp{border:2.5px solid;border-radius:12px;padding:14px;text-align:center;font-size:14px;transform:rotate(-1.2deg)}
.stamp:nth-child(even){transform:rotate(1deg)}.stamp b{display:block;font-size:22px;letter-spacing:.14em}
.stamp small{display:block;margin-top:4px;opacity:.85}
.pass{color:#0f7a3a;border-color:#0f7a3a;background:#e5f4ea}.fail{color:#b42318;border-color:#b42318;background:#fde8e6}
.big{font-size:17px;line-height:1.5;margin-top:18px}
"""

STEP = '<p class="note" style="font-size:15px">{}</p>'

with gr.Blocks(title="Brain-Age Audit · Live") as app:
    gr.HTML('<div id="hero"><h1>Right for the wrong reasons?</h1>'
            '<p>Live audit of a brain-age AI (SFCN, fine-tuned on 339 IXI brains from 3 London sites). '
            'Pick a brain the model has never trained on, then step through the tabs.</p></div>')
    subject = gr.Dropdown(SUBJECT_CHOICES, value=DEFAULT_SUBJECT, label="Choose a held-out brain (site · sex · real age)")

    with gr.Tab("① Predict"):
        gr.HTML(STEP.format("The model reads the whole 3D scan and outputs a probability for every age. "
                            "Its guess is the average of that distribution."))
        b1 = gr.Button("Predict brain age", variant="primary")
        p_img, p_card, p_dist = gr.Plot(label="The scan"), gr.HTML(), gr.Plot(label="Age distribution")
        b1.click(do_predict, subject, [p_img, p_card, p_dist])
        subject.change(do_predict, subject, [p_img, p_card, p_dist])
        app.load(do_predict, subject, [p_img, p_card, p_dist])

    with gr.Tab("② Where does it look?"):
        gr.HTML(STEP.format("A heat map shows which regions drive the prediction. Is it trustworthy? "
                            "Test it: blur away the regions it ranks highest and see whether the answer moves "
                            "more than when blurring random regions."))
        b2 = gr.Button("Show heat map", variant="primary")
        e_img = gr.Plot(label="Attribution map")
        with gr.Row():
            pct = gr.Slider(5, 20, value=5, step=5, label="How much of the brain to blur (%)")
            b2b = gr.Button("Run deletion test")
        e_bars, e_note = gr.Plot(label="Deletion test"), gr.HTML()
        b2.click(do_explain, subject, e_img)
        b2b.click(do_deletion, [subject, pct], [e_bars, e_note])

    with gr.Tab("③ Change only the scanner look"):
        gr.HTML(STEP.format("Same brain, same anatomy: give it the brightness/contrast profile of another "
                            "site (or blur it). If the predicted age moves, the model reacts to acquisition, "
                            "not anatomy."))
        with gr.Row():
            look = gr.Radio(SITES, value="IOP", label="Give this brain the look of")
            blur = gr.Slider(0, 1.5, value=0, step=0.5, label="Extra blur (voxels; 0.6 ≈ IOP's extra blur)")
        with gr.Row():
            b3 = gr.Button("Swap the look", variant="primary")
            b3b = gr.Button("Try all three looks")
        s_img, s_card, s_bars = gr.Plot(label="Before / after"), gr.HTML(), gr.Plot(label="All looks")
        b3.click(do_swap, [subject, look, blur], [s_img, s_card])
        b3b.click(do_all_looks, subject, s_bars)

    with gr.Tab("④ Site detective"):
        gr.HTML(STEP.format("The model was trained to predict age only. Can we still read the scanning site "
                            "from its 64 internal numbers? (Probe fitted on training brains only.)"))
        b4 = gr.Button("Read the site from the model's features", variant="primary")
        d_bars, d_card = gr.Plot(label="Site probe"), gr.HTML()
        b4.click(do_detective, subject, [d_bars, d_card])

    with gr.Tab("⑤ Verdict"):
        gr.HTML(VERDICT)

if __name__ == "__main__":
    app.launch(server_port=ARGS.port, share=ARGS.share, inbrowser=True, css=CSS,
               theme=gr.themes.Soft(primary_hue="blue"))
