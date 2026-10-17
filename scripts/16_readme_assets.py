"""Build the animated README assets in figures/readme/ from saved results (no model run).

    scanner_look.gif   one held-out HH brain, its intensity profile morphing between the
                       age/sex-standardised looks of HH, Guys and IOP (Phase 9b references),
                       with the model's predicted age for each look (Phase 9b predictions).
                       The subject is the HH brain whose IOP-look shift is closest to the HH
                       median, so the example is typical, not the most extreme.
    brain_sweep.gif    axial slices sweeping through one held-out brain with its Phase 7
                       attribution overlay.

Usage:
    python scripts/16_readme_assets.py --project-dir D:/ML_Project
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ft = _load("ft", "04_finetune_sfcn.py")
st = _load("st", "09_acquisition_stress_test.py")

W, H = 780, 400
BG = (11, 13, 17)
INK = (240, 239, 236)
MUTED = (150, 154, 162)
SITE_COL = {"Guys": (57, 135, 229), "HH": (217, 89, 38), "IOP": (25, 158, 112)}


def font(size, bold=False):
    for f in (["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/seguisb.ttf"] if bold else []) + \
             ["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def to_gray(sl, hi):
    return (np.clip(sl / hi, 0, 1) * 255).astype(np.uint8)


def hot_rgba(x):
    r = np.clip(x / 0.4, 0, 1); g = np.clip((x - 0.4) / 0.4, 0, 1); b = np.clip((x - 0.8) / 0.2, 0, 1)
    a = np.clip(x * 1.3, 0, 0.85)
    return (np.stack([r, g, b, a], -1) * 255).astype(np.uint8)


def slice_img(sl_gray, size):
    im = Image.fromarray(np.rot90(sl_gray).copy(), "L").convert("RGB")
    return im.resize((size, int(size * im.height / im.width)), Image.LANCZOS)


def args_():
    p = argparse.ArgumentParser(description="README animated assets.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--out-dir", type=Path, default=HERE.parent / "figures" / "readme")
    return p.parse_args()


def scanner_look(P, out):
    R = P / "results"
    s = pd.read_csv(R / "phase9b_stress_test" / "stress_predictions.csv")
    m = s[s.condition.str.startswith("match")].pivot(index=["IXI_ID", "site", "age"], columns="condition",
                                                       values="predicted_age").reset_index()
    hh = m[m.site == "HH"].assign(d=lambda x: x["match IOP"] - x["match HH"])
    row = hh.iloc[(hh.d - hh.d.median()).abs().argsort().iloc[0]]
    ref = np.load(R / "phase9b_stress_test" / "site_reference_quantiles.npz")
    v = st.load_normalised(P / ft.DEFAULT_VOLUME_SUBDIR, int(row.IXI_ID))
    looks = {site: st.match_to(v, ref[site]) for site in ["HH", "Guys", "IOP"]}
    hi = np.percentile(v[v > 0], 99.5)
    imgs = {k: slice_img(to_gray(x[:, :, 72], hi), 300) for k, x in looks.items()}
    preds = {k: float(row[f"match {k}"]) for k in looks}
    print(f"  scanner_look: IXI{int(row.IXI_ID):03d} (HH, age {row.age:.1f}); "
          + ", ".join(f"{k} look {p:.1f}" for k, p in preds.items()) + f"; median HH->IOP {hh.d.median():+.2f}")

    f_big, f_mid, f_small = font(54, True), font(22, True), font(16)
    order = ["HH", "Guys", "IOP", "HH"]
    frames, durations = [], []
    for a, b in zip(order[:-1], order[1:]):
        for t in np.linspace(0, 1, 9):
            hold = t in (0.0,)
            img = Image.blend(imgs[a], imgs[b], float(t))
            pred = preds[a] + (preds[b] - preds[a]) * t
            site = a if t < 0.5 else b
            canvas = Image.new("RGB", (W, H), BG)
            canvas.paste(img, (40, (H - img.height) // 2))
            d = ImageDraw.Draw(canvas)
            x0 = 380
            d.text((x0, 52), "SAME BRAIN · ONLY THE LOOK CHANGES", font=f_small, fill=MUTED)
            d.rounded_rectangle((x0, 96, x0 + 16, 112), 4, fill=SITE_COL[site])
            d.text((x0 + 26, 88), f"looks like {site}", font=f_mid, fill=INK)
            d.text((x0, 140), "predicted age", font=f_small, fill=MUTED)
            d.text((x0, 160), f"{pred:.1f}", font=f_big, fill=INK)
            d.text((x0, 236), f"real age {row.age:.1f}", font=f_small, fill=MUTED)
            bar_x, bar_w = x0, 290
            lo, hi_ = 30, 70
            px = lambda yrs: bar_x + (min(max(yrs, lo), hi_) - lo) / (hi_ - lo) * bar_w
            d.line((bar_x, 290, bar_x + bar_w, 290), fill=(60, 64, 72), width=3)
            d.line((px(row.age), 278, px(row.age), 302), fill=INK, width=2)
            d.ellipse((px(pred) - 8, 282, px(pred) + 8, 298), fill=SITE_COL[site])
            for t_ in (30, 40, 50, 60, 70):
                d.text((px(t_) - 8, 308), str(t_), font=f_small, fill=MUTED)
            d.text((x0, 350), f"typical HH brain (median shift) · average HH→IOP look shift {hh.d.median():+.1f} yrs",
                   font=font(12), fill=MUTED)
            frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))
            durations.append(1600 if t == 0 else 70)
    frames[0].save(out / "scanner_look.gif", save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=True, disposal=2)


def brain_sweep(P, out):
    R = P / "results"
    unc = pd.read_csv(R / "phase9f_uncertainty" / "main_model_uncertainty.csv")
    r = unc[(unc.site == "Guys") & unc.AGE.between(55, 65)].iloc[0]
    v = np.load(ft.volume_path(P / ft.DEFAULT_VOLUME_SUBDIR, int(r.IXI_ID))).astype(np.float32)
    brain = v > 0
    hi = np.percentile(v[brain], 99.5)
    cam = zoom(np.load(R / "phase7" / "gradcam_conv_2_lowres.npz")[f"IXI{int(r.IXI_ID):03d}"], 8, order=1) * brain
    cam = np.clip(cam / np.percentile(cam[brain], 99.5), 0, 1)
    f_mid, f_small = font(20, True), font(15)
    frames = []
    zs = list(range(34, 124, 4))
    for z in zs + zs[::-1][1:-1]:
        base = slice_img(to_gray(v[:, :, z], hi), 330).convert("RGBA")
        ov = Image.fromarray(np.rot90(hot_rgba(cam[:, :, z])).copy(), "RGBA").resize(base.size, Image.BILINEAR)
        im = Image.alpha_composite(base, ov).convert("RGB")
        canvas = Image.new("RGB", (W, H), BG)
        canvas.paste(im, (40, (H - im.height) // 2))
        d = ImageDraw.Draw(canvas)
        d.text((410, 70), "WHERE THE PREDICTION IS SENSITIVE", font=f_small, fill=MUTED)
        d.text((410, 96), "gradient × activation · 8 mm", font=f_mid, fill=INK)
        d.text((410, 150), f"axial slice {z:3d} / 160", font=f_small, fill=MUTED)
        d.rounded_rectangle((410, 180, 410 + 280 * (z - 34) / 88, 186), 3, fill=(57, 135, 229))
        d.text((410, 230), "Brightest around the ventricles and deep", font=f_small, fill=INK)
        d.text((410, 252), "grey matter, regions known to change", font=f_small, fill=INK)
        d.text((410, 274), "with age. Passes randomisation and", font=f_small, fill=INK)
        d.text((410, 296), "deletion tests.", font=f_small, fill=INK)
        d.text((410, 340), f"real age {r.AGE:.0f} · predicted {r.pred:.0f}", font=f_small, fill=MUTED)
        frames.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=128))
    frames[0].save(out / "brain_sweep.gif", save_all=True, append_images=frames[1:], duration=90, loop=0,
                   optimize=True, disposal=2)
    print(f"  brain_sweep: IXI{int(r.IXI_ID):03d}, {len(frames)} frames")


# Headline numbers shown in the SVG banner and scorecard (from PROJECT_GUIDE.md, Phases 5-9).
THEMES = {
    "light": {"bg": "#fcfcfb", "panel": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "line": "#e1e0d9", "ring": "#c3c2b7", "accent": "#2a78d6", "good": "#0ca30c", "bad": "#d03b3b",
              "open": "#c98500", "guys": "#2a78d6", "hh": "#eb6834", "iop": "#1baf7a"},
    "dark": {"bg": "#0d0d0d", "panel": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "line": "#2c2c2a", "ring": "#383835", "accent": "#3987e5", "good": "#0ca30c", "bad": "#e66767",
             "open": "#eda100", "guys": "#3987e5", "hh": "#d95926", "iop": "#199e70"},
}
FONT = "system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def hero_svg(c):
    stats = [("3.71 yrs", "test error (MAE)"), ("0", "detectable sex gap"),
             ("−5.1 yrs", "IOP brain-age offset"), ("91%", "scanner read from features")]
    cards = "".join(
        f'<g class="card" style="animation-delay:{0.35 + 0.18 * i:.2f}s" transform="translate({60 + i * 172},300)">'
        f'<rect width="160" height="78" rx="12" fill="{c["panel"]}" stroke="{c["line"]}"/>'
        f'<text x="16" y="38" font-size="26" font-weight="700" fill="{c["ink"]}">{v}</text>'
        f'<text x="16" y="60" font-size="12.5" fill="{c["muted"]}">{l}</text></g>' for i, (v, l) in enumerate(stats))
    rings = "".join(f'<circle cx="1000" cy="210" r="{r}" fill="none" stroke="{c["line"]}" stroke-width="1"/>'
                    for r in (60, 100, 140, 180))
    dots = "".join(f'<circle class="blip" style="animation-delay:{d}s" cx="{x}" cy="{y}" r="6" fill="{c[s]}"/>'
                   for x, y, s, d in [(925, 140, "guys", 0.2), (1085, 175, "hh", 1.1), (1040, 300, "iop", 2.0),
                                      (960, 265, "guys", 2.9), (1110, 250, "hh", 3.6), (890, 205, "iop", 4.4)])
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 420" width="1200" height="420" font-family="{FONT}">
<style>
.card{{animation:up .7s ease backwards}}
@keyframes up{{from{{opacity:0;transform:translateY(12px)}}to{{opacity:1}}}}
.sweep{{transform-origin:1000px 210px;animation:spin 6s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
.blip{{opacity:.3;animation:blip 6s ease-out infinite}}
@keyframes blip{{0%{{opacity:.15}}8%{{opacity:1}}40%{{opacity:.3}}100%{{opacity:.15}}}}
.pulse{{animation:pulse 3s ease-in-out infinite}}
@keyframes pulse{{50%{{opacity:.55}}}}
@media (prefers-reduced-motion:reduce){{.card,.sweep,.blip,.pulse{{animation:none}}}}
</style>
<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{c["accent"]}" stop-opacity="0"/><stop offset="1" stop-color="{c["accent"]}" stop-opacity=".35"/></linearGradient></defs>
<rect width="1200" height="420" rx="24" fill="{c["bg"]}"/>
<rect x=".5" y=".5" width="1199" height="419" rx="24" fill="none" stroke="{c["line"]}"/>
<text x="60" y="78" font-size="14" letter-spacing="2.5" fill="{c["muted"]}">BRAIN-AGE MODEL AUDIT · IXI MRI · 485 ADULTS · 3 SITES</text>
<text x="60" y="146" font-size="56" font-weight="800" fill="{c["ink"]}" letter-spacing="-1.5">Right for the</text>
<text x="60" y="206" font-size="56" font-weight="800" fill="{c["ink"]}" letter-spacing="-1.5">Wrong Reasons?</text>
<text x="60" y="250" font-size="20" fill="{c["ink2"]}">Auditing a brain-age deep network beyond accuracy.</text>
{cards}
{rings}
<path class="sweep" d="M1000 210 L1180 210 A180 180 0 0 0 1155.9 120 Z" fill="url(#g)"/>
{dots}
<circle cx="1000" cy="210" r="46" fill="{c["panel"]}" stroke="{c["ring"]}"/>
<text class="pulse" x="1000" y="207" text-anchor="middle" font-size="22" font-weight="800" fill="{c["ink"]}">Partly.</text>
<text x="1000" y="228" text-anchor="middle" font-size="11" fill="{c["muted"]}">the verdict</text>
<g font-size="12.5" fill="{c["ink2"]}"><circle cx="905" cy="398" r="5" fill="{c["guys"]}"/><text x="915" y="402">Guys</text>
<circle cx="975" cy="398" r="5" fill="{c["hh"]}"/><text x="985" y="402">HH</text>
<circle cx="1035" cy="398" r="5" fill="{c["iop"]}"/><text x="1045" y="402">IOP</text></g>
</svg>'''


def verdict_svg(c):
    rows = [("good", "✓", "Accurate", "Test MAE 3.71 yrs, r 0.966 (test set used once)."),
            ("good", "✓", "Fair across sex", "No difference in brain-age gap, error or attribution."),
            ("good", "✓", "Faithful heatmaps, honest uncertainty", "Pass randomisation + deletion tests; uncertainty doubles where it should."),
            ("bad", "✕", "Acquisition-invariant", "Site read from features at 91%; scan look alone shifts age 3–4 yrs."),
            ("open", "?", "IOP offset explained", "−5.1 yrs, replicated; not intensity, not blur, not the feature fingerprint.")]
    body = ""
    for i, (k, sym, title, note) in enumerate(rows):
        y = 66 + i * 82
        body += (f'<g>'
                 f'<rect x="24" y="{y}" width="1152" height="72" rx="12" fill="{c["panel"]}" stroke="{c["line"]}"/>'
                 f'<circle cx="58" cy="{y + 36}" r="17" fill="{c[k]}" fill-opacity=".15"/>'
                 f'<text x="58" y="{y + 43}" text-anchor="middle" font-size="19" font-weight="800" fill="{c[k]}">{sym}</text>'
                 f'<text x="90" y="{y + 31}" font-size="21" font-weight="700" fill="{c["ink"]}">{title}</text>'
                 f'<text x="90" y="{y + 56}" font-size="17" fill="{c["ink2"]}">{note}</text></g>')
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 494" width="1200" height="494" font-family="{FONT}">
<rect width="1200" height="494" rx="20" fill="{c["bg"]}"/><rect x=".5" y=".5" width="1199" height="493" rx="20" fill="none" stroke="{c["line"]}"/>
<text x="24" y="44" font-size="14" letter-spacing="2.5" fill="{c["muted"]}">THE AUDIT SCORECARD</text>
{body}</svg>'''


def main():
    a = args_()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for theme, c in THEMES.items():
        (a.out_dir / f"hero-{theme}.svg").write_text(hero_svg(c), encoding="utf-8")
        (a.out_dir / f"verdict-{theme}.svg").write_text(verdict_svg(c), encoding="utf-8")
    scanner_look(a.project_dir, a.out_dir)
    brain_sweep(a.project_dir, a.out_dir)
    for p in sorted(a.out_dir.glob("*.gif")):
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
