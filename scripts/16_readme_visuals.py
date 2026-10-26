"""Build the README's visuals (assets/readme/): animated SVGs (light + dark) and GIFs from real data.

    hero_{light,dark}.svg          animated title banner with a real MRI slice and its attribution map
    scorecard_{light,dark}.svg     the audit's eight questions and answers, revealed in sequence
    casefile_{light,dark}.svg      the IOP investigation: three explanations ruled out, one open
    scanner_swap.gif               one held-out brain given each site's intensity look, with the
                                   predicted age the frozen model actually returned (Phase 9b)
    attribution_sweep.gif          a sweep through one brain's axial slices with its attribution map

All numbers are read from saved results; images come from the preprocessed volumes, the Phase 7
attribution maps and the Phase 9b site references. No model is run. GitHub plays SVG animations
when the file is shown as an image; the <picture> tags in README.md pick the light or dark file.

Usage:
    python scripts/16_readme_visuals.py --project-dir D:/ML_Project
"""

import argparse
import base64
import importlib.util
import io
import json
import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ft = _load("ft", "04_finetune_sfcn.py")
st = _load("st", "09_acquisition_stress_test.py")

re_minus = re.compile(r"(?<![\w])-(?=\d)")
OUT = Path(__file__).resolve().parent.parent / "assets" / "readme"
SITES = ["Guys", "HH", "IOP"]
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": {"bg": "#ffffff", "bg2": "#f6f8fa", "card": "#f6f8fa", "line": "#d1d9e0", "ink": "#1f2328",
              "ink2": "#424a53", "muted": "#59636e", "accent": "#2a78d6", "good": "#1a7f37", "warn": "#bf8700",
              "bad": "#cf222e", "guys": "#2a78d6", "hh": "#eb6834", "iop": "#1baf7a", "glow": "rgba(42,120,214,0.10)"},
    "dark": {"bg": "#0d1117", "bg2": "#111826", "card": "#161b22", "line": "#30363d", "ink": "#e6edf3",
             "ink2": "#c9d1d9", "muted": "#8b949e", "accent": "#4c9aff", "good": "#3fb950", "warn": "#d29922",
             "bad": "#f85149", "guys": "#3987e5", "hh": "#d95926", "iop": "#199e70", "glow": "rgba(76,154,255,0.14)"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Build README visuals.")
    p.add_argument("--project-dir", type=Path, default=ft.DEFAULT_PROJECT_DIR)
    p.add_argument("--volume-dir", type=Path, default=None)
    return p.parse_args()


def font(size, bold=False):
    for name in (["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def hot_rgba(x):
    r, g, b = np.clip(x / 0.4, 0, 1), np.clip((x - 0.4) / 0.4, 0, 1), np.clip((x - 0.8) / 0.2, 0, 1)
    return (np.stack([r, g, b, np.clip(x * 1.4, 0, 0.85)], -1) * 255).astype(np.uint8)


def to_uri(img):
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gray_img(sl, hi, scale=2):
    a = (np.clip(np.rot90(sl) / hi, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a, "L").resize((a.shape[1] * scale, a.shape[0] * scale), Image.LANCZOS)


def cam_img(sl, scale=2):
    a = hot_rgba(np.rot90(sl))
    return Image.fromarray(a, "RGBA").resize((a.shape[1] * scale, a.shape[0] * scale), Image.LANCZOS)


def esc(s):
    s = re_minus.sub("\u2212", s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tspans(text, x, width, lh):
    lines = textwrap.wrap(text, width)
    return "".join(f'<tspan x="{x}" dy="{0 if i == 0 else lh}">{esc(l)}</tspan>' for i, l in enumerate(lines))


# ---------------------------------------------------------------- SVGs

def hero_svg(t, brain_uri, cam_uri, real, pred):
    W, H = 1200, 440
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Right for the wrong reasons? Auditing a brain-age model beyond accuracy">
<defs>
 <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{t['bg2']}"/><stop offset="1" stop-color="{t['bg']}"/></linearGradient>
 <radialGradient id="halo" cx="0.5" cy="0.5" r="0.5"><stop offset="0" stop-color="{t['accent']}" stop-opacity="0.35"/><stop offset="1" stop-color="{t['accent']}" stop-opacity="0"/></radialGradient>
 <clipPath id="clip"><rect x="790" y="40" width="340" height="340" rx="28"/></clipPath>
 <linearGradient id="scan" x1="0" x2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset="0.5" stop-color="#fff" stop-opacity="0.7"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
</defs>
<style>
 text {{ font-family: {FONT}; }}
 .up {{ animation: up .9s cubic-bezier(.2,.7,.2,1) both; }}
 .d1 {{ animation-delay: .15s; }} .d2 {{ animation-delay: .35s; }} .d3 {{ animation-delay: .6s; }}
 .d4 {{ animation-delay: .9s; }} .d5 {{ animation-delay: 1.1s; }} .d6 {{ animation-delay: 1.3s; }}
 @keyframes up {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: none; }} }}
 .cam {{ animation: breathe 4.5s ease-in-out infinite; }}
 @keyframes breathe {{ 0%,100% {{ opacity: .2; }} 50% {{ opacity: .95; }} }}
 .scan {{ animation: sweep 5s linear infinite; }}
 @keyframes sweep {{ from {{ transform: translateY(0); }} to {{ transform: translateY(330px); }} }}
 .halo {{ animation: pulse 4.5s ease-in-out infinite; transform-origin: 960px 210px; }}
 @keyframes pulse {{ 0%,100% {{ opacity: .5; transform: scale(.94); }} 50% {{ opacity: 1; transform: scale(1.04); }} }}
 @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
<rect width="{W}" height="{H}" rx="24" fill="url(#bg)" stroke="{t['line']}"/>
<g opacity="0.5" stroke="{t['line']}">{''.join(f'<line x1="{x}" y1="0" x2="{x}" y2="{H}"/>' for x in range(60, 760, 60))}</g>
<text class="up d1" x="60" y="92" font-size="14" letter-spacing="2.5" fill="{t['muted']}">BRAIN-AGE MODEL AUDIT · IXI MRI · 485 ADULTS · 3 LONDON SITES</text>
<text class="up d2" x="56" y="168" font-size="66" font-weight="700" fill="{t['ink']}" letter-spacing="-1.5">Right for the</text>
<text class="up d3" x="56" y="240" font-size="66" font-weight="700" fill="{t['ink']}" letter-spacing="-1.5">wrong reasons?</text>
<text class="up d4" x="60" y="288" font-size="21" fill="{t['muted']}">Auditing a deep brain-age model beyond accuracy.</text>
<g class="up d5">
 <rect x="60" y="322" width="560" height="62" rx="14" fill="{t['card']}" stroke="{t['line']}"/>
 <text x="82" y="362" font-size="26" font-weight="700" fill="{t['ink']}">Partly.</text>
 <text x="190" y="348" font-size="15" fill="{t['ink2']}">Accurate, and fair across sex. But it recognises the</text>
 <text x="190" y="370" font-size="15" fill="{t['ink2']}">scanner, and its answer moves when only the look changes.</text>
</g>
<circle class="halo" cx="960" cy="210" r="250" fill="url(#halo)"/>
<g clip-path="url(#clip)">
 <rect x="790" y="40" width="340" height="340" fill="#050505"/>
 <image href="{brain_uri}" x="800" y="50" width="320" height="320" preserveAspectRatio="xMidYMid meet"/>
 <image class="cam" href="{cam_uri}" x="800" y="50" width="320" height="320" preserveAspectRatio="xMidYMid meet"/>
 <rect class="scan" x="790" y="45" width="340" height="2" fill="url(#scan)"/>
 <text x="808" y="364" font-size="13" fill="#cfd6df">attribution · axial slice</text>
 <text x="1112" y="364" font-size="13" fill="#cfd6df" text-anchor="end">real <tspan font-weight="700" fill="#fff" font-size="16">{real:.0f}</tspan> · predicted <tspan font-weight="700" fill="#fff" font-size="16">{pred:.0f}</tspan></text>
</g>
<rect x="790" y="40" width="340" height="340" rx="28" fill="none" stroke="{t['line']}"/>
</svg>"""


def scorecard_svg(t, rows):
    W, top, rh = 1200, 92, 66
    H = top + rh * len(rows) + 28
    parts = [f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Audit scorecard: eight questions and answers">
<style>
 text {{ font-family: {FONT}; }}
 .row {{ animation: in .7s cubic-bezier(.2,.7,.2,1) both; }}
 @keyframes in {{ from {{ opacity: 0; transform: translateX(-18px); }} to {{ opacity: 1; transform: none; }} }}
 .bar {{ animation: grow 1s cubic-bezier(.2,.7,.2,1) both; transform-box: fill-box; transform-origin: left; }}
 @keyframes grow {{ from {{ transform: scaleX(0); }} to {{ transform: scaleX(1); }} }}
 @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
<rect width="{W}" height="{H}" rx="24" fill="{t['bg']}" stroke="{t['line']}"/>
<text x="40" y="52" font-size="24" font-weight="700" fill="{t['ink']}">The audit in eight questions</text>
<text x="{W - 40}" y="52" font-size="14" fill="{t['muted']}" text-anchor="end"><tspan fill="{t['good']}" font-weight="700">✓</tspan> holds up<tspan dx="18" fill="{t['warn']}" font-weight="700">!</tspan> needs caution</text>"""]
    for i, (ok, q, big, detail) in enumerate(rows):
        y = top + i * rh
        col = t["good"] if ok else t["warn"]
        d = 0.25 + i * 0.22
        parts.append(f"""<g class="row" style="animation-delay:{d:.2f}s">
 <rect x="24" y="{y}" width="{W - 48}" height="{rh - 10}" rx="14" fill="{t['card']}" stroke="{t['line']}"/>
 <rect class="bar" style="animation-delay:{d + .1:.2f}s" x="24" y="{y}" width="5" height="{rh - 10}" rx="2.5" fill="{col}"/>
 <circle cx="62" cy="{y + (rh - 10) / 2}" r="15" fill="{col}" fill-opacity="0.16" stroke="{col}" stroke-width="1.5"/>
 <text x="62" y="{y + (rh - 10) / 2 + 6}" font-size="17" font-weight="700" fill="{col}" text-anchor="middle">{'✓' if ok else '!'}</text>
 <text x="94" y="{y + 34}" font-size="18" font-weight="600" fill="{t['ink']}">{esc(q)}</text>
 <text x="660" y="{y + 36}" font-size="26" font-weight="700" fill="{t['ink']}" text-anchor="end">{esc(big)}</text>
 <text x="684" y="{y + 34}" font-size="15" fill="{t['muted']}">{esc(detail)}</text>
</g>""")
    parts.append("</svg>")
    return "\n".join(parts)


def casefile_svg(t, cards, clue):
    W, H, gap, x0, y0 = 1200, 488, 16, 24, 88
    cw = (W - 2 * x0 - 3 * gap) / 4
    parts = [f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Case file: what explains the IOP offset">
<style>
 text {{ font-family: {FONT}; }}
 .card {{ animation: rise .8s cubic-bezier(.2,.7,.2,1) both; }}
 @keyframes rise {{ from {{ opacity: 0; transform: translateY(16px); }} to {{ opacity: 1; transform: none; }} }}
 .stamp {{ animation: stamp .45s cubic-bezier(.3,1.6,.5,1) both; transform-box: fill-box; transform-origin: center; }}
 @keyframes stamp {{ from {{ opacity: 0; transform: scale(2.4) rotate(-14deg); }} to {{ opacity: .92; transform: scale(1) rotate(-9deg); }} }}
 .q {{ animation: qp 2.4s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }}
 @keyframes qp {{ 0%,100% {{ opacity: .45; transform: scale(.94); }} 50% {{ opacity: 1; transform: scale(1.06); }} }}
 @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; }} }}
</style>
<rect width="{W}" height="{H}" rx="24" fill="{t['bg']}" stroke="{t['line']}"/>
<text x="40" y="52" font-size="24" font-weight="700" fill="{t['ink']}">Case file: why do IOP brains look 5 years younger?</text>"""]
    for i, c in enumerate(cards):
        x, d = x0 + i * (cw + gap), 0.3 + i * 0.35
        open_ = c.get("open", False)
        parts.append(f"""<g class="card" style="animation-delay:{d:.2f}s">
 <rect x="{x}" y="{y0}" width="{cw}" height="304" rx="16" fill="{t['card']}" stroke="{t['accent'] if open_ else t['line']}" stroke-width="{1.5 if open_ else 1}" {'stroke-dasharray="6 5"' if open_ else ''}/>
 <text x="{x + 20}" y="{y0 + 32}" font-size="12" letter-spacing="1.8" fill="{t['muted']}">{esc(c['tag'])}</text>
 <text x="{x + 20}" y="{y0 + 62}" font-size="20" font-weight="700" fill="{t['ink']}">{esc(c['title'])}</text>
 <text x="{x + 20}" y="{y0 + 92}" font-size="14" fill="{t['ink2']}">{tspans(c['test'], x + 20, 33, 20)}</text>
 <text x="{x + 20}" y="{y0 + 258}" font-size="30" font-weight="700" fill="{t['ink']}">{esc(c['big'])}</text>
 <text x="{x + 20}" y="{y0 + 284}" font-size="13" fill="{t['muted']}">{esc(c['small'])}</text>
</g>""")
        if open_:
            parts.append(f'<text class="q" x="{x + cw - 44}" y="{y0 + 268}" font-size="64" font-weight="700" fill="{t["accent"]}" text-anchor="middle">?</text>')
        else:
            parts.append(f"""<g class="stamp" style="animation-delay:{d + .7:.2f}s">
 <rect x="{x + 20}" y="{y0 + 170}" width="132" height="40" rx="6" fill="none" stroke="{t['bad']}" stroke-width="3"/>
 <text x="{x + 86}" y="{y0 + 197}" font-size="17" font-weight="800" letter-spacing="2" fill="{t['bad']}" text-anchor="middle">RULED OUT</text>
</g>""")
    parts.append(f"""<g class="card" style="animation-delay:1.9s">
 <rect x="{x0}" y="{y0 + 320}" width="{W - 2 * x0}" height="56" rx="14" fill="{t['glow']}" stroke="{t['line']}"/>
 <text x="{x0 + 22}" y="{y0 + 354}" font-size="15" fill="{t['ink2']}"><tspan font-weight="700" fill="{t['ink']}">Clue  </tspan>{esc(clue)}</text>
</g>
</svg>""")
    return "\n".join(parts)


# ---------------------------------------------------------------- GIFs

def rounded_mask(size, r):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], r, fill=255)
    return m


def scanner_swap_gif(vol, refs, preds, real, subj_id, out):
    W, H, S = 900, 420, 340
    bg, card, ink, muted = (13, 17, 23), (22, 27, 34), (230, 237, 243), (139, 148, 158)
    colors = {"Guys": (57, 135, 229), "HH": (217, 89, 38), "IOP": (25, 158, 112)}
    order = ["HH", "IOP", "Guys", "HH"]
    looks = {s: st.match_to(vol, refs[s]) for s in SITES}
    hi = np.percentile(looks["HH"][looks["HH"] > 0], 99.5)
    tiles = {s: gray_img(looks[s][:, :, 72], hi, 1).resize((S, S), Image.LANCZOS) for s in SITES}
    f_big, f_mid, f_sm, f_xs = font(64, True), font(22, True), font(18), font(15)
    mask = rounded_mask((S, S), 22)
    lo_age, hi_age = 20, 90

    def frame(img, active, shown_age, marker_age):
        im = Image.new("RGB", (W, H), bg)
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([12, 12, W - 13, H - 13], 22, fill=card, outline=(48, 54, 61))
        im.paste(img.convert("RGB"), (40, 40), mask)
        x0 = 420
        d.text((x0, 44), "SAME BRAIN · ONLY THE SCANNER LOOK CHANGES", font=f_xs, fill=muted)
        d.text((x0, 72), f"{subj_id}  ·  scanned at HH  ·  real age {real:.0f}", font=f_sm, fill=ink)
        cx = x0
        for s in SITES:
            on = s == active
            label = f"looks like {s}"
            w = d.textlength(label, font=f_xs) + 28
            d.rounded_rectangle([cx, 116, cx + w, 148], 16, fill=colors[s] if on else bg,
                                outline=colors[s], width=2)
            d.text((cx + 14, 122), label, font=f_xs, fill=(255, 255, 255) if on else muted)
            cx += w + 10
        d.text((x0, 182), "predicted age", font=f_sm, fill=muted)
        d.text((x0, 206), "—" if shown_age is None else f"{shown_age:.1f}", font=f_big, fill=ink)
        if shown_age is not None:
            delta = shown_age - preds["HH"]
            note = "its own scanner" if active == "HH" else f"{delta:+.1f} years vs its own scanner".replace("-", "\u2212")
            d.text((x0, 290), note, font=f_mid, fill=colors[active] if active != "HH" else muted)
        gx0, gx1, gy = x0, W - 50, 360
        xa = lambda a: gx0 + (a - lo_age) / (hi_age - lo_age) * (gx1 - gx0)
        d.line([gx0, gy, gx1, gy], fill=(48, 54, 61), width=4)
        for a in range(20, 91, 10):
            d.text((xa(a) - 9, gy + 12), str(a), font=f_xs, fill=muted)
        d.line([xa(real), gy - 18, xa(real), gy + 6], fill=ink, width=3)
        d.text((xa(real) - 14, gy - 40), "real", font=f_xs, fill=ink)
        mx = xa(marker_age)
        d.ellipse([mx - 9, gy - 9, mx + 9, gy + 9], fill=colors[active], outline=card, width=3)
        return im

    frames, durations = [], []
    for a, b in zip(order[:-1], order[1:]):
        frames.append(frame(tiles[a], a, preds[a], preds[a]))
        durations.append(2200)
        for k in range(1, 9):
            w = k / 9
            img = Image.blend(tiles[a], tiles[b], w)
            frames.append(frame(img, b if w > 0.5 else a, None, preds[a] + (preds[b] - preds[a]) * (w * w * (3 - 2 * w))))
            durations.append(70)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)


def attribution_sweep_gif(vol, cam, out):
    S = 300
    brain = vol > 0
    hi = np.percentile(vol[brain], 99.5)
    zs = list(range(40, 130, 5))
    zs = zs + zs[::-1][1:-1]
    f_xs = font(14)
    frames = []
    sag = gray_img(vol[80], hi, 1).resize((120, 144), Image.LANCZOS).convert("RGB")
    for z in zs:
        base = gray_img(vol[:, :, z], hi, 1).convert("RGBA")
        over = cam_img(cam[:, :, z], 1)
        img = Image.alpha_composite(base, over).resize((S, int(S * base.height / base.width)), Image.LANCZOS)
        im = Image.new("RGB", (S + 150, max(img.height, 200) + 20), (5, 5, 5))
        im.paste(img.convert("RGB"), (10, 10))
        th = sag.copy()
        dd = ImageDraw.Draw(th)
        yline = th.height - 1 - int(z / 160 * th.height)
        dd.line([0, yline, th.width, yline], fill=(255, 196, 0), width=2)
        im.paste(th, (S + 20, 20))
        d = ImageDraw.Draw(im)
        d.text((S + 20, 172), f"slice {z}", font=f_xs, fill=(200, 205, 212))
        d.text((S + 20, 192), "attribution", font=f_xs, fill=(255, 170, 60))
        frames.append(im.quantize(colors=96, method=Image.Quantize.MEDIANCUT))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=130, loop=0, optimize=True)


def main():
    args = parse_args()
    P = args.project_dir
    R = P / "results"
    volume_dir = args.volume_dir or P / ft.DEFAULT_VOLUME_SUBDIR
    OUT.mkdir(parents=True, exist_ok=True)
    maps = np.load(R / "phase7" / "gradcam_conv_2_lowres.npz")

    def load(i):
        v = np.load(ft.volume_path(volume_dir, i)).astype(np.float32)
        return v / (v.sum() / ft.SFCN_NORMALISATION_VOXELS)

    def cam_of(i, v):
        c = zoom(maps[f"IXI{int(i):03d}"], 8, order=1) * (v > 0)
        return np.clip(c / max(np.percentile(c[v > 0], 99.5), 1e-12), 0, 1)

    # hero: a held-out Guys brain over 55, as on the website
    unc = pd.read_csv(R / "phase9f_uncertainty" / "main_model_uncertainty.csv")
    cand = unc[(unc.site == "Guys") & (unc.AGE > 55)]
    h = cand.loc[(cand.pred - cand.AGE).abs().idxmin()]
    v = load(h.IXI_ID)
    hi = np.percentile(v[v > 0], 99.5)
    c = cam_of(h.IXI_ID, v)
    brain_uri, cam_uri = to_uri(gray_img(v[:, :, 72], hi)), to_uri(cam_img(c[:, :, 72]))

    # numbers
    p5 = json.loads((R / "phase5" / "summary.json").read_text())
    eff = pd.read_csv(R / "phase6" / "age_adjusted_effects.csv").set_index(["outcome", "term"])
    loso = pd.read_csv(R / "phase6b" / "loso_effects.csv").set_index("heldout_site")
    probe = json.loads((R / "phase9a_site_probe" / "summary.json").read_text())["probe"]
    hm = json.loads((R / "phase9e_harmonisation" / "summary.json").read_text())["results"]
    un = json.loads((R / "phase9f_uncertainty" / "summary.json").read_text())
    dl = pd.read_csv(R / "phase9g_deletion_test" / "deletion_summary.csv").set_index("fraction")
    sw = json.loads((R / "phase9b_stress_test" / "summary.json").read_text())["headline_vs_own_site_control"]
    bl = json.loads((R / "phase9d_blur_to_iop" / "summary.json").read_text())["groups"]

    sex = eff.loc[("bag_corrected", "Male vs Female")]
    iop = eff.loc[("bag_corrected", "IOP vs Guys")]
    seen = [g for g in un["loso"]["IOP"]["by_group"] if "unseen" not in g["group"]][0]["mean_sd"]
    unseen = [g for g in un["loso"]["IOP"]["by_group"] if "unseen" in g["group"]][0]["mean_sd"]
    ratio = dl.loc[0.05, "top_mean"] / dl.loc[0.05, "random_mean"]
    hh_iop, iop_hh = sw["HH -> IOP (vs own-site match)"]["mean_delta"], sw["IOP -> HH (vs own-site match)"]["mean_delta"]
    rows = [
        (True, "Is it accurate?", f"{p5['raw']['metrics']['mae']:.2f} yrs", f"test MAE, r {p5['raw']['metrics']['r']:.3f}, evaluated once"),
        (True, "Is it fair across sex?", "no gap", f"male vs female {sex.estimate:+.2f} yrs [{sex.boot_ci_low:+.2f}, {sex.boot_ci_high:+.2f}]"),
        (False, "Is it fair across sites?", f"{iop.estimate:+.1f} yrs", "IOP brain-age gap vs Guys; replicated in validation and test"),
        (False, "Does it generalise to an unseen site?", f"{loso.loc['IOP', 'abs_error_effect']:+.1f} yrs", "extra error on IOP when IOP is left out of training"),
        (False, "Does it know which scanner it saw?", f"{100 * probe['sfcn_finetuned_resid']['balanced_accuracy']:.0f}%", "site read from its features, age and sex removed (chance 33%)"),
        (False, "Does its answer move with the scan's look?", f"{abs(hh_iop):.1f} yrs", f"same brain, another site's intensity profile (IOP → HH: {iop_hh:+.1f})"),
        (True, "Does it know when it is unsure?", f"{unseen / seen:.1f}×", "uncertainty on a never-seen site; its 90% range still holds"),
        (True, "Are its heatmaps faithful?", f"{ratio:.1f}×", "prediction change: top-ranked vs random regions removed"),
    ]
    g_i, i_g = sw["Guys -> IOP (vs own-site match)"]["mean_delta"], sw["IOP -> Guys (vs own-site match)"]["mean_delta"]
    cards = [
        {"tag": "SUSPECT 1", "title": "Intensity profile", "test": "Give Guys brains IOP's look and IOP brains Guys' look, with references standardised for age and sex.",
         "big": f"≤ {max(abs(g_i), abs(i_g)):.1f} yrs", "small": "shift, against a 5-year offset"},
        {"tag": "SUSPECT 2", "title": "Blurrier images", "test": "IOP is ~2 SD smoother at the same resolution. Blur Guys brains to exactly IOP's level.",
         "big": f"{bl['Guys']['matched']['mean']:+.1f} yrs", "small": "older, not younger"},
        {"tag": "SUSPECT 3", "title": "Scanner fingerprint", "test": f"Remove it from the features (ComBat): site read {hm['original']['site_probe_balanced_accuracy'] * 100:.0f}% → {hm['harmonised']['site_probe_balanced_accuracy'] * 100:.0f}%, accuracy unchanged.",
         "big": f"{hm['original']['effects']['IOP vs Guys']['estimate']:+.1f} → {hm['harmonised']['effects']['IOP vs Guys']['estimate']:+.1f}", "small": "IOP offset, years: it persists"},
        {"tag": "STILL OPEN", "title": "Something else", "test": "Untested acquisition properties, or real differences in who IOP recruited. IXI cannot separate them.",
         "big": "", "small": "needs travelling-subject data", "open": True},
    ]
    clue = "the offset is absent in the brains the model trained on (−1.0) and appears in new ones (−4.1 validation, −5.1 test)."

    for name, t in THEMES.items():
        (OUT / f"hero_{name}.svg").write_text(hero_svg(t, brain_uri, cam_uri, h.AGE, h.pred), encoding="utf-8")
        (OUT / f"scorecard_{name}.svg").write_text(scorecard_svg(t, rows), encoding="utf-8")
        (OUT / f"casefile_{name}.svg").write_text(casefile_svg(t, cards, clue), encoding="utf-8")

    # scanner swap: the held-out HH brain whose prediction moves most under IOP's look
    sp = pd.read_csv(R / "phase9b_stress_test" / "stress_predictions.csv")
    m = sp[sp.condition.str.startswith("match ")].pivot(index="IXI_ID", columns="condition", values="predicted_age")
    meta = sp.drop_duplicates("IXI_ID").set_index("IXI_ID")
    hh = m[meta.loc[m.index, "site"] == "HH"]
    pick = (hh["match IOP"] - hh["match HH"]).idxmin()
    refs = dict(np.load(R / "phase9b_stress_test" / "site_reference_quantiles.npz"))
    preds = {s: float(m.loc[pick, f"match {s}"]) for s in SITES}
    scanner_swap_gif(load(pick), refs, preds, float(meta.loc[pick, "age"]), f"IXI{int(pick):03d}", OUT / "scanner_swap.gif")
    print(f"scanner swap: IXI{int(pick):03d} real {meta.loc[pick, 'age']:.1f} | " +
          ", ".join(f"{s} look {preds[s]:.1f}" for s in SITES))

    v2 = load(h.IXI_ID)
    attribution_sweep_gif(v2, cam_of(h.IXI_ID, v2), OUT / "attribution_sweep.gif")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1024:7.0f} KB")


if __name__ == "__main__":
    main()
