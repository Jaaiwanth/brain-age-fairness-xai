"""Verify what each model receives (orientation, slice range, tensor/intensity, subjects and splits).

python phase2_corrected/verify_inputs.py            # checks 1-4 (label-free + CSV based), ~minutes
python phase2_corrected/verify_inputs.py --sweep    # + DeepBrainNet orientation sweep with CORRECT labels (slow)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

O = C.OUT_ROOT
REF = C.EXTERNAL / "DeepBrainNet" / "Data"
OUT = C.REPO / "phase2_corrected" / "reports"
LINES = []


def say(s=""):
    print(s, flush=True)
    LINES.append(s)


def load(p):
    return np.asarray(nib.load(str(p)).dataobj, dtype=np.float32)


def profile(vols, axis):
    """mean count of brain (non-zero) voxels along one axis, normalised to sum 1"""
    ax = tuple(i for i in range(3) if i != axis)
    p = np.mean([(v > 0).sum(axis=ax) for v in vols], axis=0).astype(float)
    return p / p.sum()


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def check1_2():
    rng = np.random.default_rng(0)
    qc = pd.read_csv(O / "qc" / "qc_summary.csv")
    ids = sorted(rng.choice(qc[qc.status.isin(["PASS", "WARN"])].IXI_ID.to_numpy(), 40, replace=False))
    ref_imgs = [nib.load(str(REF / f"Subject{i}_T1_BrainAligned.nii.gz")) for i in range(1, 5)]
    ref = [np.asarray(i.dataobj, dtype=np.float32) for i in ref_imgs]
    ours_img = nib.load(str(O / "deepbrainnet" / f"IXI{ids[0]:03d}.nii.gz"))
    say("## 1. Front-back flip in `deepbrainnet/` vs the original DeepBrainNet data convention\n")
    say("The original `Slicer.py` does no re-orienting: it takes `data[:, :, z]` of the NIfTI exactly as stored, so the convention is whatever "
        "the shipped reference volumes use. Comparison with those four reference volumes:\n")
    say("| | reference sample (`Subject1`) | our `deepbrainnet/IXI###.nii.gz` | match |\n|---|---|---|---|")
    ra, oa = ref_imgs[0].affine, ours_img.affine
    say(f"| shape | {ref_imgs[0].shape} | {ours_img.shape} | {ref_imgs[0].shape == ours_img.shape} |")
    say(f"| axis codes | {''.join(nib.aff2axcodes(ra))} | {''.join(nib.aff2axcodes(oa))} | {nib.aff2axcodes(ra) == nib.aff2axcodes(oa)} |")
    say(f"| affine (voxel size, origin) | diag {np.diag(ra)[:3].tolist()} origin {ra[:3, 3].tolist()} | diag {np.diag(oa)[:3].tolist()} origin {oa[:3, 3].tolist()} | {np.allclose(ra, oa)} |")
    ours_d = [load(O / "deepbrainnet" / f"IXI{i:03d}.nii.gz") for i in ids]
    ours_s = [load(O / "sfcn" / f"IXI{i:03d}.nii.gz") for i in ids]
    exact = all(np.array_equal(d, s[:, ::-1, :]) for d, s in zip(ours_d, ours_s))
    say(f"\n- `deepbrainnet/` array == `sfcn/` array with y flipped, exactly, for {len(ids)} random subjects: **{exact}**")
    say("- Label-free anatomical check: the front-back profile of brain tissue is asymmetric (frontal pole vs occipital pole/cerebellum), so it identifies "
        "the direction. Correlation of our mean profile with the reference samples' profile (axis 1), as stored vs reversed:\n")
    say("| axis | as stored | reversed |\n|---|---|---|")
    res = {}
    for ax, nm in [(0, "x (left-right)"), (1, "y (front-back)"), (2, "z (up-down)")]:
        pr, po = profile(ref, ax), profile(ours_d, ax)
        res[ax] = (corr(pr, po), corr(pr, po[::-1]))
        say(f"| {nm} | **{res[ax][0]:.3f}** | {res[ax][1]:.3f} |")
    ok_y = res[1][0] > res[1][1] and res[2][0] > res[2][1]
    say(f"\n**Result:** orientation, shape and affine identical to the reference data; y and z profiles match only in the stored direction -> flip is correct: **{ok_y and exact}**.")
    say("(Left-right is nearly symmetric, so it cannot be tested this way; it matters little: flipping it changed SFCN's MAE by 0.2 y.)\n")

    # ---- 2. slice range ----
    say("## 2. Slices 45-124 (80 axial slices)\n")
    k0, k1 = C.DBN_SLICE_START, C.DBN_SLICE_START + C.DBN_N_SLICES - 1
    z0, z1 = ra[2, 3] + ra[2, 2] * k0, ra[2, 3] + ra[2, 2] * k1
    say(f"- Slice axis is axis 2, code `{nib.aff2axcodes(ra)[2]}` (index increases toward Superior), 1 mm voxels, so slice k is at z = {ra[2, 3]:.0f} + k mm (MNI): "
        f"slice {k0} -> z = {z0:.0f} mm, slice {k1} -> z = {z1:.0f} mm.")
    say("- Original `Slicer.py` uses `range(0, 80)` offset by 45 (i.e. 45..124). antspynet's `brain_age()` uses `range(45, 125)` as well, noting that the paper "
        "only specifies 80 slices and this range spans the centre of the brain.")
    zlo = lambda v: np.argwhere(v.any(axis=(0, 1)))[[0, -1], 0]
    ze = np.array([zlo(v) for v in ours_d]); zr = np.array([zlo(v) for v in ref])
    cov = lambda v: float((v[:, :, k0:k1 + 1] > 0).sum() / (v > 0).sum())
    say(f"- Brain extent along z (slice index): ours {ze[:, 0].mean():.0f}-{ze[:, 1].mean():.0f}, reference {zr[:, 0].mean():.0f}-{zr[:, 1].mean():.0f}; "
        f"the 80 slices contain **{100 * np.mean([cov(v) for v in ours_d]):.0f}%** of our brain voxels vs **{100 * np.mean([cov(v) for v in ref]):.0f}%** of the reference brains'.")
    ztop, zbot = ze[:, 1].mean() - 72, ze[:, 0].mean() - 72
    say(f"- Our brains span z = {zbot:.0f} to {ztop:.0f} mm, so the 80 slices (z = {z0:.0f} to {z1:.0f} mm) cover the middle of the brain: they skip the lowest "
        f"{z0 - zbot:.0f} mm (brainstem/lower cerebellum/temporal base) and the top {ztop - z1:.0f} mm (vertex). The reference brains are covered by the same range. See the figure." + "\n")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a_, (v, t) in zip(ax[:2], [(ref[0], "reference sample (Subject1)"), (ours_d[0], f"ours (IXI{ids[0]:03d}, deepbrainnet/)")]):
        a_.imshow(v[v.shape[0] // 2].T, cmap="gray", origin="lower", vmin=0, vmax=np.percentile(v[v > 0], 99.5))
        a_.axhline(k0, c="#E45756", lw=1.5); a_.axhline(k1, c="#E45756", lw=1.5)
        a_.set_title(f"{t}\nmid-sagittal, red = slices {k0} and {k1}"); a_.set_xlabel("y index (reference LPS: posterior to the right)")
    pr, po = profile(ref, 1), profile(ours_d, 1)
    ax[2].plot(pr, label="reference (mean of 4)"); ax[2].plot(po, label="ours, as stored"); ax[2].plot(po[::-1], "--", label="ours, reversed")
    ax[2].set(title=f"front-back tissue profile (r as stored {res[1][0]:.3f}, reversed {res[1][1]:.3f})", xlabel="y index"); ax[2].legend()
    fig.tight_layout(); OUT.mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "input_verification.png", dpi=110)
    say("![orientation and slice range](input_verification.png)\n")
    return ref


def check3(ref):
    import infer
    say("## 3. Tensor dimensions and intensity ranges seen by each model\n")
    s = pd.read_csv(O / "inference" / "sfcn_predictions_all.csv")
    d = pd.read_csv(O / "inference" / "dbn_predictions_all.csv")
    # reference-data inputs for comparison
    ref_d = [infer.dbn_input(v.astype(np.float64))[0] for v in ref]
    ctx_vals = []
    for v in ref:
        x = v[:, ::-1, :].astype(np.float64); x = x / x.mean(); x = x[11:171, 13:205, 11:171]
        ctx_vals.append((x.min(), x.max(), x.mean()))
    say("| | expected (reference code / reference data) | observed over all subjects |\n|---|---|---|")
    say(f"| SFCN tensor | `(1,1,160,192,160)` float32 | {s.in_shape.value_counts().to_dict()} {s.in_dtype.unique().tolist()} |")
    say(f"| SFCN values after `/mean` | reference brains: min 0, max {max(c[1] for c in ctx_vals):.1f}, mean {np.mean([c[2] for c in ctx_vals]):.2f} | min {s.in_min.min():.1f}, max {s.in_max.min():.1f}-{s.in_max.max():.1f}, mean {s.in_mean.mean():.2f} (fixed by crop geometry: 182x218x182 / 160x192x160 = 1.469, so it only confirms the crop) |")
    say(f"| SFCN probabilities | sum to 1 | sum in [{s.prob_sum.min():.4f}, {s.prob_sum.max():.4f}] |")
    say(f"| DeepBrainNet tensor | `(80,256,256,3)` float32 (80 slices, 256x256, RGB) | {d.in_shape.value_counts().to_dict()} {d.in_dtype.unique().tolist()} |")
    say(f"| DeepBrainNet values | [0,1] (uint8/255; p97 -> 185/255 = 0.725); reference brains: max {max(r.max() for r in ref_d):.2f}, mean {np.mean([r.mean() for r in ref_d]):.3f} | min {d.in_min.min():.1f}, max {d.in_max.max():.2f} ({int((d.in_max < 0.7).sum())} subjects with max below 0.7), mean {d.in_mean.mean():.3f} (sd {d.in_mean.std():.3f}) |")
    say(f"| DeepBrainNet p97 (raw) | positive, finite | min {d.p97_raw.min():.0f}, max {d.p97_raw.max():.0f} (scanner scales differ; removed by the x185/p97 rescale), all finite {np.isfinite(d.p97_raw).all()} |")
    say(f"| per-slice predictions | 80 finite values | slice-range min {d.slice_pred_min.min():.1f}, max {d.slice_pred_max.max():.1f} |")
    say("\nThe SFCN checkpoint was loaded with `strict=True` (all keys matched) and the DeepBrainNet file by SHA256 against the repo's LFS pointer, "
        "so the networks themselves are the released ones.\n")
    return s, d


def check4(s, d):
    say("## 4. Same subjects and same splits for both models\n")
    meta = pd.read_csv(C.METADATA_PATH)
    spl = {n: pd.read_csv(C.PROJECT_DIR / f"ixi_{n}.csv") for n in ("train", "val", "test")}
    sets = {n: set(v.IXI_ID) for n, v in spl.items()}
    disj = not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    union = sets["train"] | sets["val"] | sets["test"]
    say(f"- Split files: train {len(sets['train'])}, val {len(sets['val'])}, test {len(sets['test'])}; disjoint **{disj}**; union equals the {len(meta)} corrected-label subjects **{union == set(meta.IXI_ID)}**.")
    same_ids = set(s.IXI_ID) == set(d.IXI_ID)
    say(f"- SFCN predicted {len(s)} subjects, DeepBrainNet {len(d)}; identical subject sets **{same_ids}**; no duplicates **{not (s.IXI_ID.duplicated().any() or d.IXI_ID.duplicated().any())}**.")
    j = s.merge(d, on="IXI_ID", suffixes=("_s", "_d"))
    say(f"- Same labels in both prediction files: ages identical **{np.allclose(j.actual_age_s, j.actual_age_d)}**, matching the corrected metadata **{np.allclose(j.set_index('IXI_ID').actual_age_s, meta.set_index('IXI_ID').loc[j.IXI_ID].AGE)}**.")
    sm = {i: n for n, st in sets.items() for i in st}
    say(f"- `split` column in both files equals the split files: SFCN **{(s.IXI_ID.map(sm) == s.split).all()}**, DeepBrainNet **{(d.IXI_ID.map(sm) == d.split).all()}**; counts "
        f"{s.split.value_counts().to_dict()} / {d.split.value_counts().to_dict()}.")
    say(f"- Same QC status for both: **{(j.qc_status_s == j.qc_status_d).all()}**.")
    say("- Both models are zero-shot (no training), so the splits are used only to report accuracy per split; no model has seen any split.\n")


def sweep():
    import infer
    from scipy import stats
    d = pd.read_csv(O / "inference" / "dbn_predictions_all.csv").sort_values("actual_age")
    sub = d.iloc[np.linspace(0, len(d) - 1, 25).astype(int)]
    model, _ = infer.load_dbn()
    vols = {int(r.IXI_ID): load(O / "deepbrainnet" / f"IXI{int(r.IXI_ID):03d}.nii.gz").astype(np.float64) for r in sub.itertuples()}
    rows = []
    for fx in (0, 1):
        for fy in (0, 1):
            for tr in (0, 1):
                preds = []
                for i, v in vols.items():
                    w = v[::-1] if fx else v
                    w = w[:, ::-1] if fy else w
                    w = w.transpose(1, 0, 2) if tr else w
                    preds.append(infer.dbn_predict(model, np.ascontiguousarray(w))[0])
                age = sub.actual_age.to_numpy(); p = np.array(preds)
                rows.append(dict(flip_rows=fx, flip_cols=fy, transpose=tr, MAE=np.abs(p - age).mean(), r=np.corrcoef(age, p)[0, 1], slope=stats.linregress(age, p).slope))
                print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "input_verification_sweep.csv", index=False)
    return t


def print_sweep(t):
    t = t.copy()
    t["as stored"] = np.where((t.flip_rows == 0) & (t.flip_cols == 0) & (t["transpose"] == 0), "<-- pipeline", "")
    say("| flip rows (x) | flip cols (y) | transpose | MAE (y) | Pearson r | slope | |\n|---|---|---|---|---|---|---|")
    for r in t.itertuples():
        say(f"| {r.flip_rows} | {r.flip_cols} | {r.transpose} | {r.MAE:.2f} | {r.r:.3f} | {r.slope:.2f} | {r._7} |")
    say("")


def orient(v, fx, fy, tr):
    w = v[::-1] if fx else v
    w = w[:, ::-1] if fy else w
    return np.ascontiguousarray(w.transpose(1, 0, 2) if tr else w)


def sweep_reference():
    import itertools
    import infer
    model, _ = infer.load_dbn()
    ages = pd.read_csv(REF.parent / "Sample_Data.csv")
    vols = [np.asarray(nib.load(str(REF / f"{i}_T1_BrainAligned.nii.gz")).dataobj, dtype=np.float64) for i in ages.ID]
    rows = []
    for fx, fy, tr in itertools.product((0, 1), (0, 1), (0, 1)):
        e = np.array([infer.dbn_predict(model, orient(v, fx, fy, tr))[0] for v in vols]) - ages.Age.to_numpy()
        rows.append(dict(flip_rows=fx, flip_cols=fy, transpose=tr, MAE=np.abs(e).mean(), bias=e.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "input_verification_reference_sweep.csv", index=False)
    return t


def paired_100():
    import infer
    d = pd.read_csv(O / "inference" / "dbn_predictions_all.csv")
    sub = d.sample(100, random_state=1)
    model, _ = infer.load_dbn()
    rows = []
    for r in sub.itertuples():
        v = load(O / "deepbrainnet" / f"IXI{int(r.IXI_ID):03d}.nii.gz").astype(np.float64)
        rows.append(dict(IXI_ID=r.IXI_ID, actual_age=r.actual_age, as_stored=infer.dbn_predict(model, v)[0],
                         transposed=infer.dbn_predict(model, orient(v, 0, 0, 1))[0]))
        print(len(rows), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "input_verification_paired100.csv", index=False)
    return t


def print_reference(t):
    say("### 5a. DeepBrainNet's own reference brains (known ages 45-58), all 8 orientations\n")
    say("These are the authors' data in their convention, so the orientation in which the model reproduces their ages is the one it expects.\n")
    say("| flip rows (x) | flip cols (y) | transpose | MAE (y) | bias (y) | |\n|---|---|---|---|---|---|")
    for r in t.itertuples():
        mark = "<-- pipeline" if (r.flip_rows, r.flip_cols, r.transpose) == (0, 0, 0) else ""
        say(f"| {r.flip_rows} | {r.flip_cols} | {r.transpose} | {r.MAE:.2f} | {r.bias:+.2f} | {mark} |")
    say("")


def print_paired(t):
    from scipy import stats
    a, b, age = t.as_stored.to_numpy(), t.transposed.to_numpy(), t.actual_age.to_numpy()
    rng = np.random.default_rng(0)
    dm, dr = [], []
    for _ in range(3000):
        i = rng.integers(0, len(t), len(t))
        dm.append(np.abs(a[i] - age[i]).mean() - np.abs(b[i] - age[i]).mean())
        dr.append(np.corrcoef(age[i], a[i])[0, 1] - np.corrcoef(age[i], b[i])[0, 1])
    say("### 5b. Paired comparison on 100 random subjects: as stored vs transposed\n")
    say("| orientation | MAE (y) | bias (y) | Pearson r | slope |\n|---|---|---|---|---|")
    for nm, p in (("as stored (pipeline)", a), ("transposed (rows/cols swapped)", b)):
        say(f"| {nm} | {np.abs(p - age).mean():.2f} | {(p - age).mean():+.2f} | {np.corrcoef(age, p)[0, 1]:.3f} | {stats.linregress(age, p).slope:.2f} |")
    say("\n" + f"Paired bootstrap, as stored minus transposed: MAE difference {np.mean(dm):+.2f} y (95% CI {np.percentile(dm, 2.5):+.2f} to {np.percentile(dm, 97.5):+.2f}), "
        f"r difference {np.mean(dr):+.3f} (95% CI {np.percentile(dr, 2.5):+.3f} to {np.percentile(dr, 97.5):+.3f}).\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--paired", action="store_true")
    a = ap.parse_args()
    say("# Input verification: what SFCN and DeepBrainNet actually receive\n")
    ref = check1_2()
    s, d = check3(ref)
    check4(s, d)
    csvp = OUT / "input_verification_sweep.csv"
    if a.sweep:
        sweep_t = sweep()
    elif csvp.exists():
        sweep_t = pd.read_csv(csvp)
    else:
        sweep_t = None
    if sweep_t is not None:
        say("## 5. DeepBrainNet orientation sweep with the CORRECT labels (25 age-spread subjects)" + "\n")
        say("The earlier sweep was scored against the wrong labels and proved nothing. Here the same 8 flip/transpose combinations are applied to the volume "
            "before the full reference chain (JPEG, 256 resize, ...). The row marked as pipeline is what the pipeline uses." + "\n")
        print_sweep(sweep_t)
    rp, pp = OUT / "input_verification_reference_sweep.csv", OUT / "input_verification_paired100.csv"
    if a.reference:
        sweep_reference()
    if a.paired:
        paired_100()
    if rp.exists():
        print_reference(pd.read_csv(rp))
    if pp.exists():
        print_paired(pd.read_csv(pp))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "input_verification.md").write_text("\n".join(LINES), encoding="utf8")
    print("wrote", OUT / "input_verification.md")
