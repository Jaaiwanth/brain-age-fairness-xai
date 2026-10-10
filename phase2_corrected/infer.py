"""Pretrained-model inference on the corrected volumes. NO training / fine-tuning anywhere.

SFCN  : reproduces examples.ipynb -> data/data.mean() -> dpu.crop_center(160,192,160) ->
        model.eval()/no_grad -> exp(log-softmax) @ bin_centers.
DBN   : reproduces Slicer.py + Model_Test.py -> data*(185/p97) -> axial slices 45..124 ->
        PIL float->RGB (trunc/clip 0..255) -> JPEG round-trip (default quality) ->
        Keras-default resize to 256x256 (nearest) -> /255 -> model.predict -> median over 80 slices.

Only subjects whose QC record is PASS are run. FAIL / missing subjects are written to
<model>_skipped_qc.csv and printed -- never silently dropped.

Usage:
  python phase2_corrected/infer.py --model sfcn --subset pilot
  python phase2_corrected/infer.py --model dbn  --subset all
  python phase2_corrected/infer.py --model sfcn --golden      # DBN's shipped sample volumes (known ages)
"""
import argparse
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


# ----------------------------------------------------------------------------- SFCN
def load_sfcn(device=None):
    import torch
    sys.path.insert(0, str(C.EXTERNAL / "UKBiobank_deep_pretrain"))
    from dp_model.model_files.sfcn import SFCN
    from dp_model import dp_utils as dpu
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(SFCN())
    sd = torch.load(str(C.EXTERNAL / "UKBiobank_deep_pretrain" / "brain_age" / "run_20190719_00_epoch_best_mae.p"),
                    map_location="cpu")
    res = model.load_state_dict(sd)   # strict=True: raises on any missing/unexpected key
    assert not res.missing_keys and not res.unexpected_keys, res
    model.to(device).eval()
    _, bc = dpu.num2vect(np.array([50.0]), C.SFCN_BIN_RANGE, C.SFCN_BIN_STEP, C.SFCN_SIGMA)
    info = dict(load_result=str(res), eval_mode=not model.training, device=device,
                n_params=sum(p.numel() for p in model.parameters()), bin_centers=f"{bc[0]}..{bc[-1]} (n={len(bc)})")
    return model, bc, dpu, device, info


def sfcn_predict(ctx, vol):
    import torch
    model, bc, dpu, device, _ = ctx
    data = vol / vol.mean()                                   # reference: data = data/data.mean()
    data = dpu.crop_center(data, C.SFCN_CROP)                 # reference: crop_center(data,(160,192,160))
    x = torch.tensor(data.reshape((1, 1) + data.shape), dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(x)
    prob = np.exp(out[0].cpu().numpy().reshape(-1))
    pred = float(prob @ bc)                                   # reference: pred = prob@bc
    log = dict(in_shape="x".join(map(str, tuple(x.shape))), in_dtype=str(x.dtype), in_min=float(data.min()),
               in_max=float(data.max()), in_mean=float(data.mean()), in_std=float(data.std()),
               prob_sum=float(prob.sum()), prob_argmax_age=float(bc[prob.argmax()]))
    return pred, log


# ----------------------------------------------------------------------------- DeepBrainNet
def dbn_input(vol):
    """vol: 182x218x182 float array in the DBN sample convention (LPS). -> (80,256,256,3) float32 in [0,1]."""
    p97 = np.percentile(vol, 97)
    if not np.isfinite(p97) or p97 <= 0:
        raise ValueError(f"invalid 97th percentile {p97}")
    data = vol * (C.DBN_P97_TARGET / p97)
    slices = []
    for sl in range(C.DBN_N_SLICES):
        clipped = data[:, :, C.DBN_SLICE_START + sl]
        img = Image.fromarray(clipped).convert("RGB")            # Slicer.py (F -> RGB: truncate, clip 0..255)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")                              # Slicer.py saves .jpg (PIL default quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")                      # keras load_img(color_mode='rgb')
        img = img.resize((C.DBN_IMG_SIZE[1], C.DBN_IMG_SIZE[0]), Image.NEAREST)   # target_size=(256,256), nearest
        slices.append(np.asarray(img, dtype=np.float32) * (1.0 / 255))            # rescale=1./255
    return np.stack(slices), float(p97)


def load_dbn():
    import tf_keras
    w = C.DBN_WEIGHTS
    if not w.exists() or w.stat().st_size != C.DBN_WEIGHTS_SIZE:
        raise SystemExit(f"DeepBrainNet weights missing or wrong size at {w} "
                         f"(need {C.DBN_WEIGHTS_SIZE} bytes; it is a Git-LFS file).")
    h = hashlib.sha256(w.read_bytes()).hexdigest()
    if h != C.DBN_WEIGHTS_SHA256:
        raise SystemExit(f"DBN_model.h5 sha256 {h} != LFS pointer oid {C.DBN_WEIGHTS_SHA256}")
    model = tf_keras.models.load_model(str(w), compile=False)
    info = dict(sha256_ok=True, input_shape=str(model.input_shape), output_shape=str(model.output_shape),
                n_params=model.count_params(), n_layers=len(model.layers), last_layer=model.layers[-1].name)
    return model, info


def dbn_predict(model, vol):
    batch, p97 = dbn_input(vol)
    preds = model.predict(batch, batch_size=C.DBN_N_SLICES, verbose=0).reshape(-1)   # keras predict = inference mode
    assert preds.shape == (C.DBN_N_SLICES,)
    log = dict(in_shape="x".join(map(str, batch.shape)), in_dtype=str(batch.dtype), in_min=float(batch.min()),
               in_max=float(batch.max()), in_mean=float(batch.mean()), in_std=float(batch.std()), p97_raw=p97,
               slice_pred_min=float(preds.min()), slice_pred_max=float(preds.max()), slice_pred_std=float(preds.std()))
    return float(np.median(preds)), log


# ----------------------------------------------------------------------------- driver
def split_map():
    m = {}
    for s in ("train", "val", "test"):
        for i in pd.read_csv(C.PROJECT_DIR / f"ixi_{s}.csv").IXI_ID:
            m[int(i)] = s
    return m


def run(args):
    import nibabel as nib
    out_root = Path(args.out)
    vol_dir = out_root / ("sfcn" if args.model == "sfcn" else "deepbrainnet")
    ctx = load_sfcn() if args.model == "sfcn" else load_dbn()
    print("Checkpoint:", ctx[-1] if args.model == "sfcn" else ctx[1], flush=True)

    rows, skipped = [], []
    if args.golden:
        ages = pd.read_csv(C.EXTERNAL / "DeepBrainNet" / "Sample_Data.csv").set_index("ID").Age
        items = []
        for sid, age in ages.items():
            img, arr = C.load_dir_volume(C.EXTERNAL / "DeepBrainNet" / "Data" / f"{sid}_T1_BrainAligned.nii.gz")
            if args.model == "sfcn":
                arr = arr[:, ::-1, :]      # sample is LPS -> FSL LAS grid (exact flip of y)
            items.append((sid, float(age), arr))
        tag = "golden"
    else:
        meta = pd.read_csv(C.METADATA_PATH)
        qc = pd.read_csv(out_root / "qc" / "qc_summary.csv")
        if args.subset == "pilot":
            want = pd.read_csv(out_root / "qc" / "pilot_ids.csv").IXI_ID.astype(int).tolist()
        elif args.subset == "all":
            want = meta.IXI_ID.astype(int).tolist()
        else:
            want = args.ids
        smap, items = split_map(), []
        for ixi in want:
            r = qc[qc.IXI_ID == ixi]
            f = vol_dir / f"IXI{ixi:03d}.nii.gz"
            if r.empty or r.iloc[0].status not in ("PASS", "WARN") or not f.exists():
                why = "no QC record" if r.empty else (r.iloc[0].fail_reasons if r.iloc[0].status == "FAIL" else "volume missing")
                skipped.append(dict(IXI_ID=ixi, reason=why))
                print(f"!!! SKIPPED IXI{ixi:03d}: {why}", flush=True)
                continue
            m = meta[meta.IXI_ID == ixi].iloc[0]
            _, arr = C.load_dir_volume(f)
            items.append((ixi, float(m.AGE), arr, dict(split=smap.get(ixi), site=m.site, sex=m.sex_label, qc_status=r.iloc[0].status)))
        tag = args.subset if args.subset != "ids" else "ids"

    for k, it in enumerate(items, 1):
        sid, age, arr = it[0], it[1], it[2]
        extra = it[3] if len(it) > 3 else {}
        pred, log = sfcn_predict(ctx, arr) if args.model == "sfcn" else dbn_predict(ctx[0], arr)
        rows.append(dict(IXI_ID=sid, **extra, actual_age=age, predicted_age=pred, **log))
        print(f"[{k}/{len(items)}] {sid}: actual={age:.1f} pred={pred:.1f}  input {log['in_shape']} "
              f"[{log['in_min']:.3g}, {log['in_max']:.3g}] mean={log['in_mean']:.3g}", flush=True)

    (out_root / "inference").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "inference" / f"{args.model}_predictions_{tag}.csv", index=False)
    sk = pd.DataFrame(skipped, columns=["IXI_ID", "reason"])
    sk.to_csv(out_root / "inference" / f"{args.model}_skipped_qc_{tag}.csv", index=False)
    if not args.golden:
        assert len(df) + len(sk) == len(want), "subjects unaccounted for"
        print(f"\nPredicted {len(df)} | skipped by QC {len(sk)} | requested {len(want)}")
    if len(df) > 1:
        e = df.predicted_age - df.actual_age
        from scipy import stats
        print(f"MAE={e.abs().mean():.2f}  Pearson r={np.corrcoef(df.actual_age, df.predicted_age)[0, 1]:.3f}  "
              f"Spearman={stats.spearmanr(df.actual_age, df.predicted_age)[0]:.3f}  "
              f"slope={stats.linregress(df.actual_age, df.predicted_age).slope:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["sfcn", "dbn"], required=True)
    ap.add_argument("--subset", choices=["pilot", "all", "ids"], default="pilot")
    ap.add_argument("--ids", type=int, nargs="+")
    ap.add_argument("--golden", action="store_true", help="run on DeepBrainNet's shipped sample volumes")
    ap.add_argument("--out", default=str(C.OUT_ROOT))
    run(ap.parse_args())
