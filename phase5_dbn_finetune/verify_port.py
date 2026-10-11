"""Check that the PyTorch port reproduces the released Keras DeepBrainNet.

1. weights: every converted tensor equals the HDF5 value (shape and layout)
2. random inputs and real cached slices: Keras vs PyTorch outputs (fp32), plus the fp16-autocast deviation used in training
3. subject level: median-of-80 PyTorch predictions vs the reference Keras predictions already saved in dbn_predictions_all.csv
python phase5_dbn_finetune/verify_port.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dbn_ft as D  # noqa: E402

R = []


def check(name, ok, detail=""):
    R.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


def main():
    import tf_keras
    check("0. released weights file size and SHA256", D.WEIGHTS.stat().st_size == D.C.DBN_WEIGHTS_SIZE and D.sha256(D.WEIGHTS) == D.C.DBN_WEIGHTS_SHA256, "matches the Git-LFS pointer")
    km = tf_keras.models.load_model(str(D.WEIGHTS), compile=False)
    tm = D.load_pretrained().eval()
    n_k, n_t = km.count_params(), sum(p.numel() for p in tm.parameters())
    check("1. parameter count", n_k == n_t, f"Keras {n_k:,} vs PyTorch {n_t:,}")
    # 1. tensor-by-tensor
    vgg = km.get_layer("vgg16")
    bad = []
    for name, mod in tm.features.named_children():
        if isinstance(mod, torch.nn.Conv2d):
            k, b = vgg.get_layer(name).get_weights()
            bad += [] if (np.array_equal(k.transpose(3, 2, 0, 1), mod.weight.detach().numpy()) and np.array_equal(b, mod.bias.detach().numpy())) else [name]
    for tn, kn in (("fc1", "dense_3"), ("fc2", "dense_4")):
        k, b = km.get_layer(kn).get_weights()
        lin = getattr(tm.head, tn)
        bad += [] if (np.array_equal(k.T, lin.weight.detach().numpy()) and np.array_equal(b, lin.bias.detach().numpy())) else [kn]
    check("1b. every conv/dense tensor identical (layout converted)", not bad, f"mismatching layers: {bad}")
    # 2. outputs
    rng = np.random.default_rng(0)
    x = rng.random((8, 256, 256, 1), dtype=np.float32)
    yk = km.predict(np.repeat(x, 3, -1), verbose=0).reshape(-1)
    with torch.no_grad():
        yt = tm(torch.from_numpy(x.transpose(0, 3, 1, 2))).numpy()
    d = np.abs(yk - yt)
    check("2a. random inputs: Keras vs PyTorch (fp32)", d.max() < 1e-3 * max(1, np.abs(yk).max()), f"max |diff| {d.max():.2e} on outputs of magnitude {np.abs(yk).mean():.2f}")
    idx = D.index()
    ids = idx[idx.split == "val"].IXI_ID.head(2).tolist() + idx[idx.split == "train"].IXI_ID.head(1).tolist()
    sl = np.concatenate([np.load(D.CACHE / f"IXI{i:03d}.npy") for i in ids]).astype(np.float32) / 255.0      # (240,256,256)
    yk = km.predict(np.repeat(sl[..., None], 3, -1), batch_size=40, verbose=0).reshape(-1)
    with torch.no_grad():
        yt = torch.cat([tm(torch.from_numpy(sl[i:i + 40, None])) for i in range(0, len(sl), 40)]).numpy()
    d = np.abs(yk - yt)
    check("2b. 240 real slices: Keras vs PyTorch (fp32)", d.max() < 0.01, f"max |diff| {d.max():.2e} y, mean {d.mean():.2e} y (predictions are in years)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmg = tm.to(dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        yh = torch.cat([tmg(torch.from_numpy(sl[i:i + 40, None]).to(dev)).float().cpu() for i in range(0, len(sl), 40)]).numpy()
    dh = np.abs(yk - yh)
    check("2c. fp16 autocast (used in training) vs Keras fp32", dh.max() < 0.25, f"max |diff| {dh.max():.3f} y, mean {dh.mean():.3f} y")
    # 3. subject level against the reference predictions
    ref = pd.read_csv(D.C.OUT_ROOT / "inference" / "dbn_predictions_all.csv").set_index("IXI_ID")
    pick = idx.groupby("split").head(8).IXI_ID.tolist()
    ds = D.SubjectData("train", ids=pick)
    ds.idx = D.index().set_index("IXI_ID").loc[pick].reset_index()
    ld = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    m, df = D.evaluate(tmg, ld, dev)
    diff = (df.set_index("IXI_ID").predicted_age - ref.loc[df.IXI_ID, "predicted_age"]).abs()
    check("3. subject-level median predictions vs the saved reference (Keras) predictions", diff.max() < 0.25, f"{len(df)} subjects from all three splits: max |diff| {diff.max():.3f} y, mean {diff.mean():.3f} y")
    print("\n" + ("PORT VERIFIED" if all(R) else "PORT CHECK FAILED"))
    sys.exit(0 if all(R) else 1)


if __name__ == "__main__":
    main()
