"""Transfer-learning utilities for the pretrained DeepBrainNet (Bashyam et al., 2020).

Architecture (read from the released DBN_model.h5):
    VGG16 conv base (13 convs, 5 max-pools) -> GlobalAveragePooling2D -> Dense(1024, ReLU) -> Dropout(0.9) -> Dense(1, linear)
The output is already a scalar age in years, so the head is NOT replaced. The network is re-implemented in PyTorch (so it can train on the GPU;
TensorFlow on native Windows is CPU-only), loaded directly from the released HDF5 file, and checked numerically against the Keras model.

Training signal: every axial slice (80 per subject, reference chain: p97*185 scaling, slices 45-124, PIL float->RGB, JPEG round trip, 256x256 nearest)
is labelled with the subject's age, as in the original DeepBrainNet training. A subject's prediction is the MEDIAN over its 80 slice predictions (reference).
"""
import hashlib
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "phase2_corrected"))
import common as C  # noqa: E402

WEIGHTS = C.DBN_WEIGHTS
CACHE = REPO / "phase5_dbn_finetune" / "cache"
N_SLICES = C.DBN_N_SLICES        # 80
IMG = 256
VGG_CFG = [(1, [(3, 64), (64, 64)]), (2, [(64, 128), (128, 128)]), (3, [(128, 256), (256, 256), (256, 256)]),
           (4, [(256, 512), (512, 512), (512, 512)]), (5, [(512, 512), (512, 512), (512, 512)])]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class DeepBrainNet(nn.Module):
    """PyTorch twin of the released Keras model. Input: (B, 1, 256, 256) in [0,1] (the 3 identical RGB channels are expanded inside)."""

    def __init__(self, dropout=0.9):
        super().__init__()
        layers = OrderedDict()
        for b, convs in VGG_CFG:
            for i, (cin, cout) in enumerate(convs, 1):
                layers[f"block{b}_conv{i}"] = nn.Conv2d(cin, cout, 3, padding=1)
                layers[f"block{b}_relu{i}"] = nn.ReLU(inplace=True)
            layers[f"block{b}_pool"] = nn.MaxPool2d(2, 2)
        self.features = nn.Sequential(layers)
        self.head = nn.Sequential(OrderedDict(fc1=nn.Linear(512, 1024), relu=nn.ReLU(inplace=True), dropout=nn.Dropout(dropout), fc2=nn.Linear(1024, 1)))

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        x = self.features(x)
        x = x.mean(dim=(2, 3))                       # GlobalAveragePooling2D
        return self.head(x).squeeze(1)


def load_pretrained(path=WEIGHTS, dropout=0.9):
    """Read the released weights straight from the HDF5 file (kernels (H,W,Cin,Cout) -> (Cout,Cin,H,W); dense (in,out) -> (out,in))."""
    m = DeepBrainNet(dropout)
    with h5py.File(str(path), "r") as f:
        g = f["model_weights"]
        for name, mod in m.features.named_children():
            if isinstance(mod, nn.Conv2d):
                k, b = g["vgg16"][f"{name}_1/kernel:0"][()], g["vgg16"][f"{name}_1/bias:0"][()]
                mod.weight.data = torch.from_numpy(np.ascontiguousarray(k.transpose(3, 2, 0, 1)))
                mod.bias.data = torch.from_numpy(b)
        for torch_name, keras_name in (("fc1", "dense_3"), ("fc2", "dense_4")):
            k, b = g[keras_name][f"{keras_name}_1/kernel:0"][()], g[keras_name][f"{keras_name}_1/bias:0"][()]
            lin = getattr(m.head, torch_name)
            lin.weight.data = torch.from_numpy(np.ascontiguousarray(k.T))
            lin.bias.data = torch.from_numpy(b)
    return m


# ---------------------------------------------------------------------------------------------- stages
STAGES = {"head": ["head."], "last1": ["head.", "features.block5"], "last2": ["head.", "features.block4", "features.block5"], "full": [""]}


def set_stage(model, stage):
    pre = STAGES[stage]
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(x) for x in pre)
    return sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters())


def compute_loss(kind, pred, age, delta=3.0):
    if kind == "huber":
        return F.huber_loss(pred, age, delta=delta)
    if kind == "l1":
        return F.l1_loss(pred, age)
    if kind == "mse":
        return F.mse_loss(pred, age)
    raise ValueError(kind)


# ---------------------------------------------------------------------------------------------- data
def index():
    return pd.read_csv(CACHE / "index.csv")


class SliceData(torch.utils.data.Dataset):
    """Training: individual (subject, slice) images with light augmentation (left-right flip, +-8 px shift, +-5% gain)."""

    def __init__(self, split, train=True, ids=None, limit=None, slices_per_subject=None):
        idx = index()
        idx = idx[idx.split == split] if ids is None else idx[idx.IXI_ID.isin(ids)]
        idx = idx.sort_values("IXI_ID").reset_index(drop=True)
        if limit:
            idx = idx.sample(min(limit, len(idx)), random_state=0).sort_values("IXI_ID").reset_index(drop=True)
        self.idx, self.train, self._mm = idx, train, {}
        ks = np.arange(N_SLICES) if slices_per_subject is None else np.linspace(0, N_SLICES - 1, slices_per_subject).astype(int)
        self.pairs = [(i, int(k)) for i in range(len(idx)) for k in ks]

    def _arr(self, ixi):
        if ixi not in self._mm:
            self._mm[ixi] = np.load(CACHE / f"IXI{ixi:03d}.npy", mmap_mode="r")
        return self._mm[ixi]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, j):
        i, k = self.pairs[j]
        r = self.idx.iloc[i]
        a = np.array(self._arr(int(r.IXI_ID))[k], dtype=np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:
                a = a[::-1]                                           # rows = left-right axis: label-preserving mirror
            dy, dx = np.random.randint(-8, 9, 2)
            a = np.roll(a, (dy, dx), (0, 1))
            a = np.clip(a * np.random.uniform(0.95, 1.05), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(a))[None], torch.tensor(float(r.AGE), dtype=torch.float32)


class SubjectData(torch.utils.data.Dataset):
    """Evaluation: all 80 slices of one subject, no augmentation."""

    def __init__(self, split, ids=None, limit=None):
        idx = index()
        idx = idx[idx.split == split] if ids is None else idx[idx.IXI_ID.isin(ids)]
        idx = idx.sort_values("IXI_ID").reset_index(drop=True)
        if limit:
            idx = idx.sample(min(limit, len(idx)), random_state=0).sort_values("IXI_ID").reset_index(drop=True)
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx.iloc[i]
        a = np.load(CACHE / f"IXI{int(r.IXI_ID):03d}.npy").astype(np.float32) / 255.0
        return torch.from_numpy(a)[:, None], torch.tensor(float(r.AGE), dtype=torch.float32), int(r.IXI_ID)


def metrics(age, pred):
    age, pred = np.asarray(age, float), np.asarray(pred, float)
    e = pred - age
    out = dict(n=len(age), MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()))
    if len(age) > 2 and np.std(pred) > 0:
        out["r"] = float(np.corrcoef(age, pred)[0, 1])
        out["slope"] = float(np.polyfit(age, pred, 1)[0])
    else:
        out["r"] = out["slope"] = float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device, return_slices=False, amp=False):
    """Subject-level evaluation exactly as the reference: median over the 80 slice predictions (mean also reported). fp32 by default (amp=True only for speed tests)."""
    model.eval()
    ids, ages, med, mean, sl = [], [], [], [], []
    for x, y, i in loader:
        b = x.shape[0]
        x = x.flatten(0, 1).to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            p = torch.cat([model(x[c:c + 40]).float() for c in range(0, len(x), 40)]).view(b, N_SLICES)      # 40 images per forward: keeps peak VRAM ~2 GB (160 at once spills past 6 GB and runs 4x slower)
        ids += i.tolist(); ages += y.tolist()
        med += torch.quantile(p, 0.5, dim=1).cpu().tolist(); mean += p.mean(dim=1).cpu().tolist()      # np.median semantics (mean of the two middle values for 80 slices); torch.median returns the lower one
        if return_slices:
            sl.append(p.cpu().numpy())
    df = pd.DataFrame(dict(IXI_ID=ids, actual_age=ages, predicted_age=med, predicted_mean=mean))
    m = metrics(df.actual_age, df.predicted_age)
    m["MAE_meanagg"] = float(np.abs(df.predicted_mean - df.actual_age).mean())
    return (m, df, np.concatenate(sl)) if return_slices else (m, df)
