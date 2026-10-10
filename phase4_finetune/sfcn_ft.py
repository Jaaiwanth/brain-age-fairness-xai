"""Transfer-learning utilities for the pretrained UK Biobank SFCN (Peng et al. 2021).

Design (see README.md for the reasoning):
  * Keep the pretrained network as is. Its output is already a differentiable age estimate:
        age = sum_i softmax(logits)_i * bin_centre_i      (40 one-year bins, 42-82 in the original)
  * The only change: the last 1x1x1 conv (64 -> 40) is extended to 70 bins covering 20-90 so IXI's young subjects are
    reachable. Overlapping bins copy the pretrained weights; new bins start with a large negative bias so the starting
    model behaves like the pretrained one.
  * Freezing stages: head (classifier conv only) -> partial (+ last two feature blocks) -> full.
    BatchNorm running statistics stay frozen (batch size is tiny on a 6 GB GPU).
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "UKBiobank_deep_pretrain"))
from dp_model.model_files.sfcn import SFCN  # noqa: E402

PRETRAINED = REPO / "external" / "UKBiobank_deep_pretrain" / "brain_age" / "run_20190719_00_epoch_best_mae.p"
CACHE = REPO / "phase4_finetune" / "cache"
PRE_START, PRE_BINS = 42, 40                 # pretrained label bins: 42..82
NEW_START, NEW_END = 20, 90                  # extended label bins: 20..90 (IXI is 20.0-86.3)
NEW_BINS = NEW_END - NEW_START
OFFSET = PRE_START - NEW_START               # index of age 42 in the new bins (22)
CENTERS = NEW_START + 0.5 + np.arange(NEW_BINS)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pretrained40():
    """Original 40-bin SFCN with the released weights (strict load, DataParallel prefix stripped)."""
    m = SFCN(output_dim=PRE_BINS)
    sd = torch.load(str(PRETRAINED), map_location="cpu")
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    res = m.load_state_dict(sd)
    assert not res.missing_keys and not res.unexpected_keys, res
    return m


def extend_head(m40, new_bin_penalty=6.0):
    """70-bin SFCN initialised from the 40-bin one. Returns the new model."""
    m = SFCN(output_dim=NEW_BINS)
    sd = {k: v.clone() for k, v in m40.state_dict().items()}
    wk, bk = "classifier.conv_6.weight", "classifier.conv_6.bias"
    w40, b40 = sd.pop(wk), sd.pop(bk)
    w = torch.zeros(NEW_BINS, *w40.shape[1:])
    b = torch.zeros(NEW_BINS)
    for i in range(NEW_BINS):
        j = min(max(i - OFFSET, 0), PRE_BINS - 1)          # nearest pretrained bin (edge replicate outside 42..82)
        w[i], b[i] = w40[j], b40[j]
        if not 0 <= i - OFFSET < PRE_BINS:
            b[i] -= new_bin_penalty                          # new bins start (almost) switched off
    res = m.load_state_dict({**sd, wk: w, bk: b}, strict=True)
    assert not res.missing_keys and not res.unexpected_keys
    return m


class AgeModel(nn.Module):
    """SFCN wrapper returning (log-probabilities over bins, expected age)."""

    def __init__(self, sfcn, centers=CENTERS):
        super().__init__()
        self.net = sfcn
        self.register_buffer("centers", torch.tensor(centers, dtype=torch.float32))

    def forward(self, x):
        logp = self.net(x)[0].flatten(1).float()
        age = (logp.exp() * self.centers).sum(1)
        return logp, age


STAGES = {"head": ["net.classifier.conv_6"],
          "partial": ["net.classifier.conv_6", "net.feature_extractor.conv_4", "net.feature_extractor.conv_5"],
          "full": [""]}


def set_stage(model, stage):
    """Freeze everything except the stage's modules. Returns (n_trainable, n_total)."""
    prefixes = STAGES[stage]
    for n, p in model.named_parameters():
        p.requires_grad = any(n.startswith(pre) for pre in prefixes)
    return (sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters()))


def train_mode(model):
    """train() but BatchNorm statistics stay frozen; dropout stays active."""
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def soft_labels(age, sigma=1.0):
    """Gaussian soft labels over the extended bins (SFCN's original training target)."""
    c = torch.tensor(CENTERS, dtype=torch.float32, device=age.device)
    lo, hi = c - 0.5, c + 0.5
    n = torch.distributions.Normal(age[:, None], sigma)
    p = n.cdf(hi[None]) - n.cdf(lo[None])
    return p / p.sum(1, keepdim=True).clamp_min(1e-12)


def compute_loss(kind, logp, age_pred, age, delta=3.0, kl_weight=0.0):
    if kind == "l1":
        main = F.l1_loss(age_pred, age)
    elif kind == "huber":
        main = F.huber_loss(age_pred, age, delta=delta)
    elif kind == "kl":
        main = F.kl_div(logp, soft_labels(age), reduction="batchmean")
    else:
        raise ValueError(kind)
    if kl_weight > 0 and kind != "kl":
        main = main + kl_weight * F.kl_div(logp, soft_labels(age), reduction="batchmean")
    return main


class CachedData(torch.utils.data.Dataset):
    """Volumes cached by build_cache.py: (168,200,168). Train: random +-4 voxel crop jitter + random left-right flip."""

    def __init__(self, split, train, ids=None, limit=None):
        idx = pd.read_csv(CACHE / "index.csv")
        idx = idx[idx.split == split] if ids is None else idx[idx.IXI_ID.isin(ids)]
        idx = idx.sort_values("IXI_ID").reset_index(drop=True)
        if limit:
            idx = idx.sample(min(limit, len(idx)), random_state=0).reset_index(drop=True)
        self.idx, self.train = idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        r = self.idx.iloc[i]
        a = np.load(CACHE / f"IXI{int(r.IXI_ID):03d}.npy")
        if self.train:
            o = np.random.randint(0, 9, 3)
            a = a[o[0]:o[0] + 160, o[1]:o[1] + 192, o[2]:o[2] + 160]
            if np.random.rand() < 0.5:
                a = a[::-1]
        else:
            a = a[4:164, 4:196, 4:164]
        return torch.from_numpy(np.ascontiguousarray(a))[None], torch.tensor(float(r.AGE), dtype=torch.float32), int(r.IXI_ID)


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
def evaluate(model, loader, device):
    model.eval()
    ids, ages, preds = [], [], []
    for x, y, i in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            _, p = model(x.to(device, non_blocking=True))
        ids += i.tolist(); ages += y.tolist(); preds += p.float().cpu().tolist()
    df = pd.DataFrame(dict(IXI_ID=ids, actual_age=ages, predicted_age=preds))
    return metrics(df.actual_age, df.predicted_age), df
