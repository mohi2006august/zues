"""M4: U-Net super-resolution downscaler (optional; needs `pip install torch`).

The whole region grid is one image. Input channels per day:
  - interpolated block field for each modelled variable (8)
  - block value of each cell's own block for each variable (8)
  - static layers: terrain, land cover, distance to water, lat/lon (16)
  - season: day-of-year sin/cos (2)
Output: the residual (fine - interpolated) for each modelled variable, in
standardised units (rain in log1p space). Loss is masked to active cells.

Unlike M3, the U-Net sees spatial context directly (neighbouring cells), which lets
it learn patterns such as rain shadows. It is slower to train and needs more data;
it is offered as an experimental alternative and must beat M3 on validation to be
worth using.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.dataset import STATIC_FEATURES, RegionContext, split_mask
from ..store import RegionStore
from ..variables import MODELLED, MODELLED_VARS
from .base import Downscaler, Prediction, VarPrediction

try:
    import torch
    from torch import nn

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


def _pad_to(n: int, k: int = 8) -> int:
    return int(np.ceil(n / k) * k)


if HAVE_TORCH:

    class _Block(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self, cin: int, cout: int, width: int = 32):
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = _Block(cin, w), _Block(w, 2 * w), _Block(2 * w, 4 * w)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, stride=2)
            self.d2 = _Block(4 * w, 2 * w)
            self.u1 = nn.ConvTranspose2d(2 * w, w, 2, stride=2)
            self.d1 = _Block(2 * w, w)
            self.head = nn.Conv2d(w, cout, 1)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
            d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
            return self.head(d1)


class _Layout:
    """Scatter active-cell vectors onto a padded (H, W) image and back."""

    def __init__(self, ctx: RegionContext):
        g = ctx.grid
        self.ny, self.nx = g.ny, g.nx
        self.H, self.W = _pad_to(g.ny), _pad_to(g.nx)
        flat = ctx.weights.active_cells
        self.rows, self.cols = flat // g.nx, flat % g.nx
        self.mask = np.zeros((self.H, self.W), np.float32)
        self.mask[self.rows, self.cols] = 1

    def to_img(self, a: np.ndarray) -> np.ndarray:
        """(..., n_active) -> (..., H, W)"""
        out = np.zeros((*a.shape[:-1], self.H, self.W), np.float32)
        out[..., self.rows, self.cols] = a
        return out

    def from_img(self, img: np.ndarray) -> np.ndarray:
        return img[..., self.rows, self.cols]


def _encode(var: str, fine_or_pred, interp):
    """Residual target in model space (log1p for rain)."""
    if MODELLED[var].kind == "rain":
        return np.log1p(np.maximum(fine_or_pred, 0)) - np.log1p(np.maximum(interp, 0))
    return fine_or_pred - interp


def _decode(var: str, residual, interp):
    if MODELLED[var].kind == "rain":
        return np.maximum(np.expm1(np.log1p(np.maximum(interp, 0)) + residual), 0)
    return interp + residual


class UNetModel(Downscaler):
    model_id = "M4"
    name = "U-Net (deep learning)"

    def __init__(self, net, stats: dict, ctx_static: np.ndarray, layout: _Layout):
        self.net = net.eval()
        self.stats = stats
        self.static = ctx_static  # (S, H, W) standardised
        self.layout = layout

    @staticmethod
    def dynamic(ctx, blk, stats) -> np.ndarray:
        """Standardised block-derived channels on active cells: (D, 16, n_active)."""
        chans = []
        for v in MODELLED_VARS:
            mu, sd = stats["in"][v]
            chans.append((ctx.interp(blk[v]) - mu) / sd)
            chans.append((ctx.copy(blk[v]) - mu) / sd)
        return np.stack(chans, axis=1).astype(np.float32)

    @staticmethod
    def assemble(dyn: np.ndarray, doy: np.ndarray, static: np.ndarray, layout: _Layout) -> np.ndarray:
        """Batch of network inputs (B, C, H, W) from active-cell channels."""
        B = len(dyn)
        season = np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1)
        season = np.broadcast_to(season[:, :, None, None], (B, 2, layout.H, layout.W)) * layout.mask
        stat = np.broadcast_to(static[None], (B, *static.shape))
        mask = np.broadcast_to(layout.mask[None, None], (B, 1, layout.H, layout.W))
        return np.concatenate([layout.to_img(dyn), stat, season, mask], axis=1).astype(np.float32)

    def predict(self, ctx: RegionContext, blk, dates) -> Prediction:
        dyn = self.dynamic(ctx, blk, self.stats)
        doy = dates.dayofyear.to_numpy()
        with torch.no_grad():
            y = np.concatenate([
                self.net(torch.from_numpy(self.assemble(dyn[i:i + 32], doy[i:i + 32], self.static, self.layout))).numpy()
                for i in range(0, len(dyn), 32)
            ])
        out: Prediction = {}
        for k, v in enumerate(MODELLED_VARS):
            mu, sd = self.stats["out"][v]
            res = self.layout.from_img(y[:, k]) * sd + mu
            out[v] = VarPrediction(_decode(v, res, ctx.interp(blk[v])))
        return out


def _static_stack(ctx: RegionContext, layout: _Layout) -> np.ndarray:
    chans = []
    for name in STATIC_FEATURES:
        a = ctx.static[name]
        chans.append((a - a.mean()) / (a.std() + 1e-6))
    return layout.to_img(np.stack(chans))


def train_unet(ctx: RegionContext, store: RegionStore, blk: dict[str, np.ndarray],
               dates: pd.DatetimeIndex, out_dir: Path, epochs: int = 25, batch: int = 16,
               seed: int = 7, log=print) -> dict:
    if not HAVE_TORCH:
        raise RuntimeError("M4 needs PyTorch: pip install torch")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    layout = _Layout(ctx)
    splits = ctx.meta["splits"]
    tr = np.where(split_mask(dates, tuple(splits["train"])))[0]
    va = np.where(split_mask(dates, tuple(splits["val"])))[0]

    stats = {"in": {}, "out": {}}
    targets = []
    for v in MODELLED_VARS:
        _, fine = store.load_fine_active(v, ctx.weights)
        interp = ctx.interp(blk[v])
        res = _encode(v, np.nan_to_num(fine, nan=0.0), interp)
        stats["in"][v] = (float(np.mean(blk[v][tr])), float(np.std(blk[v][tr]) + 1e-6))
        stats["out"][v] = (float(res[tr].mean()), float(res[tr].std() + 1e-6))
        targets.append(((res - stats["out"][v][0]) / stats["out"][v][1]).astype(np.float32))
        del fine
    tgt = np.stack(targets, axis=1)  # (T, 8, n_active): kept compact, images built per batch
    static = _static_stack(ctx, layout)
    dyn = UNetModel.dynamic(ctx, blk, stats)
    doy = dates.dayofyear.to_numpy()
    cin = dyn.shape[1] + static.shape[0] + 3

    net = UNet(cin, len(MODELLED_VARS))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mask = torch.from_numpy(layout.mask)[None, None]

    def loss_on(idx):
        xb = torch.from_numpy(UNetModel.assemble(dyn[idx], doy[idx], static, layout))
        yb = torch.from_numpy(layout.to_img(tgt[idx]))
        return (((net(xb) - yb) ** 2) * mask).sum() / (mask.sum() * len(idx) * yb.shape[1])

    best, best_state, history = float("inf"), None, []
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(tr)
        tl = []
        for i in range(0, len(perm), batch):
            opt.zero_grad()
            loss = loss_on(perm[i:i + batch])
            loss.backward()
            opt.step()
            tl.append(loss.item())
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(np.mean([float(loss_on(va[i:i + 64])) for i in range(0, len(va), 64)])) if len(va) else np.mean(tl)
        history.append((float(np.mean(tl)), vl))
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
        log(f"[M4] epoch {ep + 1}/{epochs} train {np.mean(tl):.4f} val {vl:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "unet.pt")
    meta = {"stats": stats, "cin": int(cin), "epochs": epochs, "best_val_loss": best,
            "history": history, "train_seconds": round(time.time() - t0, 1)}
    (out_dir / "unet.json").write_text(json.dumps(meta))
    return meta


def load_unet(ctx: RegionContext, d: Path) -> UNetModel | None:
    if not HAVE_TORCH or not (d / "unet.pt").exists():
        return None
    meta = json.loads((d / "unet.json").read_text())
    layout = _Layout(ctx)
    net = UNet(meta["cin"], len(MODELLED_VARS))
    net.load_state_dict(torch.load(d / "unet.pt", weights_only=True))
    stats = {k: {v: tuple(x) for v, x in d_.items()} for k, d_ in meta["stats"].items()}
    return UNetModel(net, stats, _static_stack(ctx, layout), layout)
