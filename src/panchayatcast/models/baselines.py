"""Baseline downscalers.

M0 Copy           every panchayat gets its block's value (current practice)
M1 Interpolation  inverse-distance interpolation of block values
M2 Lapse rate     M1 + elevation correction for temperature, rates fitted per month
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.dataset import RegionContext
from ..variables import MODELLED, MODELLED_VARS
from .base import Downscaler, Prediction, VarPrediction

DEFAULT_LAPSE_C_PER_M = 0.0065


class CopyModel(Downscaler):
    model_id = "M0"
    name = "Copy block value"

    def predict(self, ctx, blk, dates) -> Prediction:
        return {v: VarPrediction(ctx.copy(blk[v])) for v in MODELLED_VARS}


class IDWModel(Downscaler):
    model_id = "M1"
    name = "Interpolation (IDW)"

    def predict(self, ctx, blk, dates) -> Prediction:
        return {v: VarPrediction(ctx.interp(blk[v])) for v in MODELLED_VARS}


class LapseRateModel(Downscaler):
    model_id = "M2"
    name = "Interpolation + lapse rate"

    def __init__(self, gammas: dict[str, np.ndarray] | None = None):
        # gammas[var]: (12,) °C per metre, one per calendar month
        self.gammas = gammas or {}

    def predict(self, ctx, blk, dates) -> Prediction:
        out = IDWModel().predict(ctx, blk, dates)
        dz = ctx.interp_elev - ctx.static["elevation"]  # (n_active,) metres
        months = dates.month.to_numpy() - 1
        for v, spec in MODELLED.items():
            if not spec.lapse:
                continue
            g = self.gammas.get(v, np.full(12, DEFAULT_LAPSE_C_PER_M))
            out[v].value = out[v].value + g[months][:, None] * dz[None, :]
        return out


def fit_lapse_rates(
    ctx: RegionContext, fine: np.ndarray, blk: np.ndarray, dates: pd.DatetimeIndex
) -> np.ndarray:
    """Least-squares lapse rate (°C/m) per month from residuals vs. interpolation."""
    resid = fine - ctx.interp(blk)
    dz = ctx.interp_elev - ctx.static["elevation"]
    months = dates.month.to_numpy()
    out = np.full(12, DEFAULT_LAPSE_C_PER_M)
    denom = float(np.sum(dz**2))
    if denom <= 0:
        return out
    for m in range(1, 13):
        r = resid[months == m]
        if r.size == 0:
            continue
        r_mean = np.nanmean(r, axis=0)
        ok = np.isfinite(r_mean)
        if ok.sum() < 10:
            continue
        g = float(np.sum(dz[ok] * r_mean[ok]) / np.sum(dz[ok] ** 2))
        out[m - 1] = float(np.clip(g, 0.0, 0.012))
    return out
