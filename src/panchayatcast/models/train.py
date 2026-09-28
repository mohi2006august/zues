"""Training entry point."""

from __future__ import annotations

import time

import numpy as np

from ..config import ModelConfig
from ..features.dataset import RegionContext, split_mask
from ..store import RegionStore
from ..variables import MODELLED_VARS
from .gbm import train_gbm_and_lapse
from .registry import save_bundle
from .station import fit_station_correction


def block_fields(ctx: RegionContext, store: RegionStore) -> dict[str, np.ndarray]:
    """Block means of every modelled variable over the whole period (the model's inputs)."""
    blk = {}
    for var in MODELLED_VARS:
        _, fine = store.load_fine_active(var, ctx.weights)
        blk[var] = ctx.block_means(fine)
        del fine
    return blk


def train_region(region_id: str, cfg: ModelConfig | None = None, log=print,
                 stations: bool = True) -> str:
    cfg = cfg or ModelConfig.load()
    t0 = time.time()
    ctx = RegionContext.load(region_id)
    store = RegionStore(region_id)
    dates = store.fine_dates()

    log(f"[train] region={region_id} days={len(dates)} cells={ctx.n_active} "
        f"blocks={ctx.n_block} gps={ctx.weights.n_gp}")
    blk = block_fields(ctx, store)
    gbm, gammas = train_gbm_and_lapse(ctx, store, blk, dates, cfg, log=log)

    station = None
    if stations:
        train_mask = split_mask(dates, tuple(ctx.meta["splits"]["train"]))
        station = fit_station_correction(ctx, store, gbm, blk, dates, train_mask, seed=cfg.seed, log=log)

    version = save_bundle(
        region_id,
        gbm,
        gammas,
        extra_meta={
            "n_active_cells": ctx.n_active,
            "region_built_at": ctx.meta.get("built_at"),
            "data_source": ctx.meta.get("data_source", ctx.meta.get("source")),
            "splits": ctx.meta["splits"],
            "config": cfg.raw,
            "train_seconds": round(time.time() - t0, 1),
        },
        station=station,
    )
    log(f"[train] saved model version {version} in {time.time() - t0:.0f}s")
    return version
