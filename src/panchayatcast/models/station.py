"""M3S: M3 plus a correction learned from weather-station observations.

Why: with real data the gridded "fine" training truth is coarse (ERA5-Land ~9 km),
so M3 can only learn detail that the grid contains. Stations measure the true point
weather. M3S learns how station observations differ from M3's prediction, as a
function of terrain, land cover, season and the day's weather, and applies that
correction everywhere.

Honesty guard: the correction is fitted with leave-stations-out cross-validation.
For each variable a shrink factor in {0, 0.25, 0.5, 0.75, 1} is chosen on the
held-out stations; 0 means the correction did not help and is switched off.

Variables: Tmax, Tmin, RH max, RH min (rain and wind corrections are left for later).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features.dataset import RegionContext, make_features
from ..geometry import nearest_active_cell
from ..store import RegionStore
from .base import Downscaler, Prediction, VarPrediction
from .gbm import GBMModel

STATION_VARS = ["tmax_c", "tmin_c", "rh_max_pct", "rh_min_pct"]
SHRINKS = (0.0, 0.25, 0.5, 0.75, 1.0)
PARAMS = {
    "objective": "regression", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
    "verbose": -1, "num_threads": 0,
}
ROUNDS = 250


@dataclass
class StationCorrection:
    boosters: dict[str, lgb.Booster]
    shrink: dict[str, float]
    features: dict[str, list[str]]
    report: dict = field(default_factory=dict)

    def delta(self, ctx, blk, dates, day_idx, cell_idx, var: str, base_value: np.ndarray) -> np.ndarray:
        """Correction for arbitrary (day, cell) pairs given the base (M3) prediction there."""
        s = self.shrink.get(var, 0.0)
        if s <= 0 or var not in self.boosters:
            return np.zeros(len(base_value))
        X = _features(ctx, blk, dates, day_idx, cell_idx, var, base_value)
        return s * self.boosters[var].predict(X[self.features[var]])


def _features(ctx, blk, dates, day_idx, cell_idx, var, base_value) -> pd.DataFrame:
    X, _ = make_features(ctx, blk, dates, day_idx, cell_idx, var, None)
    X["m3"] = base_value.astype(np.float32)
    return X


class StationCorrectedModel(Downscaler):
    model_id = "M3S"
    name = "M3 + station correction"

    def __init__(self, base: GBMModel, corr: StationCorrection):
        self.base = base
        self.corr = corr

    def predict(self, ctx: RegionContext, blk, dates) -> Prediction:
        out = self.base.predict(ctx, blk, dates)
        D = len(dates)
        day_idx = np.repeat(np.arange(D), ctx.n_active)
        cell_idx = np.tile(np.arange(ctx.n_active), D)
        for var in self.corr.boosters:
            if self.corr.shrink.get(var, 0.0) <= 0:
                continue
            vp = out[var]
            delta = self.corr.delta(ctx, blk, dates, day_idx, cell_idx, var, vp.value.ravel()).reshape(D, -1)
            out[var] = VarPrediction(
                vp.value + delta,
                None if vp.p10 is None else vp.p10 + delta,
                None if vp.p90 is None else vp.p90 + delta,
            )
        return out


def station_table(ctx: RegionContext, store: RegionStore) -> tuple[pd.DataFrame, np.ndarray]:
    """Station metadata and the active-cell index of each station."""
    st = store.stations()
    if st.empty:
        return st, np.array([], int)
    cells = nearest_active_cell(ctx.grid, ctx.weights, st["lon"].to_numpy(), st["lat"].to_numpy())
    return st.reset_index(drop=True), cells


def _training_rows(ctx, store, base: GBMModel, blk, dates, day_mask, stations, st_cells, var):
    obs = store.observations()
    obs = obs[obs["variable"] == var]
    wide = obs.pivot_table(index="date", columns="station_id", values="value")
    wide = wide.reindex(index=dates, columns=stations["station_id"]).to_numpy()
    days = np.where(day_mask)[0]
    d = np.repeat(days, len(st_cells))
    s = np.tile(np.arange(len(st_cells)), len(days))
    y = wide[d, s]
    ok = np.isfinite(y)
    d, s, y = d[ok], s[ok], y[ok]
    c = st_cells[s]
    base_val, _, _ = base.predict_cells(ctx, blk, dates, d, c, var)
    X = _features(ctx, blk, dates, d, c, var, base_val)
    return X, y - base_val, s


def fit_station_correction(
    ctx: RegionContext,
    store: RegionStore,
    base: GBMModel,
    blk: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    day_mask: np.ndarray,
    station_subset: np.ndarray | None = None,
    folds: int = 5,
    seed: int = 7,
    log=print,
) -> StationCorrection | None:
    """Fit per-variable corrections; shrink chosen by leave-stations-out CV."""
    stations, st_cells = station_table(ctx, store)
    if len(stations) < 8:
        log("[M3S] fewer than 8 stations; station correction skipped")
        return None
    use = np.arange(len(stations)) if station_subset is None else np.asarray(station_subset)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(stations), int)
    fold_of[use] = rng.permutation(len(use)) % folds

    boosters, shrink, feats, report = {}, {}, {}, {}
    for var in STATION_VARS:
        X, r, s = _training_rows(ctx, store, base, blk, dates, day_mask, stations, st_cells, var)
        keep = np.isin(s, use)
        X, r, s = X[keep], r[keep], s[keep]
        if len(r) < 1000:
            continue
        cv_pred = np.zeros_like(r)
        for f in range(folds):
            te = fold_of[s] == f
            if te.sum() == 0 or (~te).sum() == 0:
                continue
            b = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X[~te], r[~te]), ROUNDS)
            cv_pred[te] = b.predict(X[te])
        rmse = {k: float(np.sqrt(np.mean((r - k * cv_pred) ** 2))) for k in SHRINKS}
        best = min(rmse, key=rmse.get)
        shrink[var] = best
        feats[var] = list(X.columns)
        boosters[var] = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(X, r), ROUNDS)
        report[var] = {"cv_rmse_m3": round(rmse[0.0], 3), "cv_rmse_m3s": round(rmse[best], 3),
                       "shrink": best, "rows": int(len(r))}
        log(f"[M3S] {var}: leave-stations-out RMSE {rmse[0.0]:.3f} -> {rmse[best]:.3f} (shrink {best})")
    if not boosters:
        return None
    return StationCorrection(boosters, shrink, feats, report)
