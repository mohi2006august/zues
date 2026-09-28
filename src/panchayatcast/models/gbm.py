"""M3: gradient-boosted residual downscaler (LightGBM).

Additive variables: learn   fine - interpolated_block_signal   from terrain, land
cover, season, the day's block values and a per-cell climatology. Quantile models
give p10/p90.

Rainfall (zero-inflated): an occurrence classifier (P(rain > wet threshold)) and an
amount regressor on log1p(rain) for wet cells, combined with a threshold tuned for
CSI on the validation period. An optional quantile mapping of wet amounts (fitted
and selected on separate halves of the validation data) corrects the regressor's
tendency to under-predict heavy rain. Quantile models give p10/p90.

Training options used by spatial cross-validation: restrict training to a subset of
cells, and train without the per-cell climatology (for locations with no history).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import ModelConfig
from ..features.dataset import (
    RegionContext,
    all_pairs,
    compute_climatology,
    make_features,
    split_mask,
)
from ..store import RegionStore
from ..variables import MODELLED, MODELLED_VARS
from .base import Downscaler, Prediction, VarPrediction
from .baselines import fit_lapse_rates


@dataclass
class VarBoosters:
    kind: str
    features: list[str]
    clim: np.ndarray | None  # (12, n_active); None when trained without climatology
    q_lo: lgb.Booster | None = None
    q_hi: lgb.Booster | None = None
    mean: lgb.Booster | None = None  # additive
    occ: lgb.Booster | None = None  # rain occurrence
    amt: lgb.Booster | None = None  # rain amount, log1p space
    rain_threshold: float = 0.5
    qm: tuple[list[float], list[float]] | None = None  # rain: (predicted quantiles, observed quantiles)
    report: dict = field(default_factory=dict)

    def parts(self) -> dict[str, lgb.Booster]:
        return {
            name: b
            for name in ("q_lo", "q_hi", "mean", "occ", "amt")
            if (b := getattr(self, name)) is not None
        }


def apply_qm(x: np.ndarray, qm: tuple[list[float], list[float]]) -> np.ndarray:
    """Map predicted wet amounts through (pred_q -> obs_q); scale linearly above the top quantile."""
    pq, oq = np.asarray(qm[0]), np.asarray(qm[1])
    out = np.interp(x, pq, oq)
    top = x > pq[-1]
    if top.any() and pq[-1] > 0:
        out[top] = x[top] * (oq[-1] / pq[-1])
    return out


class GBMModel(Downscaler):
    model_id = "M3"
    name = "Gradient boosting (LightGBM)"

    def __init__(self, vars: dict[str, VarBoosters], wet_threshold: float = 0.1):
        self.vars = vars
        self.wet = wet_threshold

    def predict_cells(self, ctx: RegionContext, blk, dates, day_idx, cell_idx, var: str):
        """Predict one variable for arbitrary (day, cell) pairs -> (value, p10, p90) 1-D arrays."""
        vb = self.vars[var]
        X, interp = make_features(ctx, blk, dates, day_idx, cell_idx, var, vb.clim)
        X = X[vb.features]
        lo = vb.q_lo.predict(X) if vb.q_lo is not None else None
        hi = vb.q_hi.predict(X) if vb.q_hi is not None else None
        if vb.kind == "rain":
            p = vb.occ.predict(X)
            amt = np.expm1(vb.amt.predict(X))
            if vb.qm is not None:
                amt = apply_qm(amt, vb.qm)
            val = np.where(p >= vb.rain_threshold, np.maximum(amt, self.wet), 0.0)
            dry_block = X[f"blk_{var}"].to_numpy() < self.wet
            val[dry_block] = 0.0
            if lo is not None:
                lo = np.where(dry_block, 0.0, np.maximum(lo, 0.0))
                hi = np.where(dry_block, 0.0, np.maximum(hi, 0.0))
        else:
            val = interp + vb.mean.predict(X)
            if lo is not None:
                lo, hi = interp + lo, interp + hi
        if lo is not None:
            lo, hi = np.minimum(lo, val), np.maximum(hi, val)
        return val, lo, hi

    def predict(self, ctx: RegionContext, blk, dates) -> Prediction:
        D = len(dates)
        day_idx, cell_idx = all_pairs(D, ctx.n_active)
        out: Prediction = {}
        for var in MODELLED_VARS:
            val, lo, hi = self.predict_cells(ctx, blk, dates, day_idx, cell_idx, var)
            out[var] = VarPrediction(
                val.reshape(D, -1),
                None if lo is None else lo.reshape(D, -1),
                None if hi is None else hi.reshape(D, -1),
            )
        return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _sample_rows(
    fine: np.ndarray,
    blk_cell: np.ndarray | None,
    days: np.ndarray,
    cells: np.ndarray,
    n: int,
    wet: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Random (day, cell) pairs with a finite target; rain rows only where the block is wet."""
    got_d, got_c, have = [], [], 0
    for _ in range(8):
        k = max(int((n - have) * 1.6), 1000)
        d = days[rng.integers(0, len(days), k)]
        c = cells[rng.integers(0, len(cells), k)]
        ok = np.isfinite(fine[d, c])
        if blk_cell is not None:
            ok &= blk_cell[d, c] >= wet
        got_d.append(d[ok])
        got_c.append(c[ok])
        have += int(ok.sum())
        if have >= n:
            break
    d, c = np.concatenate(got_d)[:n], np.concatenate(got_c)[:n]
    return d, c


def _train(params: dict, X, y, Xv, yv, rounds: int, es: int) -> lgb.Booster:
    dtr = lgb.Dataset(X, y, free_raw_data=True)
    if len(yv) > 0:
        dva = lgb.Dataset(Xv, yv, reference=dtr)
        return lgb.train(
            params, dtr, num_boost_round=rounds, valid_sets=[dva],
            callbacks=[lgb.early_stopping(es, verbose=False)],
        )
    return lgb.train(params, dtr, num_boost_round=rounds)


def _rmse(a, b) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def _csi(pred: np.ndarray, obs: np.ndarray) -> float:
    hits = np.sum(pred & obs)
    denom = hits + np.sum(pred & ~obs) + np.sum(~pred & obs)
    return float(hits / denom) if denom else float("nan")


def _select_qm(pred_wet_amt, y_wet, occ_ok, y_all, pred_all_fn, rng) -> tuple[tuple | None, dict]:
    """Fit QM on half of the validation rows, keep it only if it helps on the other half."""
    n = len(y_all)
    if n < 2000 or occ_ok.sum() < 500:
        return None, {"qm": "skipped (too little validation data)"}
    idx = rng.permutation(n)
    a, b = idx[: n // 2], idx[n // 2 :]
    fit_rows = a[occ_ok[a] & (y_all[a] >= 0.1)]
    if len(fit_rows) < 300:
        return None, {"qm": "skipped (too few wet rows)"}
    qs = np.linspace(0.01, 0.995, 60)
    qm = (np.quantile(pred_wet_amt[fit_rows], qs).tolist(), np.quantile(y_wet[fit_rows], qs).tolist())
    base, mapped = pred_all_fn(None)[b], pred_all_fn(qm)[b]
    yb = y_all[b]
    score = lambda p: np.nanmean([_csi(p >= t, yb >= t) for t in (15.6, 64.5)])  # noqa: E731
    s0, s1 = score(base), score(mapped)
    r0, r1 = _rmse(base, yb), _rmse(mapped, yb)
    use = bool(s1 > s0 and r1 <= r0 * 1.03)
    return (qm if use else None), {"qm_used": use, "qm_csi_heavy": [round(s0, 3), round(s1, 3)],
                                   "qm_rmse": [round(r0, 3), round(r1, 3)]}


def train_gbm_and_lapse(
    ctx: RegionContext,
    store: RegionStore,
    blk: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    cfg: ModelConfig,
    log=print,
    train_cells: np.ndarray | None = None,
    use_clim: bool = True,
    with_quantiles: bool = True,
) -> tuple[GBMModel, dict[str, np.ndarray]]:
    splits = ctx.meta["splits"]
    tr_mask = split_mask(dates, splits["train"])
    va_mask = split_mask(dates, splits["val"])
    tr_days, va_days = np.where(tr_mask)[0], np.where(va_mask)[0]
    if len(tr_days) == 0:
        raise ValueError("No training days in the configured train split.")
    cells = np.arange(ctx.n_active) if train_cells is None else np.asarray(train_cells)

    rng = np.random.default_rng(cfg.seed)
    n_tr = int(cfg.training.get("train_rows", 300_000))
    n_va = int(cfg.training.get("val_rows", 60_000))
    wet = cfg.wet_threshold
    rounds, es = cfg.rounds, int(cfg.rounds.get("early_stopping", 50))
    q_lo, q_hi = cfg.quantiles
    base = {"verbose": -1, "seed": cfg.seed, "num_threads": 0, **cfg.lightgbm}

    boosters: dict[str, VarBoosters] = {}
    gammas: dict[str, np.ndarray] = {}
    for var in MODELLED_VARS:
        spec = MODELLED[var]
        _, fine = store.load_fine_active(var, ctx.weights)
        if fine.shape[0] != len(dates):
            raise ValueError(f"{var}: fine data has {fine.shape[0]} days, expected {len(dates)}")

        if spec.lapse:
            gammas[var] = fit_lapse_rates(ctx, fine[tr_mask], blk[var][tr_mask], dates[tr_mask])

        clim = (
            compute_climatology(ctx, var, fine[tr_mask], blk[var][tr_mask], dates[tr_mask])
            if use_clim else None
        )
        blk_cell = ctx.copy(blk[var]) if spec.kind == "rain" else None
        d_tr, c_tr = _sample_rows(fine, blk_cell, tr_days, cells, n_tr, wet, rng)
        d_va, c_va = (
            _sample_rows(fine, blk_cell, va_days, cells, n_va, wet, rng)
            if len(va_days) else (np.array([], int), np.array([], int))
        )
        Xtr, itr = make_features(ctx, blk, dates, d_tr, c_tr, var, clim)
        Xva, iva = make_features(ctx, blk, dates, d_va, c_va, var, clim)
        ytr, yva = fine[d_tr, c_tr], fine[d_va, c_va]
        features = list(Xtr.columns)
        del fine

        def quantile_models(ytr_, yva_):
            if not with_quantiles:
                return None, None
            lo = _train({**base, "objective": "quantile", "alpha": q_lo}, Xtr, ytr_, Xva, yva_,
                        rounds["quantile"], es)
            hi = _train({**base, "objective": "quantile", "alpha": q_hi}, Xtr, ytr_, Xva, yva_,
                        rounds["quantile"], es)
            return lo, hi

        if spec.kind == "rain":
            wet_tr, wet_va = ytr >= wet, yva >= wet
            occ = _train({**base, "objective": "binary"}, Xtr, wet_tr.astype(int),
                         Xva, wet_va.astype(int), rounds["mean"], es)
            amt = _train({**base, "objective": "regression"}, Xtr[wet_tr], np.log1p(ytr[wet_tr]),
                         Xva[wet_va], np.log1p(yva[wet_va]), rounds["mean"], es)
            lo, hi = quantile_models(ytr, yva)
            tau, qm, report = 0.5, None, {}
            if len(yva):
                p = occ.predict(Xva)
                taus = np.linspace(0.2, 0.8, 25)
                tau = float(taus[int(np.nanargmax([_csi(p >= t, wet_va) for t in taus]))])
                amt_va = np.expm1(amt.predict(Xva))
                occ_ok = p >= tau

                def pred_va(qm_):
                    a = apply_qm(amt_va, qm_) if qm_ is not None else amt_va
                    return np.where(occ_ok, np.maximum(a, wet), 0.0)

                qm, qm_report = _select_qm(amt_va, yva, occ_ok, yva, pred_va, rng)
                pred = pred_va(qm)
                report = {
                    "val_rmse_model": _rmse(pred, yva),
                    "val_rmse_interp": _rmse(iva, yva),
                    "val_csi_model": _csi(pred >= wet, wet_va),
                    "val_csi_interp": _csi(iva >= wet, wet_va),
                    "rain_threshold": tau,
                    **qm_report,
                }
            vb = VarBoosters("rain", features, clim, lo, hi, occ=occ, amt=amt,
                             rain_threshold=tau, qm=qm, report=report)
        else:
            rtr, rva = ytr - itr, yva - iva
            mean = _train({**base, "objective": "regression"}, Xtr, rtr, Xva, rva, rounds["mean"], es)
            lo, hi = quantile_models(rtr, rva)
            report = {}
            if len(yva):
                report = {
                    "val_rmse_model": _rmse(iva + mean.predict(Xva), yva),
                    "val_rmse_interp": _rmse(iva, yva),
                }
            vb = VarBoosters("additive", features, clim, lo, hi, mean=mean, report=report)

        report["train_rows"] = int(len(ytr))
        report["val_rows"] = int(len(yva))
        boosters[var] = vb
        msg = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in report.items())
        log(f"[train] {var}: {msg}")

    return GBMModel(boosters, wet), gammas
