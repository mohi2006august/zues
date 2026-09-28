"""Operational pipeline: block forecast -> panchayat forecast + advisories, stored."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .advisory.engine import AdvisoryEngine
from .config import ModelConfig, paths
from .downscale.aggregate import aggregate_to_gp
from .downscale.engine import downscale, to_dataset
from .features.dataset import RegionContext
from .ingest.block_forecast import BlockForecast, parse_block_forecast
from .models.registry import ModelBundle
from .storage.db import Repository
from .store import RegionStore
from .variables import INPUT_RANGES, INPUT_VARS, MODELLED_VARS, uv_to_wind


@dataclass
class RunResult:
    run_id: str
    issue_date: pd.Timestamp
    model_id: str
    model_version: str | None
    n_gps: int
    n_advisories: int
    warnings: list[str] = field(default_factory=list)


def run_dir(run_id: str) -> Path:
    return paths().runs / run_id


def resolve_model(available: ModelBundle | Iterable[str], cfg: ModelConfig, model_id: str | None) -> str:
    """The requested model if available, else the configured default (M2 as a last resort)."""
    ids = set(available.models) if isinstance(available, ModelBundle) else set(available)
    if model_id:
        if model_id not in ids:
            raise KeyError(f"Model {model_id} not available (have {sorted(ids)})")
        return model_id
    return cfg.default_model if cfg.default_model in ids else "M2"


def validate_forecast(region_id: str, src) -> BlockForecast:
    ctx = RegionContext.load(region_id)
    return parse_block_forecast(src, ctx.weights.block_ids)


def execute_run(
    run_id: str,
    region_id: str,
    fc: BlockForecast,
    model_id: str,
    repo: Repository,
    cfg: ModelConfig | None = None,
    version: str | None = None,
) -> RunResult:
    """Downscale a validated forecast for an already-created run row; marks it done/failed."""
    cfg = cfg or ModelConfig.load()
    try:
        repo.update_run(run_id, status="running")
        ctx = RegionContext.load(region_id)
        bundle = ModelBundle.load(ctx, version)
        blk = fc.to_blk(ctx.weights.block_ids)
        pred = downscale(ctx, bundle, blk, fc.dates, model_id, cfg)

        d = run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        to_dataset(ctx, pred, fc.dates, fc.lead_days, attrs={
            "run_id": run_id, "region_id": region_id, "issue_date": fc.issue_date.date(),
            "model_id": model_id, "model_version": bundle.version,
            "data_source": ctx.meta.get("data_source", ctx.meta.get("source")),
        }).to_netcdf(d / "grid.nc")

        gp_df = aggregate_to_gp(ctx, pred, fc.dates, fc.lead_days, model_id)
        repo.insert_block_forecasts(run_id, fc.table, INPUT_VARS)
        repo.insert_gp_forecasts(run_id, gp_df)

        engine = AdvisoryEngine.for_region(ctx.meta)
        items = engine.generate(gp_df, ctx.panchayats[["gp_lgd", "block_lgd"]])
        repo.insert_advisories(run_id, items)

        notes = "; ".join(fc.warnings) or None
        repo.update_run(run_id, status="done", notes=notes)
        return RunResult(run_id, fc.issue_date, model_id, bundle.version, ctx.weights.n_gp,
                         len(items), fc.warnings)
    except Exception as e:
        repo.update_run(run_id, status="failed", notes=f"{type(e).__name__}: {e}")
        raise


def run_forecast(
    region_id: str,
    src,
    source: str = "upload",
    model_id: str | None = None,
    repo: Repository | None = None,
    cfg: ModelConfig | None = None,
    version: str | None = None,
) -> RunResult:
    """Validate, create a run, downscale, store. Raises ForecastValidationError on bad input."""
    cfg = cfg or ModelConfig.load()
    repo = repo or Repository()
    ctx = RegionContext.load(region_id)
    bundle = ModelBundle.load(ctx, version)
    model_id = resolve_model(bundle, cfg, model_id)
    fc = parse_block_forecast(src, ctx.weights.block_ids)
    run_id = str(uuid.uuid4())
    repo.create_run(run_id, region_id, fc.issue_date, source, model_id, bundle.version)
    return execute_run(run_id, region_id, fc, model_id, repo, cfg, bundle.version)


# ---------------------------------------------------------------------------
# Emulated block forecasts (for demos and testing without an official feed)
# ---------------------------------------------------------------------------

# Forecast error standard deviations at lead day L: a + b * L
_ERROR = {
    "tmax_c": (0.4, 0.25),
    "tmin_c": (0.4, 0.2),
    "rh_max_pct": (2.5, 1.2),
    "rh_min_pct": (3.0, 1.5),
    "wind_u": (1.0, 0.5),
    "wind_v": (1.0, 0.5),
    "cloud_okta": (0.5, 0.2),
}


def add_forecast_error(
    blk: dict[str, np.ndarray], lead_days: np.ndarray, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Add realistic block-forecast error that grows with lead time.

    blk values are (n_days, n_block); lead_days is (n_days,) or a scalar. Errors are
    spatially correlated across blocks (a common part plus a block part). Rain gets a
    multiplicative (log-normal) error.
    """
    out = {k: v.copy() for k, v in blk.items()}
    D, B = blk["tmax_c"].shape
    L = np.broadcast_to(np.asarray(lead_days, dtype=float).reshape(-1, 1), (D, 1))
    for var, (a, b) in _ERROR.items():
        sd = a + b * L
        out[var] = out[var] + sd * (0.8 * rng.standard_normal((D, 1)) + 0.6 * rng.standard_normal((D, B)))
    out["rain_mm"] = np.maximum(out["rain_mm"] * np.exp((0.25 + 0.1 * L) * rng.standard_normal((D, B))), 0)
    out["cloud_okta"] = np.clip(out["cloud_okta"], 0, 8)
    out["rh_max_pct"] = np.clip(out["rh_max_pct"], 0, 100)
    out["rh_min_pct"] = np.clip(np.minimum(out["rh_min_pct"], out["rh_max_pct"]), 0, 100)
    out["tmin_c"] = np.minimum(out["tmin_c"], out["tmax_c"] - 0.5)
    return out


def emulate_block_forecast(
    region_id: str,
    issue_date: str | pd.Timestamp,
    lead_days: int = 5,
    noise: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a block forecast from the region's fine historical data (+ lead-dependent error).

    Stands in for an official IMD block forecast in demos. The result is tagged by
    the caller as source="emulated".
    """
    ctx = RegionContext.load(region_id)
    store = RegionStore(region_id)
    issue = pd.Timestamp(issue_date).normalize()
    dates = pd.date_range(issue + pd.Timedelta(days=1), periods=lead_days, freq="D")
    rng = np.random.default_rng(seed)
    avail = store.fine_dates()
    if dates[0] < avail[0] or dates[-1] > avail[-1]:
        raise ValueError(
            f"No history for {dates[0].date()}..{dates[-1].date()}; this installation has "
            f"{avail[0].date()}..{avail[-1].date()} (pick an issue date at least {lead_days} days before the end)"
        )

    blk = {}
    for var in MODELLED_VARS:
        d, fine = store.load_fine_active(var, ctx.weights, str(dates[0].date()), str(dates[-1].date()))
        if len(d) != lead_days:
            raise ValueError(f"Missing days of {var} history in {dates[0].date()}..{dates[-1].date()}")
        blk[var] = ctx.block_means(fine)

    if noise:
        blk = add_forecast_error(blk, np.arange(1, lead_days + 1), rng)

    speed, direction = uv_to_wind(blk["wind_u"], blk["wind_v"])
    values = {
        "rain_mm": blk["rain_mm"],
        "tmax_c": blk["tmax_c"],
        "tmin_c": np.minimum(blk["tmin_c"], blk["tmax_c"] - 1.0),
        "rh_max_pct": blk["rh_max_pct"],
        "rh_min_pct": np.minimum(blk["rh_min_pct"], blk["rh_max_pct"] - 2.0),
        "wind_kmph": speed,
        "wind_dir_deg": direction,
        "cloud_okta": blk["cloud_okta"],
    }
    rows = []
    for i, day in enumerate(dates):
        for j, block in enumerate(ctx.weights.block_ids):
            row = {"block_lgd": int(block), "issue_date": issue.date(), "valid_date": day.date()}
            for var in INPUT_VARS:
                lo, hi = INPUT_RANGES[var]
                row[var] = float(np.clip(values[var][i, j], lo, min(hi, 359.9) if var == "wind_dir_deg" else hi))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[df["rain_mm"] < 0.1, "rain_mm"] = 0.0
    return df.round(1)
