"""REST API (FastAPI).

Run:  pcast serve     (or: uvicorn panchayatcast.api.app:create_app --factory --reload)
Docs: http://localhost:8000/docs

Write endpoints (upload/emulate runs, edit advisories) require the header
`X-API-Key` when the environment variable PCAST_API_KEY is set.
"""

from __future__ import annotations

import json
import math
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from shapely.geometry import Point

from .. import __version__
from ..config import ModelConfig, paths
from ..exports.bulletin import bulletin_pdf
from ..exports.raster import geotiff_bytes
from ..exports.sms import sms_messages
from ..exports.tabular import gp_forecast_wide, gp_geojson, records
from ..features.dataset import RegionContext
from ..geometry import EQUAL_AREA_CRS
from ..ingest.block_forecast import ForecastValidationError, parse_block_forecast
from ..ingest.nwp import nwp_block_forecast
from ..models.registry import available_models, latest_version
from ..pipeline import emulate_block_forecast, execute_run, resolve_model
from ..storage.db import Repository
from ..store import RegionStore, list_regions
from ..variables import OUTPUT_META, OUTPUT_VARS, RAIN_CATEGORIES

API_PREFIX = "/api/v1"


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _ctx(region_id: str) -> RegionContext:
    return RegionContext.load(region_id)


@lru_cache(maxsize=8)
def _geojson(region_id: str, kind: str) -> dict:
    ctx = _ctx(region_id)
    gdf = (ctx.panchayats if kind == "gp" else ctx.blocks).copy()
    gdf["geometry"] = gdf.geometry.simplify(0.0005, preserve_topology=True)
    return json.loads(gdf.to_json(na="null"))


def ctx_or_404(region_id: str) -> RegionContext:
    try:
        return _ctx(region_id)
    except FileNotFoundError as e:
        raise HTTPException(404, f"Unknown region '{region_id}'") from e


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(v) else round(float(v), 2)
    if isinstance(v, np.integer):
        return int(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _run_or_404(repo: Repository, run_id: str) -> dict:
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Unknown run '{run_id}'")
    return run


def _run_json(run: dict, repo: Repository | None = None) -> dict:
    out = {k: _clean(v) for k, v in run.items()}
    if repo is not None:
        out["advisory_counts"] = repo.advisory_counts(run["run_id"])
    return out


class AdvisoryUpdate(BaseModel):
    text_en: str | None = None
    status: Literal["draft", "approved", "rejected"] | None = None
    edited_by: str | None = None


def create_app(repo: Repository | None = None) -> FastAPI:
    app = FastAPI(
        title="PanchayatCast API",
        version=__version__,
        description="Block-to-Panchayat weather forecast downscaling for agromet advisories (SIH 26074).",
    )
    origins = os.environ.get("PCAST_CORS_ORIGINS", "*").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    app.state.repo = repo or Repository()
    cfg = ModelConfig.load()

    def get_repo() -> Repository:
        return app.state.repo

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        key = os.environ.get("PCAST_API_KEY")
        if key and x_api_key != key:
            raise HTTPException(401, "Missing or invalid X-API-Key")

    r = APIRouter(prefix=API_PREFIX)

    # ---- meta -------------------------------------------------------------
    @r.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "version": __version__}

    @r.get("/variables", tags=["meta"])
    def variables():
        return {
            "variables": [{"id": v, **OUTPUT_META[v]} for v in OUTPUT_VARS],
            "rain_categories": [
                {"name": n, "min_mm": lo, "max_mm": None if math.isinf(hi) else hi}
                for n, lo, hi in RAIN_CATEGORIES
            ],
        }

    # ---- regions ----------------------------------------------------------
    @r.get("/regions", tags=["regions"])
    def regions(repo: Repository = Depends(get_repo)):
        out = []
        for rid in list_regions():
            try:
                ctx = _ctx(rid)
            except Exception:  # noqa: BLE001  (half-built region)
                continue
            m = ctx.meta
            latest = repo.latest_run(rid)
            out.append({
                "region_id": rid, "name": m.get("name"),
                "data_source": m.get("data_source", m.get("source")),
                "bbox": m.get("bbox"), "n_blocks": ctx.n_block, "n_gps": ctx.weights.n_gp,
                "languages": m.get("languages", ["en"]), "model_version": latest_version(rid),
                "latest_run_id": latest["run_id"] if latest else None,
            })
        return out

    @r.get("/regions/{region_id}", tags=["regions"])
    def region(region_id: str):
        ctx = ctx_or_404(region_id)
        version, model_ids = available_models(region_id)
        try:
            d = RegionStore(region_id).fine_dates()
            period = [str(d[0].date()), str(d[-1].date())]
        except (FileNotFoundError, OSError, IndexError):
            period = None
        return {
            **{k: v for k, v in ctx.meta.items() if k != "grid"},
            "data_source": ctx.meta.get("data_source", ctx.meta.get("source")),
            "grid": ctx.meta["grid"],
            "n_blocks": ctx.n_block,
            "n_gps": ctx.weights.n_gp,
            "n_active_cells": ctx.n_active,
            "data_period": period,
            "model_version": version,
            "models": sorted(model_ids),
            "default_model": resolve_model(model_ids, cfg, None),
        }

    @r.get("/regions/{region_id}/panchayats/locate", tags=["regions"])
    def locate(region_id: str, lat: float, lon: float):
        """Panchayat containing a point (e.g. the farmer's GPS), else the nearest one within 10 km."""
        ctx = ctx_or_404(region_id)
        gps = ctx.panchayats
        pt = Point(lon, lat)
        hit = gps[gps.contains(pt)]
        dist_km = 0.0
        if hit.empty:
            proj = gps.to_crs(EQUAL_AREA_CRS)
            p = gpd.GeoSeries([pt], crs="EPSG:4326").to_crs(EQUAL_AREA_CRS).iloc[0]
            d = proj.distance(p)
            i = int(d.values.argmin())
            dist_km = float(d.iloc[i]) / 1000
            if dist_km > 10:
                raise HTTPException(404, f"No panchayat of this region within 10 km ({dist_km:.0f} km away)")
            hit = gps.iloc[[i]]
        g = hit.iloc[0]
        return {"gp_lgd": int(g["gp_lgd"]), "gp_name": g["gp_name"], "block_lgd": int(g["block_lgd"]),
                "block_name": g["block_name"], "distance_km": round(dist_km, 2)}

    @r.get("/regions/{region_id}/blocks", tags=["regions"])
    def blocks(region_id: str):
        ctx_or_404(region_id)
        return _geojson(region_id, "block")

    @r.get("/regions/{region_id}/panchayats", tags=["regions"])
    def panchayats(region_id: str, block_lgd: int | None = None):
        ctx_or_404(region_id)
        fc = _geojson(region_id, "gp")
        if block_lgd is not None:
            fc = {**fc, "features": [f for f in fc["features"]
                                     if f["properties"]["block_lgd"] == block_lgd]}
        return fc

    @r.get("/regions/{region_id}/panchayats/search", tags=["regions"])
    def search(region_id: str, q: str = Query(min_length=1), limit: int = 20):
        ctx = ctx_or_404(region_id)
        gps = ctx.panchayats
        hit = gps[gps["gp_name"].str.contains(q, case=False, regex=False)].head(limit)
        return records(hit[["gp_lgd", "gp_name", "block_lgd", "block_name"]])

    @r.get("/regions/{region_id}/validation", tags=["validation"])
    def validation(region_id: str, model_version: str | None = None,
                   repo: Repository = Depends(get_repo)):
        ctx_or_404(region_id)
        version = model_version or latest_version(region_id)
        df = repo.metrics(region_id, version)
        if df.empty:
            return {"model_version": version, "metrics": [], "summary": []}
        df = df.drop(columns=["id", "created_at"], errors="ignore")
        sk = df[df["metric"] == "skill_vs_M0"]
        summary = records(sk[["model_id", "variable", "level", "value"]])
        return {"model_version": version, "metrics": records(df), "summary": summary}

    # ---- runs ---------------------------------------------------------------
    @r.get("/regions/{region_id}/runs", tags=["runs"])
    def region_runs(region_id: str, limit: int = 50, repo: Repository = Depends(get_repo)):
        return [_run_json(x, repo) for x in repo.list_runs(region_id, limit)]

    @r.get("/regions/{region_id}/runs/latest", tags=["runs"])
    def latest_run(region_id: str, repo: Repository = Depends(get_repo)):
        run = repo.latest_run(region_id)
        if run is None:
            raise HTTPException(404, "No completed runs for this region yet")
        return _run_json(run, repo)

    def _start_run(region_id: str, src, source: str, model_id: str | None,
                   tasks: BackgroundTasks, repo: Repository) -> dict:
        ctx = ctx_or_404(region_id)
        version, model_ids = available_models(region_id)
        try:
            model_id = resolve_model(model_ids, cfg, model_id)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0])) from e
        try:
            fc = parse_block_forecast(src, ctx.weights.block_ids)
        except ForecastValidationError as e:
            raise HTTPException(422, {"message": "Invalid block forecast", "errors": e.errors}) from e
        run_id = str(uuid.uuid4())
        repo.create_run(run_id, region_id, fc.issue_date, source, model_id, version)
        tasks.add_task(_safe_execute, run_id, region_id, fc, model_id, repo, version)
        return {"run_id": run_id, "status": "queued", "issue_date": str(fc.issue_date.date()),
                "model_id": model_id, "warnings": fc.warnings}

    def _safe_execute(run_id, region_id, fc, model_id, repo, version):
        try:
            execute_run(run_id, region_id, fc, model_id, repo, cfg, version)
        except Exception:  # noqa: BLE001  (status already recorded as failed)
            pass

    @r.post("/regions/{region_id}/runs", status_code=202, tags=["runs"],
            dependencies=[Depends(require_key)])
    async def upload_run(region_id: str, tasks: BackgroundTasks,
                         file: UploadFile = File(..., description="Block forecast CSV"),
                         model_id: str | None = Form(default=None),
                         repo: Repository = Depends(get_repo)):
        content = await file.read()
        return _start_run(region_id, content, "upload", model_id, tasks, repo)

    @r.post("/regions/{region_id}/runs/emulate", status_code=202, tags=["runs"],
            dependencies=[Depends(require_key)])
    def emulate_run(region_id: str, tasks: BackgroundTasks, issue_date: str,
                    model_id: str | None = None, seed: int = 0,
                    repo: Repository = Depends(get_repo)):
        ctx_or_404(region_id)
        try:
            df = emulate_block_forecast(region_id, issue_date, seed=seed)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(400, "This installation has no daily history for the region, so it "
                                     "cannot emulate a forecast. Upload one or fetch a live forecast.") from e
        return _start_run(region_id, df, "emulated", model_id, tasks, repo)

    @r.post("/regions/{region_id}/runs/fetch", status_code=202, tags=["runs"],
            dependencies=[Depends(require_key)])
    def fetch_run(region_id: str, tasks: BackgroundTasks, model_id: str | None = None,
                  nwp_model: str | None = None, repo: Repository = Depends(get_repo)):
        """Fetch today's live NWP forecast (Open-Meteo) at block centroids and downscale it."""
        ctx = ctx_or_404(region_id)
        try:
            df = nwp_block_forecast(ctx, model=nwp_model)
        except Exception as e:  # noqa: BLE001  (network / provider errors)
            raise HTTPException(502, f"Could not fetch the live forecast: {e}") from e
        out = _start_run(region_id, df, "nwp", model_id, tasks, repo)
        return {**out, "provider": "open-meteo"}

    @r.get("/runs/{run_id}", tags=["runs"])
    def get_run(run_id: str, repo: Repository = Depends(get_repo)):
        return _run_json(_run_or_404(repo, run_id), repo)

    @r.get("/runs/{run_id}/map", tags=["forecast"])
    def run_map(run_id: str, variable: str = "rain_mm", valid_date: str | None = None,
                level: Literal["gp", "block"] = "gp", repo: Repository = Depends(get_repo)):
        run = _run_or_404(repo, run_id)
        if variable not in OUTPUT_VARS:
            raise HTTPException(400, f"variable must be one of {OUTPUT_VARS}")
        if valid_date is None:
            valid_date = str(pd.Timestamp(run["issue_date"]) + pd.Timedelta(days=1))[:10]
        vals = repo.map_values(run_id, variable, valid_date, level)
        if vals.empty:
            raise HTTPException(404, f"No values for {variable} on {valid_date}")
        key = "gp_lgd" if level == "gp" else "block_lgd"
        by_id = {int(row[key]): row for row in vals.to_dict(orient="records")}
        block_vals = {}
        if level == "gp":
            bv = repo.map_values(run_id, variable, valid_date, "block")
            block_vals = dict(zip(bv["block_lgd"].astype(int), bv["value"]))
        base = _geojson(run["region_id"], level)
        feats = []
        for f in base["features"]:
            p = f["properties"]
            row = by_id.get(int(p[key]))
            props = {**p, "value": _clean(row["value"]) if row else None}
            if level == "gp" and row:
                props.update(p10=_clean(row["p10"]), p90=_clean(row["p90"]),
                             confidence=row["confidence"],
                             block_value=_clean(block_vals.get(int(p["block_lgd"]))))
                if props["value"] is not None and props["block_value"] is not None:
                    props["diff_from_block"] = round(props["value"] - props["block_value"], 2)
            feats.append({**f, "properties": props})
        values = [x["properties"]["value"] for x in feats if x["properties"]["value"] is not None]
        return {"type": "FeatureCollection", "features": feats,
                "properties": {"run_id": run_id, "variable": variable, "valid_date": valid_date,
                               "level": level, **OUTPUT_META[variable],
                               "min": min(values) if values else None,
                               "max": max(values) if values else None}}

    @r.get("/runs/{run_id}/panchayats/{gp_lgd}/forecast", tags=["forecast"])
    def gp_forecast(run_id: str, gp_lgd: int, repo: Repository = Depends(get_repo)):
        run = _run_or_404(repo, run_id)
        ctx = ctx_or_404(run["region_id"])
        gps = ctx.panchayats.set_index("gp_lgd")
        if gp_lgd not in gps.index:
            raise HTTPException(404, f"Unknown panchayat {gp_lgd}")
        g = gps.loc[gp_lgd]
        df = repo.gp_forecasts(run_id, gp_lgd)
        bf = repo.block_forecasts(run_id, int(g["block_lgd"]))
        bmap = {(str(r.valid_date), r.variable): r.value for r in bf.itertuples()}
        days = []
        for d, sub in df.sort_values("lead_day").groupby("valid_date", sort=True):
            day = {"valid_date": str(d), "lead_day": int(sub["lead_day"].iloc[0])}
            for row in sub.itertuples():
                day[row.variable] = {
                    "value": _clean(row.value), "p10": _clean(row.p10), "p90": _clean(row.p90),
                    "confidence": row.confidence, "block": _clean(bmap.get((str(d), row.variable))),
                }
            days.append(day)
        return {
            "gp_lgd": gp_lgd, "gp_name": g["gp_name"], "block_lgd": int(g["block_lgd"]),
            "block_name": g["block_name"], "area_km2": _clean(g["area_km2"]),
            "run": _run_json(run), "days": days,
        }

    @r.get("/runs/{run_id}/panchayats/{gp_lgd}/advisories", tags=["advisories"])
    def gp_advisories(run_id: str, gp_lgd: int, crop: str | None = None, lang: str = "en",
                      repo: Repository = Depends(get_repo)):
        _run_or_404(repo, run_id)
        items = repo.list_advisories(run_id, gp_lgd=gp_lgd, crop=crop)
        return [_advisory_json(a, lang) for a in items if a["status"] != "rejected"]

    @r.get("/runs/{run_id}/advisories", tags=["advisories"])
    def run_advisories(run_id: str, block_lgd: int | None = None, gp_lgd: int | None = None,
                       severity: str | None = None, crop: str | None = None,
                       status: str | None = None, lang: str = "en",
                       repo: Repository = Depends(get_repo)):
        _run_or_404(repo, run_id)
        items = repo.list_advisories(run_id, gp_lgd=gp_lgd, block_lgd=block_lgd,
                                     severity=severity, crop=crop, status=status)
        return [_advisory_json(a, lang) for a in items]

    @r.patch("/advisories/{advisory_id}", tags=["advisories"], dependencies=[Depends(require_key)])
    def edit_advisory(advisory_id: str, body: AdvisoryUpdate, repo: Repository = Depends(get_repo)):
        if repo.get_advisory(advisory_id) is None:
            raise HTTPException(404, "Unknown advisory")
        values = body.model_dump(exclude_none=True)
        if not values:
            raise HTTPException(400, "Nothing to update")
        return _advisory_json(repo.update_advisory(advisory_id, **values), "en")

    @r.get("/runs/{run_id}/compare", tags=["forecast"])
    def compare(run_id: str, block_lgd: int | None = None, repo: Repository = Depends(get_repo)):
        """How much downscaling changed things: per block/day/variable, the spread of panchayat values."""
        run = _run_or_404(repo, run_id)
        ctx = ctx_or_404(run["region_id"])
        gp = repo.gp_forecasts(run_id).merge(ctx.panchayats[["gp_lgd", "block_lgd"]], on="gp_lgd")
        bf = repo.block_forecasts(run_id)
        if block_lgd is not None:
            gp, bf = gp[gp.block_lgd == block_lgd], bf[bf.block_lgd == block_lgd]
        gp = gp[gp.variable != "wind_dir_deg"]
        agg = gp.groupby(["block_lgd", "valid_date", "variable"])["value"].agg(["min", "max", "std"])
        agg = agg.reset_index().merge(
            bf.rename(columns={"value": "block_value"})[["block_lgd", "valid_date", "variable", "block_value"]],
            on=["block_lgd", "valid_date", "variable"], how="left")
        agg["range"] = agg["max"] - agg["min"]
        return records(agg.round(2))

    # ---- exports ------------------------------------------------------------
    @r.get("/runs/{run_id}/export.csv", tags=["exports"])
    def export_csv(run_id: str, repo: Repository = Depends(get_repo)):
        run = _run_or_404(repo, run_id)
        df = gp_forecast_wide(ctx_or_404(run["region_id"]), repo, run_id)
        return Response(df.to_csv(index=False), media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="panchayat_forecast_{run["issue_date"]}.csv"'})

    @r.get("/runs/{run_id}/export.geojson", tags=["exports"])
    def export_geojson(run_id: str, repo: Repository = Depends(get_repo)):
        run = _run_or_404(repo, run_id)
        fc = gp_geojson(ctx_or_404(run["region_id"]), repo, run_id)
        return JSONResponse(fc, headers={
            "Content-Disposition": f'attachment; filename="panchayat_forecast_{run["issue_date"]}.geojson"'})

    @r.get("/runs/{run_id}/bulletin.pdf", tags=["exports"])
    def export_pdf(run_id: str, block_lgd: int | None = None, lang: str = "en",
                   repo: Repository = Depends(get_repo)):
        run = _run_or_404(repo, run_id)
        pdf = bulletin_pdf(ctx_or_404(run["region_id"]), repo, run_id, block_lgd, lang=lang)
        return Response(pdf, media_type="application/pdf", headers={
            "Content-Disposition": f'inline; filename="bulletin_{run["issue_date"]}_{lang}.pdf"'})

    @r.get("/runs/{run_id}/sms.csv", tags=["exports"])
    def export_sms(run_id: str, lang: str = "en", repo: Repository = Depends(get_repo)):
        run = _run_or_404(repo, run_id)
        df = sms_messages(ctx_or_404(run["region_id"]), repo, run_id, lang)
        # UTF-8 BOM so Excel shows Hindi/Kannada correctly.
        return Response("﻿" + df.to_csv(index=False), media_type="text/csv; charset=utf-8", headers={
            "Content-Disposition": f'attachment; filename="sms_{run["issue_date"]}_{lang}.csv"'})

    @r.get("/runs/{run_id}/raster/{variable}/{valid_date}.tif", tags=["exports"])
    def export_tif(run_id: str, variable: str, valid_date: str, repo: Repository = Depends(get_repo)):
        _run_or_404(repo, run_id)
        try:
            data = geotiff_bytes(run_id, variable, valid_date)
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e)) from e
        return Response(data, media_type="image/tiff", headers={
            "Content-Disposition": f'attachment; filename="{variable}_{valid_date}.tif"'})

    app.include_router(r)

    # Serve the built dashboard (web/dist) at "/" when it exists. API routes and /docs
    # are registered first, so they take precedence over the static files.
    dist = Path(os.environ.get("PCAST_WEB_DIST", paths().root / "web" / "dist"))
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app


def _advisory_json(a: dict, lang: str) -> dict:
    out = {k: _clean(v) for k, v in a.items() if k != "text_local"}
    local = (a.get("text_local") or {}).get(lang) if lang != "en" else None
    out["text"] = local["text"] if local else a["text_en"]
    out["lang"] = lang if local else "en"
    out["machine_translated"] = bool(local and not local.get("reviewed", False))
    out["translations"] = sorted((a.get("text_local") or {}).keys())
    return out
