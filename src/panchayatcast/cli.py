"""Command-line interface: `pcast --help`."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from .config import ModelConfig, RegionConfig, paths

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="PanchayatCast: block -> panchayat forecast downscaling (SIH 26074).")
download_app = typer.Typer(no_args_is_help=True, help="Download open datasets for a real region.")
app.add_typer(download_app, name="download")

DEMO_CONFIG = "configs/region.demo.yaml"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Hindi/Kannada text on Windows consoles


def _fast(cfg: ModelConfig) -> ModelConfig:
    cfg.raw.setdefault("training", {})
    cfg.raw["training"]["train_rows"] = 40_000
    cfg.raw["training"]["val_rows"] = 10_000
    cfg.raw["rounds"] = {"mean": 200, "quantile": 100, "early_stopping": 20}
    return cfg


@app.command()
def synth(config: Path = typer.Argument(DEMO_CONFIG, help="Region YAML with source: synthetic")):
    """Generate the synthetic demo region (terrain, boundaries, 1 km weather, stations)."""
    from .ingest.synthetic import generate

    generate(RegionConfig.from_yaml(paths().resolve(config)))


@app.command("build-region")
def build_region(config: Path = typer.Argument(..., help="Region YAML with source: real")):
    """Build a real region from boundaries, DEM, land cover and daily gridded data."""
    from .ingest.real import build_real_region

    build_real_region(RegionConfig.from_yaml(paths().resolve(config)))


@app.command()
def pilot(config: Path = typer.Argument("configs/region.dharwad.yaml"),
          build: bool = typer.Option(True, help="Build the region after downloading")):
    """Download all open data for a real district (resumable), then build the region.

    Boundaries: LGD panchayat layer. Rain: CHIRPS. Temperature/humidity: ERA5-Land and
    wind/cloud: ERA5 (Open-Meteo archive). Terrain/land cover: Copernicus DEM, WorldCover.
    """
    import geopandas as gpd

    from .ingest import download
    from .ingest.lgd import fetch_district_panchayats
    from .ingest.openmeteo_archive import build_fine_fields
    from .ingest.real import build_real_region

    p = paths()
    rc = RegionConfig.from_yaml(p.resolve(config))
    r = rc.real
    start, end = r["period"]
    gp_path = p.resolve(r["boundaries"]["panchayats"]["path"])
    if not gp_path.exists():
        lgd = r.get("lgd_district") or {}
        g = fetch_district_panchayats(lgd["district"], lgd.get("state"))
        gp_path.parent.mkdir(parents=True, exist_ok=True)
        g.to_file(gp_path, driver="GeoJSON")
    gps = gpd.read_file(gp_path)

    rain = p.resolve(r["fine"]["rain_mm"])
    if not rain.exists():
        typer.echo(f"[pilot] CHIRPS rainfall {start}..{end}")
        download.download_chirps(rc.bbox, start, end, rain)
    others = [k for k in r["fine"] if k != "rain_mm"]
    if not all(p.resolve(r["fine"][k]).exists() for k in others):
        typer.echo("[pilot] ERA5-Land / ERA5 daily fields via Open-Meteo (paced, resumable)")
        build_fine_fields(gps.geometry, rc.bbox, start, end, rain.parent,
                          p.data / "cache" / "open-meteo" / rc.region_id)
    if build:
        build_real_region(rc)
        typer.echo(f"[pilot] built. Next: pcast train {rc.region_id} && pcast evaluate {rc.region_id}")


@app.command()
def train(region: str = typer.Argument("demo"),
          fast: bool = typer.Option(False, help="Small sample, fewer trees (quick test)")):
    """Train M2 (lapse rates) and M3 (LightGBM) for a region."""
    from .models.train import train_region

    cfg = ModelConfig.load()
    train_region(region, _fast(cfg) if fast else cfg)


def _evaluate_all(region: str, version: str | None, modes: list[str]) -> Path:
    from .models.registry import latest_version
    from .storage.db import Repository
    from .store import RegionStore
    from .validate import evaluate as ev
    from .validate.report import write_report

    version = version or latest_version(region)
    frames = {}
    if "test" in modes:
        frames["test"] = ev.evaluate_region(region, version=version)
    if "forecast" in modes:
        frames["forecast"] = ev.evaluate_forecast_mode(region, version=version)
    if "spatial" in modes:
        frames["spatial_cv"] = ev.evaluate_spatial_cv(region, version=version)
    if "station" in modes:
        frames["station_cv"] = ev.evaluate_station_cv(region, version=version)
    # Keep earlier results for evaluations not re-run now, so the report stays complete.
    stored = Repository().metrics(region, version)
    if not stored.empty:
        for split, df in stored.groupby("split"):
            frames.setdefault(split, df.drop(columns=["id", "created_at", "region_id", "model_version",
                                                     "split"], errors="ignore").reset_index(drop=True))
    return write_report(region, version, frames, RegionStore(region).meta)


@app.command("train-unet")
def train_unet_cmd(region: str = typer.Argument("demo"), epochs: int = 25,
                   version: str | None = None):
    """Train the optional M4 U-Net (needs `pip install torch`) into an existing model version."""
    from .features.dataset import RegionContext
    from .models.registry import latest_version, region_model_dir
    from .models.train import block_fields
    from .models.unet import train_unet
    from .store import RegionStore

    version = version or latest_version(region)
    if version is None:
        typer.echo("Train the region first: pcast train", err=True)
        raise typer.Exit(1)
    ctx, store = RegionContext.load(region), RegionStore(region)
    meta = train_unet(ctx, store, block_fields(ctx, store), store.fine_dates(),
                      region_model_dir(region) / version / "m4", epochs=epochs)
    typer.echo(f"M4 saved into model version {version} (best val loss {meta['best_val_loss']:.4f}, "
               f"{meta['train_seconds']:.0f}s). Re-run `pcast evaluate {region}` to compare it.")


@app.command()
def evaluate(region: str = typer.Argument("demo"), version: str | None = None,
             mode: list[str] = typer.Option(["test", "forecast", "spatial", "station"],
                                            help="test | forecast | spatial | station (repeatable)")):
    """Validate models (held-out period, forecast mode, spatial CV, station CV) and write a report."""
    out = _evaluate_all(region, version, mode)
    typer.echo(f"Report: {out / 'summary.md'}")


@app.command()
def emulate(region: str, issue_date: str,
            out: Path = typer.Option(None, help="CSV path (default data/forecasts/...)"),
            noise: bool = typer.Option(True, help="Add lead-dependent forecast error"),
            seed: int = 0):
    """Create an emulated block forecast CSV (stand-in for an official IMD forecast)."""
    from .pipeline import emulate_block_forecast

    df = emulate_block_forecast(region, issue_date, noise=noise, seed=seed)
    out = out or paths().data / "forecasts" / f"{region}_block_forecast_{issue_date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    typer.echo(f"Wrote {out} ({len(df)} rows)")


@app.command()
def run(region: str, forecast_csv: Path, model: str | None = None,
        source: str = typer.Option("upload", help="official | upload | emulated")):
    """Downscale a block forecast CSV to panchayats and generate advisories."""
    from .ingest.block_forecast import ForecastValidationError
    from .pipeline import run_forecast

    try:
        res = run_forecast(region, forecast_csv, source=source, model_id=model)
    except ForecastValidationError as e:
        typer.echo("Forecast file is invalid:", err=True)
        for msg in e.errors:
            typer.echo(f"  - {msg}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Run {res.run_id}: issue {res.issue_date.date()}, model {res.model_id} "
               f"({res.model_version}), {res.n_gps} panchayats, {res.n_advisories} advisories")
    for w in res.warnings:
        typer.echo(f"  warning: {w}")


@app.command()
def demo(fast: bool = typer.Option(False, help="Quick training (for a smoke test)"),
         issue_date: str = "2024-07-10"):
    """End-to-end demo: synthetic region -> train -> evaluate -> emulated forecast run."""
    from .ingest.synthetic import generate
    from .models.train import train_region
    from .pipeline import emulate_block_forecast, run_forecast

    rcfg = RegionConfig.from_yaml(paths().resolve(DEMO_CONFIG))
    generate(rcfg)
    cfg = ModelConfig.load()
    train_region(rcfg.region_id, _fast(cfg) if fast else cfg)
    modes = ["test"] if fast else ["test", "forecast", "spatial", "station"]
    out = _evaluate_all(rcfg.region_id, None, modes)
    fc = emulate_block_forecast(rcfg.region_id, issue_date, seed=1)
    res = run_forecast(rcfg.region_id, fc, source="emulated")
    typer.echo("")
    typer.echo(f"Validation report: {out / 'summary.md'}")
    typer.echo(f"Forecast run:      {res.run_id} ({res.n_gps} panchayats, {res.n_advisories} advisories)")
    typer.echo("Start the API:     pcast serve   ->  http://localhost:8000/docs")


@app.command()
def fetch(region: str = typer.Argument("demo"), model: str | None = None,
          nwp_model: str | None = typer.Option(None, help="Open-Meteo model, e.g. ecmwf_ifs025, gfs_seamless")):
    """Fetch today's live NWP forecast (Open-Meteo) at block centroids and downscale it."""
    from .features.dataset import RegionContext
    from .ingest.nwp import nwp_block_forecast
    from .pipeline import run_forecast

    df = nwp_block_forecast(RegionContext.load(region), model=nwp_model)
    res = run_forecast(region, df, source="nwp", model_id=model)
    typer.echo(f"Run {res.run_id}: issue {res.issue_date.date()} (live NWP), {res.n_gps} panchayats, "
               f"{res.n_advisories} advisories")


@app.command()
def watch(region: str = typer.Argument("demo"), days: str = typer.Option(
          "tue,fri", help="Weekdays to run (AAS bulletins go out Tue/Fri)"),
          at: str = typer.Option("06:00", help="Local time HH:MM")):
    """Keep running and fetch + downscale the live forecast on the given weekdays and time.

    For production, prefer the OS scheduler (Windows Task Scheduler / cron) calling `pcast fetch`.
    """
    import time
    from datetime import datetime

    wanted = {d.strip().lower()[:3] for d in days.split(",")}
    hh, mm = (int(x) for x in at.split(":"))
    last = None
    typer.echo(f"Watching: {sorted(wanted)} at {at}. Ctrl+C to stop.")
    while True:
        now = datetime.now()
        key = now.date()
        if now.strftime("%a").lower() in wanted and (now.hour, now.minute) >= (hh, mm) and last != key:
            try:
                fetch(region)
            except Exception as e:  # noqa: BLE001  (keep watching after a failed fetch)
                typer.echo(f"[{now:%Y-%m-%d %H:%M}] fetch failed: {e}", err=True)
            last = key
        time.sleep(60)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Start the REST API."""
    import uvicorn

    uvicorn.run("panchayatcast.api.app:create_app", factory=True, host=host, port=port,
                reload=reload)


@app.command()
def regions():
    """List built regions and their latest model versions."""
    from .models.registry import latest_version
    from .store import RegionStore, list_regions

    for rid in list_regions():
        m = RegionStore(rid).meta
        typer.echo(f"{rid:12s} {m.get('name', ''):30s} source={m.get('data_source', m.get('source'))} "
                   f"gps={m.get('n_gps')} model={latest_version(rid)}")


@app.command()
def runs(region: str | None = None, limit: int = 20):
    """List recent forecast runs."""
    from .storage.db import Repository

    for r in Repository().list_runs(region, limit):
        typer.echo(f"{r['run_id']}  {r['region_id']:8s} issue={r['issue_date']} {r['status']:8s} "
                   f"{r['source']:9s} {r['model_id']} {r['model_version'] or ''}")


@app.command("export-bundle")
def export_bundle_cmd(
    region: str,
    out: Path = typer.Argument(None, help="Output .tar.xz (default: deploy/bundles/<region>.tar.xz)"),
    fine_start: str | None = typer.Option(None, help="Keep daily history from this date (YYYY-MM-DD)"),
    fine_end: str | None = typer.Option(None, help="Keep daily history up to this date"),
    no_fine: bool = typer.Option(False, "--no-fine", help="Leave out the daily history (no emulated runs)"),
):
    """Package a built region (latest model, completed runs, validation) for a server."""
    from .storage.bundle import export_bundle

    out = paths().resolve(out or f"deploy/bundles/{region}.tar.xz")
    m = export_bundle(region, out, fine_start=fine_start, fine_end=fine_end, include_fine=not no_fine)
    typer.echo(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB): model {m['model_version']}, "
               f"{len(m['runs'])} runs, history {m['fine_window'] or ('full' if m['fine_included'] else 'none')}")


@app.command("import-bundle")
def import_bundle_cmd(bundles: list[Path] = typer.Argument(..., help="Bundle .tar.xz files")):
    """Unpack region bundles into this installation (replaces earlier imports of the same runs)."""
    from .storage.bundle import import_bundle

    for b in bundles:
        m = import_bundle(paths().resolve(b))
        typer.echo(f"Imported {m['region_id']} from {b}: model {m['model_version']}, {len(m['runs'])} runs")


# ---- downloads --------------------------------------------------------------


def _bbox(config: Path):
    return RegionConfig.from_yaml(paths().resolve(config)).bbox


@download_app.command("static")
def dl_static(config: Path):
    """Copernicus DEM + ESA WorldCover for the region bbox (public, no account needed)."""
    from .ingest import download

    rc = RegionConfig.from_yaml(paths().resolve(config))
    st = rc.real.get("static", {})
    download.download_dem(rc.bbox, paths().resolve(st.get("dem", "data/raw/static/dem.tif")))
    download.download_worldcover(rc.bbox, paths().resolve(st.get("worldcover", "data/raw/static/worldcover.tif")))


@download_app.command("chirps")
def dl_chirps(config: Path, start: str, end: str,
              out: Path = Path("data/raw/chirps/rain_mm.nc")):
    """CHIRPS daily rainfall (0.05°) for the region bbox."""
    from .ingest import download

    download.download_chirps(_bbox(config), start, end, paths().resolve(out))


@download_app.command("era5land")
def dl_era5land(config: Path, start_year: int, end_year: int,
                raw_dir: Path = Path("data/raw/era5land/hourly"),
                out_dir: Path = Path("data/raw/era5land")):
    """ERA5-Land hourly -> daily tmax/tmin/RH/wind (needs a Copernicus CDS account)."""
    from .ingest import download

    files = download.download_era5land(_bbox(config), start_year, end_year, paths().resolve(raw_dir))
    download.process_era5land(files, paths().resolve(out_dir))


@download_app.command("era5cloud")
def dl_era5cloud(config: Path, start_year: int, end_year: int,
                 raw_dir: Path = Path("data/raw/era5/hourly"),
                 out: Path = Path("data/raw/era5/cloud_okta.nc")):
    """ERA5 total cloud cover -> daily okta (needs a Copernicus CDS account)."""
    from .ingest import download

    files = download.download_era5_cloud(_bbox(config), start_year, end_year, paths().resolve(raw_dir))
    download.process_era5_cloud(files, paths().resolve(out))


if __name__ == "__main__":
    app()
