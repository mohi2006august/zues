# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project

**PanchayatCast**, built for SIH Problem Statement **26074**: downscale weather forecasts from **Block level to Gram Panchayat level** and turn them into **agro-meteorological advisories**.

## Read these first

All project docs live in `docs/` (this folder). The project overview is in the root [README.md](../README.md).

| File | Purpose |
|---|---|
| [BRAIN.md](BRAIN.md) | Problem interpretation, decisions, assumptions, open questions. **Start here.** |
| [PRD.md](PRD.md) | What we build and why: requirements, scope, success metrics |
| [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) | Components, data flow, deployment |
| [TECHNICAL.md](TECHNICAL.md) | Datasets, algorithms, storage, API, repo layout, **implementation status (§14)** |
| [DESIGN.md](DESIGN.md) | UI/UX, screens, colors, map styling |

## Current status

- **Implemented and tested:** synthetic + real region building (Dharwad pilot), models M0–M4, block-consistent downscaling, 4 validation modes, advisories (en/hi/kn), storage, exports (CSV/GeoJSON/GeoTIFF/PDF en-hi-kn/SMS), REST API, CLI, React dashboard + farmer view, live NWP fetch.
- **Untested:** Docker/PostgreSQL (no Docker on the dev machine), Copernicus CDS downloader.
- See [TECHNICAL.md §14](TECHNICAL.md) for the full status table.

## Environment

- The virtual environment lives **outside OneDrive** at `C:\Users\mohiuddin\.venvs\panchayatcast` (OneDrive locks files inside `.venv` and breaks pip). Do not create `.venv` inside the project folder.
- Frontend: `web/node_modules` is a **junction** to `C:\Users\mohiuddin\.cache\panchayatcast-web\node_modules` for the same reason. Keep it that way.
- The Browser preview uses `.claude/launch.json` (config `panchayatcast` → `pcast serve --port 8000`). Rebuild `web/dist` (`npm run build`) before previewing UI changes.
- Python 3.13. Install: `python -m venv %USERPROFILE%\.venvs\panchayatcast` then `pip install -e ".[dev]"`.
- `PCAST_ROOT` overrides the project root (tests use it). `PCAST_DATABASE_URL` switches the DB (default SQLite at `data/panchayatcast.db`). `PCAST_API_KEY` protects write endpoints.

## Tech stack (in use)

- **Pipeline/ML:** numpy, pandas, xarray, netCDF4, rasterio, geopandas, shapely, scipy, LightGBM
- **API:** FastAPI + Pydantic, SQLAlchemy Core, uvicorn
- **DB:** SQLite (dev) / PostgreSQL via `PCAST_DATABASE_URL`
- **Exports:** matplotlib, fpdf2
- **Frontend:** React 19 + TypeScript + Vite, MapLibre GL 6, Recharts (`web/`)
- **Optional:** torch (M4), psycopg (PostgreSQL), cdsapi (Copernicus)

## Commands

```bash
pcast demo [--fast]            # synthetic region -> train -> 4 validations -> emulated forecast run
pcast synth                    # (re)generate the synthetic demo region
pcast pilot configs/region.dharwad.yaml   # real district: download open data (resumable) + build
pcast train <region> [--fast]  # M2 + M3 (+ M3S when >= 8 stations)
pcast train-unet <region>      # optional M4 (needs torch) into the latest model version
pcast evaluate <region> [--mode test|forecast|spatial|station]   # writes reports/<region>/<version>/
pcast emulate demo 2024-07-10  # write an emulated block forecast CSV
pcast run <region> <forecast.csv>  # downscale a block forecast + advisories
pcast fetch <region>           # live NWP forecast (Open-Meteo) -> run
pcast watch <region> --days tue,fri --at 06:00   # keep fetching on AAS days
pcast serve                    # dashboard at http://127.0.0.1:8000, API docs at /docs
pcast regions | pcast runs     # list regions / runs
pcast export-bundle dharwad --fine-start 2024-06-01 --fine-end 2024-10-31   # -> deploy/bundles/dharwad.tar.xz
pcast import-bundle deploy/bundles/*.tar.xz   # load pre-built regions (the Docker build does this)
pcast download static|chirps|era5land|era5cloud <config> ...
pcast build-region configs/region.<id>.yaml
pytest -q                      # tests (a few minutes; builds a tiny synthetic region in a temp dir)
ruff check src tests           # lint
cd web && npm run dev          # frontend dev server (proxies /api to :8000)
cd web && npx tsc -p . && npm run build   # type-check + production build
```

## Rules

### Data
- `data/`, `models/`, `reports/` are **git-ignored**. Never commit datasets, rasters, model weights or credentials.
- `data/raw/` is **immutable**. Derived files go to `data/processed/`.
- **Never fabricate observations.** Synthetic or emulated data must be labelled (`data_source: synthetic`, run `source: emulated`).
- **LGD codes** are the primary keys for blocks and panchayats. Blocks are always derived as the union of their GPs.

### Units and conventions
- Rainfall: `mm/day`. Temperature: `°C`. RH: `%`. Wind speed: `km/h`. Wind direction: degrees, meteorological (direction wind comes FROM). Cloud: `okta`.
- Variable names: `rain_mm`, `tmax_c`, `tmin_c`, `rh_max_pct`, `rh_min_pct`, `wind_kmph`, `wind_dir_deg`, `cloud_okta` (+ internal `wind_u`, `wind_v`).
- Dates: ISO 8601 (`YYYY-MM-DD`). Time zone: **IST (Asia/Kolkata)**. Rain day = 08:30 IST to 08:30 IST (IMD convention).
- CRS: store in **EPSG:4326**. Use **EPSG:7755** (India NSF LCC) for area/distance calculations.
- Models handle wind as u/v components internally and convert to speed/direction only for output.

### Modelling
- Every new model must be compared with **M0 (copy block value)** and **M1 (interpolation)** on the **held-out test split** before it becomes the default.
- Temporal train/val/test splits only; climatology features come from training years only.
- Output must stay **block-consistent** (area-weighted GP mean = official block forecast) unless the config disables it.
- Model artifacts are versioned in `models/<region>/<version>/` with metadata; a region rebuild invalidates old models (checked on load).

### Code style
- Python: type hints, `ruff` for lint, `pytest` for tests. Functions stay small and pure where possible.
- Configuration (region, model settings, advisory rules, crop calendars, translations) lives in `configs/*.yaml`. Variable definitions live in `variables.py` because they are part of the model contract.
- Advisory rule expressions go through the safe evaluator in `advisory/expr.py`; never use `eval`.
- Comments explain *why*, not *what*.

### Documentation
- Record important decisions in the **BRAIN.md decision log** (D-xxx).
- Keep docs in sync when architecture, schemas or APIs change (TECHNICAL.md §6–§7, §14).
- All docs go in `docs/`. Only `README.md` stays in the repo root.
