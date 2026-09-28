# PanchayatCast

**Block-to-Panchayat weather forecast downscaling for agro-meteorological advisories**

Smart India Hackathon · Problem Statement **26074**

> *Downscaling of weather forecast from Block level to Panchayat level: Inferring high-resolution plots/ data/ information from low-resolution plot /data /information /variables for agro-meteorological advisory services.*

---

## The problem

India's Agromet Advisory Service issues weather forecasts at **block level**. One block can hold 30–100 Gram Panchayats, and weather inside it varies with terrain, land cover and local rain patterns. Today every village in a block gets the **same forecast and the same advice**.

## Our solution

PanchayatCast takes the official block forecast and produces a **forecast for every Gram Panchayat**, then turns it into **crop-specific advisories**.

1. **Learn** each area's local weather signature (terrain, land cover, water bodies, climatology) from 10+ years of historical data.
2. **Downscale** every new block forecast to a ~1 km grid and aggregate it to panchayat boundaries.
3. **Validate** against weather stations and show the improvement over "copy the block value".
4. **Advise:** a rule engine turns panchayat forecasts into crop advisories (in local languages).
5. **Deliver** results as **plots** (maps, charts), **data** (CSV, GeoJSON, GeoTIFF, API) and **information** (advisory text, PDF bulletins).

```
Block forecast ──► Downscaling engine ──► Panchayat forecast ──► Advisory engine ──► Dashboard / exports / farmer view
                   (terrain-aware ML)      (rain, Tmax, Tmin,      (crop + stage rules)
                                            RH, wind, cloud)
```

**Variables:** rainfall, max/min temperature, max/min relative humidity, wind speed and direction, cloud cover · **Lead time:** 5 days

---

## Documentation

| Doc | What's inside |
|---|---|
| [docs/BRAIN.md](docs/BRAIN.md) | **Start here.** Problem breakdown, requirements, decisions, assumptions, open questions |
| [docs/PRD.md](docs/PRD.md) | Product requirements: users, user stories, features, success metrics, phases |
| [docs/SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) | Components, data flow, sequence diagrams, deployment |
| [docs/TECHNICAL.md](docs/TECHNICAL.md) | Datasets, models, maths, DB schema, API, advisory rules, validation |
| [docs/DESIGN.md](docs/DESIGN.md) | UI/UX: screens, colours, typography, accessibility |
| [docs/CLAUDE.md](docs/CLAUDE.md) | Conventions and rules for contributors and Claude Code |

---

## Quick start

Requires Python 3.11+. On Windows, keep the virtual environment **outside OneDrive** (OneDrive locks its files).

```bash
python -m venv %USERPROFILE%\.venvs\panchayatcast
%USERPROFILE%\.venvs\panchayatcast\Scripts\activate
pip install -e ".[dev]"
cd web && npm install && npm run build && cd ..   # build the dashboard (Node 20+)

pcast demo      # synthetic demo district -> train -> 4 validations -> forecast run (~15 min)
pcast serve     # dashboard at http://127.0.0.1:8000  ·  API docs at /docs
pytest -q       # tests
```

Other useful commands:

```bash
pcast fetch demo                          # today's live NWP forecast -> panchayat forecast
pcast pilot configs/region.dharwad.yaml   # REAL district: download open data + build (~2 h, resumable)
pcast train dharwad && pcast evaluate dharwad
pcast train-unet demo                     # optional deep-learning model (pip install torch)
```

**Deploy (Render):** Dashboard → New → Blueprint → this repo (`render.yaml`). The Docker build bakes in the synthetic demo and imports the pre-built real regions in `deploy/bundles/` (made with `pcast export-bundle dharwad --fine-start 2024-06-01 --fine-end 2024-10-31`), so the site serves the Dharwad pilot without re-downloading anything. The free plan's disk is ephemeral: new runs and advisory edits reset on restart.

> The **demo** district is synthetic (generated terrain, boundaries and weather): it proves the pipeline end to end, and its scores are **not** real-world skill. The **Dharwad pilot** uses real LGD panchayat boundaries, real terrain and land cover, and real reanalysis/satellite weather (see [docs/TECHNICAL.md §16](docs/TECHNICAL.md)).

## Real pilot: Dharwad district, Karnataka

145 LGD-coded Gram Panchayats in 8 blocks, built entirely from open data (CHIRPS rain, ERA5-Land/ERA5, Copernicus DEM, ESA WorldCover), July 2021 to December 2024. On the held-out year 2024 (truth: panchayat means of those gridded fields), LightGBM cuts the error of copying the block value by:

| Max temp | Min temp | Humidity | Wind | Rain | Heavy-rain CSI |
|---:|---:|---:|---:|---:|---:|
| −64% | −53% | ≈ −50% | −51% | −17% | 0.19 → 0.53 |

Honest caveats: with realistic forecast errors the gain is small (≤ 7% on day 1, ≈ 0 by day 3–5); in blocks never seen in training, temperature and humidity still improve but rain and wind do no better than interpolation; and the gridded truth (5–25 km) understates village-scale differences, so station data is the next test. Full details: [docs/TECHNICAL.md §16](docs/TECHNICAL.md).

## What you get

- **Dashboard:** panchayat forecast map for every variable and day, block-vs-panchayat compare slider, panchayat detail with charts and uncertainty, advisory review (approve/edit), validation results, forecast upload / live fetch.
- **Farmer view:** phone-friendly page in English, Hindi or Kannada: find your panchayat by name or GPS, today's forecast, what to do, read-aloud.
- **Exports:** CSV, GeoJSON, GeoTIFF, PDF bulletins (English/Hindi/Kannada), bulk-SMS list.
- **Validation:** held-out year, forecast mode by lead day, spatial cross-validation (unseen places), station cross-validation.

## How it works

| Model | What it does |
|---|---|
| M0 | Copy the block value to every panchayat (current practice, the baseline to beat) |
| M1 | Smooth interpolation between block values |
| M2 | M1 + temperature correction for elevation (lapse rate) |
| **M3** | **LightGBM** learns each location's difference from the block signal from terrain, land cover, water, season and climatology (default) |
| M3S | M3 + a correction learned from weather stations (switched on only where cross-validation shows it helps) |
| M4 | U-Net deep-learning model over the whole grid (optional, experimental) |

The output is **block-consistent**: the area-weighted average over a block's panchayats equals the official block forecast. Each value has a p10–p90 range and a confidence level.

## Tech stack

| Layer | Tools |
|---|---|
| Data & ML | Python, numpy, xarray, rasterio, geopandas, LightGBM, PyTorch (optional) |
| Backend | FastAPI, SQLAlchemy (SQLite by default, PostgreSQL supported) |
| Frontend | React, TypeScript, Vite, MapLibre GL, Recharts |
| Exports | CSV, GeoJSON, GeoTIFF, PDF (fpdf2 + Noto fonts + HarfBuzz), charts (matplotlib) |
| Deployment | `pcast serve` on one machine; Dockerfile + docker-compose (PostgreSQL) provided |

## Data sources

IMD block forecasts (CSV upload) · LGD Gram Panchayat boundaries · Copernicus DEM · ESA WorldCover · CHIRPS rainfall · ERA5-Land / ERA5 (via Open-Meteo or Copernicus CDS) · live NWP forecasts (Open-Meteo) · IMD and state AWS stations (when available). See [docs/TECHNICAL.md §3](docs/TECHNICAL.md).

---

## Project structure

```
.
├── README.md
├── pyproject.toml  Dockerfile  docker-compose.yml
├── docs/                 # all project documentation (incl. DEMO_SCRIPT.md for the pitch)
├── configs/              # region configs (demo, dharwad, example), models, advisory rules, crop calendar, translations
├── src/panchayatcast/    # ingest, features, models, downscale, validate, advisory, storage, exports, api, cli
├── web/                  # dashboard + farmer view (React)
├── tests/                # unit + end-to-end tests
└── data/ models/ reports/  # generated, git-ignored
```

## Status

- ✅ Backend, models M0–M4, four validation modes, advisories (en/hi/kn), exports incl. Indic PDFs and SMS, REST API, CLI.
- ✅ Dashboard and farmer view.
- ✅ Real-data pilot for Dharwad district (real boundaries, terrain, satellite/reanalysis weather).
- ⏳ Needs data access: official IMD block-forecast archive and station observations; expert review of advisory rules and translations.
- Details: [docs/TECHNICAL.md §14](docs/TECHNICAL.md).

## Team

_TBD_
