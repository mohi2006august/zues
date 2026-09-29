# TECHNICAL.md: Technical Specification

**Version:** 0.2 · **Date:** 2026-09-26 · Core backend implemented (see §14 for status)

---

## 1. Problem formulation

Given a block-level forecast $y_b(v, d)$ for block $b$, variable $v$ and lead day $d$, estimate the panchayat-level value $\hat{y}_p(v, d)$ for every Gram Panchayat $p \in b$.

We estimate on a fine grid (~1 km, cells $i$) and aggregate:

$$
\hat{y}_i = \underbrace{I(y_{b}, y_{b'} \dots)_i}_{\text{interpolated coarse signal}} + \underbrace{f_v(\mathbf{x}_i, t)}_{\text{learned local correction}}
\qquad
\hat{y}_p = \sum_{i \in p} w_{i,p}\, \hat{y}_i
$$

- $I$: inverse-distance interpolation of block values (so values change smoothly across block borders)
- $\mathbf{x}_i$: static and seasonal features of cell $i$ (elevation, land cover, climatology…)
- $f_v$: model trained on historical data
- $w_{i,p}$: area fraction of panchayat $p$ that falls in cell $i$ (sums to 1 over $i$)

Everything is matrix algebra on "active" cells (cells overlapping at least one panchayat), see `src/panchayatcast/geometry.py`:

| Operation | Matrix | Shape |
|---|---|---|
| panchayat means | `W_gp` (sparse, rows sum to 1) | n_gp × n_cells |
| block means | `W_block` (sparse, rows sum to 1) | n_block × n_cells |
| interpolation | `idw` (dense, rows sum to 1) | n_cells × n_block |

Blocks are defined as the **union of their panchayats**, so block means are exactly the area-weighted means of panchayat values.

---

## 2. Variables

| Name | Unit | Modelled as | Notes |
|---|---|---|---|
| `rain_mm` | mm/day | `rain_mm` (two-stage) | 08:30–08:30 IST rain day |
| `tmax_c` | °C | `tmax_c` | Lapse-rate sensitive |
| `tmin_c` | °C | `tmin_c` | Lapse rate + valley cold-pooling |
| `rh_max_pct` | % | `rh_max_pct` | Morning RH |
| `rh_min_pct` | % | `rh_min_pct` | Afternoon RH |
| `wind_kmph` | km/h | `wind_u`, `wind_v` | Speed derived from aggregated u/v |
| `wind_dir_deg` | ° (from) | `wind_u`, `wind_v` | Meteorological convention |
| `cloud_okta` | okta 0–8 | `cloud_okta` | Mostly interpolated |

Definitions and valid ranges live in `src/panchayatcast/variables.py` (tied to the model contract, so kept in code).

---

## 3. Data sources

| Dataset | Resolution | Period | Use | Access | In code |
|---|---|---|---|---|---|
| **IMD block-level forecasts** | Block | Recent | Operational input; training pairs if an archive is provided | IMD / DAMU (via mentor) | CSV upload |
| **LGD Gram Panchayat polygons** | Polygons | 2024 | Real boundaries with LGD codes | india-geodata / bharatlas GeoParquet (CC0), range reads | `ingest/lgd.py` |
| **ERA5-Land / ERA5 via Open-Meteo** | 0.1° / 0.25° daily | 1950– / 1940– | Tmax, Tmin, RH (ERA5-Land); wind, cloud (ERA5), no account | Open-Meteo archive API (CC BY 4.0) | `ingest/openmeteo_archive.py` |
| **Live NWP forecast via Open-Meteo** | ~9–25 km | Today +7 d | Live block forecasts at block centroids (stand-in for the IMD feed) | Open-Meteo forecast API | `ingest/nwp.py` |
| **ERA5-Land** | 0.1° hourly | 1950– | Fine target: Tmax/Tmin, RH (from dewpoint), wind | Copernicus CDS (free account) | `pcast download era5land` |
| **ERA5** | 0.25° hourly | 1940– | Cloud cover | Copernicus CDS | `pcast download era5cloud` |
| **CHIRPS v2** | 0.05° daily | 1981– | Fine rainfall target | UCSB CHC (public) | `pcast download chirps` |
| **Copernicus DEM GLO-30** | 30 m (read at ~90 m) | Static | Elevation, slope, aspect, TPI | AWS open data (public COG) | `pcast download static` |
| **ESA WorldCover 2021** | 10 m (read at ~90 m) | Static | Land-cover fractions | AWS open data (public COG) | `pcast download static` |
| **Station data** | Point | Varies | **Validation ground truth** | IMD AWS/ARG; KSNDMC (Karnataka); TSDPS (Telangana) | CSV in config |
| **Admin boundaries** | Polygons | Current | GPs with **LGD codes** (blocks derived) | LGD + Bhuvan / state GIS; confirm with mentor | config |
| IMD gridded rain/temp | 0.25° / 1° | 1901– / 1951– | Cross-check | `imdlib` | planned |
| GFS / NCUM | 0.25° / 12 km | Live | Emulated block forecasts from real NWP | NOAA / NCMRWF | planned |
| NDVI, distance to coast | — | — | Extra features | MODIS / Sentinel-2 | planned |

### 3.1 Derived quantities (implemented in `ingest/download.py`)
- Hourly UTC data is shifted to IST (+5:30) and aggregated per calendar day.
- RH from 2 m temperature and dewpoint (Magnus formula); RH max/min are the daily max/min of hourly RH.
- Tmax/Tmin: daily max/min of hourly 2 m temperature.
- Wind: daily mean of 10 m u/v (converted to km/h).
- Cloud: daily mean total cloud cover × 8 → okta.

> **Limitation to state in the pitch:** with real data, the "fine" training truth is only as fine as ERA5-Land (~9 km) and CHIRPS (~5 km). Terrain/land-cover features are at ~1 km, and station observations are the point-scale check. Adding station data (or IMD high-resolution products) to training is a planned improvement.

---

## 4. Preprocessing (implemented)

1. **Region config** `configs/region.<id>.yaml`: bbox, grid resolution, date splits, languages, crop calendar, data paths. Templates: `region.demo.yaml` (synthetic), `region.example.yaml` (real).
2. **Target grid:** 0.01° (~1.1 km) regular lat/lon grid, rows north → south (`grid.py`).
3. **Boundaries:** GP polygons are read and fixed (`make_valid`); multi-row GPs are dissolved. Blocks = dissolve of GPs by `block_lgd`.
4. **Weights:** exact cell × GP polygon intersection with GeoPandas `overlay` in EPSG:7755 (equal area) → `W_gp`, `W_block`, majority block per cell, IDW matrix (power 2, min distance 1 km).
5. **Static layers** on the grid: elevation (area average of ~90 m DEM), slope, aspect sin/cos, dz/dx, dz/dy, TPI at ~3 km and ~15 km windows, land-cover fractions (crop/open vegetation, tree, built, water, bare), distance to water. Lat/lon are derived from the grid.
6. **Fine daily fields** → grid by linear interpolation (nearest-neighbour fill at edges); only dates common to all variables are kept.
7. **Emulated block inputs** (training): each day's fine field is area-averaged over each block → exactly what an official block forecast looks like.
8. **Stations:** mapped to the nearest active grid cell.

On-disk layout: `data/processed/<region>/` (see `store.py` docstring).

---

## 5. Models

### 5.1 Model ladder (all implemented)

| ID | Name | Description | Role |
|---|---|---|---|
| **M0** | Copy | every cell takes its block's value (current practice) | Baseline to beat |
| **M1** | Interpolation | IDW of block values from block centroids | Baseline 2 |
| **M2** | + lapse rate | M1 + $\Gamma_m (z_{interp} - z_i)$ for Tmax/Tmin; $\Gamma_m$ fitted per calendar month by least squares, clipped to [0, 12] °C/km | Cheap physics |
| **M3** | **Residual LightGBM** (default) | learns $y_i - I(y_b)_i$; quantile models for p10/p90 | Main model |
| **M3S** | M3 + station correction | LightGBM on station residuals (Tmax, Tmin, RH), shrink chosen by leave-stations-out CV (§5.7) | For real data with coarse gridded truth |
| **M4** | U-Net | CNN over the whole region grid (§5.6); optional, needs `torch` | Experimental; must beat M3 on validation |

### 5.2 M3 features (`features/dataset.py::make_features`)

| Group | Features |
|---|---|
| Day's block values | `blk_<var>` for all 8 modelled variables (context: e.g. block rain and cloud help Tmin) |
| Coarse signal | `interp`, `interp_minus_blk` |
| Terrain | elevation, slope, aspect sin/cos, dz/dx, dz/dy, TPI small/large, `dz_block` (vs block mean elevation), `dz_interp` |
| Orography × wind | `upslope` = block u·dz/dx + block v·dz/dy (windward lifting) |
| Surface | land-cover fractions, distance to water |
| Location / season | lat, lon, day-of-year sin/cos |
| Climatology | per cell & month, from training years only: mean(fine − interp) for additive variables, (Σfine+1)/(Σinterp+1) for rain |
| Rain only | log1p(block rain), log1p(interp) |

Training samples `train_rows` random (day, cell) pairs per variable (default 300k) from the train split; the validation split is used for early stopping and for tuning the rain threshold.

### 5.3 Rainfall (implemented)
1. **Occurrence:** LightGBM binary classifier, P(rain ≥ 0.1 mm), trained on cells of wet blocks only.
2. **Amount:** LightGBM regressor on log1p(rain) for wet cells.
3. **Combine:** rain = expm1(amount) if P ≥ τ else 0; τ tuned on validation to maximise CSI.
4. **Dry block ⇒ 0 mm** everywhere in the block (strict mode).
5. **Quantile mapping (auto-selected):** a mapping from predicted to observed wet-day amount quantiles is fitted on one half of the validation rows and kept only if, on the other half, it improves CSI at 15.6 and 64.5 mm without raising RMSE by more than 3%. The decision is recorded in the model metadata (`qm_used`).

### 5.4 Wind
u and v are modelled separately (residual GBM). Speed = √(u²+v²); direction (from) = atan2(−u, −v) mod 360°. Panchayat wind = speed/direction of the area-mean u/v.

### 5.5 Post-processing (`downscale/postprocess.py`)
1. **Block consistency** (default on; 3 iterations to handle cells that straddle block borders):
   - Additive variables: shift every cell of block $b$ by $-(\text{mean}_b(\hat y) - y_b)$.
   - Rain: if the block total is too high, scale down. If too low, scale up by at most `rain_max_scale` (3×) and spread the remaining deficit evenly over the block. This avoids piling a block's rain onto a few cells.
2. **Physical limits:** clip to valid ranges; enforce Tmin < Tmax and RHmin < RHmax (then clip again).
3. **Uncertainty:** p10/p90 from LightGBM quantile models (α = 0.1, 0.9), shifted/scaled with the value. Confidence per panchayat from the p90−p10 width: high / medium / low (thresholds per variable in `variables.py`).

### 5.6 M4 U-Net (`models/unet.py`, `pcast train-unet`)
- The whole region grid is one image (padded to a multiple of 8). 35 input channels per day: interpolated block field and own-block value for each of the 8 variables (16), 16 static layers, day-of-year sin/cos, active-cell mask.
- Output: 8 channels, the standardised residual (fine − interpolated) per variable; rain in log1p space.
- Architecture: 3-level U-Net (32/64/128 channels, BatchNorm + GELU), AdamW + cosine schedule, masked MSE, best epoch by validation loss.
- Memory-light: inputs are stored per active cell and assembled into images per batch.
- Output goes through the same post-processing (block consistency, limits) as every other model.

### 5.7 M3S station correction (`models/station.py`)
- For Tmax, Tmin, RH max and RH min: target = station observation − M3 prediction at the station's cell, over training-period days.
- Features: M3's `make_features` (without climatology) + the M3 prediction itself. Small, strongly regularised LightGBM (15 leaves, min 200 rows per leaf).
- **Leave-stations-out CV** (5 folds) picks a shrink factor per variable from {0, 0.25, 0.5, 0.75, 1}; 0 switches the correction off. Needs at least 8 stations.
- Purpose: with real data the gridded training truth is coarse (~9–25 km); stations carry the point-scale signal.

---

## 6. Storage

**Database** (`storage/db.py`, SQLAlchemy Core): **SQLite** by default at `data/panchayatcast.db`; set `PCAST_DATABASE_URL` for PostgreSQL. Geometries stay in the region's GeoJSON files (not in the DB), so the schema is identical on both backends.

| Table | Key columns |
|---|---|
| `forecast_runs` | run_id (PK), region_id, issue_date, source (`official`/`upload`/`emulated`), model_id, model_version, status (`queued`/`running`/`done`/`failed`), created_at, finished_at, notes |
| `block_forecasts` | (run_id, block_lgd, valid_date, variable) PK, lead_day, value |
| `panchayat_forecasts` | (run_id, gp_lgd, valid_date, variable) PK, lead_day, value, p10, p90, confidence, model_id |
| `advisories` | advisory_id (PK), run_id, gp_lgd, block_lgd, crop, crop_stage, valid_from, valid_to, rule_id, category, severity, text_en, text_local (JSON: `{lang: {text, reviewed}}`), params (JSON), status (`draft`/`approved`/`rejected`), edited_by, updated_at |
| `validation_metrics` | region_id, model_version, model_id, variable, level (`station`/`gp`), split, metric, threshold, value, n |

No migrations yet: after a schema change, delete `data/panchayatcast.db` in development.

**Files:**

| Path | Content |
|---|---|
| `data/processed/<region>/` | region.json, blocks.geojson, panchayats.geojson, static.nc, fine/<var>.nc, weights.npz, stations.csv, observations.parquet |
| `models/<region>/<version>/` | meta.json, m2.json (lapse rates), m3/<var>__<part>.txt (LightGBM), m3/clim.npz; `models/<region>/LATEST` |
| `data/runs/<run_id>/grid.nc` | gridded forecast (valid_date, lat, lon) incl. p10/p90 |
| `reports/<region>/<version>/` | metrics.csv, summary.md, skill_vs_copy.png, rain_csi.png |

---

## 7. REST API (FastAPI, `api/app.py`)

Base: `/api/v1`. Interactive docs at `/docs`. Start with `pcast serve`.
Write endpoints need header `X-API-Key` **if** env var `PCAST_API_KEY` is set. CORS origins: `PCAST_CORS_ORIGINS` (default `*`).

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + version |
| GET | `/variables` | Variable labels/units + IMD rain categories |
| GET | `/regions` | Built regions, model version, latest run id |
| GET | `/regions/{region_id}` | Region metadata, grid, available models |
| GET | `/regions/{region_id}/blocks` | Block polygons (GeoJSON, simplified) |
| GET | `/regions/{region_id}/panchayats?block_lgd=` | GP polygons (GeoJSON, simplified) |
| GET | `/regions/{region_id}/panchayats/search?q=` | Name search |
| GET | `/regions/{region_id}/panchayats/locate?lat=&lon=` | GP containing a point (farmer GPS), else nearest within 10 km |
| GET | `/regions/{region_id}/validation?model_version=` | Metrics (all splits: test, forecast by lead day, spatial_cv, station_cv) + skill summary |
| GET | `/regions/{region_id}/runs` · `/runs/latest` | Run list (with advisory counts) / latest completed run |
| POST | `/regions/{region_id}/runs` | Upload block forecast CSV (multipart `file`, optional `model_id`) → 202 + run_id; 422 with an error list if invalid |
| POST | `/regions/{region_id}/runs/emulate?issue_date=` | Demo: create a run from an emulated block forecast |
| POST | `/regions/{region_id}/runs/fetch?nwp_model=` | Live: today's NWP forecast (Open-Meteo) at block centroids → downscaled run (`source=nwp`) |
| GET | `/runs/{run_id}` | Run status + advisory counts by severity |
| GET | `/runs/{run_id}/map?variable=&valid_date=&level=gp\|block` | GeoJSON with value, p10, p90, confidence, block_value, diff_from_block + min/max |
| GET | `/runs/{run_id}/panchayats/{gp_lgd}/forecast` | 5-day forecast, all variables, with block values |
| GET | `/runs/{run_id}/panchayats/{gp_lgd}/advisories?crop=&lang=` | Advisories for one GP |
| GET | `/runs/{run_id}/advisories?block_lgd=&gp_lgd=&severity=&crop=&status=&lang=` | Advisory list |
| PATCH | `/advisories/{advisory_id}` | Edit `text_en` / set `status` / `edited_by` |
| GET | `/runs/{run_id}/compare?block_lgd=` | Per block/day/variable: block value vs GP min/max/std/range |
| GET | `/runs/{run_id}/export.csv` · `/export.geojson` | Downloads |
| GET | `/runs/{run_id}/bulletin.pdf?lang=en\|hi\|kn&block_lgd=` | PDF bulletin in English, Hindi or Kannada |
| GET | `/runs/{run_id}/sms.csv?lang=` | One SMS per panchayat (day-1 forecast + top advisory, segment count) for bulk gateways |
| GET | `/runs/{run_id}/raster/{variable}/{valid_date}.tif` | Gridded GeoTIFF |
| GET | `/` | The web dashboard (served from `web/dist` when built) |

**Example:** `GET /api/v1/runs/{run_id}/panchayats/99000001/forecast`
```json
{
  "gp_lgd": 99000001, "gp_name": "Demo GP G-01", "block_lgd": 990007, "block_name": "Demo Block G",
  "run": {"run_id": "…", "issue_date": "2024-07-10", "source": "emulated", "model_id": "M3", "status": "done"},
  "days": [
    {"valid_date": "2024-07-11", "lead_day": 1,
     "rain_mm": {"value": 18.2, "p10": 6.0, "p90": 31.5, "confidence": "medium", "block": 11.0},
     "tmax_c":  {"value": 25.0, "p10": 24.6, "p90": 25.5, "confidence": "high", "block": 25.1}}
  ]
}
```

Advisory JSON includes `text` (in the requested `lang`, English fallback), `machine_translated` (true until a translation file is marked reviewed) and `translations` (available languages).

---

## 8. Advisory engine (`advisory/`)

### 8.1 Inputs
- Panchayat forecast (5 days, all variables)
- Crop calendar: crop → sowing date → stage durations (`configs/crop_calendar/<district>.yaml`). A crop is included if it is in the field at any point in the forecast window.
- Rules: `configs/advisory_rules.yaml` (13 illustrative rules, to be reviewed with the DAMU/KVK)
- Translations: `configs/i18n/advisories.<lang>.yaml` (Hindi and Kannada provided, `reviewed: false`)

### 8.2 IMD rainfall categories (24 h)
| Category | mm |
|---|---|
| Very light | 0.1 – 2.4 |
| Light | 2.5 – 15.5 |
| Moderate | 15.6 – 64.4 |
| Heavy | 64.5 – 115.5 |
| Very heavy | 115.6 – 204.4 |
| Extremely heavy | ≥ 204.5 |

### 8.3 Rule format
```yaml
- id: HEAT_STRESS_FLOWERING
  category: temperature
  crops: [paddy, maize, cotton, soybean, groundnut, chickpea, sorghum]
  stages: [flowering]
  when: "max(tmax_c[d1:d3]) >= 36"
  severity: orange
  window: [d1, d3]
  params:
    tmax: "max(tmax_c[d1:d3])"
    day: "date_of_max(tmax_c, d1, d3)"
  text_en: "High temperature ({tmax:.0f}°C) on {day} during the flowering stage of {crop}. Apply light irrigation in the evening to reduce heat stress."
```
- **Expression language** (safe AST evaluator, **no `eval`**): variables, `d1..d5`, inclusive slices (`d1:d3` = 3 days), chained comparisons (`22 <= mean(tmax_c[d1:d3]) <= 32`), `and/or/not`, `in`, and the functions `max min mean sum any all count abs date_of_max first_date_below first_date_above`. Attribute access, lambdas, comprehensions and other functions are rejected when the rule file loads.
- **Templates:** `{name}` / `{name:.0f}` from `params`; `{crop}` and `{stage}` are built in (translated per language). Attribute/index placeholders are rejected.
- Crop-agnostic rules omit `crops`; stage filters require a crop list.
- Each advisory stores its `params`, so the PDF bulletin can merge many panchayats into one line with value ranges (e.g. "Rain of up to 18-54 mm…").

### 8.4 Translation
- Hindi and Kannada templates are written by hand and flagged `reviewed: false` (shown as "unreviewed translation") until a native speaker/agromet expert checks them.
- Dates inside regional text are localised ("30 Sep" → "30 ಸೆಪ್ಟೆಂಬರ್").
- PDF bulletins render in English, Hindi and Kannada with bundled Noto fonts (SIL OFL, `src/panchayatcast/assets/fonts/`) and HarfBuzz shaping (`uharfbuzz`), so conjuncts and vowel signs are correct.
- SMS export: one message per panchayat (day-1 forecast + most severe advisory) with a segment count (GSM-7 for English, Unicode for Hindi/Kannada).

---

## 9. Repository layout (actual)

```
.
├── README.md
├── pyproject.toml            # package + deps + ruff/pytest config; CLI entry point `pcast`
├── docs/                     # BRAIN, CLAUDE, PRD, DESIGN, SYSTEM_ARCHITECTURE, TECHNICAL
├── Dockerfile  docker-compose.yml  .dockerignore  render.yaml
├── deploy/bundles/           # pre-built real regions (`pcast export-bundle`), imported by the Docker build
├── configs/
│   ├── region.demo.yaml      # synthetic demo region
│   ├── region.dharwad.yaml   # REAL pilot: Dharwad district, Karnataka
│   ├── region.example.yaml   # template for another real district
│   ├── models.yaml           # training + post-processing settings
│   ├── advisory_rules.yaml
│   ├── crop_calendar/demo.yaml
│   └── i18n/advisories.{hi,kn}.yaml
├── src/panchayatcast/
│   ├── config.py  variables.py  grid.py  geometry.py  store.py  pipeline.py  cli.py
│   ├── ingest/               # synthetic, real, download (DEM/WorldCover/CHIRPS/ERA5), lgd (boundaries),
│   │                         # openmeteo_archive (ERA5-Land/ERA5), nwp (live forecast), block_forecast
│   ├── features/             # static.py (terrain), dataset.py (context, features, climatology)
│   ├── models/               # baselines (M0-M2), gbm (M3), station (M3S), unet (M4), registry, train
│   ├── downscale/            # engine.py, postprocess.py, aggregate.py
│   ├── validate/             # metrics.py, evaluate.py (4 modes), report.py
│   ├── advisory/             # expr.py, rules.py, crops.py, engine.py
│   ├── storage/db.py         # SQLAlchemy tables + Repository
│   ├── exports/              # tabular (CSV/GeoJSON), raster (GeoTIFF), bulletin (PDF en/hi/kn), sms
│   ├── assets/fonts/         # Noto fonts for Indic PDFs (OFL)
│   └── api/app.py            # FastAPI (+ serves web/dist)
├── web/                      # React + TypeScript + Vite + MapLibre dashboard and farmer view
├── tests/                    # unit + end-to-end tests on a tiny synthetic region
└── data/  models/  reports/  # generated, git-ignored
```

---

## 10. Validation protocol

All implemented in `validate/evaluate.py`; run with `pcast evaluate <region>` (all four) or `--mode`.

| Evaluation (`split`) | What it answers | How |
|---|---|---|
| `test` | How much does downscaling add? | Held-out year; block inputs are the true block means (isolates downscaling error) |
| `forecast` | Does it still help with an imperfect forecast? | Held-out year; block inputs get realistic error growing with lead day 1–5 (same error model as emulated runs); scored per lead day |
| `spatial_cv` | Does it work where it has never been trained? | Blocks split into folds; M3 retrained without each fold's cells **and without climatology**; scored only on held-out panchayats/stations |
| `station_cv` | Does the station correction really help? | M3S fitted without a fold of stations, scored on those stations in the test year |

- **Truths:** station observations (at the station's grid cell) and panchayat area-means of the fine field.
- **Metrics:** RMSE, MAE, bias, Pearson r; skill vs a reference = 1 − RMSE/RMSE_ref; rain POD, FAR, CSI, HSS at 2.5, 15.6, 64.5 mm; wind-direction angular MAE.
- **Splits:** temporal train / val / test from the region config (no shuffling across time).
- **Report:** `reports/<region>/<version>/summary.md` with charts (skill by variable, rain CSI, skill by lead day); metrics are stored in the DB (`lead_day` column for forecast mode) and shown on the dashboard's Validation page.
- **Still open:** full-chain test with real IMD block-forecast archives (needs IMD data).

---

## 11. Performance (measured on a laptop CPU)
- Synthetic demo (1,907 cells, 140 GPs, 7 years): generation ~15 s, full training ~2 min (8 variables × 4 LightGBM models, 300k rows each), evaluation of 4 models on 366 days ~20 s, one 5-day forecast run ~3 s.
- National scale (~3.3 M cells): aggregation is a sparse matrix multiply; LightGBM inference is batch and parallel per district.

---

## 12. Key libraries (in use)

| Purpose | Library |
|---|---|
| Arrays / rasters | numpy, xarray, netCDF4, rasterio |
| Vector | geopandas, shapely, pyproj, pyogrio |
| ML | lightgbm, scikit-learn, scipy; optional `torch` (M4) |
| API | fastapi, pydantic, sqlalchemy, uvicorn, python-multipart; optional `psycopg` (PostgreSQL) |
| Reports / exports | matplotlib, fpdf2 + uharfbuzz (Indic shaping), pyarrow |
| CLI / config | typer, pyyaml |
| Data access | requests, pyarrow (remote GeoParquet); optional `cdsapi` (ERA5), `imdlib` (IMD) |
| Frontend | React 19, TypeScript, Vite, MapLibre GL 6, Recharts |
| Quality | ruff, pytest, httpx, tsc |

---

## 13. Synthetic demo region

`configs/region.demo.yaml` → `pcast synth` generates a complete, clearly-labelled **synthetic** district (0.6° × 0.6°, 1 km grid, 7 blocks, 140 GPs, 30 stations, 2018–2024 daily weather). Its weather contains known local effects: lapse rate, valley cold-pooling on clear nights, lake/river cooling and humidity, urban warmth, south-facing slope warmth, windward orographic rain, and ridge wind speed-up. Annual rainfall is ~700–900 mm, mostly June–September.

Purpose: run and test the full pipeline before real data arrives, and show that the model recovers local structure. **Scores on it are not real-world skill.** Everything generated is tagged `data_source: synthetic`, and the PDF bulletin prints a DEMO warning.

---

## 14. Implementation status

| Area | Status |
|---|---|
| Synthetic region, grid, weights, static features | ✅ Done, tested |
| Real-region builder (boundaries, DEM, WorldCover, NetCDF fields, stations) | ✅ Done; used for the Dharwad pilot |
| Real LGD panchayat boundaries (remote GeoParquet, range reads) | ✅ Done (Dharwad: 145 GPs, 8 blocks) |
| Downloaders: DEM, WorldCover, CHIRPS (COG windows, parallel), ERA5-Land/ERA5 via Open-Meteo | ✅ Used for the pilot |
| Downloader: ERA5-Land via Copernicus CDS | ⚠️ Written, untested (needs a CDS account); Open-Meteo path used instead |
| M0, M1, M2, M3 (+ auto rain QM), M3S, registry | ✅ Done, tested |
| M4 U-Net | ✅ Done (optional, needs torch) |
| Block consistency, limits, uncertainty/confidence | ✅ Done, tested |
| Block forecast CSV validation | ✅ Done, tested |
| Live NWP forecast (Open-Meteo) + `pcast fetch` / `pcast watch` | ✅ Done, tested (mocked in tests, live-checked manually) |
| Advisory engine (rules, crop calendar, hi/kn templates, localised dates) | ✅ Done, tested; rules/translations need expert review |
| Validation: test, forecast mode, spatial CV, station CV + report | ✅ Done, tested |
| SQLite storage; CSV/GeoJSON/GeoTIFF/PDF (en/hi/kn)/SMS exports | ✅ Done, tested |
| REST API | ✅ Done, tested |
| Frontend: map, compare, advisories, validation, runs, farmer view (en/hi/kn) | ✅ Done, checked in the browser |
| Docker image | ✅ Builds and runs on Render (free plan, https://panchayatcast-cnvx.onrender.com) with the demo and the Dharwad bundle baked in |
| PostgreSQL (docker-compose) | ⚠️ Written, **untested** (no Docker on the dev machine) |
| Real IMD block-forecast archive, station data, NDVI feature | ⏳ Needs data access |

---

## 15. Frontend (`web/`)

React 19 + TypeScript + Vite, MapLibre GL 6 (OSM raster basemap, muted), Recharts. Built into `web/dist`, which `pcast serve` serves at `/`.

| Page | Route | What it shows |
|---|---|---|
| Forecast map | `#/` | Panchayat choropleth for any variable and day; Panchayat / Block / Difference views; hover tooltip (value, block value, difference, likely range, confidence); click → detail panel with 5-day cards, temperature and rain charts (with block values and p10–p90 band), panchayat-vs-block table and advisories (en/hi/kn); spread-inside-block card; alert list |
| Compare | `#/compare` | Swipe divider between block and panchayat maps (shared colour scale, synced pan/zoom) + per-block spread table |
| Advisories | `#/advisories` | Filter by block/crop/severity/status/text; approve, reject, edit; bulk approve; PDF bulletin (en/hi/kn) and SMS CSV |
| Validation | `#/validation` | All four evaluations; stations vs panchayat means; skill tiles, charts, RMSE and rain-event tables; synthetic-data warning |
| Runs | `#/runs` | Upload a block forecast CSV (with validation errors), emulated demo run, live NWP fetch; run list with downloads |
| Farmer view | `#/farmer/<gp>` | Mobile-first, English/Hindi/Kannada; find panchayat by search or GPS; today's forecast, "what to do" advisories by crop, read-aloud (browser speech), next 4 days |

Development: `npm run dev` in `web/` (proxies `/api` to `pcast serve` on :8000). Note: MapLibre 6's worker is bundled via `?worker&url` + `setWorkerUrl`. On this machine `web/node_modules` is a junction to `%USERPROFILE%\.cache\panchayatcast-web\node_modules` (outside OneDrive).

---

## 16. Real pilot: Dharwad district, Karnataka

Built with `pcast pilot configs/region.dharwad.yaml` (all open data, resumable), then `pcast train dharwad` and `pcast evaluate dharwad`. Model version `20260927-101426`; report in `reports/dharwad/<version>/summary.md`.

| Item | Value |
|---|---|
| Boundaries | 145 Gram Panchayats in 8 blocks, LGD-coded (LGD panchayat layer, CC0). Urban wards and town councils (Dharwad/Hubballi city, Navalgund TMC) are not GPs and are excluded (66 rows without GP codes). |
| Grid | 0.01° (~1.1 km), 3,497 active cells |
| Terrain / land cover | Copernicus DEM GLO-30, ESA WorldCover 2021 (elevation 467–738 m; ~87% crop/open vegetation) |
| Daily fields, 2021-07-01 to 2024-12-31 | Rain: CHIRPS (0.05°). Tmax/Tmin/RH: ERA5-Land (0.1°, 53 points). Wind/cloud: ERA5 (0.25°, 12 points), via Open-Meteo |
| Splits | train Jul 2021–Jun 2023 · val Jul–Dec 2023 · test 2024 |
| Stations | none yet (so no station CV and no M3S) |
| Training time | ~6 min on a laptop CPU (8 variables, 300k rows each); rain quantile mapping auto-selected (heavy-rain CSI on validation 0.45 → 0.55) |

**Results (held-out 2024, truth = panchayat means of the gridded fields):**

| Evaluation | Finding |
|---|---|
| Perfect block input | RMSE reduction vs copying the block value: Tmax −64%, Tmin −53%, RH max −49%, RH min −50%, wind −51%, cloud −34%, rain −17%. Heavy rain (≥ 64.5 mm) CSI 0.19 → 0.53 (interpolation alone reaches 0.54). |
| Forecast mode | Day 1: rain −5%, Tmax −6%, RH max −7%; ≈ 0% by days 3–5; wind and cloud ≈ −1% worse. The emulated block-forecast error is much larger than the sub-block variability in these smooth fields. |
| Spatial CV (no climatology) | Tmax −36%, Tmin −28%, RH −30%/−31% still; rain −12% and wind −17% are *worse* than plain interpolation (−16%, −25%). |

**Honest interpretation.** The fine "truth" is 5–25 km gridded data, so the measured within-block variability is small (e.g. Tmax ≈ 0.3 °C) and partly consists of those grids' fixed patterns, which the per-cell climatology learns easily. The real village-scale differences are larger and need station data (IMD AWS, KSNDMC) to measure; M3S is ready for that.

**Runs created:** an emulated run for 20 Jul 2024 (monsoon: panchayat day-1 rain ranges 12–144 mm across the district, up to 53 mm within one block), and a live NWP run for the current date. One `pcast run` (145 GPs, 5 days, advisories) takes ~25 s including start-up; `pcast fetch` ~1 min.

**M4 (U-Net) on the synthetic demo:** close to M3 but not better (rain RMSE 5.02 vs 4.60 at stations; temperature/RH within 0.01–0.02), best validation epoch 9 of 20, ~50 min CPU training. M3 stays the default.
