# PanchayatCast: API + dashboard in one image.
#   docker compose up --build            (API + UI on http://localhost:8000, PostgreSQL)
#   docker compose run --rm api pcast demo   (build the synthetic demo region + models)

# ---- 1. Build the dashboard ------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# ---- 2. Python runtime --------------------------------------------------------
FROM python:3.12-slim
# libgomp1: OpenMP runtime needed by LightGBM
# libexpat1: XML parser the GDAL bundled in the rasterio wheel links against; the slim
#            image no longer ships it ("ImportError: libexpat.so.1" otherwise)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PCAST_ROOT=/app PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[postgres]"
# Fail fast (with a clear message) if a native library is still missing.
RUN python -c "import rasterio, pyogrio, netCDF4, lightgbm, geopandas, uharfbuzz; print('native libs OK, GDAL', rasterio.__gdal_version__)"
COPY configs/ configs/
COPY --from=web /web/dist web/dist
# Bake data into the image so hosts with ephemeral disks (Render) serve it from the first
# request: the synthetic demo region (built here, ~5 min, ~210 MB) and the pre-built real
# regions in deploy/bundles (`pcast export-bundle`; Dharwad takes hours to rebuild).
# docker-compose sets this to 0 because it mounts ./data, ./models and ./reports instead.
ARG PCAST_BAKE_DEMO=1
RUN if [ "$PCAST_BAKE_DEMO" = "1" ]; then pcast demo --fast; fi
COPY deploy/bundles/ deploy/bundles/
RUN if [ "$PCAST_BAKE_DEMO" = "1" ]; then \
      for b in deploy/bundles/*.tar.xz; do if [ -f "$b" ]; then pcast import-bundle "$b"; fi; done; \
    fi
VOLUME ["/app/data", "/app/models", "/app/reports"]
EXPOSE 8000
# Hosts like Render inject $PORT; default to 8000 elsewhere.
CMD ["sh", "-c", "exec pcast serve --host 0.0.0.0 --port ${PORT:-8000}"]
