"""Portable region bundles: ship a built region to a server that cannot rebuild it.

A real region takes hours to download and train (and hits API rate limits), so hosts
like Render import a pre-built bundle instead. A bundle is a tar.xz with:

    manifest.json
    files/...              region files (boundaries, static layers, weights), the latest
                           model version, its validation report and the run grids,
                           laid out relative to the project root
    db/<table>.jsonl       database rows for the region: runs, forecasts, advisories,
                           and the validation metrics of the bundled model version

The daily fine-resolution history can be cut to a window (`fine_start`/`fine_end`):
serving only needs it for emulated forecasts, and the full archive is large.
"""

from __future__ import annotations

import io
import json
import shutil
import tarfile
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

import xarray as xr
from sqlalchemy import JSON, Date, DateTime, Table, delete, insert, select

from ..config import paths
from ..models.registry import latest_version, region_model_dir
from ..store import RegionStore
from .db import (
    Repository,
    advisories,
    block_forecasts,
    forecast_runs,
    panchayat_forecasts,
    validation_metrics,
)

FORMAT = 1
RUN_TABLES = (block_forecasts, panchayat_forecasts, advisories)


def _json_default(v):
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    raise TypeError(f"Cannot serialise {type(v).__name__}")


def _dump_rows(tar: tarfile.TarFile, table: Table, rows: list[dict]) -> None:
    data = "\n".join(json.dumps(r, default=_json_default, ensure_ascii=False) for r in rows).encode()
    info = tarfile.TarInfo(f"db/{table.name}.jsonl")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _load_rows(table: Table, text: str) -> list[dict]:
    """Parse JSONL rows back into the column types the database expects."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for col in table.columns:
            v = r.get(col.name)
            if v is None or isinstance(col.type, JSON):
                continue
            if isinstance(col.type, DateTime):
                r[col.name] = datetime.fromisoformat(v)
            elif isinstance(col.type, Date):
                r[col.name] = date.fromisoformat(v)
        rows.append(r)
    return rows


def export_bundle(region_id: str, out: Path, repo: Repository | None = None,
                  fine_start: str | None = None, fine_end: str | None = None,
                  include_fine: bool = True) -> dict:
    """Write a bundle for `region_id` (latest model version, all completed runs)."""
    repo = repo or Repository()
    p = paths()
    store = RegionStore(region_id)
    if not store.exists():
        raise FileNotFoundError(f"Region {region_id!r} is not built")
    version = latest_version(region_id)

    with repo.engine.connect() as c:
        runs = [dict(r) for r in c.execute(
            select(forecast_runs).where(forecast_runs.c.region_id == region_id,
                                        forecast_runs.c.status == "done")).mappings()]
        run_ids = [r["run_id"] for r in runs]
        per_table = {
            t.name: [dict(r) for r in c.execute(select(t).where(t.c.run_id.in_(run_ids))).mappings()]
            for t in RUN_TABLES
        } if run_ids else {t.name: [] for t in RUN_TABLES}
        q = select(validation_metrics).where(validation_metrics.c.region_id == region_id)
        if version:
            q = q.where(validation_metrics.c.model_version == version)
        metrics = [{k: v for k, v in dict(r).items() if k != "id"} for r in c.execute(q).mappings()]

    out.parent.mkdir(parents=True, exist_ok=True)
    fine_window = None
    with tarfile.open(out, "w:xz") as tar, tempfile.TemporaryDirectory() as tmp:
        def add(path: Path) -> None:
            tar.add(path, arcname=f"files/{path.relative_to(p.root).as_posix()}")

        for f in sorted(store.dir.iterdir()):
            if f.name != "fine":
                add(f)
        if include_fine and (store.dir / "fine").is_dir():
            for f in sorted((store.dir / "fine").glob("*.nc")):
                if fine_start or fine_end:
                    # Cut the history to the window and re-compress it into a temp copy.
                    with xr.open_dataarray(f) as da:
                        cut = da.sel(time=slice(fine_start, fine_end)).load()
                    if not cut.sizes.get("time"):
                        raise ValueError(f"{f.name}: no data between {fine_start} and {fine_end}")
                    fine_window = [str(cut.time.values[0])[:10], str(cut.time.values[-1])[:10]]
                    tmp_f = Path(tmp) / f.name
                    cut.to_netcdf(tmp_f, encoding={cut.name: {"zlib": True, "complevel": 6}})
                    tar.add(tmp_f, arcname=f"files/{f.relative_to(p.root).as_posix()}")
                else:
                    add(f)
        if version:
            add(region_model_dir(region_id) / version)
            add(region_model_dir(region_id) / "LATEST")
            report = p.reports / region_id / version
            if report.is_dir():
                add(report)
        for rid in run_ids:
            d = p.runs / rid
            if d.is_dir():
                add(d)

        _dump_rows(tar, forecast_runs, runs)
        for t in RUN_TABLES:
            _dump_rows(tar, t, per_table[t.name])
        _dump_rows(tar, validation_metrics, metrics)

        manifest = {
            "format": FORMAT, "region_id": region_id, "model_version": version,
            "runs": run_ids, "fine_window": fine_window, "fine_included": include_fine,
            "created_at": datetime.now(UTC).isoformat(),
            "rows": {"forecast_runs": len(runs), **{k: len(v) for k, v in per_table.items()},
                     "validation_metrics": len(metrics)},
        }
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return manifest


def import_bundle(src: Path, repo: Repository | None = None) -> dict:
    """Unpack a bundle into the project root and load its rows (replacing earlier copies)."""
    repo = repo or Repository()
    p = paths()
    with tarfile.open(src, "r:xz") as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read())
        if manifest.get("format") != FORMAT:
            raise ValueError(f"Unsupported bundle format {manifest.get('format')}")
        region_id = manifest["region_id"]
        with tempfile.TemporaryDirectory() as tmp:
            members = [m for m in tar.getmembers() if m.name.startswith("files/")]
            # "data" filter: no absolute paths, no escaping the target, no special files.
            tar.extractall(tmp, members=members, filter="data")
            files = Path(tmp) / "files"
            for f in sorted(x for x in files.rglob("*") if x.is_file()):
                dest = p.root / f.relative_to(files)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        rows = {t.name: _load_rows(t, tar.extractfile(f"db/{t.name}.jsonl").read().decode())
                for t in (forecast_runs, *RUN_TABLES, validation_metrics)}

    run_ids = manifest["runs"]
    with repo.engine.begin() as c:
        if run_ids:
            for t in RUN_TABLES:
                c.execute(delete(t).where(t.c.run_id.in_(run_ids)))
            c.execute(delete(forecast_runs).where(forecast_runs.c.run_id.in_(run_ids)))
        c.execute(delete(validation_metrics).where(
            validation_metrics.c.region_id == region_id,
            validation_metrics.c.model_version == manifest["model_version"]))
        for t in (forecast_runs, *RUN_TABLES, validation_metrics):
            batch = rows[t.name]
            for i in range(0, len(batch), 2000):
                c.execute(insert(t), batch[i:i + 2000])
    return manifest
