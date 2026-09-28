"""Tests for evaluation modes, station correction, QM, live NWP parsing, SMS, i18n PDFs."""

import numpy as np
import pandas as pd
import pytest

from panchayatcast.advisory.engine import localize_dates
from panchayatcast.ingest.nwp import daily_from_hourly
from panchayatcast.models.gbm import apply_qm

# ---- small pure functions ------------------------------------------------------


def test_localize_dates():
    assert localize_dates("30 Sep ರಂದು ಮಳೆ", "kn") == "30 ಸೆಪ್ಟೆಂಬರ್ ರಂದು ಮಳೆ"
    assert localize_dates("11 Jul को", "hi") == "11 जुलाई को"
    assert localize_dates("11 Jul", "en") == "11 Jul"


def test_apply_qm_is_monotonic_and_extends():
    qm = ([1.0, 5.0, 10.0], [1.0, 8.0, 20.0])
    y = apply_qm(np.array([1.0, 3.0, 5.0, 10.0, 20.0]), qm)
    assert np.all(np.diff(y) > 0)
    assert y[2] == pytest.approx(8.0) and y[-1] == pytest.approx(40.0)  # scaled above top quantile


def test_daily_from_hourly_ist_aggregation():
    times = pd.date_range("2024-07-10 00:00", periods=48, freq="h")
    hourly = {
        "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
        "temperature_2m": [20 + (h % 24) / 2 for h in range(48)],
        "relative_humidity_2m": [60 + (h % 24) for h in range(48)],
        "precipitation": [1.0] * 48,
        "wind_speed_10m": [10.0] * 48,
        "wind_direction_10m": [270.0] * 48,
        "cloud_cover": [50.0] * 48,
    }
    d = daily_from_hourly(hourly)
    assert len(d) == 2
    assert d["rain_mm"].iloc[0] == pytest.approx(24.0)
    assert d["tmax_c"].iloc[0] == pytest.approx(31.5) and d["tmin_c"].iloc[0] == pytest.approx(20.0)
    assert d["wind_kmph"].iloc[0] == pytest.approx(10.0)
    assert d["wind_dir_deg"].iloc[0] == pytest.approx(270.0)
    assert d["cloud_okta"].iloc[0] == pytest.approx(4.0)


# ---- models and evaluation on the tiny region ------------------------------------------


def test_station_correction_in_bundle(trained):
    from panchayatcast.features.dataset import RegionContext
    from panchayatcast.models.registry import ModelBundle

    b = ModelBundle.load(RegionContext.load("tiny"))
    assert "M3S" in b.models
    rep = b.meta["m3s"]["report"]
    assert set(rep) <= {"tmax_c", "tmin_c", "rh_max_pct", "rh_min_pct"} and rep
    for r in rep.values():
        assert r["cv_rmse_m3s"] <= r["cv_rmse_m3"] + 1e-9  # shrink 0 is always allowed


def test_forecast_mode_evaluation(trained, repo):
    from panchayatcast.validate.evaluate import evaluate_forecast_mode

    df = evaluate_forecast_mode("tiny", leads=(1, 5), repo=repo, log=lambda *a: None)
    assert set(df["lead_day"].dropna().astype(int)) == {1, 5}
    sk = df[(df.metric == "skill_vs_M0") & (df.model_id == "M3") & (df.level == "gp") & (df.variable == "tmax_c")]
    assert (sk["value"] > 0).all()  # downscaling still helps with an imperfect block forecast
    stored = repo.metrics("tiny")
    assert (stored["split"] == "forecast").any() and stored["lead_day"].notna().any()


def test_spatial_cv_evaluation(trained, repo):
    from panchayatcast.validate.evaluate import evaluate_spatial_cv

    df = evaluate_spatial_cv("tiny", folds=3, repo=repo, train_rows=6000, log=lambda *a: None)
    rm = df[(df.metric == "rmse") & (df.level == "gp") & (df.variable == "tmax_c")]
    by = dict(zip(rm.model_id, rm.value))
    assert set(by) >= {"M0", "M3"}
    assert np.isfinite(by["M3"])


def test_station_cv_evaluation(trained, repo):
    from panchayatcast.validate.evaluate import evaluate_station_cv

    df = evaluate_station_cv("tiny", folds=4, repo=repo, log=lambda *a: None)
    assert set(df["model_id"]) >= {"M3", "M3S"}


def test_report_with_all_sections(trained, repo):
    from panchayatcast.store import RegionStore
    from panchayatcast.validate.evaluate import evaluate_forecast_mode, evaluate_region
    from panchayatcast.validate.report import write_report

    frames = {
        "test": evaluate_region("tiny", repo=repo, log=lambda *a: None),
        "forecast": evaluate_forecast_mode("tiny", leads=(1, 3, 5), repo=repo, log=lambda *a: None),
    }
    out = write_report("tiny", trained, frames, RegionStore("tiny").meta)
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "Forecast mode" in text and (out / "forecast_leads.png").exists()


# ---- exports and API ------------------------------------------------------------


def test_sms_messages(forecast_run, repo):
    from panchayatcast.exports.sms import sms_messages
    from panchayatcast.features.dataset import RegionContext

    res, _ = forecast_run
    ctx = RegionContext.load("tiny")
    en = sms_messages(ctx, repo, res.run_id, "en")
    kn = sms_messages(ctx, repo, res.run_id, "kn")
    assert len(en) == len(kn) == 18
    assert en["message"].str.contains("PanchayatCast").all()
    assert (kn["sms_segments"] >= 1).all() and kn["message"].str.contains("ಮಳೆ").all()


@pytest.mark.parametrize("lang", ["en", "hi", "kn"])
def test_bulletin_languages(forecast_run, repo, lang):
    from panchayatcast.exports.bulletin import bulletin_pdf
    from panchayatcast.features.dataset import RegionContext

    res, _ = forecast_run
    assert bulletin_pdf(RegionContext.load("tiny"), repo, res.run_id, lang=lang)[:4] == b"%PDF"


@pytest.fixture(scope="module")
def client(forecast_run, repo):
    from fastapi.testclient import TestClient

    from panchayatcast.api.app import create_app

    return TestClient(create_app(repo))


def test_api_locate_sms_and_fetch(client, forecast_run, monkeypatch):
    from panchayatcast.api import app as app_module
    from panchayatcast.features.dataset import RegionContext

    ctx = RegionContext.load("tiny")
    c = ctx.panchayats.to_crs("EPSG:7755").geometry.iloc[0].centroid
    pt = pd.Series([c], dtype=object)
    import geopandas as gpd

    ll = gpd.GeoSeries(pt, crs="EPSG:7755").to_crs("EPSG:4326").iloc[0]
    r = client.get("/api/v1/regions/tiny/panchayats/locate", params={"lat": ll.y, "lon": ll.x})
    assert r.status_code == 200 and r.json()["gp_lgd"] == int(ctx.panchayats["gp_lgd"].iloc[0])
    assert client.get("/api/v1/regions/tiny/panchayats/locate", params={"lat": 0, "lon": 0}).status_code == 404

    res, df = forecast_run
    sms = client.get(f"/api/v1/runs/{res.run_id}/sms.csv", params={"lang": "hi"})
    assert sms.status_code == 200 and "बारिश" in sms.content.decode("utf-8-sig")

    # Live NWP fetch with the provider mocked: reuse the emulated forecast as "today's" forecast.
    fake = df.copy()
    monkeypatch.setattr(app_module, "nwp_block_forecast", lambda ctx, model=None: fake)
    r = client.post("/api/v1/regions/tiny/runs/fetch")
    assert r.status_code == 202
    assert client.get(f"/api/v1/runs/{r.json()['run_id']}").json()["source"] == "nwp"


# ---- deployment bundles -----------------------------------------------------------


def test_bundle_round_trip(forecast_run, repo, project, tmp_path, monkeypatch):
    import shutil

    from sqlalchemy import func, select

    from panchayatcast.models.registry import available_models, latest_version
    from panchayatcast.pipeline import emulate_block_forecast
    from panchayatcast.storage.bundle import export_bundle, import_bundle
    from panchayatcast.storage.db import Repository, advisories, validation_metrics
    from panchayatcast.store import RegionStore

    res, _ = forecast_run
    bundle = tmp_path / "tiny.tar.xz"
    m = export_bundle("tiny", bundle, repo=repo, fine_start="2023-07-01", fine_end="2023-08-31")
    assert res.run_id in m["runs"] and m["fine_window"] == ["2023-07-01", "2023-08-31"]
    version = latest_version("tiny")

    # A fresh installation: empty root and database.
    root = tmp_path / "server"
    shutil.copytree(project / "configs", root / "configs")
    (root / "pyproject.toml").write_text("[project]\nname='t'\n")
    monkeypatch.setenv("PCAST_ROOT", str(root))
    fresh = Repository(f"sqlite:///{(root / 'server.db').as_posix()}")
    import_bundle(bundle, repo=fresh)
    import_bundle(bundle, repo=fresh)  # re-importing replaces rather than duplicates

    assert RegionStore("tiny").exists()
    got_version, ids = available_models("tiny")
    assert got_version == version and {"M0", "M1", "M2", "M3"} <= set(ids)
    assert fresh.get_run(res.run_id)["status"] == "done"
    count = lambda t: fresh.engine.connect().execute(select(func.count()).select_from(t)).scalar()  # noqa: E731
    assert count(advisories) == m["rows"]["advisories"] > 0
    assert count(validation_metrics) == m["rows"]["validation_metrics"]
    assert len(emulate_block_forecast("tiny", "2023-07-10")) > 0
    with pytest.raises(ValueError, match="No history"):
        emulate_block_forecast("tiny", "2023-03-01")
