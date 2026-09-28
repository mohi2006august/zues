"""Versioned model storage.

    models/<region_id>/LATEST                 name of the default version
    models/<region_id>/<version>/meta.json    training metadata + validation report
    models/<region_id>/<version>/m2.json      monthly lapse rates
    models/<region_id>/<version>/m3/<var>__<part>.txt   LightGBM boosters
    models/<region_id>/<version>/m3/clim.npz  per-cell climatology per variable
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

from ..config import paths
from ..features.dataset import RegionContext
from ..variables import MODELLED_VARS
from .base import Downscaler
from .baselines import CopyModel, IDWModel, LapseRateModel
from .gbm import GBMModel, VarBoosters
from .station import StationCorrectedModel, StationCorrection
from .unet import HAVE_TORCH, load_unet

MODEL_IDS = ["M0", "M1", "M2", "M3", "M3S", "M4"]


def region_model_dir(region_id: str) -> Path:
    return paths().models / region_id


def latest_version(region_id: str) -> str | None:
    p = region_model_dir(region_id) / "LATEST"
    return p.read_text().strip() if p.exists() else None


def save_bundle(
    region_id: str,
    gbm: GBMModel,
    gammas: dict[str, np.ndarray],
    extra_meta: dict,
    set_latest: bool = True,
    station: StationCorrection | None = None,
) -> str:
    version = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    d = region_model_dir(region_id) / version
    (d / "m3").mkdir(parents=True, exist_ok=True)
    if station is not None:
        (d / "m3s").mkdir(exist_ok=True)
        for var, b in station.boosters.items():
            b.save_model(str(d / "m3s" / f"{var}.txt"))
        extra_meta = {**extra_meta, "m3s": {"shrink": station.shrink, "features": station.features,
                                            "report": station.report}}

    (d / "m2.json").write_text(json.dumps({v: g.tolist() for v, g in gammas.items()}, indent=2))
    clim = {}
    var_meta = {}
    for var, vb in gbm.vars.items():
        for part, booster in vb.parts().items():
            booster.save_model(str(d / "m3" / f"{var}__{part}.txt"))
        if vb.clim is not None:
            clim[var] = vb.clim
        var_meta[var] = {
            "kind": vb.kind,
            "features": vb.features,
            "rain_threshold": vb.rain_threshold,
            "qm": vb.qm,
            "parts": list(vb.parts()),
            "report": vb.report,
        }
    np.savez_compressed(d / "m3" / "clim.npz", **clim)

    meta = {
        "version": version,
        "region_id": region_id,
        "created_at": datetime.now(UTC).isoformat(),
        "wet_threshold": gbm.wet,
        "variables": var_meta,
        **extra_meta,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if set_latest:
        (region_model_dir(region_id) / "LATEST").write_text(version)
    return version


def available_models(region_id: str, version: str | None = None) -> tuple[str | None, list[str]]:
    """(version, model ids) a bundle would offer, read from metadata without loading weights.

    Loading every LightGBM booster takes seconds and ~100 MB, far too much for a request
    that only lists what is available (small hosts time out or run out of memory).
    """
    version = version or latest_version(region_id)
    ids = ["M0", "M1", "M2"]
    if version is None:
        return None, ids
    d = region_model_dir(region_id) / version
    meta = json.loads((d / "meta.json").read_text())
    ids.append("M3")
    if meta.get("m3s"):
        ids.append("M3S")
    if HAVE_TORCH and (d / "m4" / "unet.pt").exists():
        ids.append("M4")
    return version, ids


class ModelBundle:
    def __init__(self, region_id: str, version: str | None, meta: dict, models: dict[str, Downscaler]):
        self.region_id = region_id
        self.version = version
        self.meta = meta
        self.models = models

    def get(self, model_id: str) -> Downscaler:
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not available (have {sorted(self.models)})")
        return self.models[model_id]

    @property
    def label(self) -> str:
        return f"{self.version or 'untrained'}"

    @classmethod
    def load(cls, ctx: RegionContext, version: str | None = None) -> ModelBundle:
        """Load a trained bundle; if none exists, only the untrained baselines are available."""
        region_id = ctx.region_id
        version = version or latest_version(region_id)
        models: dict[str, Downscaler] = {"M0": CopyModel(), "M1": IDWModel(), "M2": LapseRateModel()}
        if version is None:
            return cls(region_id, None, {}, models)

        d = region_model_dir(region_id) / version
        meta = json.loads((d / "meta.json").read_text())
        if meta.get("n_active_cells") not in (None, ctx.n_active):
            raise ValueError(
                f"Model {version} was trained for {meta['n_active_cells']} cells but the region "
                f"now has {ctx.n_active}. The region was rebuilt; retrain with `pcast train`."
            )
        gammas = {v: np.array(g) for v, g in json.loads((d / "m2.json").read_text()).items()}
        models["M2"] = LapseRateModel(gammas)

        clim = np.load(d / "m3" / "clim.npz")
        vars_: dict[str, VarBoosters] = {}
        for var in MODELLED_VARS:
            vm = meta["variables"][var]
            parts = {
                p: lgb.Booster(model_file=str(d / "m3" / f"{var}__{p}.txt")) for p in vm["parts"]
            }
            vars_[var] = VarBoosters(
                kind=vm["kind"],
                features=vm["features"],
                clim=clim[var] if var in clim.files else None,
                q_lo=parts.get("q_lo"),
                q_hi=parts.get("q_hi"),
                mean=parts.get("mean"),
                occ=parts.get("occ"),
                amt=parts.get("amt"),
                rain_threshold=vm["rain_threshold"],
                qm=tuple(vm["qm"]) if vm.get("qm") else None,
                report=vm.get("report", {}),
            )
        models["M3"] = GBMModel(vars_, meta.get("wet_threshold", 0.1))
        m3s = meta.get("m3s")
        if m3s:
            boosters = {v: lgb.Booster(model_file=str(d / "m3s" / f"{v}.txt")) for v in m3s["shrink"]}
            corr = StationCorrection(boosters, m3s["shrink"], m3s["features"], m3s.get("report", {}))
            models["M3S"] = StationCorrectedModel(models["M3"], corr)
        m4 = load_unet(ctx, d / "m4")  # optional deep-learning model (needs torch)
        if m4 is not None:
            models["M4"] = m4
        return cls(region_id, version, meta, models)
