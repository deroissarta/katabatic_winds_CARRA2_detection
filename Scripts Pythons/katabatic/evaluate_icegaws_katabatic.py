"""Evaluate detected katabatic events against ICE-GAWS station records."""

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm

try:
    from pykrige.ok import OrdinaryKriging
except Exception:
    OrdinaryKriging = None

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from shared.icegaws import (
    _build_station_carra_spatial_temperature_reference,
    _resolve_detector_columns,
    _year_from_l3_filename,
    build_station_pixel_mapping,
    discover_available_station_years,
    load_icegaws_l3_csv,
    load_station_locations,
)
from shared.cache_utils import compute_signature, load_manifest, outputs_exist, path_fingerprint, write_manifest
from shared.map_launcher import plot_iceland
from shared.paths import (
    DATA_DIR,
    DEFAULT_CARRA_DETECTOR_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_KATABATIC_DATA_DIR,
    DEFAULT_NPY_DATA_DIR,
    resolve_glacier_mask_path,
)
from shared.reader import crop_iceland, load_static_orography, open_t2m_netcdf, open_uv_netcdf


@dataclass
class ThresholdSet:
    dT_threshold: float
    wind_threshold: float
    direction_threshold: float


def _parse_float_list(raw: str):
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError(f"Expected at least one numeric value, got: {raw}")
    return values


def _discover_latest_per_kat_path():
    yearly_candidates = sorted(DEFAULT_NPY_DATA_DIR.glob("yearly/*/aggregated/*/per_kat_*.npy"))
    if yearly_candidates:
        return yearly_candidates[-1]

    concat_candidates = sorted(DEFAULT_NPY_DATA_DIR.glob("concat/*/*/per_kat_*.npy"))
    if concat_candidates:
        return concat_candidates[-1]

    raise FileNotFoundError(
        f"No per_kat files found under {DEFAULT_NPY_DATA_DIR / 'yearly'} or {DEFAULT_NPY_DATA_DIR / 'concat'}"
    )


def _relative_to_project(path: Path | str):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path)


def _path_payload(path: Path | str):
    resolved = Path(path).resolve()
    return {
        "absolute": str(resolved),
        "relative": _relative_to_project(resolved),
    }


def _build_eval_cache_signature_payload(
    *,
    mode: str,
    carra_data_dir: Path,
    mapping_path: Path,
    per_kat_path: Path,
    detector_json_path: Path | None,
    station_subset,
    allowed_years,
    args,
    param_tag: str,
    domain_bounds,
    available_station_years,
):
    return {
        "schema_version": 1,
        "mode": str(mode),
        "paths": {
            "carra_data_dir": path_fingerprint(carra_data_dir),
            "mapping_json": path_fingerprint(mapping_path),
            "per_kat": path_fingerprint(per_kat_path),
            "detector_json": None if detector_json_path is None else path_fingerprint(detector_json_path),
        },
        "param_tag": str(param_tag),
        "stations_requested": sorted(list(station_subset)) if station_subset is not None else None,
        "years_requested": sorted(list(allowed_years)) if allowed_years is not None else None,
        "resample_step": str(args.resample_step),
        "match_tolerance": str(args.match_tolerance),
        "reference_window_size": int(args.reference_window_size),
        "reduced_domain": domain_bounds,
        "min_valid_fraction": float(args.min_valid_fraction),
        "min_valid_steps": int(args.min_valid_steps),
        "dT_thresholds": sorted(float(v) for v in _parse_float_list(args.dT_thresholds)),
        "wind_thresholds": sorted(float(v) for v in _parse_float_list(args.wind_thresholds)),
        "direction_thresholds": sorted(float(v) for v in _parse_float_list(args.direction_thresholds)),
        "available_station_years": available_station_years,
        "generate_stat_maps": bool(args.generate_stat_maps),
        "annotate_point_values": bool(args.annotate_point_values),
        "kriging_min_stations": int(args.kriging_min_stations),
    }


def _is_cache_hit(manifest_path: Path, signature: str, required_outputs):
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return False
    if manifest.get("signature") != signature:
        return False
    return outputs_exist(required_outputs)


def _load_mapping_json(mapping_path: Path, carra_data_dir: Path, station_codes=None):
    if mapping_path.exists():
        return json.loads(mapping_path.read_text(encoding="utf-8"))

    reference = load_static_orography(carra_data_dir, year="2022", month="June", file_index=2)
    station_locations = load_station_locations()
    return build_station_pixel_mapping(
        carra_ds=reference,
        station_locations=station_locations,
        station_codes=station_codes,
    )


def _compute_station_domain_bounds(mapping_stations: dict, grid_shape, station_subset=None, padding: int = 0):
    y_max_grid, x_max_grid = int(grid_shape[0]), int(grid_shape[1])
    ys = []
    xs = []
    for station_name, station_payload in mapping_stations.items():
        if station_subset is not None and station_name not in station_subset:
            continue
        pixel = station_payload.get("nearest_carra_pixel", {})
        if "y" not in pixel or "x" not in pixel:
            continue
        ys.append(int(pixel["y"]))
        xs.append(int(pixel["x"]))

    if not ys or not xs:
        return None

    y0 = max(0, min(ys) - int(padding))
    y1 = min(y_max_grid, max(ys) + int(padding) + 1)
    x0 = max(0, min(xs) - int(padding))
    x1 = min(x_max_grid, max(xs) + int(padding) + 1)
    if y1 <= y0 or x1 <= x0:
        return None

    return {
        "y_start": int(y0),
        "y_end": int(y1),
        "x_start": int(x0),
        "x_end": int(x1),
        "padding": int(padding),
        "shape": [int(y1 - y0), int(x1 - x0)],
    }


def _crop_monthly_to_domain(ds_month, domain_bounds):
    if domain_bounds is None:
        return ds_month
    if "y" not in ds_month.dims or "x" not in ds_month.dims:
        return ds_month
    return ds_month.isel(
        y=slice(int(domain_bounds["y_start"]), int(domain_bounds["y_end"])),
        x=slice(int(domain_bounds["x_start"]), int(domain_bounds["x_end"])),
    )


def _load_carra_uv_jja_for_year(carra_data_dir: Path, year: str, domain_bounds=None):
    monthly = []
    for month in ("June", "July", "August"):
        folder = carra_data_dir / str(year) / f"CARRA2_{month}"
        if not folder.exists():
            continue
        try:
            ds_month = crop_iceland(open_uv_netcdf(folder), "uv")
        except (FileNotFoundError, ValueError):
            continue
        if "u10" not in ds_month.data_vars or "v10" not in ds_month.data_vars:
            continue
        ds_month = _crop_monthly_to_domain(ds_month, domain_bounds=domain_bounds)
        monthly.append(ds_month[["u10", "v10"]].sortby("time"))

    if not monthly:
        return None
    return monthly


def _load_carra_t2m_jja_for_year(carra_data_dir: Path, year: str, domain_bounds=None):
    monthly = []
    for month in ("June", "July", "August"):
        folder = carra_data_dir / str(year) / f"CARRA2_{month}"
        if not folder.exists():
            continue
        try:
            ds_month = crop_iceland(open_t2m_netcdf(folder), "t2m")
        except (FileNotFoundError, ValueError):
            continue
        if "t2m" not in ds_month.data_vars:
            continue
        ds_month = _crop_monthly_to_domain(ds_month, domain_bounds=domain_bounds)
        monthly.append(ds_month[["t2m"]].sortby("time"))

    if not monthly:
        return None
    return monthly


def _cos_alignment(direction_deg: pd.Series, slope_direction_deg: float):
    return np.cos(np.deg2rad(pd.to_numeric(direction_deg, errors="coerce") - float(slope_direction_deg)))


def _angular_diff_deg(model_deg: pd.Series, obs_deg: pd.Series):
    model = pd.to_numeric(model_deg, errors="coerce")
    obs = pd.to_numeric(obs_deg, errors="coerce")
    return ((model - obs + 180.0) % 360.0) - 180.0


def _continuous_metrics(obs: pd.Series, model: pd.Series, include_correlations: bool = True):
    obs = pd.to_numeric(obs, errors="coerce")
    model = pd.to_numeric(model, errors="coerce")
    valid = obs.notna() & model.notna()
    if int(valid.sum()) == 0:
        return {
            "n": 0,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "relative_error_pct": np.nan,
            "relative_error_n": 0,
        }

    obs_v = obs[valid]
    model_v = model[valid]
    err = model_v - obs_v
    rmse = float(np.sqrt(np.mean(np.square(err))))

    obs_np = obs_v.to_numpy(dtype=float)
    model_np = model_v.to_numpy(dtype=float)
    nonzero_mask = np.abs(obs_np) > 1e-8
    relative_error_pct = np.nan
    relative_error_n = 0
    if np.any(nonzero_mask):
        rel = ((model_np[nonzero_mask] - obs_np[nonzero_mask]) / obs_np[nonzero_mask]) * 100.0
        finite_rel = rel[np.isfinite(rel)]
        if finite_rel.size > 0:
            relative_error_pct = float(np.mean(finite_rel))
            relative_error_n = int(finite_rel.size)

    def _safe_pearson(x: np.ndarray, y: np.ndarray):
        if x.size < 2 or y.size < 2:
            return np.nan
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_centered = x - x_mean
        y_centered = y - y_mean
        denom = math.sqrt(float(np.sum(x_centered * x_centered)) * float(np.sum(y_centered * y_centered)))
        if denom <= 0.0:
            return np.nan
        return float(np.sum(x_centered * y_centered) / denom)

    pearson_r = np.nan
    spearman_r = np.nan
    if include_correlations and int(valid.sum()) >= 2:
        try:
            pearson_r = _safe_pearson(obs_np.astype(np.float64, copy=False), model_np.astype(np.float64, copy=False))
            obs_rank = pd.Series(obs_np).rank(method="average").to_numpy(dtype=float)
            model_rank = pd.Series(model_np).rank(method="average").to_numpy(dtype=float)
            spearman_r = _safe_pearson(obs_rank, model_rank)
        except Exception:
            pearson_r = np.nan
            spearman_r = np.nan

    return {
        "n": int(valid.sum()),
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": rmse,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "relative_error_pct": relative_error_pct,
        "relative_error_n": relative_error_n,
    }


def _param_tag_from_per_kat_path(per_kat_path: Path):
    stem = per_kat_path.stem
    if stem.startswith("per_kat_"):
        return stem.replace("per_kat_", "", 1)
    return stem


def _resolve_stat_output_dir(stat_output_dir: str | None, param_tag: str):
    if stat_output_dir is not None:
        return Path(stat_output_dir)
    return DEFAULT_FIGURES_DIR / "yearly" / str(param_tag) / "aggregated" / "STAT"


def _metric_value_for_station(summary: dict, variable: str, metric: str):
    metrics = summary.get("continuous_metrics", {})
    if variable in ("temperature", "wind_speed"):
        return metrics.get(variable, {}).get(metric)

    if variable == "wind_direction":
        circular = metrics.get("wind_direction", {})
        raw = metrics.get("wind_direction_raw", {})
        if metric == "bias":
            return circular.get("bias_deg")
        if metric == "mae":
            return circular.get("mae_deg")
        if metric == "rmse":
            return circular.get("rmse_deg")
        if metric == "pearson_r":
            return raw.get("pearson_r")
        if metric == "relative_error_pct":
            return raw.get("relative_error_pct")
    return None


def _station_metric_table(station_static_summary: dict, mapping_stations: dict):
    rows = []
    for station_name, summary in station_static_summary.items():
        location = mapping_stations.get(station_name, {}).get("location", {})
        row = {
            "station": station_name,
            "lat": location.get("lat"),
            "lon": location.get("lon"),
            "eligible": summary.get("eligible"),
            "valid_steps": summary.get("valid_steps"),
            "valid_fraction": summary.get("valid_fraction"),
        }
        for variable in ("temperature", "wind_speed", "wind_direction"):
            for metric in ("bias", "mae", "rmse", "pearson_r", "relative_error_pct"):
                row[f"{variable}_{metric}"] = _metric_value_for_station(summary, variable, metric)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("station") if rows else pd.DataFrame()


def _build_points_for_metric(station_static_summary: dict, mapping_stations: dict, variable: str, metric: str):
    points = []
    for station_name, summary in station_static_summary.items():
        if not bool(summary.get("eligible", False)):
            continue
        value = _metric_value_for_station(summary, variable, metric)
        if value is None or pd.isna(value):
            continue

        loc = mapping_stations.get(station_name, {}).get("location", {})
        lat = loc.get("lat")
        lon = loc.get("lon")
        if lat is None or lon is None:
            continue

        points.append(
            {
                "station": station_name,
                "label": station_name,
                "lat": float(lat),
                "lon": float(lon),
                "value": float(value),
            }
        )
    return points


def _build_norm(values: np.ndarray, metric: str):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)

    if metric == "pearson_r":
        return Normalize(vmin=-1.0, vmax=1.0)

    if metric in ("bias", "relative_error_pct"):
        vmax = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    vmax = float(np.nanquantile(finite, 0.95)) if finite.size > 2 else float(np.nanmax(finite))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return Normalize(vmin=0.0, vmax=vmax)


def _idw_interpolate(lon_points, lat_points, val_points, lon_grid, lat_grid, power=2.0):
    lon_flat = lon_grid.ravel()
    lat_flat = lat_grid.ravel()

    dlon = lon_flat[:, None] - lon_points[None, :]
    dlat = lat_flat[:, None] - lat_points[None, :]
    dist = np.sqrt(dlon * dlon + dlat * dlat)

    exact = dist <= 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = 1.0 / np.power(np.maximum(dist, 1e-12), power)
        interp = np.sum(weights * val_points[None, :], axis=1) / np.sum(weights, axis=1)

    if np.any(exact):
        exact_rows = np.where(np.any(exact, axis=1))[0]
        for idx in exact_rows:
            interp[idx] = float(val_points[np.where(exact[idx])[0][0]])

    return interp.reshape(lon_grid.shape)


def _kriging_or_idw(points, lon_grid, lat_grid):
    lon_points = np.asarray([p["lon"] for p in points], dtype=float)
    lat_points = np.asarray([p["lat"] for p in points], dtype=float)
    val_points = np.asarray([p["value"] for p in points], dtype=float)

    if OrdinaryKriging is not None:
        try:
            ok = OrdinaryKriging(
                lon_points,
                lat_points,
                val_points,
                variogram_model="linear",
                verbose=False,
                enable_plotting=False,
                coordinates_type="geographic",
            )
            z, _ = ok.execute("points", lon_grid.ravel(), lat_grid.ravel())
            return np.asarray(z, dtype=float).reshape(lon_grid.shape), "ordinary_kriging"
        except Exception:
            pass

    return _idw_interpolate(lon_points, lat_points, val_points, lon_grid, lat_grid), "idw"


def _generate_stat_maps(
    station_static_summary: dict,
    mapping_stations: dict,
    ref,
    per_kat,
    stat_dir: Path,
    min_kriging_stations: int,
    annotate_points: bool,
):
    stat_dir.mkdir(parents=True, exist_ok=True)
    lon_grid = np.asarray(ref["longitude"].values, dtype=float)
    lat_grid = np.asarray(ref["latitude"].values, dtype=float)

    try:
        glaciers = np.load(resolve_glacier_mask_path())
    except Exception:
        glaciers = None

    variable_specs = {
        "temperature": {"cmap": "coolwarm", "unit": "degC"},
        "wind_speed": {"cmap": "viridis", "unit": "m s-1"},
        "wind_direction": {"cmap": "twilight", "unit": "deg"},
    }
    metrics = ("bias", "mae", "rmse", "pearson_r", "relative_error_pct")

    statuses = []
    for variable, spec in variable_specs.items():
        for metric in metrics:
            points = _build_points_for_metric(
                station_static_summary=station_static_summary,
                mapping_stations=mapping_stations,
                variable=variable,
                metric=metric,
            )
            if not points:
                statuses.append(
                    {
                        "variable": variable,
                        "metric": metric,
                        "status": "skipped_no_station_values",
                    }
                )
                continue

            values = np.asarray([p["value"] for p in points], dtype=float)
            norm = _build_norm(values, metric=metric)
            point_map_path = stat_dir / f"station_points_{variable}_{metric}.png"
            plot_iceland(
                per_kat,
                "per_kat",
                typ="mask",
                ref=ref,
                unit={"per_kat": "0-1"},
                cmap="Greys",
                glaciers=glaciers,
                points=points,
                points_cmap=spec["cmap"],
                points_norm=norm,
                points_size=55,
                points_edgecolor="white",
                annotate_points=annotate_points,
                save_path=point_map_path,
                show=False,
                title=f"Stations - {variable} - {metric}",
                coasts="white",
            )

            if len(points) < int(min_kriging_stations):
                statuses.append(
                    {
                        "variable": variable,
                        "metric": metric,
                        "status": "kriging_skipped_not_enough_stations",
                        "station_count": int(len(points)),
                        "required_station_count": int(min_kriging_stations),
                        "point_map": str(point_map_path),
                        "point_map_relative": _relative_to_project(point_map_path),
                    }
                )
                continue

            interpolated, method_used = _kriging_or_idw(points, lon_grid=lon_grid, lat_grid=lat_grid)
            krig_path = stat_dir / f"kriging_{method_used}_{variable}_{metric}.png"
            plot_iceland(
                interpolated,
                "per_kat",
                typ="mask",
                ref=ref,
                unit={"per_kat": spec["unit"]},
                cmap=spec["cmap"],
                norm=norm,
                glaciers=glaciers,
                points=points,
                points_cmap=spec["cmap"],
                points_norm=norm,
                points_size=25,
                points_edgecolor="black",
                annotate_points=False,
                save_path=krig_path,
                show=False,
                title=f"Interpolated ({method_used}) - {variable} - {metric}",
                coasts="white",
            )
            statuses.append(
                {
                    "variable": variable,
                    "metric": metric,
                    "status": "ok",
                    "station_count": int(len(points)),
                    "point_map": str(point_map_path),
                    "point_map_relative": _relative_to_project(point_map_path),
                    "interpolation_map": str(krig_path),
                    "interpolation_map_relative": _relative_to_project(krig_path),
                    "interpolation_method": method_used,
                }
            )

    status_path = stat_dir / "evaluation_map_status.json"
    status_path.write_text(
        json.dumps(
            {
                "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
                "kriging_min_stations": int(min_kriging_stations),
                "stat_output_dir": str(stat_dir.resolve()),
                "stat_output_dir_relative": _relative_to_project(stat_dir),
                "entries": statuses,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return status_path


def _direction_metrics(obs_deg: pd.Series, model_deg: pd.Series):
    delta = _angular_diff_deg(model_deg, obs_deg)
    valid = delta.notna()
    if int(valid.sum()) == 0:
        return {
            "n": 0,
            "bias_deg": np.nan,
            "mae_deg": np.nan,
            "rmse_deg": np.nan,
        }

    err = delta[valid]
    return {
        "n": int(valid.sum()),
        "bias_deg": float(np.mean(err)),
        "mae_deg": float(np.mean(np.abs(err))),
        "rmse_deg": float(np.sqrt(np.mean(np.square(err)))),
    }


def _to_celsius_if_kelvin(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values

    median_value = float(np.nanmedian(finite))
    if median_value > 150.0:
        return values - 273.15
    return values


def _event_metrics(obs_event: pd.Series, model_event: pd.Series, valid_mask: pd.Series):
    valid = valid_mask.astype(bool)
    if int(valid.sum()) == 0:
        return {
            "valid_steps": 0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "pod": np.nan,
            "far": np.nan,
            "csi": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "obs_event_fraction": np.nan,
            "model_event_fraction": np.nan,
        }

    obs = obs_event[valid].astype(bool)
    model = model_event[valid].astype(bool)

    tp = int((obs & model).sum())
    tn = int((~obs & ~model).sum())
    fp = int((~obs & model).sum())
    fn = int((obs & ~model).sum())

    pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = pod
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        "valid_steps": int(valid.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pod": float(pod) if not math.isnan(pod) else np.nan,
        "far": float(far) if not math.isnan(far) else np.nan,
        "csi": float(csi) if not math.isnan(csi) else np.nan,
        "precision": float(precision) if not math.isnan(precision) else np.nan,
        "recall": float(recall) if not math.isnan(recall) else np.nan,
        "f1": float(f1) if not math.isnan(f1) else np.nan,
        "obs_event_fraction": float(obs.mean()),
        "model_event_fraction": float(model.mean()),
    }


def _station_slope_direction_deg(orography_2d: np.ndarray, y: int, x: int):
    dh_dy, dh_dx = np.gradient(orography_2d)
    slope_vector_y = -float(dh_dy[int(y), int(x)])
    slope_vector_x = -float(dh_dx[int(y), int(x)])
    slope_dir_math_rad = np.arctan2(slope_vector_y, slope_vector_x)
    # Convert to meteorological convention (from north, clockwise).
    return float((90.0 - np.rad2deg(slope_dir_math_rad) + 360.0) % 360.0)


def _carra_pixel_series(ds_t2m_monthly, ds_uv_monthly, y: int, x: int):
    if ds_t2m_monthly is None or ds_uv_monthly is None:
        return pd.DataFrame(columns=["model_t2m", "model_ws", "model_wd"])

    t2m_series_parts = []
    for ds_t2m in ds_t2m_monthly:
        time_index = pd.DatetimeIndex(pd.to_datetime(ds_t2m["time"].values, utc=True))
        t2m_series_parts.append(
            pd.Series(np.asarray(ds_t2m["t2m"][:, int(y), int(x)].values, dtype=float), index=time_index)
        )

    uv_frames = []
    for ds_uv in ds_uv_monthly:
        time_index = pd.DatetimeIndex(pd.to_datetime(ds_uv["time"].values, utc=True))
        u10 = np.asarray(ds_uv["u10"][:, int(y), int(x)].values, dtype=float)
        v10 = np.asarray(ds_uv["v10"][:, int(y), int(x)].values, dtype=float)
        ws = np.sqrt(u10 ** 2 + v10 ** 2)
        # Convert vector components to meteorological wind direction.
        wd = (270.0 - np.rad2deg(np.arctan2(v10, u10))) % 360.0
        uv_frames.append(pd.DataFrame({"model_ws": ws, "model_wd": wd}, index=time_index))

    if not t2m_series_parts:
        return pd.DataFrame(columns=["model_t2m", "model_ws", "model_wd"])

    t2m_series = pd.concat(t2m_series_parts).sort_index()
    uv_df = pd.concat(uv_frames).sort_index() if uv_frames else pd.DataFrame(columns=["model_ws", "model_wd"])
    out = pd.DataFrame({"model_t2m": t2m_series})
    out = out.join(uv_df, how="outer").sort_index()
    return out[~out.index.duplicated(keep="first")]


def _station_temp_reference_from_monthlies(
    station_index: pd.DatetimeIndex,
    t2m_monthly,
    y: int,
    x: int,
    window_size: int,
    match_tolerance: str,
):
    if t2m_monthly is None:
        return pd.Series(index=station_index, dtype=float)

    parts = []
    half_window = int(window_size) // 2
    for ds_t2m in t2m_monthly:
        if "t2m" not in ds_t2m.data_vars:
            continue
        t2m_da = ds_t2m["t2m"]
        if t2m_da.ndim != 3:
            continue

        y_size = int(t2m_da.sizes.get("y", 0))
        x_size = int(t2m_da.sizes.get("x", 0))
        if y_size == 0 or x_size == 0:
            continue

        # Clamp indexed window with nearest-neighbor behavior at domain boundaries.
        y_idx = np.clip(np.arange(int(y) - half_window, int(y) + half_window + 1), 0, y_size - 1)
        x_idx = np.clip(np.arange(int(x) - half_window, int(x) + half_window + 1), 0, x_size - 1)

        local_mean_da = t2m_da.isel(y=y_idx, x=x_idx).mean(dim=("y", "x"), skipna=True)
        values = np.asarray(local_mean_da.values, dtype=np.float32)
        time_index = pd.DatetimeIndex(pd.to_datetime(ds_t2m["time"].values, utc=True))
        parts.append(pd.Series(values, index=time_index, dtype=float))

    if not parts:
        return pd.Series(index=station_index, dtype=float)

    carra_series = pd.concat(parts).sort_index()
    carra_series = carra_series[~carra_series.index.duplicated(keep="first")]
    return carra_series.reindex(station_index, method="nearest", tolerance=pd.Timedelta(match_tolerance))


def _build_station_timeseries(
    station_name: str,
    station_payload: dict,
    carra_data_dir: Path,
    resample_step: str,
    match_tolerance: str,
    reference_window_size: int,
    allowed_years: set,
    t2m_cache: dict,
    uv_cache: dict,
    available_station_years: dict | None = None,
    domain_bounds: dict | None = None,
):
    y_global = int(station_payload["nearest_carra_pixel"]["y"])
    x_global = int(station_payload["nearest_carra_pixel"]["x"])
    if domain_bounds is None:
        y = y_global
        x = x_global
    else:
        y = y_global - int(domain_bounds["y_start"])
        x = x_global - int(domain_bounds["x_start"])
        if y < 0 or x < 0 or y >= int(domain_bounds["shape"][0]) or x >= int(domain_bounds["shape"][1]):
            return pd.DataFrame()

    chunks = []
    station_years = set()
    if available_station_years is not None:
        station_years = {str(y) for y in available_station_years.get(station_name, [])}

    for l3_file in station_payload.get("l3_files", []):
        l3_path = Path(l3_file)
        if not l3_path.exists():
            continue
        year = _year_from_l3_filename(l3_path)
        if year is None:
            continue
        if allowed_years is not None and str(year) not in allowed_years:
            continue
        if available_station_years is not None and str(year) not in station_years:
            continue

        station_df = load_icegaws_l3_csv(l3_path, resample_step=resample_step)

        if year not in t2m_cache:
            t2m_cache[year] = _load_carra_t2m_jja_for_year(carra_data_dir, year, domain_bounds=domain_bounds)
        if year not in uv_cache:
            uv_cache[year] = _load_carra_uv_jja_for_year(carra_data_dir, year, domain_bounds=domain_bounds)

        ds_t2m = t2m_cache[year]
        ds_uv = uv_cache[year]
        if ds_t2m is None or ds_uv is None:
            continue

        temp_ref = _station_temp_reference_from_monthlies(
            station_index=station_df.index,
            t2m_monthly=ds_t2m,
            y=y,
            x=x,
            window_size=reference_window_size,
            match_tolerance=match_tolerance,
        )

        model_df = _carra_pixel_series(ds_t2m, ds_uv, y=y, x=x)
        model_df = model_df.reindex(
            station_df.index,
            method="nearest",
            tolerance=pd.Timedelta(match_tolerance),
        )

        t_col, ws_col, wd_col, missing = _resolve_detector_columns(
            station_df,
            temperature_column="t_2m",
            wind_speed_column="ws_4m",
            wind_direction_column="wd_4m",
        )
        if missing:
            continue

        chunk = pd.DataFrame(index=station_df.index)
        chunk["station"] = station_name
        chunk["year"] = int(year)
        chunk["obs_t2m"] = pd.to_numeric(station_df[t_col], errors="coerce")
        chunk["obs_ws"] = pd.to_numeric(station_df[ws_col], errors="coerce")
        chunk["obs_wd"] = pd.to_numeric(station_df[wd_col], errors="coerce")
        chunk["temp_ref"] = pd.to_numeric(temp_ref, errors="coerce")
        chunk["model_t2m"] = pd.to_numeric(model_df["model_t2m"], errors="coerce")
        chunk["obs_t2m"] = _to_celsius_if_kelvin(chunk["obs_t2m"])
        chunk["temp_ref"] = _to_celsius_if_kelvin(chunk["temp_ref"])
        chunk["model_t2m"] = _to_celsius_if_kelvin(chunk["model_t2m"])
        chunk["model_ws"] = pd.to_numeric(model_df["model_ws"], errors="coerce")
        chunk["model_wd"] = pd.to_numeric(model_df["model_wd"], errors="coerce")
        chunk["month"] = chunk.index.month
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _apply_detector(df: pd.DataFrame, slope_direction_deg: float, threshold_set: ThresholdSet):
    jja = df["month"].isin([6, 7, 8])
    valid_event = (
        jja
        & df["obs_t2m"].notna()
        & df["obs_ws"].notna()
        & df["obs_wd"].notna()
        & df["model_t2m"].notna()
        & df["model_ws"].notna()
        & df["model_wd"].notna()
        & df["temp_ref"].notna()
    )

    obs_cold = (df["obs_t2m"] - df["temp_ref"]) < -float(threshold_set.dT_threshold)
    model_cold = (df["model_t2m"] - df["temp_ref"]) < -float(threshold_set.dT_threshold)
    obs_strong = df["obs_ws"] > float(threshold_set.wind_threshold)
    model_strong = df["model_ws"] > float(threshold_set.wind_threshold)

    obs_align = _cos_alignment(df["obs_wd"], slope_direction_deg) > float(threshold_set.direction_threshold)
    model_align = _cos_alignment(df["model_wd"], slope_direction_deg) > float(threshold_set.direction_threshold)

    obs_event = valid_event & obs_cold & obs_strong & obs_align
    model_event = valid_event & model_cold & model_strong & model_align
    return obs_event, model_event, valid_event


def _combo_key(combo: ThresholdSet):
    return (
        f"dt={combo.dT_threshold:g}|"
        f"wind={combo.wind_threshold:g}|"
        f"dir={combo.direction_threshold:g}"
    )


def _evaluate_from_existing_detector(
    mapping: dict,
    detector_json_path: Path,
    per_kat_path: Path,
    output_json: Path,
    output_csv: Path,
    station_subset=None,
):
    print("[evaluation] detector-mode: load detector json", flush=True)
    detector = json.loads(detector_json_path.read_text(encoding="utf-8"))
    print("[evaluation] detector-mode: load per_kat", flush=True)
    per_kat = np.asarray(np.load(per_kat_path), dtype=float)
    mapping_stations = mapping.get("stations", {})
    detector_stations = detector.get("stations", {})
    print(
        f"[evaluation] detector-mode: mapping stations={len(mapping_stations)} detector stations={len(detector_stations)}",
        flush=True,
    )

    rows = []
    for station_name, station_payload in mapping_stations.items():
        if station_subset is not None and station_name not in station_subset:
            continue
        if station_name not in detector_stations:
            continue

        pixel = station_payload.get("nearest_carra_pixel", {})
        y = int(pixel.get("y"))
        x = int(pixel.get("x"))
        if y < 0 or x < 0 or y >= per_kat.shape[0] or x >= per_kat.shape[1]:
            continue

        summary = detector_stations[station_name].get("detector_summary", {})
        obs_fraction = summary.get("event_fraction")
        valid_steps = summary.get("valid_step_count")
        event_count = summary.get("event_count")
        if obs_fraction is None or pd.isna(obs_fraction):
            continue

        model_fraction = float(per_kat[y, x])
        if pd.isna(model_fraction):
            continue

        rows.append(
            {
                "station": station_name,
                "obs_event_fraction": float(obs_fraction),
                "model_event_fraction": model_fraction,
                "bias_model_minus_obs": model_fraction - float(obs_fraction),
                "abs_error": abs(model_fraction - float(obs_fraction)),
                "valid_step_count": None if valid_steps is None else int(valid_steps),
                "event_count": None if event_count is None else int(event_count),
                "distance_km": pixel.get("distance_km"),
                "y": y,
                "x": x,
            }
        )

    if not rows:
        raise RuntimeError(
            "No overlapping stations between mapping and detector JSON with valid event_fraction values."
        )

    station_df = pd.DataFrame(rows).sort_values("station")
    print(f"[evaluation] detector-mode: station rows={len(station_df)}", flush=True)
    station_df.to_csv(output_csv, index=False)
    print("[evaluation] detector-mode: station CSV written", flush=True)

    global_metrics = _continuous_metrics(
        station_df["obs_event_fraction"],
        station_df["model_event_fraction"],
        include_correlations=False,
    )
    print("[evaluation] detector-mode: global metrics computed", flush=True)
    if station_df["valid_step_count"].notna().any():
        weights = station_df["valid_step_count"].fillna(0).astype(float)
        if float(weights.sum()) > 0:
            weighted_bias = float(np.average(station_df["bias_model_minus_obs"], weights=weights))
            weighted_mae = float(np.average(station_df["abs_error"], weights=weights))
        else:
            weighted_bias = np.nan
            weighted_mae = np.nan
    else:
        weighted_bias = np.nan
        weighted_mae = np.nan

    payload = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "mode": "existing_detector_json",
        "inputs": {
            "detector_json": str(detector_json_path),
            "mapping_json": mapping.get("_source_path"),
            "per_kat_path": str(per_kat_path),
            "stations_requested": sorted(list(station_subset)) if station_subset is not None else None,
            "detector_metadata": detector.get("detector", {}),
        },
        "station_count": int(len(station_df)),
        "fraction_metrics_global": {
            **global_metrics,
            "weighted_bias": weighted_bias,
            "weighted_mae": weighted_mae,
        },
        "station_table_csv": str(output_csv),
        "stations": station_df.to_dict(orient="records"),
    }

    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[evaluation] detector-mode: JSON written", flush=True)
    print(f"[evaluation] Wrote detector-based metrics JSON: {output_json}")
    print(f"[evaluation] Wrote detector-based station CSV: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ICE-GAWS katabatic relevance against CARRA for the provided detector thresholds."
        )
    )
    parser.add_argument(
        "--carra-data-dir",
        default=str(DEFAULT_KATABATIC_DATA_DIR),
        help="CARRA data root used to read JJA t2m/u10/v10 files.",
    )
    parser.add_argument(
        "--mapping-json",
        default=str(DATA_DIR / "ICE-GAWS" / "icegaws_carra_pixel_basin_mapping.json"),
        help="Station-to-CARRA mapping JSON. Built in-memory if missing.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DATA_DIR / "ICE-GAWS" / "icegaws_katabatic_validation_metrics.json"),
        help="Output JSON for station and global metrics.",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DATA_DIR / "ICE-GAWS" / "icegaws_katabatic_threshold_sweep.csv"),
        help="Output CSV for evaluated threshold combinations; with one value per threshold option, this is a single-config metrics table.",
    )
    parser.add_argument(
        "--detector-json",
        default=None,
        help=(
            "Optional precomputed ICE-GAWS detector JSON. "
            "If provided, run a fast station-fraction evaluation without recomputing events."
        ),
    )
    parser.add_argument(
        "--carra-per-kat-path",
        default=None,
        help=(
            "CARRA per_kat .npy used with --detector-json. "
            "If omitted, the latest available per_kat file is auto-discovered."
        ),
    )
    parser.add_argument(
        "--station",
        nargs="*",
        default=None,
        help="Optional station subset. If omitted, use all mapped stations.",
    )
    parser.add_argument(
        "--resample-step",
        default="3h",
        help="Resampling step for ICE-GAWS observations.",
    )
    parser.add_argument(
        "--match-tolerance",
        default="2h",
        help="Nearest-time tolerance for aligning station and CARRA data.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.70,
        help="Minimum fraction of valid event-evaluation steps to keep a station.",
    )
    parser.add_argument(
        "--min-valid-steps",
        type=int,
        default=120,
        help="Minimum number of valid event-evaluation steps to keep a station.",
    )
    parser.add_argument(
        "--dT-thresholds",
        default="0.5,1.0,1.5,2.0,2.5,3.0",
        help="Comma-separated temperature anomaly thresholds.",
    )
    parser.add_argument(
        "--wind-thresholds",
        default="1.5,2.0,2.5,3.0,4.0,5.0",
        help="Comma-separated wind speed thresholds.",
    )
    parser.add_argument(
        "--direction-thresholds",
        default="0.70,0.80,0.87,0.90,0.95",
        help="Comma-separated cosine alignment thresholds.",
    )
    parser.add_argument(
        "--years",
        default=None,
        help="Optional comma-separated years (e.g. 2011,2012) to restrict evaluation.",
    )
    parser.add_argument(
        "--reference-window-size",
        type=int,
        default=15,
        help="Spatial window size used to build the CARRA temperature reference field.",
    )
    parser.add_argument(
        "--param-tag",
        default=None,
        help="Optional parameter tag used for STAT output directory naming.",
    )
    parser.add_argument(
        "--stat-output-dir",
        default=None,
        help="Optional explicit output directory for maps/summary tables.",
    )
    parser.add_argument(
        "--kriging-min-stations",
        type=int,
        default=6,
        help="Minimum number of valid stations required to create interpolated maps.",
    )
    parser.add_argument(
        "--annotate-point-values",
        action="store_true",
        help="Annotate station points with station code and metric value.",
    )
    parser.add_argument(
        "--no-annotate-point-values",
        dest="annotate_point_values",
        action="store_false",
        help="Disable point-value annotation on station maps.",
    )
    parser.add_argument(
        "--generate-stat-maps",
        dest="generate_stat_maps",
        action="store_true",
        help=(
            "Generate cartopy/matplotlib station maps. Enabled by default; disable with "
            "--no-generate-stat-maps if needed."
        ),
    )
    parser.add_argument(
        "--no-generate-stat-maps",
        dest="generate_stat_maps",
        action="store_false",
        help="Disable cartopy/matplotlib station maps generation.",
    )
    parser.add_argument(
        "--reuse-if-unchanged",
        dest="reuse_if_unchanged",
        action="store_true",
        help="Reuse existing outputs when configuration and inputs are unchanged.",
    )
    parser.add_argument(
        "--no-reuse-if-unchanged",
        dest="reuse_if_unchanged",
        action="store_false",
        help="Disable cache reuse and recompute this evaluation run.",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation even if cache signature matches.",
    )
    parser.set_defaults(annotate_point_values=False)
    parser.set_defaults(reuse_if_unchanged=True)
    parser.set_defaults(generate_stat_maps=True)

    args = parser.parse_args()

    carra_data_dir = Path(args.carra_data_dir)
    mapping_path = Path(args.mapping_json)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    station_subset = None
    if args.station:
        station_subset = {str(code).strip() for code in args.station}

    allowed_years = None
    if args.years is not None:
        allowed_years = {token.strip() for token in str(args.years).split(",") if token.strip()}
        if not allowed_years:
            raise ValueError("--years was provided but no valid year token was parsed.")

    mapping = _load_mapping_json(mapping_path, carra_data_dir=carra_data_dir, station_codes=args.station)
    mapping["_source_path"] = str(mapping_path)

    available_station_years = discover_available_station_years()

    if args.detector_json is not None:
        detector_json_path = Path(args.detector_json)
        if not detector_json_path.exists():
            raise FileNotFoundError(f"Detector JSON not found: {detector_json_path}")
        per_kat_path = Path(args.carra_per_kat_path) if args.carra_per_kat_path else _discover_latest_per_kat_path()
        if not per_kat_path.exists():
            raise FileNotFoundError(f"per_kat file not found: {per_kat_path}")

        param_tag = args.param_tag if args.param_tag is not None else _param_tag_from_per_kat_path(per_kat_path)
        detector_dir = Path(DEFAULT_CARRA_DETECTOR_DIR)
        detector_dir.mkdir(parents=True, exist_ok=True)
        detector_manifest_path = detector_dir / f"icegaws_evaluation_manifest_detector_mode_{param_tag}.json"
        detector_manifest_latest_path = detector_dir / "icegaws_evaluation_manifest_detector_mode_latest.json"

        cache_payload = _build_eval_cache_signature_payload(
            mode="existing_detector_json",
            carra_data_dir=carra_data_dir,
            mapping_path=mapping_path,
            per_kat_path=per_kat_path,
            detector_json_path=detector_json_path,
            station_subset=station_subset,
            allowed_years=allowed_years,
            args=args,
            param_tag=param_tag,
            domain_bounds=None,
            available_station_years=available_station_years,
        )
        cache_signature = compute_signature(cache_payload)
        if args.reuse_if_unchanged and (not args.force_recompute):
            if _is_cache_hit(
                detector_manifest_path,
                cache_signature,
                required_outputs=[output_json, output_csv],
            ):
                print(f"[evaluation] Cache hit, skip detector-mode recomputation: {detector_manifest_path}")
                return

        _evaluate_from_existing_detector(
            mapping=mapping,
            detector_json_path=detector_json_path,
            per_kat_path=per_kat_path,
            output_json=output_json,
            output_csv=output_csv,
            station_subset=station_subset,
        )
        write_manifest(
            detector_manifest_path,
            {
                "signature": cache_signature,
                "signature_payload": cache_payload,
                "mode": "existing_detector_json",
                "paths": {
                    "evaluation_json": _path_payload(output_json),
                    "threshold_csv": _path_payload(output_csv),
                },
            },
        )
        write_manifest(
            detector_manifest_latest_path,
            {
                "signature": cache_signature,
                "signature_payload": cache_payload,
                "mode": "existing_detector_json",
                "paths": {
                    "evaluation_json": _path_payload(output_json),
                    "threshold_csv": _path_payload(output_csv),
                },
            },
        )
        return

    per_kat_path = Path(args.carra_per_kat_path) if args.carra_per_kat_path else _discover_latest_per_kat_path()
    if not per_kat_path.exists():
        raise FileNotFoundError(f"per_kat file not found: {per_kat_path}")

    param_tag = args.param_tag if args.param_tag is not None else _param_tag_from_per_kat_path(per_kat_path)
    stat_dir = _resolve_stat_output_dir(args.stat_output_dir, param_tag=param_tag)
    stat_dir.mkdir(parents=True, exist_ok=True)

    reference_orog = load_static_orography(carra_data_dir, year="2022", month="June", file_index=2)
    orog_2d = np.asarray(reference_orog["orog"].isel(time=0).values, dtype=float)
    domain_padding = max(2, int(math.ceil(int(args.reference_window_size) / 2)))
    domain_bounds = _compute_station_domain_bounds(
        mapping_stations=mapping.get("stations", {}),
        grid_shape=orog_2d.shape,
        station_subset=station_subset,
        padding=domain_padding,
    )
    if domain_bounds is not None:
        print(
            "[evaluation] Reduced domain enabled: "
            f"y={domain_bounds['y_start']}:{domain_bounds['y_end']} "
            f"x={domain_bounds['x_start']}:{domain_bounds['x_end']} "
            f"shape={domain_bounds['shape']}"
        )

    stat_summary_path = stat_dir / "icegaws_evaluation_summary.json"
    station_metric_csv = stat_dir / "icegaws_station_metrics_table.csv"
    map_status_path = stat_dir / "evaluation_map_status.json"

    detector_dir = Path(DEFAULT_CARRA_DETECTOR_DIR)
    detector_dir.mkdir(parents=True, exist_ok=True)
    detector_manifest_path = detector_dir / f"icegaws_evaluation_manifest_{param_tag}.json"
    detector_manifest_latest_path = detector_dir / "icegaws_evaluation_manifest_latest.json"

    cache_payload = _build_eval_cache_signature_payload(
        mode="full_evaluation",
        carra_data_dir=carra_data_dir,
        mapping_path=mapping_path,
        per_kat_path=per_kat_path,
        detector_json_path=None,
        station_subset=station_subset,
        allowed_years=allowed_years,
        args=args,
        param_tag=param_tag,
        domain_bounds=domain_bounds,
        available_station_years=available_station_years,
    )
    cache_signature = compute_signature(cache_payload)
    if args.reuse_if_unchanged and (not args.force_recompute):
        if _is_cache_hit(
            detector_manifest_path,
            cache_signature,
            required_outputs=[output_json, output_csv, stat_summary_path, map_status_path],
        ):
            print(f"[evaluation] Cache hit, skip full recomputation: {detector_manifest_path}")
            return

    per_kat = np.asarray(np.load(per_kat_path), dtype=float)

    station_data = {}
    t2m_cache = {}
    uv_cache = {}
    for station_name, station_payload in mapping.get("stations", {}).items():
        if station_subset is not None and station_name not in station_subset:
            continue
        df_station = _build_station_timeseries(
            station_name=station_name,
            station_payload=station_payload,
            carra_data_dir=carra_data_dir,
            resample_step=args.resample_step,
            match_tolerance=args.match_tolerance,
            reference_window_size=int(args.reference_window_size),
            allowed_years=allowed_years,
            t2m_cache=t2m_cache,
            uv_cache=uv_cache,
            available_station_years=available_station_years,
            domain_bounds=domain_bounds,
        )
        if df_station.empty:
            continue
        y = int(station_payload["nearest_carra_pixel"]["y"])
        x = int(station_payload["nearest_carra_pixel"]["x"])
        slope_direction = _station_slope_direction_deg(orog_2d, y=y, x=x)
        station_data[station_name] = {
            "frame": df_station,
            "slope_direction_deg": slope_direction,
            "nearest_pixel": station_payload.get("nearest_carra_pixel", {}),
        }

    if not station_data:
        raise RuntimeError("No station with enough data could be loaded for evaluation.")

    dT_values = _parse_float_list(args.dT_thresholds)
    wind_values = _parse_float_list(args.wind_thresholds)
    direction_values = _parse_float_list(args.direction_thresholds)
    combinations = [
        ThresholdSet(dT_threshold=dt, wind_threshold=ws, direction_threshold=dr)
        for dt, ws, dr in itertools.product(dT_values, wind_values, direction_values)
    ]

    station_static_summary = {}
    eligible_stations = []
    excluded_stations = []
    for station_name, payload in station_data.items():
        frame = payload["frame"]
        jja = frame["month"].isin([6, 7, 8])
        event_valid = (
            jja
            & frame["obs_t2m"].notna()
            & frame["obs_ws"].notna()
            & frame["obs_wd"].notna()
            & frame["model_t2m"].notna()
            & frame["model_ws"].notna()
            & frame["model_wd"].notna()
            & frame["temp_ref"].notna()
        )

        total_jja = int(jja.sum())
        valid_steps = int(event_valid.sum())
        coverage = (valid_steps / total_jja) if total_jja > 0 else 0.0
        is_eligible = valid_steps >= int(args.min_valid_steps) and coverage >= float(args.min_valid_fraction)

        station_static_summary[station_name] = {
            "valid_steps": valid_steps,
            "total_jja_steps": total_jja,
            "valid_fraction": float(coverage),
            "eligible": bool(is_eligible),
            "continuous_metrics": {
                "temperature": _continuous_metrics(frame["obs_t2m"], frame["model_t2m"]),
                "wind_speed": _continuous_metrics(frame["obs_ws"], frame["model_ws"]),
                "wind_direction": _direction_metrics(frame["obs_wd"], frame["model_wd"]),
                "wind_direction_raw": _continuous_metrics(frame["obs_wd"], frame["model_wd"]),
            },
            "slope_direction_deg": float(payload["slope_direction_deg"]),
            "nearest_pixel": payload["nearest_pixel"],
        }

        if is_eligible:
            eligible_stations.append(station_name)
        else:
            excluded_stations.append(station_name)

    if not eligible_stations:
        raise RuntimeError(
            "No eligible station after coverage filtering. "
            "Try lowering --min-valid-fraction or --min-valid-steps."
        )

    threshold_rows = []
    best_record = None
    for combo in combinations:
        combo_key = _combo_key(combo)
        per_station_event = {}

        tp = tn = fp = fn = valid_steps_sum = 0
        obs_all = []
        model_all = []
        valid_all = []
        for station_name in eligible_stations:
            payload = station_data[station_name]
            frame = payload["frame"]
            obs_event, model_event, valid_event = _apply_detector(
                frame,
                slope_direction_deg=payload["slope_direction_deg"],
                threshold_set=combo,
            )
            metrics = _event_metrics(obs_event, model_event, valid_event)
            per_station_event[station_name] = metrics

            tp += metrics["tp"]
            tn += metrics["tn"]
            fp += metrics["fp"]
            fn += metrics["fn"]
            valid_steps_sum += metrics["valid_steps"]

            obs_all.append(obs_event)
            model_all.append(model_event)
            valid_all.append(valid_event)

        global_metrics = _event_metrics(
            pd.concat(obs_all),
            pd.concat(model_all),
            pd.concat(valid_all),
        )

        record = {
            "combo": combo_key,
            "dT_threshold": combo.dT_threshold,
            "wind_threshold": combo.wind_threshold,
            "direction_threshold": combo.direction_threshold,
            "eligible_station_count": len(eligible_stations),
            "valid_steps": valid_steps_sum,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "pod": global_metrics["pod"],
            "far": global_metrics["far"],
            "csi": global_metrics["csi"],
            "precision": global_metrics["precision"],
            "recall": global_metrics["recall"],
            "f1": global_metrics["f1"],
            "obs_event_fraction": global_metrics["obs_event_fraction"],
            "model_event_fraction": global_metrics["model_event_fraction"],
            "per_station_event_metrics": per_station_event,
        }
        threshold_rows.append(record)

        if best_record is None:
            best_record = record
            continue

        current_score = (
            -1.0 if pd.isna(record["csi"]) else float(record["csi"]),
            -1.0 if pd.isna(record["f1"]) else float(record["f1"]),
            -999999.0 if pd.isna(record["far"]) else -float(record["far"]),
        )
        best_score = (
            -1.0 if pd.isna(best_record["csi"]) else float(best_record["csi"]),
            -1.0 if pd.isna(best_record["f1"]) else float(best_record["f1"]),
            -999999.0 if pd.isna(best_record["far"]) else -float(best_record["far"]),
        )
        if current_score > best_score:
            best_record = record

    ranked_df = pd.DataFrame(
        [
            {
                "combo": row["combo"],
                "dT_threshold": row["dT_threshold"],
                "wind_threshold": row["wind_threshold"],
                "direction_threshold": row["direction_threshold"],
                "eligible_station_count": row["eligible_station_count"],
                "valid_steps": row["valid_steps"],
                "tp": row["tp"],
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "pod": row["pod"],
                "far": row["far"],
                "csi": row["csi"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "obs_event_fraction": row["obs_event_fraction"],
                "model_event_fraction": row["model_event_fraction"],
            }
            for row in threshold_rows
        ]
    ).sort_values(by=["csi", "f1", "far"], ascending=[False, False, True], na_position="last")
    ranked_df.to_csv(output_csv, index=False)

    eligible_frames = [station_data[s]["frame"] for s in eligible_stations]
    global_frame = pd.concat(eligible_frames).sort_index()

    payload = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "cache": {
            "signature": cache_signature,
            "reuse_if_unchanged": bool(args.reuse_if_unchanged),
            "force_recompute": bool(args.force_recompute),
        },
        "inputs": {
            "carra_data_dir": str(carra_data_dir),
            "carra_data_dir_relative": _relative_to_project(carra_data_dir),
            "mapping_json": str(mapping_path),
            "mapping_json_relative": _relative_to_project(mapping_path),
            "per_kat_path": str(per_kat_path),
            "per_kat_path_relative": _relative_to_project(per_kat_path),
            "param_tag": str(param_tag),
            "stat_output_dir": str(stat_dir),
            "stat_output_dir_relative": _relative_to_project(stat_dir),
            "stations_requested": sorted(list(station_subset)) if station_subset is not None else None,
            "years_requested": sorted(list(allowed_years)) if allowed_years is not None else None,
            "resample_step": args.resample_step,
            "match_tolerance": args.match_tolerance,
            "reference_window_size": int(args.reference_window_size),
            "reduced_domain": domain_bounds,
            "min_valid_fraction": float(args.min_valid_fraction),
            "min_valid_steps": int(args.min_valid_steps),
        },
        "station_filtering": {
            "eligible_stations": eligible_stations,
            "excluded_stations": excluded_stations,
            "station_count_total_loaded": len(station_data),
            "station_count_eligible": len(eligible_stations),
        },
        "continuous_metrics_global": {
            "temperature": _continuous_metrics(global_frame["obs_t2m"], global_frame["model_t2m"]),
            "wind_speed": _continuous_metrics(global_frame["obs_ws"], global_frame["model_ws"]),
            "wind_direction": _direction_metrics(global_frame["obs_wd"], global_frame["model_wd"]),
            "wind_direction_raw": _continuous_metrics(global_frame["obs_wd"], global_frame["model_wd"]),
        },
        "station_static_summary": station_static_summary,
        "threshold_sweep": {
            "dT_thresholds": dT_values,
            "wind_thresholds": wind_values,
            "direction_thresholds": direction_values,
            "combination_count": len(combinations),
            "best_combination": best_record,
            "ranking_csv": str(output_csv),
            "ranking_csv_relative": _relative_to_project(output_csv),
        },
    }

    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    station_metric_table = _station_metric_table(
        station_static_summary=station_static_summary,
        mapping_stations=mapping.get("stations", {}),
    )
    station_metric_csv = stat_dir / "icegaws_station_metrics_table.csv"
    if not station_metric_table.empty:
        station_metric_table.to_csv(station_metric_csv, index=False)

    map_status_path = stat_dir / "evaluation_map_status.json"
    if args.generate_stat_maps:
        try:
            map_status_path = _generate_stat_maps(
                station_static_summary=station_static_summary,
                mapping_stations=mapping.get("stations", {}),
                ref=reference_orog,
                per_kat=per_kat,
                stat_dir=stat_dir,
                min_kriging_stations=int(args.kriging_min_stations),
                annotate_points=bool(args.annotate_point_values),
            )
        except Exception as exc:
            map_status_path.write_text(
                json.dumps(
                    {
                        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
                        "status": "failed",
                        "error": str(exc),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
    else:
        map_status_path.write_text(
            json.dumps(
                {
                    "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
                    "status": "skipped",
                    "reason": "Map generation disabled. Re-run with --generate-stat-maps to enable.",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    stat_summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_payload = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "signature": cache_signature,
        "signature_payload": cache_payload,
        "param_tag": str(param_tag),
        "paths": {
            "evaluation_json": _path_payload(output_json),
            "threshold_csv": _path_payload(output_csv),
            "stat_summary_json": _path_payload(stat_summary_path),
            "station_metrics_csv": _path_payload(station_metric_csv),
            "map_status_json": _path_payload(map_status_path) if map_status_path is not None else None,
            "stat_directory": _path_payload(stat_dir),
        },
        "station_filtering": payload.get("station_filtering", {}),
        "best_combination": payload.get("threshold_sweep", {}).get("best_combination"),
    }
    detector_manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    detector_manifest_latest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[evaluation] Wrote metrics JSON: {output_json}")
    print(f"[evaluation] Wrote threshold ranking CSV: {output_csv}")
    print(f"[evaluation] Wrote STAT summary JSON: {stat_summary_path}")
    print(f"[evaluation] Wrote station metrics CSV: {station_metric_csv}")
    if map_status_path is not None:
        print(f"[evaluation] Wrote map status JSON: {map_status_path}")
    print(f"[evaluation] Wrote detector manifest JSON: {detector_manifest_path}")
    print(f"[evaluation] Wrote detector latest manifest JSON: {detector_manifest_latest_path}")
    print(
        "[evaluation] Best thresholds: "
        f"dT={best_record['dT_threshold']:g}, "
        f"wind={best_record['wind_threshold']:g}, "
        f"dir={best_record['direction_threshold']:g}, "
        f"CSI={best_record['csi']:.3f}, F1={best_record['f1']:.3f}, FAR={best_record['far']:.3f}"
    )


if __name__ == "__main__":
    main()