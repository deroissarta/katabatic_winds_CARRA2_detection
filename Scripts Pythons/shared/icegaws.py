"""Read ICE-GAWS station data and build spatial mapping files."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from shared.paths import DATA_DIR


ICEGAWS_LOCATION_SUMMARY = DATA_DIR / "ICE-GAWS" / "ICE-GAWS-location-summary.csv"
ICEGAWS_DATA_DIR = DATA_DIR / "ICE-GAWS"
ICEGAWS_L3_GLOB = "ICE-GAWS_*_L3*.csv"


def _normalize_station_code(value: str) -> str:
    return str(value).strip()


def _station_code_from_l3_filename(path: Path) -> Optional[str]:
    name = path.name
    match = re.match(r"ICE-GAWS_(?P<station>.+?)_L3(?:_.+)?_(?P<year>\d{4})\.csv$", name, flags=re.IGNORECASE)
    if match:
        return _normalize_station_code(match.group("station"))

    match = re.match(r"ICE-GAWS_(?P<station>.+?)_L3(?:_.+)?\.csv$", name, flags=re.IGNORECASE)
    if match:
        return _normalize_station_code(match.group("station"))

    return None


def _year_from_l3_filename(path: Path) -> Optional[str]:
    name = Path(path).name
    match = re.match(r"ICE-GAWS_.+?_L3(?:_.+)?_(?P<year>\d{4})\.csv$", name, flags=re.IGNORECASE)
    if match:
        return match.group("year")
    return None


def discover_available_station_years(data_dir=ICEGAWS_DATA_DIR):
    files_by_station = discover_l3_files(data_dir=data_dir)
    available_years = {}
    for station, files in files_by_station.items():
        years = sorted({year for path in files if (year := _year_from_l3_filename(path)) is not None})
        if years:
            available_years[station] = years
    return available_years


def discover_l3_files(data_dir=ICEGAWS_DATA_DIR):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"ICE-GAWS data directory not found: {data_dir}")

    files = {}
    for path in sorted(data_dir.glob(ICEGAWS_L3_GLOB)):
        station = _station_code_from_l3_filename(path)
        if station is None:
            continue
        files.setdefault(station, []).append(path)
    return files


def load_station_locations(location_summary_path=ICEGAWS_LOCATION_SUMMARY):
    location_summary_path = Path(location_summary_path)
    if not location_summary_path.exists():
        raise FileNotFoundError(f"ICE-GAWS location summary not found: {location_summary_path}")

    df = pd.read_csv(location_summary_path)
    expected = {"station", "lat_ave", "lon_ave", "ele_ave"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(
            f"ICE-GAWS location summary missing columns: {', '.join(sorted(missing))}"
        )

    df = df.copy()
    df["station"] = df["station"].astype(str).map(_normalize_station_code)
    df["lat"] = pd.to_numeric(df["lat_ave"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon_ave"], errors="coerce")
    df["ele"] = pd.to_numeric(df["ele_ave"], errors="coerce")
    if "no_years" in df.columns:
        df["no_years"] = pd.to_numeric(df["no_years"], errors="coerce")
    return df


def _ensure_utc_index(df: pd.DataFrame, time_column: str = "Time", source_timezone: str = "UTC"):
    if time_column not in df.columns:
        raise ValueError(f"Missing time column '{time_column}' in ICE-GAWS file")

    out = df.copy()
    out[time_column] = pd.to_datetime(out[time_column], errors="coerce")
    out = out.dropna(subset=[time_column])
    if out[time_column].dt.tz is None:
        out[time_column] = out[time_column].dt.tz_localize(source_timezone)
    else:
        out[time_column] = out[time_column].dt.tz_convert(source_timezone)

    out = out.sort_values(time_column).set_index(time_column)
    out.index.name = "time"
    return out


def _circular_mean_degrees(values: pd.Series, weights: Optional[pd.Series] = None):
    series = pd.to_numeric(values, errors="coerce")
    valid_mask = series.notna().to_numpy()
    series = series[valid_mask]
    if series.empty:
        return np.nan

    radians = np.deg2rad(series.to_numpy(dtype=float))
    if weights is not None:
        if isinstance(weights, pd.Series):
            weight_values = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
        else:
            weight_values = np.asarray(weights, dtype=float)
        if weight_values.shape[0] != valid_mask.shape[0]:
            weight_values = np.asarray(weight_values).reshape(-1)
        weight_values = np.nan_to_num(weight_values, nan=0.0)
        w = weight_values[valid_mask]
        if np.allclose(w.sum(), 0.0):
            w = None

    if weights is None:
        sin_mean = np.nanmean(np.sin(radians))
        cos_mean = np.nanmean(np.cos(radians))
    else:
        sin_mean = np.average(np.sin(radians), weights=w)
        cos_mean = np.average(np.cos(radians), weights=w)

    angle = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360.0
    return float(angle)


def _is_wind_direction_column(column_name: str) -> bool:
    name = str(column_name).strip().lower()
    return name == "wd" or name.startswith("wd_")


def _paired_wind_speed_column(direction_column: str, available_columns) -> Optional[str]:
    direction_name = str(direction_column).strip().lower()
    available_map = {str(col).strip().lower(): col for col in available_columns}

    if direction_name == "wd":
        candidates = ["ws", "ws_4m", "ws_2m"]
    elif direction_name.startswith("wd_"):
        candidates = ["ws_" + direction_name[3:], "ws", "ws_4m", "ws_2m"]
    else:
        candidates = []

    for candidate in candidates:
        if candidate in available_map:
            return available_map[candidate]
    return None


def resample_icegaws_l3_to_carra_step(
    df: pd.DataFrame,
    step: str = "3H",
    smoothing_half_window_hours: int = 0,
):
    if df.index.name != "time":
        raise ValueError("ICE-GAWS dataframe must be indexed by time before resampling")

    if smoothing_half_window_hours < 0:
        raise ValueError("smoothing_half_window_hours must be >= 0")

    normalized_step = str(step).strip().lower()
    if normalized_step.endswith("h"):
        normalized_step = normalized_step[:-1] + "h"
    elif normalized_step.endswith("h"):
        normalized_step = normalized_step[:-1] + "h"
    if normalized_step in {"3h"}:
        normalized_step = "3h"

    def aggregate(group: pd.DataFrame):
        if isinstance(group, pd.Series):
            group = group.to_frame().T

        result = {}
        for column in group.columns:
            series = pd.to_numeric(group[column], errors="coerce")
            if series.notna().sum() == 0:
                continue

            if column.lower() == "precip":
                result[column] = series.sum(min_count=1)
            elif _is_wind_direction_column(column):
                speed_column = _paired_wind_speed_column(column, group.columns)
                weights = None
                if speed_column is not None:
                    weights = pd.to_numeric(group[speed_column], errors="coerce")
                result[column] = _circular_mean_degrees(series, weights=weights)
            else:
                result[column] = series.mean()
        return pd.Series(result)

    def aggregate_with_smoothing(group: pd.DataFrame):
        if isinstance(group, pd.Series):
            group = group.to_frame().T

        if smoothing_half_window_hours == 0:
            return aggregate(group)

        centered = group.copy()
        for column in centered.columns:
            series = pd.to_numeric(centered[column], errors="coerce")
            if series.notna().sum() == 0:
                continue
            if column.lower() == "precip":
                centered[column] = series.rolling(
                    window=2 * smoothing_half_window_hours + 1,
                    center=True,
                    min_periods=1,
                ).sum()
            elif _is_wind_direction_column(column):
                speed_column = _paired_wind_speed_column(column, centered.columns)
                weights = None
                if speed_column is not None:
                    weights = pd.to_numeric(centered[speed_column], errors="coerce")
                values = []
                for idx in range(len(centered)):
                    local_slice = centered.iloc[max(0, idx - smoothing_half_window_hours):min(len(centered), idx + smoothing_half_window_hours + 1)]
                    local_series = pd.to_numeric(local_slice[column], errors="coerce")
                    local_weights = None
                    if speed_column is not None:
                        local_weights = pd.to_numeric(local_slice[speed_column], errors="coerce")
                    values.append(_circular_mean_degrees(local_series, weights=local_weights))
                centered[column] = values
            else:
                centered[column] = pd.to_numeric(centered[column], errors="coerce").rolling(
                    window=2 * smoothing_half_window_hours + 1,
                    center=True,
                    min_periods=1,
                ).mean()
        return aggregate(centered)

    grouped = df.groupby(pd.Grouper(freq=normalized_step, origin="start_day", label="left", closed="left"))
    rows = []
    index = []
    for timestamp, group in grouped:
        if group.empty:
            continue
        rows.append(aggregate_with_smoothing(group))
        index.append(timestamp)

    if not rows:
        return df.iloc[0:0].copy()

    resampled = pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="time"))
    return resampled.sort_index()


def load_icegaws_l3_csv(
    file_path,
    time_column: str = "Time",
    source_timezone: str = "UTC",
    resample_step: Optional[str] = "3h",
    resample_smoothing_half_window_hours: int = 0,
):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"ICE-GAWS L3 file not found: {file_path}")

    df = pd.read_csv(file_path)
    df = _ensure_utc_index(df, time_column=time_column, source_timezone=source_timezone)
    if resample_step is not None:
        df = resample_icegaws_l3_to_carra_step(
            df,
            step=resample_step,
            smoothing_half_window_hours=resample_smoothing_half_window_hours,
        )
    return df


def _load_carra_t2m_jja_for_year(carra_data_dir, year: str):
    from shared.reader import open_t2m_netcdf, crop_iceland
    import xarray as xr

    carra_data_dir = Path(carra_data_dir)
    monthly = []
    for month in ("June", "July", "August"):
        folder = carra_data_dir / str(year) / f"CARRA2_{month}"
        if not folder.exists():
            continue
        try:
            ds_month = crop_iceland(open_t2m_netcdf(folder), "t2m")
        except (FileNotFoundError, ValueError):
            # Some years can miss t2m for a given month; skip it instead of failing the full build.
            continue
        if "t2m" not in ds_month.data_vars:
            continue
        monthly.append(ds_month[["t2m"]])

    if not monthly:
        return None

    # Months are appended in chronological JJA order, so sorting here only forces
    # an unnecessary full-array copy of the 3D field.
    ds = xr.concat(monthly, dim="time") if len(monthly) > 1 else monthly[0]
    return ds


def _build_station_carra_spatial_temperature_reference(
    station_index: pd.DatetimeIndex,
    carra_t2m_ds,
    y: int,
    x: int,
    window_size: int,
    match_tolerance: str = "2h",
):
    if carra_t2m_ds is None or "t2m" not in carra_t2m_ds.data_vars:
        return pd.Series(index=station_index, dtype=float)

    t2m_da = carra_t2m_ds["t2m"]
    if t2m_da.ndim != 3:
        return pd.Series(index=station_index, dtype=float)

    half_window = max(0, int(window_size) // 2)
    y0 = max(0, int(y) - half_window)
    y1 = min(int(t2m_da.sizes["y"]), int(y) + half_window + 1)
    x0 = max(0, int(x) - half_window)
    x1 = min(int(t2m_da.sizes["x"]), int(x) + half_window + 1)

    if y1 <= y0 or x1 <= x0:
        return pd.Series(index=station_index, dtype=float)

    # Only compute the local spatial mean needed for this station to avoid allocating
    # a full filtered copy of the time x y x grid.
    local_window = t2m_da.isel(y=slice(y0, y1), x=slice(x0, x1))
    series_values = local_window.mean(dim=("y", "x"), skipna=True).load().to_numpy(dtype=float)

    carra_time_index = pd.DatetimeIndex(pd.to_datetime(carra_t2m_ds["time"].values, utc=True))
    carra_series = pd.Series(series_values, index=carra_time_index, dtype=float)
    aligned = carra_series.reindex(station_index, method="nearest", tolerance=pd.Timedelta(match_tolerance))
    return aligned


def _haversine_km(lon1, lat1, lon2, lat2):
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 6371.0 * (2.0 * np.arcsin(np.sqrt(a)))


def _normalize_longitudes(lon_values):
    return np.where(lon_values > 180.0, lon_values - 360.0, lon_values)


def nearest_carra_pixel(station_lon: float, station_lat: float, lon_grid, lat_grid):
    lon_grid = np.asarray(lon_grid, dtype=float)
    lat_grid = np.asarray(lat_grid, dtype=float)
    lon_grid = _normalize_longitudes(lon_grid)
    station_lon = station_lon - 360.0 if station_lon > 180.0 else station_lon

    distances = _haversine_km(station_lon, station_lat, lon_grid, lat_grid)
    flat_index = int(np.nanargmin(distances))
    y, x = np.unravel_index(flat_index, distances.shape)
    return {
        "y": int(y),
        "x": int(x),
        "carra_lon": float(lon_grid[y, x]),
        "carra_lat": float(lat_grid[y, x]),
        "distance_km": float(distances[y, x]),
    }


def build_station_pixel_mapping(
    carra_ds,
    station_locations: Optional[pd.DataFrame] = None,
    station_codes: Optional[Iterable[str]] = None,
    data_dir=ICEGAWS_DATA_DIR,
    location_summary_path=ICEGAWS_LOCATION_SUMMARY,
):
    if station_locations is None:
        station_locations = load_station_locations(location_summary_path)

    if "longitude" not in carra_ds.variables and "lon" not in carra_ds.variables:
        raise ValueError("CARRA reference dataset must expose longitude/latitudes")

    lon_name = "longitude" if "longitude" in carra_ds.variables else "lon"
    lat_name = "latitude" if "latitude" in carra_ds.variables else "lat"
    lon_grid = carra_ds[lon_name].values
    lat_grid = carra_ds[lat_name].values

    if lon_grid.ndim == 1 and lat_grid.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    station_codes_set = {str(code).strip() for code in station_codes} if station_codes is not None else None

    files_by_station = discover_l3_files(data_dir)
    mapping = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "carra_reference": {
            "grid_shape": [int(carra_ds.sizes.get("y", lon_grid.shape[0])), int(carra_ds.sizes.get("x", lon_grid.shape[1]))],
            "lon_name": lon_name,
            "lat_name": lat_name,
        },
        "stations": {},
    }

    for _, row in station_locations.iterrows():
        station = str(row["station"]).strip()
        if station_codes_set is not None and station not in station_codes_set:
            continue
        if station not in files_by_station:
            continue

        lat = row.get("lat")
        lon = row.get("lon")
        if pd.isna(lat) or pd.isna(lon):
            continue

        pixel = nearest_carra_pixel(float(lon), float(lat), lon_grid, lat_grid)
        mapping["stations"][station] = {
            "location": {
                "lat": float(lat),
                "lon": float(lon),
                "ele": None if pd.isna(row.get("ele")) else float(row.get("ele")),
                "no_years": None if "no_years" not in row or pd.isna(row.get("no_years")) else int(row.get("no_years")),
            },
            "nearest_carra_pixel": pixel,
            "l3_files": [str(p) for p in files_by_station.get(station, [])],
        }

    return mapping


def build_station_static_mapping_json(
    carra_orography_ds,
    station_locations: Optional[pd.DataFrame] = None,
    station_codes: Optional[Iterable[str]] = None,
    data_dir=ICEGAWS_DATA_DIR,
    location_summary_path=ICEGAWS_LOCATION_SUMMARY,
    basins_path=None,
):
    station_locations = load_station_locations(location_summary_path) if station_locations is None else station_locations
    pixel_mapping = build_station_pixel_mapping(
        carra_ds=carra_orography_ds,
        station_locations=station_locations,
        station_codes=station_codes,
        data_dir=data_dir,
        location_summary_path=location_summary_path,
    )
    basins = _load_or_build_basins(basins_path)

    static_json = {
        "generated_at_utc": pixel_mapping["generated_at_utc"],
        "carra_reference": pixel_mapping["carra_reference"],
        "stations": {},
    }

    for station_name, station_payload in pixel_mapping["stations"].items():
        pixel = station_payload["nearest_carra_pixel"]
        y = int(pixel["y"])
        x = int(pixel["x"])
        basin_label = int(basins[y, x]) if basins.ndim == 2 else 0
        if basin_label <= 0:
            basin_label = _nearest_nonzero_basin_label(basins, y, x, search_radius=5)

        static_json["stations"][station_name] = {
            **station_payload,
            "katabatic_basin_label": int(basin_label),
        }

    return static_json


def write_station_static_mapping_json(
    output_path,
    carra_orography_ds,
    station_locations: Optional[pd.DataFrame] = None,
    station_codes: Optional[Iterable[str]] = None,
    data_dir=ICEGAWS_DATA_DIR,
    location_summary_path=ICEGAWS_LOCATION_SUMMARY,
    basins_path=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_station_static_mapping_json(
        carra_orography_ds=carra_orography_ds,
        station_locations=station_locations,
        station_codes=station_codes,
        data_dir=data_dir,
        location_summary_path=location_summary_path,
        basins_path=basins_path,
    )
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_station_pixel_mapping_json(
    output_path,
    carra_ds,
    station_locations: Optional[pd.DataFrame] = None,
    station_codes: Optional[Iterable[str]] = None,
    data_dir=ICEGAWS_DATA_DIR,
    location_summary_path=ICEGAWS_LOCATION_SUMMARY,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = build_station_pixel_mapping(
        carra_ds=carra_ds,
        station_locations=station_locations,
        station_codes=station_codes,
        data_dir=data_dir,
        location_summary_path=location_summary_path,
    )
    output_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def build_station_overlay_points_from_json(
    static_mapping_json,
    detector_json=None,
    station_codes: Optional[Iterable[str]] = None,
    value_key: str = "event_percentage",
    normalize_value: bool = True,
    value_scale: float = 100.0,
):
    station_subset = {str(code).strip() for code in station_codes} if station_codes is not None else None
    detector_stations = {}
    if detector_json is not None:
        detector_stations = detector_json.get("stations", {})

    points = []
    for station_name, station_payload in static_mapping_json.get("stations", {}).items():
        if station_subset is not None and station_name not in station_subset:
            continue

        location = station_payload.get("location", {})
        detector_payload = detector_stations.get(station_name, {})
        detector_summary = detector_payload.get("detector_summary", {})
        raw_value = detector_summary.get(value_key)
        if raw_value is None and value_key == "event_fraction":
            fallback_percent = detector_summary.get("event_percentage")
            if fallback_percent is not None and not pd.isna(fallback_percent):
                raw_value = float(fallback_percent) / 100.0
        value = None
        if raw_value is not None and not pd.isna(raw_value):
            value = float(raw_value)
            if normalize_value and value_scale not in (0, None):
                value = value / float(value_scale)

        event_fraction = detector_summary.get("event_fraction")
        if event_fraction is None:
            event_percentage = detector_summary.get("event_percentage")
            if event_percentage is not None and not pd.isna(event_percentage):
                event_fraction = float(event_percentage) / 100.0

        points.append(
            {
                "station": station_name,
                "label": station_name,
                "lon": location.get("lon"),
                "lat": location.get("lat"),
                "value": value,
                "event_fraction": event_fraction,
                "event_percentage": detector_summary.get("event_percentage"),
                "katabatic_basin_label": station_payload.get("katabatic_basin_label"),
                "nearest_carra_pixel": station_payload.get("nearest_carra_pixel"),
            }
        )

    return points


def load_station_overlay_points_from_json(
    static_mapping_path,
    detector_json_path=None,
    station_codes: Optional[Iterable[str]] = None,
    value_key: str = "event_percentage",
    normalize_value: bool = True,
    value_scale: float = 100.0,
):
    static_mapping_path = Path(static_mapping_path)
    static_mapping_json = json.loads(static_mapping_path.read_text(encoding="utf-8"))
    detector_json = None
    if detector_json_path is not None:
        detector_json = json.loads(Path(detector_json_path).read_text(encoding="utf-8"))

    return build_station_overlay_points_from_json(
        static_mapping_json=static_mapping_json,
        detector_json=detector_json,
        station_codes=station_codes,
        value_key=value_key,
        normalize_value=normalize_value,
        value_scale=value_scale,
    )


def _load_carra_orography_array(carra_orography_ds):
    if "orog" not in carra_orography_ds.data_vars:
        raise ValueError("CARRA orography dataset must contain an 'orog' variable")
    orography = carra_orography_ds["orog"]
    if "time" in orography.dims:
        orography = orography.isel(time=0)
    return np.asarray(orography.values, dtype=float)


def _station_local_slope_direction_deg(orography_2d, y, x):
    dh_dy, dh_dx = np.gradient(orography_2d)
    slope_vector_y = -float(dh_dy[y, x])
    slope_vector_x = -float(dh_dx[y, x])
    slope_dir_math_rad = np.arctan2(slope_vector_y, slope_vector_x)
    slope_dir_deg = float((90.0 - np.rad2deg(slope_dir_math_rad) + 360.0) % 360.0)
    return slope_dir_deg, float(np.deg2rad(slope_dir_deg))


def _nearest_nonzero_basin_label(basins, y, x, search_radius=5):
    y0 = max(0, int(y) - int(search_radius))
    y1 = min(basins.shape[0], int(y) + int(search_radius) + 1)
    x0 = max(0, int(x) - int(search_radius))
    x1 = min(basins.shape[1], int(x) + int(search_radius) + 1)
    window = basins[y0:y1, x0:x1]
    window = window[np.asarray(window) > 0]
    if window.size == 0:
        return 0
    values, counts = np.unique(window, return_counts=True)
    return int(values[np.argmax(counts)])


def _load_or_build_basins(basins_path=None):
    if basins_path is None:
        basins_path = DATA_DIR / "NPY" / "basins.npy"
    basins_path = Path(basins_path)
    if not basins_path.exists():
        raise FileNotFoundError(
            f"Basin labels not found: {basins_path}. Run the katabatic pipeline first so it writes basins.npy."
        )
    return np.load(basins_path)


def _find_first_existing_column(df: pd.DataFrame, candidates):
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _to_celsius_if_kelvin(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values

    median_value = float(np.nanmedian(finite))
    if median_value > 150.0:
        return values - 273.15
    return values


def _iter_runs(mask_array: np.ndarray):
    start = None
    for idx, value in enumerate(mask_array):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, len(mask_array)


def _resolve_detector_columns(
    station_df: pd.DataFrame,
    temperature_column: str,
    wind_speed_column: str,
    wind_direction_column: str,
):
    temp_candidates = [temperature_column]
    if temperature_column != "t_2m":
        temp_candidates.append("t_2m")

    speed_candidates = [wind_speed_column]
    speed_candidates.extend(["ws_4m", "ws_2m", "ws"])

    direction_candidates = [wind_direction_column]
    direction_candidates.extend(["wd_4m", "wd_2m", "wd"])

    temperature_used = _find_first_existing_column(station_df, temp_candidates)
    wind_speed_used = _find_first_existing_column(station_df, speed_candidates)
    wind_direction_used = _find_first_existing_column(station_df, direction_candidates)

    missing = []
    if temperature_used is None:
        missing.append(temperature_column)
    if wind_speed_used is None:
        missing.append(wind_speed_column)
    if wind_direction_used is None:
        missing.append(wind_direction_column)

    return temperature_used, wind_speed_used, wind_direction_used, missing


def detect_station_katabatic_events(
    station_df: pd.DataFrame,
    slope_direction_deg: float,
    window_size: int = 15,
    dT_threshold: float = 1.5,
    wind_threshold: float = 2.5,
    direction_threshold: float = 0.87,
    temperature_column: str = "t_2m",
    wind_speed_column: str = "ws_4m",
    wind_direction_column: str = "wd_4m",
    temperature_reference_mode: str = "spatial_carra",
    spatial_temperature_reference: Optional[pd.Series] = None,
    season: Optional[Iterable[int]] = None,
    min_event_duration: int = 1,
):
    if not isinstance(station_df.index, pd.DatetimeIndex):
        raise ValueError("Station dataframe must be indexed by datetime")

    temperature_used, wind_speed_used, wind_direction_used, missing_columns = _resolve_detector_columns(
        station_df,
        temperature_column=temperature_column,
        wind_speed_column=wind_speed_column,
        wind_direction_column=wind_direction_column,
    )
    if missing_columns:
        raise ValueError(f"Station dataframe missing columns: {', '.join(missing_columns)}")

    data = station_df.copy()
    temperature = _to_celsius_if_kelvin(pd.to_numeric(data[temperature_used], errors="coerce"))
    wind_speed = pd.to_numeric(data[wind_speed_used], errors="coerce")
    wind_direction = pd.to_numeric(data[wind_direction_used], errors="coerce")

    if temperature_reference_mode != "spatial_carra":
        raise ValueError(
            "Only temperature_reference_mode='spatial_carra' is supported; temporal mode has been removed."
        )
    if spatial_temperature_reference is None:
        raise ValueError("Spatial temperature reference is required when temperature_reference_mode='spatial_carra'.")
    temperature_reference = _to_celsius_if_kelvin(
        pd.to_numeric(spatial_temperature_reference, errors="coerce")
    ).reindex(temperature.index)

    cold_mask = (temperature - temperature_reference) < -dT_threshold
    strong_mask = wind_speed >= wind_threshold

    slope_direction_deg = float(np.mod(float(slope_direction_deg), 360.0))
    wind_direction_deg = np.mod(wind_direction.astype(float), 360.0)
    alignment_mask = np.cos(np.deg2rad(wind_direction_deg - slope_direction_deg)) >= direction_threshold

    if season is None:
        season = [6, 7, 8]
    season_mask = temperature.index.month.isin(list(season))
    valid_mask = (
        temperature.notna()
        & wind_speed.notna()
        & wind_direction.notna()
        & temperature_reference.notna()
        & season_mask
    )
    event_candidates = valid_mask & cold_mask & strong_mask & alignment_mask

    if min_event_duration > 1:
        event_mask = event_candidates.copy()
        run_lengths = []
        current_run = 0
        for value in event_candidates.astype(bool).tolist():
            if value:
                current_run += 1
            else:
                if current_run > 0:
                    run_lengths.append(current_run)
                current_run = 0
        if current_run > 0:
            run_lengths.append(current_run)

        event_mask = pd.Series(False, index=event_candidates.index)
        for start, end in _iter_runs(event_candidates.astype(bool).to_numpy()):
            if end - start >= min_event_duration:
                event_mask.iloc[start:end] = True
    else:
        event_mask = event_candidates

    valid_steps = int(valid_mask.sum())
    event_count = int(event_mask.sum())
    event_fraction = float(event_count / valid_steps) if valid_steps > 0 else np.nan
    event_percentage = float(100.0 * event_count / valid_steps) if valid_steps > 0 else np.nan

    return {
        "event_mask": event_mask,
        "valid_mask": valid_mask,
        "temperature_reference": temperature_reference,
        "event_count": event_count,
        "valid_step_count": valid_steps,
        "event_fraction": event_fraction,
        "event_percentage": event_percentage,
        "temperature_reference_mode": temperature_reference_mode,
        "temperature_column": temperature_used,
        "wind_speed_column": wind_speed_used,
        "wind_direction_column": wind_direction_used,
    }


def build_station_katabatic_detector_json(
    carra_orography_ds=None,
    carra_ds=None,
    station_locations: Optional[pd.DataFrame] = None,
    station_codes: Optional[Iterable[str]] = None,
    data_dir=ICEGAWS_DATA_DIR,
    location_summary_path=ICEGAWS_LOCATION_SUMMARY,
    basins_path=None,
    window_size: int = 15,
    dT_threshold: float = 1.5,
    wind_threshold: float = 2.5,
    direction_threshold: float = 0.87,
    temperature_column: str = "t_2m",
    wind_speed_column: str = "ws_4m",
    wind_direction_column: str = "wd_4m",
    temperature_reference_mode: str = "spatial_carra",
    carra_t2m_data_dir=None,
    carra_time_match_tolerance: str = "2h",
    resample_step: str = "3h",
):
    if carra_orography_ds is None:
        carra_orography_ds = carra_ds
    if carra_orography_ds is None:
        raise ValueError("A CARRA dataset must be provided via carra_orography_ds or carra_ds")
    if temperature_reference_mode != "spatial_carra":
        raise ValueError(
            "Only temperature_reference_mode='spatial_carra' is supported; temporal mode has been removed."
        )
    if carra_t2m_data_dir is None:
        raise ValueError("carra_t2m_data_dir is required when temperature_reference_mode='spatial_carra'.")

    station_locations = load_station_locations(location_summary_path) if station_locations is None else station_locations
    pixel_mapping = build_station_pixel_mapping(
        carra_ds=carra_orography_ds,
        station_locations=station_locations,
        station_codes=station_codes,
        data_dir=data_dir,
        location_summary_path=location_summary_path,
    )
    basins = _load_or_build_basins(basins_path)
    orography_2d = _load_carra_orography_array(carra_orography_ds)
    files_by_station = discover_l3_files(data_dir)
    carra_t2m_cache = {}

    detector_json = {
        "detector": {
            "name": "katabatic_station_detector",
            "season_filter": "JJA",
            "window_size": int(window_size),
            "dT_threshold": float(dT_threshold),
            "wind_threshold": float(wind_threshold),
            "direction_threshold": float(direction_threshold),
            "temperature_column": temperature_column,
            "wind_speed_column": wind_speed_column,
            "wind_direction_column": wind_direction_column,
            "temperature_reference_mode": temperature_reference_mode,
            "carra_time_match_tolerance": carra_time_match_tolerance,
            "resample_step": resample_step,
        },
        "generated_at_utc": pixel_mapping["generated_at_utc"],
        "carra_reference": pixel_mapping["carra_reference"],
        "stations": {},
    }

    station_subset = set(str(code).strip() for code in station_codes) if station_codes is not None else None

    for station_name, station_payload in pixel_mapping["stations"].items():
        if station_subset is not None and station_name not in station_subset:
            continue

        pixel = station_payload["nearest_carra_pixel"]
        y = int(pixel["y"])
        x = int(pixel["x"])
        slope_direction_deg, slope_direction_rad = _station_local_slope_direction_deg(orography_2d, y, x)
        basin_label = int(basins[y, x]) if basins.ndim == 2 else 0
        if basin_label <= 0:
            basin_label = _nearest_nonzero_basin_label(basins, y, x, search_radius=5)

        file_summaries = []
        total_event_count = 0
        total_valid_steps = 0

        for file_path in files_by_station.get(station_name, []):
            station_df = load_icegaws_l3_csv(file_path, resample_step=resample_step)
            station_df = station_df[station_df.index.month.isin([6, 7, 8])]
            spatial_temperature_reference = None
            year = _year_from_l3_filename(file_path)
            if year is not None:
                if year not in carra_t2m_cache:
                    carra_t2m_cache[year] = _load_carra_t2m_jja_for_year(carra_t2m_data_dir, year)
                spatial_temperature_reference = _build_station_carra_spatial_temperature_reference(
                    station_index=station_df.index,
                    carra_t2m_ds=carra_t2m_cache[year],
                    y=y,
                    x=x,
                    window_size=window_size,
                    match_tolerance=carra_time_match_tolerance,
                )
            try:
                stats = detect_station_katabatic_events(
                    station_df,
                    slope_direction_deg=slope_direction_deg,
                    window_size=window_size,
                    dT_threshold=dT_threshold,
                    wind_threshold=wind_threshold,
                    direction_threshold=direction_threshold,
                    temperature_column=temperature_column,
                    wind_speed_column=wind_speed_column,
                    wind_direction_column=wind_direction_column,
                    temperature_reference_mode=temperature_reference_mode,
                    spatial_temperature_reference=spatial_temperature_reference,
                )
            except ValueError as exc:
                if "missing columns" not in str(exc).lower():
                    raise
                file_summaries.append(
                    {
                        "file": str(file_path),
                        "time_start": None if station_df.empty else station_df.index.min().isoformat(),
                        "time_end": None if station_df.empty else station_df.index.max().isoformat(),
                        "event_count": 0,
                        "valid_step_count": 0,
                        "event_fraction": np.nan,
                        "event_percentage": np.nan,
                        "skipped": True,
                        "skip_reason": str(exc),
                    }
                )
                continue

            total_event_count += int(stats["event_count"])
            total_valid_steps += int(stats["valid_step_count"])
            file_summaries.append(
                {
                    "file": str(file_path),
                    "time_start": None if station_df.empty else station_df.index.min().isoformat(),
                    "time_end": None if station_df.empty else station_df.index.max().isoformat(),
                    "event_count": int(stats["event_count"]),
                    "valid_step_count": int(stats["valid_step_count"]),
                    "event_fraction": float(stats["event_fraction"]),
                    "event_percentage": float(stats["event_percentage"]),
                    "temperature_reference_mode": stats.get("temperature_reference_mode"),
                    "temperature_column": stats.get("temperature_column"),
                    "wind_speed_column": stats.get("wind_speed_column"),
                    "wind_direction_column": stats.get("wind_direction_column"),
                }
            )

        station_event_fraction = float(total_event_count / total_valid_steps) if total_valid_steps > 0 else np.nan
        station_event_percentage = float(100.0 * total_event_count / total_valid_steps) if total_valid_steps > 0 else np.nan

        detector_json["stations"][station_name] = {
            "station": station_name,
            "katabatic_slope_direction_deg": float(slope_direction_deg),
            "katabatic_slope_direction_rad": float(slope_direction_rad),
            "detector_summary": {
                "event_count": int(total_event_count),
                "valid_step_count": int(total_valid_steps),
                "event_fraction": float(station_event_fraction),
                "event_percentage": float(station_event_percentage),
                "temperature_reference_mode": temperature_reference_mode,
                "per_file": file_summaries,
            },
        }

    return detector_json


def write_station_katabatic_detector_json(
    output_path,
    carra_orography_ds=None,
    carra_ds=None,
    station_locations: Optional[pd.DataFrame] = None,
    station_codes: Optional[Iterable[str]] = None,
    data_dir=ICEGAWS_DATA_DIR,
    location_summary_path=ICEGAWS_LOCATION_SUMMARY,
    basins_path=None,
    window_size: int = 15,
    dT_threshold: float = 1.5,
    wind_threshold: float = 2.5,
    direction_threshold: float = 0.87,
    temperature_column: str = "t_2m",
    wind_speed_column: str = "ws_4m",
    wind_direction_column: str = "wd_4m",
    temperature_reference_mode: str = "spatial_carra",
    carra_t2m_data_dir=None,
    carra_time_match_tolerance: str = "2h",
    resample_step: str = "3h",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_station_katabatic_detector_json(
        carra_orography_ds=carra_orography_ds,
        carra_ds=carra_ds,
        station_locations=station_locations,
        station_codes=station_codes,
        data_dir=data_dir,
        location_summary_path=location_summary_path,
        basins_path=basins_path,
        window_size=window_size,
        dT_threshold=dT_threshold,
        wind_threshold=wind_threshold,
        direction_threshold=direction_threshold,
        temperature_column=temperature_column,
        wind_speed_column=wind_speed_column,
        wind_direction_column=wind_direction_column,
        temperature_reference_mode=temperature_reference_mode,
        carra_t2m_data_dir=carra_t2m_data_dir,
        carra_time_match_tolerance=carra_time_match_tolerance,
        resample_step=resample_step,
    )
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path