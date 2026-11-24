"""Read and validate CARRA2 NetCDF files and related spatial datasets."""

import pandas as pd
import xarray as xr
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import netCDF4 as nc
import matplotlib.pyplot as plt


# CSV Data reader
def station_reader (file): # checked (usefull)
    df = pd.read_csv(file, sep=';', encoding='utf-8')
    df['decade'] = (df['year'] // 10) * 10
    return df


# Netcdf Data readers

VALID_YEARS_CACHE_FILE = "_katabatic_valid_years_cache.json"

def _list_nc_files(folder): #checked (usefull)
    # Centralized guard: all NetCDF readers rely on this file discovery.
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {folder}")
    return sorted(folder.glob("*.nc"))


def _format_time_value(value):
    try:
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit="s")
        return str(value)
    except Exception:
        return str(value)


def _print_dataset_summary(ds, var_name, file_path):
    dims = {k: int(v) for k, v in ds.sizes.items()}
    print(f"[reader] Opened '{var_name}' from: {file_path}")
    print(f"[reader] Dimensions: {dims}")

    if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
        time_values = ds["time"].values
        first = _format_time_value(time_values[0])
        last = _format_time_value(time_values[-1])
        print(f"[reader] Time coverage: {first} -> {last} ({len(time_values)} steps)")


def _prepare_step_metadata(nc_file_path): #checked (usefull)
    # Some input files contain an unexpected "dtype" NetCDF attribute on "step".
    # Removing it in place avoids downstream decoding edge cases.
    try:
        ds = nc.Dataset(str(nc_file_path), "r+")
    except Exception as exc:
        print(f"[reader] Warning: could not inspect metadata in {nc_file_path}: {exc}")
        return

    try:
        if "step" in ds.variables:
            var = ds.variables["step"]
            if "dtype" in var.ncattrs():
                var.delncattr("dtype")
                print(f"[reader] Removed unexpected 'dtype' attribute from {nc_file_path.name}")
    except Exception as exc:
        print(f"[reader] Warning: could not sanitize 'step' in {nc_file_path}: {exc}")
    finally:
        ds.close()


def _normalize_text_list(values): #checked (usefull)
    """Normalize a list of strings to lowercase."""
    return [str(v) for v in values]


def _valid_years_cache_path(path): #checked (usefull)
    """Return the path to the valid years cache file for a given data root."""
    return Path(path) / VALID_YEARS_CACHE_FILE


def write_valid_years_cache(path, months, required_vars, valid_years, strict_timestamps, source="data_quality_check"): #checked (usefull)
    """Persist computed valid years so other commands can reuse them."""
    cache_path = _valid_years_cache_path(path)
    months_n = _normalize_text_list(months)
    vars_n = _normalize_text_list(required_vars)

    payload = {}
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    entries = payload.get("entries", [])
    new_entry = {
        "months": months_n,
        "required_vars": vars_n,
        "strict_timestamps": bool(strict_timestamps),
        "valid_years": _normalize_text_list(valid_years),
        "computed_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source": source,
    }

    filtered_entries = []
    for entry in entries:
        # Replace only entries with the same key tuple, keep other cache variants.
        same_key = (
            entry.get("months") == months_n
            and entry.get("required_vars") == vars_n
            and bool(entry.get("strict_timestamps", False)) == bool(strict_timestamps)
        )
        if not same_key:
            filtered_entries.append(entry)

    filtered_entries.append(new_entry)
    payload["entries"] = filtered_entries

    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[reader] Saved valid years cache to: {cache_path}")
    return cache_path


def read_valid_years_cache(path, months, required_vars, prefer_strict_timestamps=False): #checked (usefull)
    """Read valid years computed previously by data_quality_check for matching settings.

    Preference order:
    - prefer_strict_timestamps=True: longest strict list first, then longest relaxed list
    - prefer_strict_timestamps=False: longest relaxed list first, then longest strict list
    - prefer_strict_timestamps=None: most recently stored matching entry
    """
    cache_path = _valid_years_cache_path(path)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[reader] Warning: could not parse valid years cache {cache_path}: {exc}")
        return None

    months_n = _normalize_text_list(months)
    vars_n = _normalize_text_list(required_vars)
    entries = payload.get("entries", [])

    matches = [
        e for e in entries
        if e.get("months") == months_n and e.get("required_vars") == vars_n
    ]
    if not matches:
        return None

    def _years_count(entry):
        years = entry.get("valid_years", [])
        return len(years) if isinstance(years, list) else 0

    if prefer_strict_timestamps is True:
        strict_matches = [e for e in matches if bool(e.get("strict_timestamps", False))]
        if strict_matches:
            preferred = max(strict_matches, key=_years_count)
        else:
            relaxed_matches = [e for e in matches if not bool(e.get("strict_timestamps", False))]
            preferred = max(relaxed_matches, key=_years_count) if relaxed_matches else None
    elif prefer_strict_timestamps is False:
        relaxed_matches = [e for e in matches if not bool(e.get("strict_timestamps", False))]
        if relaxed_matches:
            preferred = max(relaxed_matches, key=_years_count)
        else:
            strict_matches = [e for e in matches if bool(e.get("strict_timestamps", False))]
            preferred = max(strict_matches, key=_years_count) if strict_matches else None
    else:
        # Keep full backward compatibility for custom callers that want "latest entry" behavior.
        preferred = matches[-1] if matches else None

    if preferred is None:
        preferred = matches[-1]

    years = preferred.get("valid_years", [])
    if not isinstance(years, list):
        return None

    strict_flag = bool(preferred.get("strict_timestamps", False))
    source = preferred.get("source", "unknown")
    computed_at = preferred.get("computed_at_utc", "unknown")
    print(
        "[reader] Using cached valid years "
        f"(strict_timestamps={strict_flag}, source={source}, computed_at={computed_at})"
    )
    print(f"[reader] Cached valid years: {years}")
    return years


def _discover_years_with_direct_scan(path, months, required_vars):
    """Fall back to a simple directory scan when strict quality checks are too strict."""
    root = Path(path)
    discovered = []

    for year_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue

        month_ok = True
        for month in months:
            month_dir = year_dir / f"CARRA2_{month}"
            if not month_dir.exists():
                month_ok = False
                break

            nc_files = sorted(month_dir.glob("*.nc"))
            if not nc_files:
                month_ok = False
                break

            has_required_vars = False
            for nc_file in nc_files:
                try:
                    with xr.open_dataset(nc_file, decode_timedelta=False) as ds:
                        if all(var in ds.data_vars for var in required_vars):
                            has_required_vars = True
                            break
                except Exception:
                    continue

            if not has_required_vars:
                month_ok = False
                break

        if month_ok:
            discovered.append(year_dir.name)

    return discovered


def resolve_available_years(
    path,
    required_months,
    required_vars=None,
    requested_year="multi",
    expected_min_years=None,
    prefer_strict_timestamps=False,
): #checked (usefull)
    """Resolve the years to process and optionally enforce a minimum count."""
    months = list(required_months)
    vars_required = list(required_vars) if required_vars is not None else []

    if isinstance(requested_year, (list, tuple, set)):
        years = [str(y) for y in requested_year]
    elif requested_year == "multi":
        years = read_valid_years_cache(
            path,
            months,
            vars_required,
            prefer_strict_timestamps=prefer_strict_timestamps,
        )
        if not years:
            from katabatic.data_quality_check import run_quality_check

            # Use the dedicated quality check to validate years before processing.
            strict_mode = bool(prefer_strict_timestamps)
            years, _ = run_quality_check(
                Path(path),
                months,
                vars_required,
                strict_timestamps=strict_mode,
            )

            if not years:
                years = _discover_years_with_direct_scan(path, months, vars_required)

            write_valid_years_cache(
                path=Path(path),
                months=months,
                required_vars=vars_required,
                valid_years=years,
                strict_timestamps=strict_mode,
                source="shared.reader.resolve_available_years",
            )
        if not years:
            raise ValueError(
                "No available years found with complete monthly data and required variables "
                f"in {path} for months: {', '.join(months)}"
            )
    else:
        years = [str(requested_year)]

    years = sorted(years, key=lambda y: (0, int(y)) if str(y).isdigit() else (1, str(y)))

    if expected_min_years is not None and len(years) < expected_min_years:
        raise ValueError(
            f"Expected at least {expected_min_years} years to process, but found {len(years)}: {years}"
        )

    return years

def open_t2m_netcdf(folder): #checked (usefull)
    folder = Path(folder)
    nc_files = _list_nc_files(folder)
    print(f"[reader] Opening t2m from folder: {folder}")
    print(f"[reader] Candidate files: {len(nc_files)}")

    for nc_file in nc_files:
        # Metadata sanitation is idempotent and done once per candidate file.
        _prepare_step_metadata(nc_file)

    for nc_file in nc_files:
        try:
            ds = xr.open_dataset(nc_file, decode_timedelta=False)
            if "t2m" in ds.data_vars:
                _print_dataset_summary(ds, "t2m", nc_file)
                return ds
            ds.close()
        except Exception as exc:
            print(f"[reader] Warning: could not open {nc_file}: {exc}")

    raise ValueError(f"No .nc file containing 't2m' was found in {folder}.")

def open_orography_netcdf(folder): #checked (usefull)
    folder = Path(folder)
    nc_files = _list_nc_files(folder)
    print(f"[reader] Opening orography from folder: {folder}")
    print(f"[reader] Candidate files: {len(nc_files)}")

    for nc_file in nc_files:
        try:
            ds = xr.open_dataset(nc_file, decode_timedelta=False)
            if "orog" in ds.data_vars:
                _print_dataset_summary(ds, "orog", nc_file)
                return ds
            ds.close()
        except Exception as exc:
            print(f"[reader] Warning: could not open {nc_file}: {exc}")

    raise ValueError(f"No .nc file containing 'orog' was found in {folder}.")

def open_uv_netcdf(folder): #checked (usefull)
    folder = Path(folder)
    nc_files = _list_nc_files(folder)
    print(f"[reader] Opening wind components from folder: {folder}")
    print(f"[reader] Candidate files: {len(nc_files)}")

    for nc_file in nc_files:
        try:
            ds = xr.open_dataset(nc_file, decode_timedelta=False)
            if "u10" in ds.data_vars and "v10" in ds.data_vars:
                _print_dataset_summary(ds, "u10/v10", nc_file)
                return ds
            ds.close()
        except Exception as exc:
            print(f"[reader] Warning: could not open {nc_file}: {exc}")

    raise ValueError(f"No .nc file containing both 'u10' and 'v10' was found in {folder}.")

# Cropper

def crop_iceland(ds, param): # checked (usefull)
    # Fixed CARRA index window used throughout this project for Iceland extraction.
    # d = ds.isel(
    # y=slice(450, 720),
    # x=slice(1591, 1874)
    # )

    # plt.plot(d["longitude"].values, d["latitude"].values, "o", markersize=1)

    # if param == "t2m":
    #     arr = ds["t2m"]

    #     # On prend une seule carte 2D pour le debug visuel
    #     if "time" in arr.dims:
    #         arr2d = arr.isel(time=0).values
    #     else:
    #         arr2d = arr.values

    #     nans_mask = np.isnan(arr2d)

    #     y0, y1 = 450, 720
    #     x0, x1 = 1591, 1874

    #     lon = ds["longitude"].values
    #     lat = ds["latitude"].values
    #     lon = np.where(lon > 180, lon - 360, lon)  # passe en [-180, 180]

    #     # 4 bords du carré isel (sans dupliquer les coins)
    #     top_lon,    top_lat    = lon[y0, x0:x1],     lat[y0, x0:x1]
    #     bottom_lon, bottom_lat = lon[y1-1, x0:x1],   lat[y1-1, x0:x1]
    #     left_lon,   left_lat   = lon[y0:y1, x0],     lat[y0:y1, x0]
    #     right_lon,  right_lat  = lon[y0:y1, x1-1],   lat[y0:y1, x1-1]

    #     print(f"[reader] isel border indices: x={x0}:{x1}, y={y0}:{y1}")
    #     # print(f"[reader] coordonnées du bord bas du carré isel: lon={lon[y0, x0:x1]}, lat={lat[y0, x0:x1]}")
    #     # print(f"[reader] coordonnées du bord haut du carré isel: lon={lon[y1-1, x0:x1]}, lat={lat[y1-1, x0:x1]}")
    #     # print(f"[reader] coordonnées du bord gauche du carré isel: lon={lon[y0:y1, x0]}, lat={lat[y0:y1, x0]}")
    #     # print(f"[reader] coordonnées du bord droit du carré isel: lon={lon[y0:y1, x1-1]}, lat={lat[y0:y1, x1-1]}")
    #     print(f"[reader] isel angles coordinates: SW=({lon[y0, x0]:.4f}, {lat[y0, x0]:.4f}), SE=({lon[y0, x1-1]:.4f}, {lat[y0, x1-1]:.4f}), NW=({lon[y1-1, x0]:.4f}, {lat[y1-1, x0]:.4f}), NE=({lon[y1-1, x1-1]:.4f}, {lat[y1-1, x1-1]:.4f})")

    #     lon_border = np.concatenate([top_lon, bottom_lon, left_lon, right_lon])
    #     lat_border = np.concatenate([top_lat, bottom_lat, left_lat, right_lat])

    #     west  = float(np.nanmin(lon_border))
    #     east  = float(np.nanmax(lon_border))
    #     south = float(np.nanmin(lat_border))
    #     north = float(np.nanmax(lat_border))

    #     print(f"[reader] isel border bounds lon/lat: W={west:.4f}, E={east:.4f}, S={south:.4f}, N={north:.4f}")

    #     fig, ax = plt.subplots(figsize=(8, 8))
    #     ax.imshow(nans_mask, origin="lower", cmap="gray_r")
    #     rect = plt.Rectangle(
    #         (x0, y0), x1 - x0, y1 - y0,
    #         fill=False, edgecolor="red", linewidth=2
    #     )
    #     ax.add_patch(rect)

    #     ax.set_title("NaN mask + isel square (red)")
    #     ax.set_xlabel("x index")
    #     ax.set_ylabel("y index")
    #     plt.tight_layout()
    #     plt.show()


    # print(f"[reader] Cropped {param} to Iceland region: {d['latitude'].min().values} -> {d['latitude'].max().values}, "
    #       f"{d['longitude'].min().values} -> {d['longitude'].max().values}")

    return ds

def load_static_orography(path, year="2022", month="June", file_index=2):
    """Load the static CARRA2 orography field from a specific year/month folder."""
    folder = Path(path) / str(year) / f"CARRA2_{month}"
    target_file = folder / f"data_{file_index}.nc"
    if target_file.exists():
        ds_orog = xr.open_dataset(target_file, decode_timedelta=False)
        if "orog" not in ds_orog.data_vars:
            raise ValueError(f"Missing 'orog' in file: {target_file}")
        return crop_iceland(ds_orog, "orog")

    return crop_iceland(open_orography_netcdf(folder), "orog")

def load_land_mask(path):
    p = Path(path)
    if p.is_dir():
        p = p / "land_mask.nc"
    if not p.exists():
        raise FileNotFoundError(f"Land mask file not found: {p}")

    _prepare_step_metadata(p)

    with xr.open_dataset(p, decode_timedelta=False) as ds:
        if "land_mask" in ds.data_vars:
            da = ds["land_mask"]
        else:
            if not ds.data_vars:
                raise ValueError(f"No data variables found in land mask file: {p}")
            first_var = next(iter(ds.data_vars))
            da = ds[first_var]

        if "time" in da.dims:
            da = da.isel(time=0)

        da = crop_iceland(da, "land_mask")
        return da.load()


