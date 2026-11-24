"""Organize raw NetCDF files into the year-and-month layout used by the tools."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import shutil
import tempfile
import zipfile

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_DIR = ROOT / "Data" / "unordered_Brut_netcdf_files"
TARGET_DIR = ROOT / "Data" / "Brut_netcdf_files"
TEMP_FILES_DIR = ROOT / "Data" / "temporary_files"
DEBUG = False

# Cache NetCDF metadata (land-mask flag + inferred time metadata) to avoid
# reopening the same files when multiple helpers query them.
_NETCDF_METADATA_CACHE = {}

# Red-square corners (lon, lat) from the CARRA index crop diagnostics.
RED_SQUARE_CORNERS = {
    "SW": (-25.9535, 62.5445),
    "SE": (-13.4169, 61.4693),
    "NW": (-24.8401, 68.2985),
    "NE": (-9.1872, 66.9147),
}


def _red_square_bbox_with_margin(margin_lon=1.5, margin_lat=1.0):
    lons = [p[0] for p in RED_SQUARE_CORNERS.values()]
    lats = [p[1] for p in RED_SQUARE_CORNERS.values()]
    west = float(min(lons) - margin_lon)
    east = float(max(lons) + margin_lon)
    south = float(min(lats) - margin_lat)
    north = float(max(lats) + margin_lat)
    return west, east, south, north


def _metadata_cache_key(path):
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return str(path.resolve())
    return (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))

def _plot_crop_debug(lon_norm,lat,mask,ys,xs,west,east,south,north,nc_name,):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Vue lon/lat: points gardés par la bbox géographique
    ax = axes[0]
    ax.pcolormesh(lon_norm.values, lat.values, mask.astype(int), shading="auto")
    ax.plot(
    [west, east, east, west, west],
    [south, south, north, north, south],
    "--r",
    linewidth=2,
    label="bbox cible",
    )
    ax.set_title(f"Mask lon/lat - {nc_name}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="best")

    # Vue index: rectangle englobant min/max en indices
    ax = axes[1]
    ax.imshow(mask.astype(int), origin="lower", cmap="gray")
    rect = Rectangle(
    (x0, y0),
    x1 - x0,
    y1 - y0,
    fill=False,
    edgecolor="gold",
    linewidth=2,
    linestyle="--",
    label="rectangle englobant index",
    )
    ax.add_patch(rect)
    ax.set_title(f"Mask index + rectangle - {nc_name}")
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


def _cdo_crop_lonlat_inplace(nc_path, margin_lon=1.5, margin_lat=1.0):
    cdo_bin = shutil.which("cdo")
    if cdo_bin is None:
        raise RuntimeError("CDO is not available in PATH. Install CDO or disable CDO cropping.")

    west, east, south, north = _red_square_bbox_with_margin(
        margin_lon=margin_lon,
        margin_lat=margin_lat,
    )

    out_path = nc_path.with_name(f"{nc_path.stem}.cropped{nc_path.suffix}")
    cmd = [
        cdo_bin,
        "-L",
        "-f",
        "nc4c",
        "-z",
        "zip_4",
        f"sellonlatbox,{west},{east},{south},{north}",
        str(nc_path),
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        msg = stderr or stdout or "unknown CDO error"
        raise RuntimeError(f"CDO crop failed for {nc_path.name}: {msg}")

    if not out_path.exists():
        raise RuntimeError(f"CDO reported success but output was not created: {out_path}")

    nc_path.unlink(missing_ok=True)
    out_path.rename(nc_path)
    print(
        "[sort_nc] CDO crop applied "
        f"({west:.4f}, {east:.4f}, {south:.4f}, {north:.4f}) to {nc_path.name}"
    )


def _python_crop_lonlat_inplace(nc_path, margin_lon=1.5, margin_lat=1.0):
    west, east, south, north = _red_square_bbox_with_margin(
        margin_lon=margin_lon,
        margin_lat=margin_lat,
    )

    print(f"[sort_nc] Python crop target bbox bounds: W={west:.4f}, E={east:.4f}, S={south:.4f}, N={north:.4f} for {nc_path.name}")

    out_path = nc_path.with_name(f"{nc_path.stem}.cropped{nc_path.suffix}")

    with xr.open_dataset(nc_path, decode_timedelta=False) as ds:
        lon_name = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
        lat_name = "latitude" if "latitude" in ds.variables else ("lat" if "lat" in ds.variables else None)
        if lon_name is None or lat_name is None:
            raise RuntimeError(
                f"Cannot crop {nc_path.name}: missing longitude/latitude coordinates"
            )

        lon = ds[lon_name]
        lat = ds[lat_name]
        lon_norm = xr.where(lon > 180.0, lon - 360.0, lon)

        if lon.ndim == 1 and lat.ndim == 1:
            lon_idx = np.where((lon_norm.values >= west) & (lon_norm.values <= east))[0]
            lat_idx = np.where((lat.values >= south) & (lat.values <= north))[0]
            if lon_idx.size == 0 or lat_idx.size == 0:
                raise RuntimeError(
                    f"No grid points found in crop bounds for {nc_path.name}"
                )

            ds_crop = ds.isel(
                {
                    lon.dims[0]: slice(int(lon_idx.min()), int(lon_idx.max()) + 1),
                    lat.dims[0]: slice(int(lat_idx.min()), int(lat_idx.max()) + 1),
                }
            )
        elif lon.ndim == 2 and lat.ndim == 2:
            if lon.dims != lat.dims:
                raise RuntimeError(
                    f"Unsupported lon/lat dims mismatch for {nc_path.name}: {lon.dims} vs {lat.dims}"
                )

            mask = (
                (lon_norm.values >= west)
                & (lon_norm.values <= east)
                & (lat.values >= south)
                & (lat.values <= north)
            )

            

            if not np.any(mask):
                raise RuntimeError(
                    f"No grid points found in crop bounds for {nc_path.name}"
                )

            ys = np.where(mask.any(axis=1))[0]
            xs = np.where(mask.any(axis=0))[0]
            y_min = int(ys.min())
            y_max = int(ys.max())
            x_min = int(xs.min())
            x_max = int(xs.max())

            lon_rect = lon_norm.values[y_min:y_max + 1, x_min:x_max + 1]
            lat_rect = lat.values[y_min:y_max + 1, x_min:x_max + 1]

            print(f"[sort_nc] Python crop indices: x={x_min}:{x_max}, y={y_min}:{y_max}")
            print(f"[sort_nc] Python crop target bbox bounds: W={west:.4f}, E={east:.4f}, S={south:.4f}, N={north:.4f}")
            print(
                "[sort_nc] Python crop kept bbox bounds: "
                f"W={np.nanmin(lon_rect):.4f}, E={np.nanmax(lon_rect):.4f}, "
                f"S={np.nanmin(lat_rect):.4f}, N={np.nanmax(lat_rect):.4f}"
            )
            if DEBUG :
                _plot_crop_debug(lon_norm=lon_norm,lat=lat,mask=mask,ys=ys,xs=xs,west=west,east=east,south=south,north=north,nc_name=nc_path.name)
            y_dim, x_dim = lon.dims
            ds_crop = ds.isel(
                {
                    y_dim: slice(y_min, y_max + 1),
                    x_dim: slice(x_min, x_max + 1),
                }
            )
        else:
            raise RuntimeError(
                f"Unsupported lon/lat dimensions for {nc_path.name}: {lon.ndim}D/{lat.ndim}D"
            )

        ds_out = ds_crop.load()

    encoding = {}
    for var_name in ds_out.data_vars:
        if np.issubdtype(ds_out[var_name].dtype, np.floating):
            ds_out[var_name] = ds_out[var_name].astype("float32")
        encoding[var_name] = {"zlib": True, "complevel": 4, "shuffle": True}

    ds_out.to_netcdf(out_path, format="NETCDF4", encoding=encoding)
    ds_out.close()

    if not out_path.exists():
        raise RuntimeError(f"Python crop reported success but output was not created: {out_path}")

    nc_path.unlink(missing_ok=True)
    out_path.rename(nc_path)
    print(
        "[sort_nc] Python crop applied "
        f"({west:.4f}, {east:.4f}, {south:.4f}, {north:.4f}) to {nc_path.name}"
    )


def _crop_lonlat_inplace(nc_path, margin_lon=1.5, margin_lat=1.0, engine="python"):
    if engine == "python":
        _python_crop_lonlat_inplace(nc_path, margin_lon=margin_lon, margin_lat=margin_lat)
        return

    if engine == "cdo":
        _cdo_crop_lonlat_inplace(nc_path, margin_lon=margin_lon, margin_lat=margin_lat)
        return

    if engine == "auto":
        cdo_bin = shutil.which("cdo")
        if cdo_bin:
            _cdo_crop_lonlat_inplace(nc_path, margin_lon=margin_lon, margin_lat=margin_lat)
        else:
            print("[sort_nc] CDO not found, falling back to Python crop.")
            _python_crop_lonlat_inplace(nc_path, margin_lon=margin_lon, margin_lat=margin_lat)
        return

    raise ValueError(f"Unknown crop engine: {engine}")


def _inspect_netcdf_metadata(path):
    """Read a NetCDF file once and infer (is_land_mask, year, month)."""
    path = Path(path)
    cache_key = _metadata_cache_key(path)
    cached = _NETCDF_METADATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    is_land_mask = "land_mask" in path.stem.lower()
    year = None
    month = None

    try:
        with xr.open_dataset(path, decode_timedelta=False) as ds:
            if "land_mask" in ds.data_vars:
                is_land_mask = True

            has_time_coord = any(key in ds.coords for key in ["time", "valid_time", "forecast_reference_time"])
            has_supported_data_var = any(
                var in ds.data_vars
                for var in [
                    "t2m",
                    "u10",
                    "v10",
                    "orog",
                    "r2",
                    "u",
                    "v",
                    "msl",
                    "tcc",
                    "10u",
                    "10v",
                    "t",
                    "q",
                    "sp",
                    "cp",
                    "ws",
                    "wdir",
                ]
            )
            if not has_time_coord or not has_supported_data_var:
                result = {"is_land_mask": is_land_mask, "year": None, "month": None}
                _NETCDF_METADATA_CACHE[cache_key] = result
                return result

            for key in ["time", "valid_time", "forecast_reference_time"]:
                if key in ds.coords:
                    values = ds[key].values
                    if values.size > 0:
                        first_value = values[0]
                        if isinstance(first_value, np.datetime64):
                            text = np.datetime_as_string(first_value, unit="D")
                            parsed = datetime.fromisoformat(text)
                            year = str(parsed.year)
                            month = parsed.strftime("%B").capitalize()
                            break

                        if hasattr(first_value, "astype"):
                            try:
                                first_value = first_value.astype("datetime64[ns]")
                            except Exception:
                                pass

                        if hasattr(first_value, "year"):
                            year = str(first_value.year)
                            month = first_value.strftime("%B").capitalize()
                            break

                        if hasattr(first_value, "item") and not isinstance(first_value, (str, bytes)):
                            try:
                                candidate = first_value.item()
                            except Exception:
                                candidate = None
                            if isinstance(candidate, (int, float)):
                                first_value = candidate

                        if isinstance(first_value, str):
                            try:
                                parsed = datetime.fromisoformat(first_value.replace("Z", "+00:00"))
                            except Exception:
                                parsed = None
                            if parsed is not None:
                                year = str(parsed.year)
                                month = parsed.strftime("%B").capitalize()
                                break
    except Exception:
        result = {"is_land_mask": is_land_mask, "year": None, "month": None}
        _NETCDF_METADATA_CACHE[cache_key] = result
        return result

    if year is None:
        year = path.parent.name if path.parent.name.isdigit() else None

    if month is None:
        month = "June"

    result = {"is_land_mask": is_land_mask, "year": year, "month": month}
    _NETCDF_METADATA_CACHE[cache_key] = result
    return result


def infer_year_month(path):
    """Infer year and month from a CARRA2-style NetCDF file using its time coordinate."""
    metadata = _inspect_netcdf_metadata(path)
    year = metadata.get("year")
    month = metadata.get("month")

    return year, month


def build_target_path(path, target_dir=TARGET_DIR):
    year, month = infer_year_month(path)
    if year is None:
        year = "unknown"
    month_dir = f"CARRA2_{month}"
    return Path(target_dir) / str(year) / month_dir


def _is_probably_zip_file(path):
    # ZIP local file header, empty ZIP header, and spanned ZIP marker.
    zip_signatures = {b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"}
    try:
        with path.open("rb") as handle:
            signature = handle.read(4)
    except OSError as exc:
        print(f"[sort_nc][warning] Cannot read archive {path}: {exc}")
        return False

    return signature in zip_signatures


def _build_unique_destination_path(destination_path):
    if not destination_path.exists():
        return destination_path

    stem = destination_path.stem
    suffix = destination_path.suffix
    counter = 1
    while True:
        candidate = destination_path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _write_unusable_sidecar(destination_path, source_path, reason):
    sidecar_path = destination_path.with_name(f"{destination_path.name}.reason.txt")
    content = (
        f"source: {source_path}\n"
        f"reason: {reason}\n"
    )
    sidecar_path.write_text(content, encoding="utf-8")
    return sidecar_path


def _move_archive_to_unusable(archive_path, unusable_dir, dry_run, reason):
    destination_path = _build_unique_destination_path(unusable_dir / archive_path.name)

    if dry_run:
        print(f"[dry-run][sort_nc][unusable] {archive_path} -> {destination_path} ({reason})")
        print(f"[dry-run][sort_nc][unusable] sidecar: {destination_path.name}.reason.txt")
        return destination_path

    unusable_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(archive_path), str(destination_path))
    sidecar_path = _write_unusable_sidecar(destination_path, archive_path, reason)
    print(f"[sort_nc][unusable] {archive_path} -> {destination_path} ({reason})")
    print(f"[sort_nc][unusable] sidecar written: {sidecar_path}")
    return destination_path


def _move_nc_to_unusable(src_path, unusable_dir, dry_run, reason):
    destination_path = _build_unique_destination_path(unusable_dir / src_path.name)

    if dry_run:
        print(f"[dry-run][sort_nc][unusable] {src_path} -> {destination_path} ({reason})")
        print(f"[dry-run][sort_nc][unusable] sidecar: {destination_path.name}.reason.txt")
        return destination_path

    unusable_dir.mkdir(parents=True, exist_ok=True)
    if src_path.exists():
        shutil.move(str(src_path), str(destination_path))
    sidecar_path = _write_unusable_sidecar(destination_path, src_path, reason)
    print(f"[sort_nc][unusable] {src_path} -> {destination_path} ({reason})")
    print(f"[sort_nc][unusable] sidecar written: {sidecar_path}")
    return destination_path


def _extract_nc_files_from_archive(archive_path, extract_dir):
    extracted_paths = []
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for index, member in enumerate(archive.namelist(), start=1):
                if member.endswith("/") or not member.lower().endswith(".nc"):
                    continue

                member_name = Path(member).name or f"member_{index}.nc"
                safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in member_name)
                output_path = _build_unique_destination_path(extract_dir / safe_name)
                with archive.open(member) as src, output_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_paths.append(output_path)
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        return None, str(exc)

    return extracted_paths, None


def _collect_target_year_nc_files(target_dir):
    target_dir = Path(target_dir)
    year_dirs = [p for p in target_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    year_dirs = sorted(year_dirs, key=lambda p: p.name)
    files = []
    root_land_mask = target_dir / "land_mask.nc"
    if root_land_mask.exists():
        files.append(root_land_mask)
    for year_dir in year_dirs:
        files.extend(sorted(year_dir.rglob("*.nc")))
    return files


def _is_inside_generated_sort_tmp(path):
    return any(part.startswith("sort_nc_") for part in Path(path).parts)


def _is_land_mask_file(path):
    return bool(_inspect_netcdf_metadata(path).get("is_land_mask", False))


def _move_and_optionally_crop(
    src_path,
    target_dir,
    unusable_dir,
    dry_run,
    do_crop,
    crop_only_land_mask,
    crop_engine,
    margin_lon,
    margin_lat,
):
    metadata = _inspect_netcdf_metadata(src_path)
    is_land_mask = bool(metadata.get("is_land_mask", False))

    if is_land_mask:
        destination_dir = Path(target_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / "land_mask.nc"
    else:
        year = metadata.get("year")
        month = metadata.get("month")
        if year is None or month is None:
            return _move_nc_to_unusable(
                src_path=src_path,
                unusable_dir=unusable_dir,
                dry_run=dry_run,
                reason="no usable CARRA2-style metadata found",
            )

        destination_dir = Path(target_dir) / str(year) / f"CARRA2_{month}"
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / src_path.name

    if dry_run:
        print(f"[dry-run] {src_path} -> {destination_path}")
        return destination_path

    src_resolved = src_path.resolve()
    dst_resolved = destination_path.resolve()

    if src_resolved != dst_resolved:
        if destination_path.exists():
            destination_path.unlink()
        shutil.move(str(src_path), str(destination_path))
        print(f"Moved: {src_path} -> {destination_path}")
    else:
        print(f"[sort_nc] Already in target location: {destination_path}")

    should_crop_this_file = do_crop and (not crop_only_land_mask or is_land_mask)
    if should_crop_this_file:
        _crop_lonlat_inplace(
            destination_path,
            margin_lon=margin_lon,
            margin_lat=margin_lat,
            engine=crop_engine,
        )

    return destination_path


def sort_nc_files(
    source_dir=SOURCE_DIR,
    target_dir=TARGET_DIR,
    dry_run=False,
    apply_cdo_crop=False,
    apply_crop=False,
    crop_engine="python",
    margin_lon=1.5,
    margin_lat=1.0,
    rescan_target_years=False,
    crop_only_land_mask=False,
):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    nc_files = [path for path in sorted(source_dir.rglob("*.nc")) if not _is_inside_generated_sort_tmp(path)]
    archives = sorted(source_dir.rglob("*.zip"))
    unusable_dir = target_dir / "sorted_but_unusable"

    if rescan_target_years:
        nc_files = _collect_target_year_nc_files(target_dir)
        archives = []
        print(f"[sort_nc] Rescan mode enabled: {len(nc_files)} .nc files found in target year folders")

    if not nc_files and not archives:
        print(f"No .nc files or .zip archives found in {source_dir}")
        return []

    moved = []
    do_crop = bool(apply_crop or apply_cdo_crop or crop_only_land_mask)
    if apply_cdo_crop and crop_engine == "python":
        # Backward compatibility for existing calls that use apply_cdo_crop only.
        crop_engine = "cdo"

    if crop_only_land_mask and not (apply_crop or apply_cdo_crop):
        print("[sort_nc] crop-only-land-mask enabled: crop will be applied only to land_mask files.")

    for path in nc_files:
        if not path.is_file():
            continue

        destination_path = _move_and_optionally_crop(
            src_path=path,
            target_dir=target_dir,
            unusable_dir=unusable_dir,
            dry_run=dry_run,
            do_crop=do_crop,
            crop_only_land_mask=crop_only_land_mask,
            crop_engine=crop_engine,
            margin_lon=margin_lon,
            margin_lat=margin_lat,
        )
        if destination_path is not None:
            moved.append((path, destination_path))

    for archive_path in archives:
        if not archive_path.is_file():
            continue

        if not _is_probably_zip_file(archive_path):
            _move_archive_to_unusable(
                archive_path=archive_path,
                unusable_dir=unusable_dir,
                dry_run=dry_run,
                reason="invalid ZIP signature",
            )
            continue

        TEMP_FILES_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="sort_nc_", dir=str(TEMP_FILES_DIR)))
        try:
            extracted_paths, extraction_error = _extract_nc_files_from_archive(archive_path, temp_dir)
            if extraction_error is not None:
                _move_archive_to_unusable(
                    archive_path=archive_path,
                    unusable_dir=unusable_dir,
                    dry_run=dry_run,
                    reason=f"zip extraction failed: {extraction_error}",
                )
                continue

            if not extracted_paths:
                print(f"No NetCDF files found inside {archive_path}")
                continue

            archive_processed = True
            for extracted_path in extracted_paths:
                destination_path = _move_and_optionally_crop(
                    src_path=extracted_path,
                    target_dir=target_dir,
                    unusable_dir=unusable_dir,
                    dry_run=dry_run,
                    do_crop=do_crop,
                    crop_only_land_mask=crop_only_land_mask,
                    crop_engine=crop_engine,
                    margin_lon=margin_lon,
                    margin_lat=margin_lat,
                )
                if destination_path is None:
                    archive_processed = False
                    continue
                if not dry_run:
                    print(f"[sort_nc] Moved from archive: {archive_path} -> {destination_path}")
                moved.append((archive_path, destination_path))

            if not dry_run and archive_processed:
                archive_path.unlink(missing_ok=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return moved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Sort CARRA NetCDF files into year/month folders, with optional CDO "
            "lon/lat crop around the red-square ROI."
        )
    )
    parser.add_argument("--source-dir", default=str(SOURCE_DIR), help="Input folder containing .nc and .zip files.")
    parser.add_argument("--target-dir", default=str(TARGET_DIR), help="Output root folder for sorted files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without modifying files.")
    parser.add_argument("--apply-crop", action="store_true", help="Apply crop after each move.")
    parser.add_argument("--apply-cdo-crop", action="store_true", help="Apply CDO sellonlatbox crop after each move.")
    parser.add_argument(
        "--crop-engine",
        choices=["python", "cdo", "auto"],
        default="python",
        help="Crop backend to use when crop is enabled.",
    )
    parser.add_argument(
        "--margin-lon",
        type=float,
        default=1.5,
        help="Longitude margin in degrees around red-square corner extrema.",
    )
    parser.add_argument(
        "--margin-lat",
        type=float,
        default=1.0,
        help="Latitude margin in degrees around red-square corner extrema.",
    )
    parser.add_argument(
        "--rescan-target-years",
        action="store_true",
        help="Scan target year folders only and re-apply sorting/cropping on found .nc files.",
    )
    parser.add_argument(
        "--crop-only-land-mask",
        action="store_true",
        help="Apply crop only to files identified as land_mask (other files are moved without crop).",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with additional plots for crop diagnostics."
    )

    args = parser.parse_args()
    DEBUG = args.debug
    sort_nc_files(
        source_dir=Path(args.source_dir),
        target_dir=Path(args.target_dir),
        dry_run=args.dry_run,
        apply_cdo_crop=args.apply_cdo_crop,
        apply_crop=args.apply_crop,
        crop_engine=args.crop_engine,
        margin_lon=args.margin_lon,
        margin_lat=args.margin_lat,
        rescan_target_years=args.rescan_target_years,
        crop_only_land_mask=args.crop_only_land_mask,
    )
