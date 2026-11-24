"""Check which CARRA2 years contain complete input data."""

import argparse
from pathlib import Path
import sys

import netCDF4 as nc
import numpy as np

from shared.paths import DEFAULT_KATABATIC_DATA_DIR
from shared.reader import write_valid_years_cache


# Red-square corners (lon, lat) from the Iceland index crop diagnostics.
RED_SQUARE_CORNERS = {
    "SW": (-25.9535, 62.5445),
    "SE": (-13.4169, 61.4693),
    "NW": (-24.8401, 68.2985),
    "NE": (-9.1872, 66.9147),
}


def _expanded_red_square_polygon(margin_lon=0.0, margin_lat=0.0):
    # Order points clockwise for polygon tests.
    pts = [
        RED_SQUARE_CORNERS["SW"],
        RED_SQUARE_CORNERS["SE"],
        RED_SQUARE_CORNERS["NE"],
        RED_SQUARE_CORNERS["NW"],
    ]
    lon_c = sum(p[0] for p in pts) / len(pts)
    lat_c = sum(p[1] for p in pts) / len(pts)

    expanded = []
    for lon, lat in pts:
        lon_sign = 1.0 if lon >= lon_c else -1.0
        lat_sign = 1.0 if lat >= lat_c else -1.0
        expanded.append((lon + lon_sign * margin_lon, lat + lat_sign * margin_lat))
    return expanded


def _point_in_polygon_mask(x, y, polygon):
    # Vectorized ray-casting for a convex/concave simple polygon.
    inside = np.zeros_like(x, dtype=bool)
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        inside ^= intersects
    return inside


def _nan_issues_in_red_square(file_path, required_vars, margin_lon=0.0, margin_lat=0.0):
    issues = []
    polygon = _expanded_red_square_polygon(margin_lon=margin_lon, margin_lat=margin_lat)

    try:
        with nc.Dataset(str(file_path), "r") as ds:
            lon_name = "longitude" if "longitude" in ds.variables else ("lon" if "lon" in ds.variables else None)
            lat_name = "latitude" if "latitude" in ds.variables else ("lat" if "lat" in ds.variables else None)
            if lon_name is None or lat_name is None:
                return [f"{file_path.name}: missing longitude/latitude coordinates"]

            lon = np.asarray(ds.variables[lon_name][:])
            lat = np.asarray(ds.variables[lat_name][:])

            lon = np.where(lon > 180.0, lon - 360.0, lon)

            if lon.ndim == 1 and lat.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)
            elif lon.ndim != 2 or lat.ndim != 2:
                return [f"{file_path.name}: unsupported lon/lat dimensions ({lon.ndim}D/{lat.ndim}D)"]

            roi_mask = _point_in_polygon_mask(lon, lat, polygon)
            if not np.any(roi_mask):
                return [f"{file_path.name}: no grid point intersects red-square ROI"]

            vars_to_check = [v for v in required_vars if v in ds.variables]
            if not vars_to_check:
                return []

            for var_name in vars_to_check:
                var = ds.variables[var_name]

                if var.ndim >= 3:
                    # Stream slices over leading dims to avoid loading the full data cube.
                    leading_shape = var.shape[:-2]
                    has_nan = np.zeros(var.shape[-2:], dtype=bool)

                    for idx in np.ndindex(*leading_shape):
                        slice_2d = var[idx + (slice(None), slice(None))]
                        slice_data = np.ma.asarray(slice_2d)

                        mask_nan = np.ma.getmaskarray(slice_data)
                        data_nan = np.isnan(np.ma.getdata(slice_data))
                        has_nan |= (mask_nan | data_nan)
                elif var.ndim == 2:
                    slice_data = np.ma.asarray(var[:, :])
                    mask_nan = np.ma.getmaskarray(slice_data)
                    data_nan = np.isnan(np.ma.getdata(slice_data))
                    has_nan = mask_nan | data_nan
                else:
                    continue

                n_nan = int(np.count_nonzero(has_nan & roi_mask))
                if n_nan > 0:
                    issues.append(
                        f"{file_path.name}: variable '{var_name}' has {n_nan} NaN points in red-square ROI"
                    )
    except Exception as exc:
        issues.append(f"{file_path.name}: red-square NaN check error: {exc}")

    return issues


def _parse_time_coverage(file_path):
    """Return first/last timestamp strings from a netCDF file when possible."""
    try:
        with nc.Dataset(str(file_path), "r") as ds:
            if "time" not in ds.variables:
                return None, None, "missing_time_var"

            tvar = ds.variables["time"]
            units = getattr(tvar, "units", None)
            calendar = getattr(tvar, "calendar", "standard")
            values = tvar[:]
            if values is None or len(values) == 0:
                return None, None, "empty_time"

            if units is None:
                return None, None, "missing_time_units"

            dt = nc.num2date(values, units=units, calendar=calendar)
            first = dt[0]
            last = dt[-1]
            return first, last, None
    except Exception as exc:
        return None, None, f"time_decode_error: {exc}"


def _iter_year_dirs(root):
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: p.name)


def _month_dir(year_dir, month):
    return year_dir / f"CARRA2_{month}"


def _scan_month(month_dir, required_vars, check_red_square_nans=False, margin_lon=0.0, margin_lat=0.0):
    """Inspect one month directory and return diagnostics."""
    if not month_dir.exists():
        return {
            "exists": False,
            "files": [],
            "vars_found": set(),
            "missing_vars": list(required_vars),
            "time_issues": ["missing_month_dir"],
        }

    nc_files = sorted(month_dir.glob("*.nc"))
    if not nc_files:
        return {
            "exists": True,
            "files": [],
            "vars_found": set(),
            "missing_vars": list(required_vars),
            "time_issues": ["no_nc_files"],
        }

    vars_found = set()
    time_issues = []
    red_square_issues = []
    file_summaries = []

    for file_path in nc_files:
        try:
            with nc.Dataset(str(file_path), "r") as ds:
                var_names = set(ds.variables.keys())
                vars_found.update(var_names)
        except Exception as exc:
            time_issues.append(f"open_error:{file_path.name}:{exc}")
            continue

        first, last, err = _parse_time_coverage(file_path)
        file_summaries.append(
            {
                "name": file_path.name,
                "first": first,
                "last": last,
                "time_error": err,
            }
        )

        if check_red_square_nans:
            red_square_issues.extend(
                _nan_issues_in_red_square(
                    file_path=file_path,
                    required_vars=required_vars,
                    margin_lon=margin_lon,
                    margin_lat=margin_lat,
                )
            )

    missing_vars = [v for v in required_vars if v not in vars_found]

    return {
        "exists": True,
        "files": file_summaries,
        "vars_found": vars_found,
        "missing_vars": missing_vars,
        "time_issues": time_issues,
        "red_square_issues": red_square_issues,
    }


def _expected_month_number(month_name):
    mapping = {
        "june": 6,
        "july": 7,
        "august": 8,
    }
    return mapping.get(month_name.lower())


def _check_timestamp_consistency(year_str, month_name, file_summary):
    """Return list of timestamp consistency issues for one file."""
    issues = []

    if file_summary["time_error"] is not None:
        issues.append(f"{file_summary['name']}: {file_summary['time_error']}")
        return issues

    first = file_summary["first"]
    last = file_summary["last"]
    if first is None or last is None:
        issues.append(f"{file_summary['name']}: missing decoded time bounds")
        return issues

    expected_year = int(year_str)
    expected_month = _expected_month_number(month_name)

    if expected_month is None:
        return issues

    if first.year != expected_year or last.year != expected_year:
        issues.append(
            f"{file_summary['name']}: unexpected year range {first.year}->{last.year}, expected {expected_year}"
        )

    if first.month != expected_month or last.month != expected_month:
        issues.append(
            f"{file_summary['name']}: unexpected month range {first.month}->{last.month}, expected {expected_month}"
        )

    return issues


def run_quality_check(
    data_root,
    months,
    required_vars,
    strict_timestamps=False,
    check_red_square_nans=True,
    red_square_margin_lon=0.0,
    red_square_margin_lat=0.0,
):
    print(f"[qc] Data root: {data_root}")
    print(f"[qc] Required months: {', '.join(months)}")
    print(f"[qc] Required variables: {', '.join(required_vars)}")
    print(f"[qc] Strict timestamps: {strict_timestamps}")
    print(f"[qc] Check red-square NaNs: {check_red_square_nans}")
    if check_red_square_nans:
        print(
            "[qc] Red-square margins "
            f"(lon={red_square_margin_lon}, lat={red_square_margin_lat})"
        )

    issues = []
    valid_years = []

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    year_dirs = _iter_year_dirs(data_root)
    if not year_dirs:
        raise ValueError(f"No year folders found in {data_root}")

    for year_dir in year_dirs:
        year = year_dir.name
        year_ok = True
        print(f"\n[qc] Checking year {year}")

        for month in months:
            mdir = _month_dir(year_dir, month)
            month_scan = _scan_month(
                mdir,
                required_vars,
                check_red_square_nans=check_red_square_nans,
                margin_lon=red_square_margin_lon,
                margin_lat=red_square_margin_lat,
            )

            if not month_scan["exists"]:
                msg = f"year {year}, {month}: missing folder {mdir.name}"
                issues.append(msg)
                year_ok = False
                print(f"  - {msg}")
                continue

            if not month_scan["files"]:
                msg = f"year {year}, {month}: no .nc files"
                issues.append(msg)
                year_ok = False
                print(f"  - {msg}")
                continue

            if month_scan["missing_vars"]:
                msg = (
                    f"year {year}, {month}: missing variables "
                    + ", ".join(month_scan["missing_vars"])
                )
                issues.append(msg)
                year_ok = False
                print(f"  - {msg}")

            if month_scan["time_issues"]:
                for t_issue in month_scan["time_issues"]:
                    msg = f"year {year}, {month}: {t_issue}"
                    issues.append(msg)
                    year_ok = False
                    print(f"  - {msg}")

            if month_scan.get("red_square_issues"):
                for rs_issue in month_scan["red_square_issues"]:
                    msg = f"year {year}, {month}: {rs_issue}"
                    issues.append(msg)
                    year_ok = False
                    print(f"  - {msg}")

            for file_summary in month_scan["files"]:
                first = file_summary["first"]
                last = file_summary["last"]
                if file_summary["time_error"] is None and first is not None and last is not None:
                    print(
                        f"  - {month}/{file_summary['name']}: "
                        f"time {first.isoformat()} -> {last.isoformat()}"
                    )

                if strict_timestamps:
                    ts_issues = _check_timestamp_consistency(year, month, file_summary)
                    for ts_issue in ts_issues:
                        issues.append(f"year {year}, {month}: {ts_issue}")
                        year_ok = False
                        print(f"  - year {year}, {month}: {ts_issue}")

        if year_ok:
            valid_years.append(year)
            print(f"[qc] Year {year}: OK")
        else:
            print(f"[qc] Year {year}: issues found")

    print("\n[qc] Summary")
    print(f"[qc] Valid years: {valid_years}")
    print(f"[qc] Total issues: {len(issues)}")

    return valid_years, issues


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone quality checker for katabatic monthly netCDF data. "
            "A year is valid only if all requested months exist and contain required variables."
        )
    )
    parser.add_argument(
        "--path",
        default=str(DEFAULT_KATABATIC_DATA_DIR),
        help="Root folder that contains year subfolders.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=["June", "July", "August"],
        help="Months that must be present in each year.",
    )
    parser.add_argument(
        "--required-vars",
        nargs="+",
        default=["t2m", "u10", "v10"],
        help="Variables required in each month.",
    )
    parser.add_argument(
        "--strict-timestamps",
        dest="strict_timestamps",
        action="store_true",
        help="Require timestamp year and month to match folder names (default).",
    )
    parser.add_argument(
        "--no-strict-timestamps",
        dest="strict_timestamps",
        action="store_false",
        help="Disable strict timestamp consistency checks.",
    )
    parser.add_argument(
        "--no-check-red-square-nans",
        action="store_true",
        help="Disable NaN checks in the red-square ROI.",
    )
    parser.add_argument(
        "--red-square-margin-lon",
        type=float,
        default=0.0,
        help="Longitude margin (degrees) around red-square corners for NaN checks.",
    )
    parser.add_argument(
        "--red-square-margin-lat",
        type=float,
        default=0.0,
        help="Latitude margin (degrees) around red-square corners for NaN checks.",
    )
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Exit with code 1 if any issue is found.",
    )

    parser.set_defaults(strict_timestamps=True)

    args = parser.parse_args()

    data_root = Path(args.path)
    valid_years, issues = run_quality_check(
        data_root=data_root,
        months=args.months,
        required_vars=args.required_vars,
        strict_timestamps=args.strict_timestamps,
        check_red_square_nans=not args.no_check_red_square_nans,
        red_square_margin_lon=args.red_square_margin_lon,
        red_square_margin_lat=args.red_square_margin_lat,
    )

    write_valid_years_cache(
        path=data_root,
        months=args.months,
        required_vars=args.required_vars,
        valid_years=valid_years,
        strict_timestamps=args.strict_timestamps,
        source="katabatic.data_quality_check",
    )

    if args.fail_on_issues and issues:
        print("[qc] Failing due to detected issues.")
        sys.exit(1)

    print("[qc] Done.")


if __name__ == "__main__":
    main()
