"""Build spatial trend maps from yearly katabatic classification results."""

import argparse
import csv
from math import erf, sqrt
import os
from pathlib import Path
import re

import matplotlib
# Trend maps are batch exports; force a non-interactive backend for stability.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from shared.map_launcher import get_ref, plot_iceland
from shared.paths import DEFAULT_FIGURES_DIR, DEFAULT_NPY_DATA_DIR, resolve_glacier_mask_path

try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover - optional dependency fallback
    ccrs = None


def _trend_debug(message):
    if os.environ.get("KATABATIC_TREND_DEBUG", "0") == "1":
        print(f"[trend][debug] {message}", flush=True)


def _norm_cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def mann_kendall_test_1d(x):
    values = np.asarray(x, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 3:
        return np.nan, np.nan, np.nan

    s = 0.0
    for i in range(n - 1):
        s += np.sign(values[i + 1 :] - values[i]).sum()

    _, counts = np.unique(values, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        return float(s), np.nan, np.nan

    if s > 0:
        z = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0

    p_value = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return float(s), float(z), float(p_value)


def sen_slope_1d(series, years):
    y = np.asarray(series, dtype=float)
    t = np.asarray(years, dtype=float)
    valid = np.isfinite(y) & np.isfinite(t)
    y = y[valid]
    t = t[valid]

    if y.size < 2:
        return np.nan

    slopes = []
    for i in range(y.size - 1):
        dt = t[i + 1 :] - t[i]
        dy = y[i + 1 :] - y[i]
        keep = dt != 0
        if np.any(keep):
            slopes.append(dy[keep] / dt[keep])

    if not slopes:
        return np.nan

    return float(np.median(np.concatenate(slopes)))


def ols_slope_1d(series, years):
    y = np.asarray(series, dtype=float)
    t = np.asarray(years, dtype=float)
    valid = np.isfinite(y) & np.isfinite(t)
    y = y[valid]
    t = t[valid]

    if y.size < 2:
        return np.nan

    # Closed-form OLS slope avoids calling LAPACK-backed polyfit on Windows,
    # which can hard-crash in some conda environments.
    t_centered = t - np.mean(t)
    y_centered = y - np.mean(y)
    denom = float(np.sum(t_centered * t_centered))
    if denom <= 0:
        return np.nan
    return float(np.sum(t_centered * y_centered) / denom)


def _load_glacier_mask():
    mask_path = resolve_glacier_mask_path()
    if not mask_path.exists():
        return None
    try:
        return np.load(mask_path)
    except Exception:
        return None


def _sanitize_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _load_group_labels(group_kind, basins_path=None, glacier_path=None):
    if group_kind == "basin":
        path = Path(basins_path) if basins_path else (DEFAULT_NPY_DATA_DIR / "basins.npy")
        if not path.exists():
            raise FileNotFoundError(f"Missing basins labels file: {path}")
        labels = np.asarray(np.load(path))
        return labels, path

    if group_kind == "glacier":
        path = Path(glacier_path) if glacier_path else resolve_glacier_mask_path()
        if not path.exists():
            raise FileNotFoundError(f"Missing glacier labels file: {path}")
        labels = np.asarray(np.load(path))
        return labels, path

    raise ValueError(f"Unknown group kind: {group_kind}")


def _compute_group_timeseries(years, stack, labels, min_pixels=5):
    if labels.shape != stack.shape[1:]:
        raise ValueError(
            f"Shape mismatch: labels={labels.shape}, field={stack.shape[1:]}"
        )

    valid_labels = np.unique(labels[np.isfinite(labels)])
    valid_labels = valid_labels[valid_labels > 0]

    series_by_group = {}
    stats_rows = []
    for group_id in valid_labels:
        mask = labels == group_id
        n_pix = int(np.sum(mask))
        if n_pix < min_pixels:
            continue

        ts = np.nanmean(stack[:, mask], axis=1)
        ts = np.asarray(ts, dtype=float)
        series_by_group[int(group_id)] = ts

        mk_s, mk_z, mk_p = mann_kendall_test_1d(ts)
        sen = sen_slope_1d(ts, years)
        ols = ols_slope_1d(ts, years)

        stats_rows.append(
            {
                "group_id": int(group_id),
                "n_pixels": n_pix,
                "mean_fraction": float(np.nanmean(ts)),
                "mk_s": None if not np.isfinite(mk_s) else float(mk_s),
                "mk_z": None if not np.isfinite(mk_z) else float(mk_z),
                "mk_p": None if not np.isfinite(mk_p) else float(mk_p),
                "sen_slope_fraction_per_year": None if not np.isfinite(sen) else float(sen),
                "ols_slope_fraction_per_year": None if not np.isfinite(ols) else float(ols),
            }
        )

    stats_rows.sort(key=lambda r: r["group_id"])
    return series_by_group, stats_rows


def _save_group_timeseries_outputs(fig_out_root, data_out_root, group_kind, years, series_by_group, stats_rows):
    fig_out_root = Path(fig_out_root) / f"by_{group_kind}"
    data_out_root = Path(data_out_root) / f"by_{group_kind}"
    fig_out_root.mkdir(parents=True, exist_ok=True)
    data_out_root.mkdir(parents=True, exist_ok=True)

    summary_csv = data_out_root / f"{group_kind}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_id",
                "n_pixels",
                "mean_fraction",
                "mk_s",
                "mk_z",
                "mk_p",
                "sen_slope_fraction_per_year",
                "ols_slope_fraction_per_year",
            ],
        )
        writer.writeheader()
        for row in stats_rows:
            writer.writerow(row)

    for group_id, ts in series_by_group.items():
        fig_folder = fig_out_root / f"{group_kind}_{_sanitize_name(group_id)}"
        data_folder = data_out_root / f"{group_kind}_{_sanitize_name(group_id)}"
        fig_folder.mkdir(parents=True, exist_ok=True)
        data_folder.mkdir(parents=True, exist_ok=True)

        csv_path = data_folder / "timeseries.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["year", "mean_kat_fraction"])
            for year, value in zip(years, ts):
                writer.writerow([int(year), float(value)])

        fig_path = fig_folder / "timeseries.png"
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(years, ts, color="#1f77b4", lw=1.8)
        ax.scatter(years, ts, color="#1f77b4", s=16)
        ax.set_title(f"{group_kind.capitalize()} {group_id} - Mean katabatic fraction")
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean katabatic fraction")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=220)
        plt.close(fig)

    print(f"Saved grouped figure outputs: {fig_out_root}")
    print(f"Saved grouped data outputs: {data_out_root}")


def _run_group_modes(
    fig_output_dir,
    data_output_dir,
    years,
    stack,
    group_mode,
    basins_path=None,
    glacier_path=None,
    min_group_pixels=5,
):
    kinds = []
    if group_mode in ("basin", "both"):
        kinds.append("basin")
    if group_mode in ("glacier", "both"):
        kinds.append("glacier")

    for kind in kinds:
        labels, labels_path = _load_group_labels(
            kind,
            basins_path=basins_path,
            glacier_path=glacier_path,
        )
        print(f"[group-mode] {kind} labels: {labels_path}")
        series_by_group, stats_rows = _compute_group_timeseries(
            years=years,
            stack=stack,
            labels=labels,
            min_pixels=min_group_pixels,
        )
        _save_group_timeseries_outputs(
            fig_out_root=fig_output_dir,
            data_out_root=data_output_dir,
            group_kind=kind,
            years=years,
            series_by_group=series_by_group,
            stats_rows=stats_rows,
        )


def _load_yearly_stack(yearly_root, param_tag, start_year, end_year):
    yearly_root = Path(yearly_root)
    years = []
    fields = []

    for year in range(start_year, end_year + 1):
        file_name = f"per_kat_{param_tag}.npy"

        # Current layout (param-first): Data/NPY/yearly/<param_tag>/<year>/...
        path = yearly_root / param_tag / str(year) / file_name
        # Legacy layout kept for backward compatibility.
        if not path.exists():
            path = yearly_root / str(year) / param_tag / file_name

        if not path.exists():
            continue

        try:
            arr = np.asarray(np.load(path), dtype=float)
        except Exception:
            continue

        if arr.ndim != 2:
            continue

        years.append(year)
        fields.append(arr)

    if not fields:
        raise RuntimeError("No yearly per_kat maps found for the requested period.")

    return np.asarray(years, dtype=int), np.stack(fields, axis=0)


def _discover_effective_years(yearly_root, param_tag, start_year, end_year):
    yearly_root = Path(yearly_root)
    years = []

    for year in range(start_year, end_year + 1):
        file_name = f"per_kat_{param_tag}.npy"
        path = yearly_root / param_tag / str(year) / file_name
        if not path.exists():
            path = yearly_root / str(year) / param_tag / file_name
        if path.exists():
            years.append(year)

    return years


def _discover_param_tags(yearly_root, start_year, end_year):
    yearly_root = Path(yearly_root)
    if not yearly_root.exists():
        return []

    discovered = []
    for child in sorted(yearly_root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        tag = child.name
        years = _discover_effective_years(yearly_root, tag, start_year, end_year)
        if years:
            discovered.append(tag)
    return discovered


def _parse_param_tag(param_tag):
    pattern = r"dt(?P<dt>[-+]?\d*\.?\d+)_ws(?P<ws>\d+)_wind(?P<wind>[-+]?\d*\.?\d+)_dir(?P<dir>[-+]?\d*\.?\d+)"
    match = re.search(pattern, str(param_tag))
    if not match:
        return {
            "dt_threshold": "N/A",
            "window_size": "N/A",
            "wind_threshold": "N/A",
            "direction_threshold": "N/A",
        }
    return {
        "dt_threshold": match.group("dt"),
        "window_size": match.group("ws"),
        "wind_threshold": match.group("wind"),
        "direction_threshold": match.group("dir"),
    }


def _period_mean(stack, years, start_year, end_year):
    mask = (years >= start_year) & (years <= end_year)
    if not np.any(mask):
        raise RuntimeError(f"No data in period {start_year}-{end_year}.")
    return np.nanmean(stack[mask], axis=0)


def _compute_trend_maps(years, stack, min_years):
    t = years.astype(float)
    nt, ny, nx = stack.shape
    flat = stack.reshape(nt, ny * nx)

    p_values = np.full(ny * nx, np.nan, dtype=float)
    sen = np.full(ny * nx, np.nan, dtype=float)
    ols = np.full(ny * nx, np.nan, dtype=float)
    valid = np.isfinite(flat)
    valid_count = valid.sum(axis=0).astype(int)
    eligible = valid_count >= int(min_years)

    if np.any(eligible):
        eligible_idx = np.flatnonzero(eligible)
        eligible_series = np.where(valid[:, eligible], flat[:, eligible], np.nan)
        # Constant-series shortcut is vectorized first to avoid Python loops.
        mins = np.nanmin(eligible_series, axis=0)
        maxs = np.nanmax(eligible_series, axis=0)
        constant_eligible = mins == maxs

        if np.any(constant_eligible):
            const_idx = eligible_idx[constant_eligible]
            p_values[const_idx] = 1.0
            sen[const_idx] = 0.0
            ols[const_idx] = 0.0

        work_idx = eligible_idx[~constant_eligible]
        n_work = int(work_idx.size)
        if n_work > 0:
            print(f"[trend] computing MK/Sen/OLS on {n_work} variable pixels (eligible={int(eligible.sum())})")
        for idx in work_idx:
            series = flat[:, idx]
            keep = np.isfinite(series)
            yy = series[keep]
            tt = t[keep]

            _, _, p_val = mann_kendall_test_1d(yy)
            p_values[idx] = p_val
            sen[idx] = sen_slope_1d(yy, tt)
            ols[idx] = ols_slope_1d(yy, tt)

    return (
        p_values.reshape(ny, nx),
        sen.reshape(ny, nx),
        ols.reshape(ny, nx),
        valid_count.reshape(ny, nx),
    )


def _filter_small_connected_components(mask, min_pixels=0, connectivity=8):
    arr = np.asarray(mask, dtype=bool)
    if min_pixels <= 1 or not np.any(arr):
        return arr

    if connectivity not in (4, 8):
        raise ValueError(f"Unsupported connectivity={connectivity}. Use 4 or 8.")

    ny, nx = arr.shape
    visited = np.zeros_like(arr, dtype=bool)
    kept = np.zeros_like(arr, dtype=bool)

    if connectivity == 8:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    else:
        neighbors = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    for iy in range(ny):
        for ix in range(nx):
            if not arr[iy, ix] or visited[iy, ix]:
                continue

            stack = [(iy, ix)]
            visited[iy, ix] = True
            component = []

            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    nyy = cy + dy
                    nxx = cx + dx
                    if nyy < 0 or nyy >= ny or nxx < 0 or nxx >= nx:
                        continue
                    if visited[nyy, nxx] or not arr[nyy, nxx]:
                        continue
                    visited[nyy, nxx] = True
                    stack.append((nyy, nxx))

            if len(component) >= int(min_pixels):
                for py, px in component:
                    kept[py, px] = True

    return kept


def _robust_symmetric_limits(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0

    # Keep a true diverging scale centered on zero.
    vmax = float(np.nanmax(np.abs(finite)))
    if vmax <= 0 or not np.isfinite(vmax):
        vmax = 1.0
    return -vmax, vmax


def _extract_lon_lat_extent(ref_ds):
    lons = np.asarray(ref_ds["longitude"].values)
    lats = np.asarray(ref_ds["latitude"].values)
    lons = np.where(lons > 180.0, lons - 360.0, lons)

    if lons.ndim == 1 and lats.ndim == 1:
        x, y = np.meshgrid(lons, lats)
    else:
        x, y = lons, lats

    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))

    extent = [max(-24.5, lon_min - 0.2), min(-13.5, lon_max + 0.2), max(62.8, lat_min - 0.15), min(66.8, lat_max + 0.15)]
    return x, y, extent


def _draw_trend_map(
    field,
    x,
    y,
    extent,
    output_path,
    title,
    colorbar_label,
    p_values=None,
    alpha=0.05,
    significance_veil_alpha=0.15,
    glacier_mask=None,
    coast_color="black",
    show_contours=False,
    mk_sig_mask=None,
    sen_amp_mask=None,
    delta_amp_mask=None,
    launch_params=None,
    sen_threshold=None,
    delta_threshold=None,
):
    vmin, vmax = _robust_symmetric_limits(field)
    ref = get_ref()
    param_key = "delta"
    if "sen" in title.lower():
        param_key = "sen"
    elif "ols" in title.lower():
        param_key = "ols"

    overlays = None
    if show_contours:
        overlays = [
            {
                "mask": mk_sig_mask,
                "color": "black",
                "linestyle": "-",
                "linewidth": 1.0,
                "zorder": 7,
                "label": f"MK significant (p < {alpha:g})",
            },
            {
                "mask": sen_amp_mask,
                "color": "#f59e0b",
                "linestyle": "--",
                "linewidth": 1.0,
                "zorder": 8,
                "label": (
                    f"Thresholded Sen slope (|Sen| >= {float(sen_threshold):g})"
                    if sen_threshold is not None
                    else "Thresholded Sen slope"
                ),
            },
            {
                "mask": delta_amp_mask,
                "color": "#7c3aed",
                "linestyle": ":",
                "linewidth": 1.0,
                "zorder": 9,
                "label": (
                    f"Thresholded Delta (|Delta| >= {float(delta_threshold):g})"
                    if delta_threshold is not None
                    else "Thresholded Delta"
                ),
            },
        ]

    plot_iceland(
        np.asarray(field, dtype=float),
        param_key,
        typ="mask",
        ref=ref,
        unit={param_key: colorbar_label},
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        glaciers=glacier_mask,
        overlays=overlays,
        save_path=output_path,
        show=False,
        title=title,
        coasts=coast_color,
        launch_params=launch_params,
        keep_open=False,
    )
    print(f"Saved figure: {output_path}")


def run_trend_mapping(
    yearly_root,
    param_tag,
    start_year,
    end_year,
    period1_start,
    period1_end,
    period2_start,
    period2_end,
    alpha,
    sen_threshold,
    delta_threshold,
    min_years,
    output_dir,
    data_output_dir,
    include_delta_threshold_line,
    significance_veil_alpha,
    show_contours,
    group_mode,
    basins_path,
    glacier_path,
    min_group_pixels,
    mk_min_pixels,
    mk_connectivity,
):
    years, stack = _load_yearly_stack(
        yearly_root=yearly_root,
        param_tag=param_tag,
        start_year=start_year,
        end_year=end_year,
    )

    print(f"[trend] years used: {years.min()}-{years.max()} ({len(years)} years)")

    delta = _period_mean(stack, years, period2_start, period2_end) - _period_mean(
        stack, years, period1_start, period1_end
    )

    p_mk, sen_map, ols_map, n_valid = _compute_trend_maps(years, stack, min_years=min_years)
    mk_sig = np.isfinite(p_mk) & (p_mk < alpha)
    if int(mk_min_pixels) > 1:
        raw_count = int(np.sum(mk_sig))
        mk_sig = _filter_small_connected_components(
            mk_sig,
            min_pixels=int(mk_min_pixels),
            connectivity=int(mk_connectivity),
        )
        kept_count = int(np.sum(mk_sig))
        print(
            "[trend] MK cleanup: "
            f"min_pixels={int(mk_min_pixels)}, connectivity={int(mk_connectivity)}, "
            f"raw={raw_count}, kept={kept_count}"
        )
    sen_amp = np.isfinite(sen_map) & (np.abs(sen_map) >= sen_threshold)
    delta_amp = np.isfinite(delta) & (np.abs(delta) >= delta_threshold)

    ref = get_ref()
    x, y, extent = _extract_lon_lat_extent(ref)

    if x.shape != delta.shape:
        print("[trend][warning] Reference lon/lat shape does not match data grid; fallback to index plotting.")
        x = None
        y = None
        extent = None

    glacier_mask = _load_glacier_mask()
    if glacier_mask is not None and glacier_mask.shape != delta.shape:
        print("[trend][warning] Glacier mask shape mismatch; glacier contour disabled.")
        glacier_mask = None

    output_dir = Path(output_dir)
    data_output_dir = Path(data_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    fig_a_path = output_dir / "figure_A_delta.png"
    fig_b_path = output_dir / "figure_B_sen.png"
    fig_c_path = output_dir / "figure_C_ols.png"
    launch_params = _parse_param_tag(param_tag)

    _draw_trend_map(
        field=delta,
        x=x,
        y=y,
        extent=extent,
        output_path=fig_a_path,
        title=(
            "Figure A - Delta katabatic fraction "
            f"({period2_start}-{period2_end}) - ({period1_start}-{period1_end})"
        ),
        colorbar_label="Delta fraction",
        p_values=p_mk,
        alpha=alpha,
        significance_veil_alpha=significance_veil_alpha,
        show_contours=show_contours,
        mk_sig_mask=mk_sig,
        sen_amp_mask=sen_amp,
        delta_amp_mask=delta_amp if include_delta_threshold_line else None,
        glacier_mask=glacier_mask,
        launch_params=launch_params,
        sen_threshold=sen_threshold,
        delta_threshold=delta_threshold,
    )

    if group_mode != "none":
        _run_group_modes(
            fig_output_dir=output_dir,
            data_output_dir=data_output_dir,
            years=years,
            stack=stack,
            group_mode=group_mode,
            basins_path=basins_path,
            glacier_path=glacier_path,
            min_group_pixels=min_group_pixels,
        )

    _draw_trend_map(
        field=sen_map,
        x=x,
        y=y,
        extent=extent,
        output_path=fig_b_path,
        title="Figure B - Sen slope (fraction/year) with MK significance",
        colorbar_label="Sen slope (fraction/year)",
        p_values=p_mk,
        alpha=alpha,
        significance_veil_alpha=significance_veil_alpha,
        show_contours=show_contours,
        mk_sig_mask=mk_sig,
        glacier_mask=glacier_mask,
        launch_params=launch_params,
        sen_threshold=sen_threshold,
        delta_threshold=delta_threshold,
    )

    _draw_trend_map(
        field=ols_map,
        x=x,
        y=y,
        extent=extent,
        output_path=fig_c_path,
        title="Figure C - OLS slope (fraction/year) with MK significance",
        colorbar_label="OLS slope (fraction/year)",
        p_values=p_mk,
        alpha=alpha,
        significance_veil_alpha=significance_veil_alpha,
        show_contours=show_contours,
        mk_sig_mask=mk_sig,
        glacier_mask=glacier_mask,
        launch_params=launch_params,
        sen_threshold=sen_threshold,
        delta_threshold=delta_threshold,
    )

    np.savez_compressed(
        data_output_dir / "trend_maps_outputs.npz",
        years=years,
        delta=delta,
        p_mk=p_mk,
        sen=sen_map,
        ols=ols_map,
        n_valid=n_valid,
        mk_sig=mk_sig.astype(np.uint8),
        sen_amp=sen_amp.astype(np.uint8),
        delta_amp=delta_amp.astype(np.uint8),
        settings=np.array(
            [
                f"alpha={alpha}",
                f"sen_threshold={sen_threshold}",
                f"delta_threshold={delta_threshold}",
                f"min_years={min_years}",
                f"param_tag={param_tag}",
            ],
            dtype=object,
        ),
    )
    print(f"Saved outputs: {data_output_dir / 'trend_maps_outputs.npz'}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build CARRA2 katabatic trend maps: Figure A (period delta), "
            "Figure B (Sen slope), Figure C (OLS slope), with MK significance overlays."
        )
    )
    parser.add_argument(
        "--yearly-root",
        default=str(DEFAULT_NPY_DATA_DIR / "yearly"),
        help="Root folder containing yearly per_kat outputs.",
    )
    parser.add_argument(
        "--param-tag",
        default="dt1.5_ws15_wind2.5_dir0.87",
        help="Katabatic parameter tag used in yearly directories.",
    )
    parser.add_argument(
        "--all-param-tags",
        action="store_true",
        help="Run trend mapping for all available parameter-tag folders under yearly-root.",
    )
    parser.add_argument("--start-year", type=int, default=1985)
    parser.add_argument("--end-year", type=int, default=2025)

    parser.add_argument("--period1-start", type=int, default=1985)
    parser.add_argument("--period1-end", type=int, default=2004)
    parser.add_argument("--period2-start", type=int, default=2005)
    parser.add_argument("--period2-end", type=int, default=2025)

    parser.add_argument("--alpha", type=float, default=0.05, help="Mann-Kendall p-value threshold.")
    parser.add_argument("--sen-threshold", type=float, default=0.001, help="Absolute Sen slope threshold.")
    parser.add_argument("--delta-threshold", type=float, default=0.03, help="Absolute period-delta threshold.")
    parser.add_argument("--min-years", type=int, default=20, help="Minimum valid years required per grid point.")

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for trend figures. Defaults to yearly/annees/<param>/<period>/maps/tendancies.",
    )
    parser.add_argument(
        "--data-output-dir",
        default=None,
        help="Output folder for non-figure trend artefacts. Defaults to Data/NPY/yearly/annees/<param>/<period>/tendancies.",
    )
    parser.add_argument(
        "--include-delta-threshold-line",
        action="store_true",
        help="Add optional third contour for |Delta| >= delta-threshold on Figure A.",
    )
    parser.add_argument(
        "--significance-veil-alpha",
        type=float,
        default=0.15,
        help="Opacity of the black overlay on significant pixels (p < alpha).",
    )
    parser.add_argument(
        "--show-contours",
        action="store_true",
        help="Show contour overlays in addition to the significance veil.",
    )
    parser.add_argument(
        "--mk-min-pixels",
        type=int,
        default=0,
        help=(
            "Remove MK-significant connected patches smaller than this size "
            "(in pixels). 0 or 1 disables cleanup."
        ),
    )
    parser.add_argument(
        "--mk-connectivity",
        type=int,
        choices=[4, 8],
        default=8,
        help="Connectivity used for MK patch cleanup (4 or 8).",
    )
    parser.add_argument(
        "--group-mode",
        choices=["none", "basin", "glacier", "both"],
        default="none",
        help="Generate time-series outputs by basin, glacier, or both.",
    )
    parser.add_argument(
        "--basins-path",
        default=str(DEFAULT_NPY_DATA_DIR / "basins.npy"),
        help="Path to basin label map .npy.",
    )
    parser.add_argument(
        "--glacier-path",
        default=str(resolve_glacier_mask_path()),
        help="Path to glacier label map .npy.",
    )
    parser.add_argument(
        "--min-group-pixels",
        type=int,
        default=5,
        help="Minimum number of pixels required to keep a basin/glacier series.",
    )

    args = parser.parse_args()

    requested_period = f"{int(args.start_year)} - {int(args.end_year)}"
    if args.all_param_tags:
        param_tags = _discover_param_tags(args.yearly_root, args.start_year, args.end_year)
        if not param_tags:
            raise RuntimeError(
                "No usable parameter-tag folders found under yearly-root for the requested period."
            )
        print(f"[trend] Found {len(param_tags)} parameter tags.")
    else:
        param_tags = [args.param_tag]

    failures = []
    for current_tag in param_tags:
        available_years = _discover_effective_years(
            yearly_root=args.yearly_root,
            param_tag=current_tag,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        if not available_years:
            failures.append((current_tag, "no yearly maps in requested period"))
            print(f"[trend][skip] {current_tag}: no yearly maps in requested period")
            continue

        effective_period = f"{int(min(available_years))} - {int(max(available_years))}"
        if effective_period != requested_period:
            print(
                "[trend] Requested period differs from available years; "
                f"requested={requested_period}, effective={effective_period}"
            )

        if args.output_dir:
            if args.all_param_tags:
                fig_output_dir = Path(args.output_dir) / current_tag / "aggregated" / effective_period / "maps" / "tendancies"
            else:
                fig_output_dir = Path(args.output_dir)
        else:
            fig_output_dir = DEFAULT_FIGURES_DIR / "yearly" / current_tag / "aggregated" / effective_period / "maps" / "tendancies"

        if args.data_output_dir:
            if args.all_param_tags:
                data_output_dir = Path(args.data_output_dir) / current_tag / "aggregated" / effective_period / "tendancies"
            else:
                data_output_dir = Path(args.data_output_dir)
        else:
            data_output_dir = DEFAULT_NPY_DATA_DIR / "yearly" / current_tag / "aggregated" / effective_period / "tendancies"

        try:
            print(f"[trend] Running param tag: {current_tag}")
            run_trend_mapping(
                yearly_root=args.yearly_root,
                param_tag=current_tag,
                start_year=args.start_year,
                end_year=args.end_year,
                period1_start=args.period1_start,
                period1_end=args.period1_end,
                period2_start=args.period2_start,
                period2_end=args.period2_end,
                alpha=args.alpha,
                sen_threshold=args.sen_threshold,
                delta_threshold=args.delta_threshold,
                min_years=args.min_years,
                output_dir=fig_output_dir,
                data_output_dir=data_output_dir,
                include_delta_threshold_line=args.include_delta_threshold_line,
                significance_veil_alpha=args.significance_veil_alpha,
                show_contours=args.show_contours,
                group_mode=args.group_mode,
                basins_path=args.basins_path,
                glacier_path=args.glacier_path,
                min_group_pixels=args.min_group_pixels,
                mk_min_pixels=args.mk_min_pixels,
                mk_connectivity=args.mk_connectivity,
            )
        except Exception as exc:
            failures.append((current_tag, str(exc)))
            print(f"[trend][error] {current_tag}: {exc}")

    if failures:
        print("[trend] Completed with failures:")
        for tag, reason in failures:
            print(f"  - {tag}: {reason}")
        if len(failures) == len(param_tags):
            raise RuntimeError("All trend-map runs failed.")


if __name__ == "__main__":
    main()
