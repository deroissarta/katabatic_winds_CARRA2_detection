"""Load CARRA2 fields, detect katabatic events, and save classification results."""

import numpy as np
import json
from shared.reader import (
    open_t2m_netcdf,
    crop_iceland,
    open_orography_netcdf,
    open_uv_netcdf,
    resolve_available_years,
    load_static_orography,
)
from shared.map_launcher import plot_iceland, animate_iceland
from shared.paths import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_KATABATIC_DATA_DIR,
    resolve_glacier_mask_path,
    DEFAULT_NPY_DATA_DIR,
    DEFAULT_CARRA_DETECTOR_DIR,
)
from katabatic.reference_map_picker import pick_pixel_from_reference_map, load_reference_map_from_cache
from shared.cache_utils import (
    compute_signature,
    load_manifest,
    outputs_exist,
    path_fingerprint,
    write_manifest,
)
from scipy.ndimage import uniform_filter
from pathlib import Path
from datetime import datetime, timezone
import os
import xarray as xr
from PIL import Image, ImageDraw, ImageFont
import matplotlib
if os.environ.get("KATABATIC_FORCE_AGG", "0") == "1":
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

def reader_launch(month="June", year="2025", path=DEFAULT_KATABATIC_DATA_DIR, mode="concat", load_orography=True, expected_min_years=None):
    """Load CARRA2 datasets in either concatenated or year-by-year mode.

    Parameters
    ----------
    month : str
        Month selector. Use 'JJA' to load June, July, and August.
    year : str
        Specific year such as '2025', or 'multi' to iterate over all available years.
    mode : str
        'concat' returns full xarray datasets, while 'yearly' yields one year at a time.
    load_orography : bool
        If True, also load the static orography field.
    """

    def load_and_crop(year_value, month_value, param_name, open_func):
        folder = Path(path) / str(year_value) / f"CARRA2_{month_value}"
        return crop_iceland(open_func(folder), param_name)

    months = ["June", "July", "August"] if month == "JJA" else [month]
    required_vars = ["t2m", "u10", "v10"]
    if load_orography:
        required_vars.append("orog")

    years = resolve_available_years(
        path,
        months,
        required_vars=required_vars,
        requested_year=year,
        expected_min_years=expected_min_years,
        prefer_strict_timestamps=True,
    )
    print(f"[katabatic] Years selected for processing: {years}")

    open_funcs = {
        "t2m": open_t2m_netcdf,
        "uv": open_uv_netcdf,
    }
    if load_orography:
        open_funcs["orog"] = open_orography_netcdf

    def load_year(year_value):
        datasets = {
            name: [load_and_crop(year_value, m, name, func) for m in months]
            for name, func in open_funcs.items()
        }
        return {
            name: xr.concat(data, dim="time") if len(data) > 1 else data[0]
            for name, data in datasets.items()
        }

    if mode == "concat":
        data_by_year = [load_year(y) for y in years]
        if len(data_by_year) == 1:
            single = data_by_year[0]
            if load_orography:
                return single["t2m"], single["orog"], single["uv"]
            return single["t2m"], single["uv"]

        if load_orography:
            return (
                xr.concat([d["t2m"] for d in data_by_year], dim="time"),
                xr.concat([d["orog"] for d in data_by_year], dim="time"),
                xr.concat([d["uv"] for d in data_by_year], dim="time"),
            )
        return (
            xr.concat([d["t2m"] for d in data_by_year], dim="time"),
            xr.concat([d["uv"] for d in data_by_year], dim="time"),
        )

    if mode == "yearly":
        def generate_yearly():
            for y in years:
                data = load_year(y)
                if load_orography:
                    yield y, data["t2m"], data["orog"], data["uv"]
                else:
                    yield y, data["t2m"], data["uv"]
        return generate_yearly()

    raise ValueError(f"Unknown mode: {mode}")

def detect_catabatic_local(ds_t2m, ds_orog, ds_uv,
                           window_size = 15,
                           dT_threshold = 1.5,
                           wind_threshold = 2.5,
                           direction_threshold = 0.87,
                           return_counts=False):
    """Detect local katabatic events from wind direction, slope, and temperature anomaly.

    The temperature anomaly is defined as the difference between the current
    2 m temperature and a spatially smoothed local mean computed over a
    neighborhood of size `window_size`. Katabatic flow is expected to be
    colder than the surrounding air, so pixels are flagged when they are at
    least `dT_threshold` degrees below that local mean.

    Wind direction is compared to the terrain downslope direction through the
    cosine of the angle difference. Values above `direction_threshold` mean
    the wind is broadly aligned with the downslope vector.
    """

    # Build the terrain gradient once and derive the downslope direction.
    h = ds_orog["orog"].isel(time=0).values.astype(float)
    dh_dy, dh_dx = np.gradient(h)

    # Angle of the steepest downslope vector at each grid cell.
    slope_dir = np.arctan2(-dh_dy, -dh_dx)

    # Loop over time and accumulate both event timestamps and occurrence counts.
    kat_times = []
    masks = {}
    ntime = len(ds_t2m.time)
    mask_sum = np.zeros(h.shape, dtype=np.uint32)

    for i, t in enumerate(ds_t2m.time.values):
        u_i = ds_uv["u10"].isel(time=i).values.astype(float)
        v_i = ds_uv["v10"].isel(time=i).values.astype(float)
        WD_i = np.arctan2(v_i, u_i)

        # Keep pixels where the wind points broadly downslope.
        downslope_i = np.cos(WD_i - slope_dir) >= direction_threshold

        t2m_i = ds_t2m["t2m"].isel(time=i).values.astype(float)
        T_local_mean_i = uniform_filter(t2m_i, size=window_size, mode="nearest")

        # Katabatic drainage should be colder than the surrounding air.
        cold_i = (t2m_i - T_local_mean_i) < -dT_threshold

        strong_i = np.sqrt(u_i**2 + v_i**2) > wind_threshold
        mask_i = downslope_i & cold_i & strong_i

        if mask_i.any():
            kat_times.append(t)
            masks[str(t)] = mask_i

        mask_sum += mask_i.astype(np.uint32)

    per_kat = (mask_sum / float(ntime)).astype(np.float32)
    if return_counts:
        return kat_times, masks, per_kat, mask_sum, ntime
    return kat_times, masks, per_kat

def detect_catabatic_zones(cat_percent, z, ref,
                           threshold=0.002, min_cluster_size=3, threshold_relief1 = 350, threshold_relief2 = 800):
    """
    Assign persistent katabatic pixels to drainage basins derived from the DEM.

    The current implementation builds basin labels from the glacier mask and
    the topographic flow paths, then maps persistent katabatic pixels onto
    those basin identifiers.
    """

    Y, X = cat_percent.shape

    # Load the basin labels generated by the katabatic pipeline.
    basins_path = DEFAULT_NPY_DATA_DIR / "basins.npy"
    if not basins_path.exists():
        raise FileNotFoundError(
            f"Basin labels not found: {basins_path}. Run the katabatic basin generation step first."
        )
    basins = np.load(basins_path)

    cluster = basins.copy()
    basins[basins == 0] = -1

    # Keep only pixels that remain active above the persistence threshold.
    mask = cat_percent > threshold
    if not mask.any():
        print("No catabatic pixels detected.")
        return None, None, np.full(cat_percent.shape, -1)

    # Map those persistent pixels to the basin label at the same grid cell.
    label_map = np.full(cat_percent.shape, -1, dtype=int)
    label_map[mask] = basins[mask]

    coords = np.column_stack(np.where(mask))
    labels = label_map[mask]

    return coords, labels, label_map


def build_param_tag(window_size, dT_threshold, wind_threshold, direction_threshold):
    return (
        f"dt{dT_threshold:g}_ws{window_size}"
        f"_wind{wind_threshold:g}_dir{direction_threshold:g}"
    )


def compute_fraction_evolution_for_basin(yearly_per_kat_by_year, label_map, basin_label):
    """Return yearly mean katabatic fraction for a basin label."""
    years = []
    values = []
    for year, per_kat in sorted(yearly_per_kat_by_year.items(), key=lambda item: str(item[0])):
        if per_kat is None or label_map is None:
            continue
        if np.asarray(per_kat).shape != np.asarray(label_map).shape:
            continue
        basin_mask = np.asarray(label_map) == int(basin_label)
        if not np.any(basin_mask):
            continue
        basin_values = np.asarray(per_kat)[basin_mask]
        valid = np.isfinite(basin_values)
        if not np.any(valid):
            continue
        years.append(str(year))
        values.append(float(np.mean(basin_values[valid])))
    return years, values


def compute_fraction_evolution_for_pixel(yearly_per_kat_by_year, pixel):
    """Return yearly katabatic fraction for a selected pixel (y, x)."""
    if pixel is None:
        raise ValueError("pixel must be a (y, x) tuple")
    y, x = int(pixel[0]), int(pixel[1])
    years = []
    values = []
    for year, per_kat in sorted(yearly_per_kat_by_year.items(), key=lambda item: str(item[0])):
        arr = np.asarray(per_kat)
        if arr.ndim != 2:
            continue
        if y < 0 or y >= arr.shape[0] or x < 0 or x >= arr.shape[1]:
            continue
        value = float(arr[y, x])
        years.append(str(year))
        values.append(value)
    return years, values


def select_pixel_on_reference_map(reference_map, title="Click a pixel on the katabatic fraction reference map",
                                selection_file=None, param_tag=None, years=None, show=True):
    """Compatibility wrapper around the dedicated reference-map picker."""
    return pick_pixel_from_reference_map(
        reference_map,
        title=title,
        fallback_prompt=True,
        selection_file=selection_file,
        param_tag=param_tag,
        years=years,
        show=show,
    )


def export_fraction_evolution_graphs(mode, yearly_per_kat_by_year, label_map=None, output_dir=None,
                                    basin_label=None, pixel=None, reference_map=None, title_prefix=""):
    """Export basin/pixel fraction-evolution figures under the active run graphs branch."""
    base_dir = Path(output_dir) if output_dir is not None else Path(DEFAULT_FIGURES_DIR) / "yearly" / "aggregated"
    graph_root = base_dir / "graphs" / str(mode) / "katabatic_fraction_evolution"
    graph_root.mkdir(parents=True, exist_ok=True)

    def _safe_save_figure(fig, out_path):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)

    def _get_default_font():
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _save_basin_context_png(out_path, labels_array, basin_id, glacier_ids=None):
        labels_array = np.asarray(labels_array)
        basin_mask = labels_array == basin_id
        scale = max(1, int(900 / max(labels_array.shape[0], labels_array.shape[1])))
        base = np.zeros(labels_array.shape + (3,), dtype=np.uint8)
        base[labels_array > 0] = (55, 70, 92)
        if glacier_ids:
            glacier_labels_path = DEFAULT_NPY_DATA_DIR / "glacier_labels.npy"
            if glacier_labels_path.exists():
                glacier_labels_arr = np.load(glacier_labels_path)
                if glacier_labels_arr.shape == labels_array.shape:
                    glacier_mask = np.isin(glacier_labels_arr, glacier_ids)
                    base[glacier_mask] = (0, 229, 255)
        base[basin_mask] = (244, 208, 63)
        image = Image.fromarray(base, mode="RGB").resize(
            (labels_array.shape[1] * scale, labels_array.shape[0] * scale),
            resample=Image.Resampling.NEAREST,
        )
        canvas = Image.new("RGB", (image.width, image.height + 110), (255, 255, 255))
        canvas.paste(image, (0, 110))
        draw = ImageDraw.Draw(canvas)
        font = _get_default_font()
        title = f"{title_prefix}Basin {basin_id} context"
        subtitle = "Glacier IDs: " + (", ".join(str(g) for g in glacier_ids) if glacier_ids else "none")
        legend = "Dark blue: other basins | Yellow: selected basin | Cyan: associated glaciers"
        draw.text((12, 12), title, fill=(0, 0, 0), font=font)
        draw.text((12, 36), subtitle, fill=(0, 0, 0), font=font)
        draw.text((12, 60), legend, fill=(0, 0, 0), font=font)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)

    def _save_fraction_line_png(out_path, years_list, values_list, graph_title):
        width = 1200
        height = 700
        left = 90
        right = 40
        top = 70
        bottom = 110
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = _get_default_font()

        draw.text((20, 20), graph_title, fill=(0, 0, 0), font=font)
        draw.line((left, top, left, height - bottom), fill=(0, 0, 0), width=2)
        draw.line((left, height - bottom, width - right, height - bottom), fill=(0, 0, 0), width=2)

        series = np.asarray(values_list, dtype=float)
        valid_mask = np.isfinite(series)
        if not np.any(valid_mask):
            draw.text((left + 20, top + 20), "No valid values", fill=(160, 0, 0), font=font)
        else:
            valid_values = series[valid_mask]
            y_min = min(0.0, float(np.min(valid_values)))
            y_max = max(1.0, float(np.max(valid_values)))
            if abs(y_max - y_min) < 1e-9:
                y_max = y_min + 1.0

            plot_w = width - left - right
            plot_h = height - top - bottom

            for frac in np.linspace(0.0, 1.0, 6):
                y_val = y_min + frac * (y_max - y_min)
                y_px = top + int((1.0 - frac) * plot_h)
                draw.line((left, y_px, width - right, y_px), fill=(220, 220, 220), width=1)
                draw.text((10, y_px - 7), f"{y_val:.2f}", fill=(0, 0, 0), font=font)

            points = []
            if len(series) == 1:
                x_positions = [left + plot_w // 2]
            else:
                x_positions = [left + int(i * plot_w / (len(series) - 1)) for i in range(len(series))]

            for i, (x_px, value) in enumerate(zip(x_positions, series)):
                if not np.isfinite(value):
                    continue
                y_px = top + int((1.0 - ((value - y_min) / (y_max - y_min))) * plot_h)
                points.append((x_px, y_px))
                draw.ellipse((x_px - 4, y_px - 4, x_px + 4, y_px + 4), fill=(31, 119, 180), outline=(31, 119, 180))
                if len(years_list) <= 12 or i % max(1, len(years_list) // 12) == 0 or i == len(years_list) - 1:
                    draw.text((x_px - 12, height - bottom + 12), str(years_list[i]), fill=(0, 0, 0), font=font)
            if len(points) >= 2:
                draw.line(points, fill=(31, 119, 180), width=3)

        draw.text((width // 2 - 20, height - 40), "Year", fill=(0, 0, 0), font=font)
        draw.text((15, top - 30), "Katabatic fraction", fill=(0, 0, 0), font=font)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

    if mode == "basin":
        if label_map is None:
            raise ValueError("label_map is required in basin mode")

        labels = np.asarray(label_map)
        if labels.ndim != 2:
            raise ValueError("label_map must be a 2D array in basin mode")

        if basin_label in (None, "all"):
            basin_labels = [int(v) for v in np.unique(labels) if int(v) > 0]
        else:
            basin_labels = [int(basin_label)]

        glacier_labels = None
        glacier_labels_path = DEFAULT_NPY_DATA_DIR / "glacier_labels.npy"
        if glacier_labels_path.exists():
            try:
                glacier_labels = np.load(glacier_labels_path)
            except Exception:
                glacier_labels = None

        exported_dirs = []
        for basin_id in basin_labels:
            years, values = compute_fraction_evolution_for_basin(yearly_per_kat_by_year, labels, basin_id)
            section_dir = graph_root / f"basin_{basin_id}"
            section_dir.mkdir(parents=True, exist_ok=True)

            if reference_map is not None:
                basin_mask = labels == basin_id
                glacier_ids = []
                if glacier_labels is not None and np.asarray(glacier_labels).shape == labels.shape:
                    glacier_values = np.asarray(glacier_labels)[basin_mask]
                    glacier_ids = [int(v) for v in np.unique(glacier_values) if int(v) > 0]
                _save_basin_context_png(section_dir / "basin_context_map.png", labels, basin_id, glacier_ids=glacier_ids)

            _save_fraction_line_png(
                section_dir / "basin_fraction_evolution.png",
                years,
                values,
                f"{title_prefix}Katabatic fraction evolution - basin {basin_id}",
            )
            exported_dirs.append(section_dir)

        if basin_label == "all":
            print(f"[katabatic] Basin fraction-evolution exported for {len(exported_dirs)} basins.")

        return exported_dirs

    if mode == "pixel":
        if pixel is None:
            raise ValueError("A pixel (y, x) must be provided in pixel mode")
        y, x = int(pixel[0]), int(pixel[1])
        years, values = compute_fraction_evolution_for_pixel(yearly_per_kat_by_year, (y, x))
        section_dir = graph_root / f"pixel_{y}_{x}"
        section_dir.mkdir(parents=True, exist_ok=True)

        if reference_map is not None:
            fig_map, ax_map = plt.subplots(figsize=(6, 6))
            ax_map.imshow(np.asarray(reference_map), cmap="plasma", origin="lower")
            ax_map.scatter(x, y, color="red", s=60, marker="o", edgecolors="black", linewidths=0.8)
            ax_map.set_title(f"{title_prefix}Selected pixel position (y={y}, x={x})")
            ax_map.set_xlabel("x")
            ax_map.set_ylabel("y")
            _safe_save_figure(fig_map, section_dir / "pixel_position.png")
            plt.close(fig_map)

        x_values = np.arange(len(years))
        fig_series, ax_series = plt.subplots(figsize=(10, 5))
        ax_series.plot(x_values, values, marker="o", linewidth=2, color="tab:blue")
        ax_series.set_title(f"{title_prefix}Katabatic fraction evolution — pixel ({y}, {x})")
        ax_series.set_xlabel("Year")
        ax_series.set_ylabel("Katabatic fraction")
        ax_series.grid(True, alpha=0.3)
        if len(years) > 12:
            step = max(1, len(years) // 12)
            ax_series.set_xticks(x_values[::step])
            ax_series.set_xticklabels([years[i] for i in range(0, len(years), step)], rotation=45, ha="right")
        else:
            ax_series.set_xticks(x_values)
            ax_series.set_xticklabels(years, rotation=45, ha="right")
        _safe_save_figure(fig_series, section_dir / "pixel_fraction_evolution.png")
        plt.close(fig_series)
        return section_dir

    raise ValueError(f"Unsupported fraction-evolution mode: {mode}")


def _build_yearly_fraction_series_by_year(base_dir, years, param_tag):
    year_series = {}
    for year in years:
        per_kat_path = Path(base_dir) / "yearly" / param_tag / str(year) / f"per_kat_{param_tag}.npy"
        if not per_kat_path.exists():
            continue
        year_series[str(year)] = np.load(per_kat_path)
    return year_series


def _export_aggregated_fraction_evolution(output_dir, years, param_tag, label_map, reference_map, mode,
                                         basin_label=None, pixel=None, allow_click=False, title_prefix=""):
    year_series = _build_yearly_fraction_series_by_year(DEFAULT_NPY_DATA_DIR, years, param_tag)
    if not year_series:
        print("[katabatic] No yearly per_kat data available for fraction-evolution export.")
        return None

    if mode == "pixel" and pixel is None and allow_click:
        try:
            pixel = select_pixel_on_reference_map(reference_map, title=f"{title_prefix}Click on the representative katabatic fraction map")
        except Exception as exc:
            print(f"[katabatic] Pixel selection cancelled: {exc}")
            return None

    return export_fraction_evolution_graphs(
        mode=mode,
        yearly_per_kat_by_year=year_series,
        label_map=label_map,
        output_dir=output_dir,
        basin_label=basin_label,
        pixel=pixel,
        reference_map=reference_map,
        title_prefix=title_prefix,
    )


def _save_yearly_katabatic_outputs(base_dir, year, param_tag, per_kat, label_map, times=None):
    out_dir = Path(base_dir) / "yearly" / param_tag / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"per_kat_{param_tag}.npy", per_kat)
    np.save(out_dir / f"label_map_{param_tag}.npy", label_map)
    if times is not None:
        np.save(out_dir / f"times_{param_tag}.npy", np.asarray(times, dtype=object))
    return out_dir


def _save_yearly_aggregate_outputs(base_dir, param_tag, period, per_kat, label_map, times=None):
    out_dir = Path(base_dir) / "yearly" / param_tag / "aggregated" / period
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"per_kat_{param_tag}.npy", per_kat)
    np.save(out_dir / f"label_map_{param_tag}.npy", label_map)
    if times is not None:
        np.save(out_dir / f"times_{param_tag}.npy", np.asarray(times, dtype=object))
    return out_dir


def _yearly_output_paths(base_dir, year, param_tag):
    out_dir = Path(base_dir) / "yearly" / param_tag / str(year)
    return {
        "dir": out_dir,
        "per_kat": out_dir / f"per_kat_{param_tag}.npy",
        "label_map": out_dir / f"label_map_{param_tag}.npy",
        "times": out_dir / f"times_{param_tag}.npy",
    }


def export_yearly_maps_from_cache(
    output_dir,
    years,
    param_tag,
    ref_ds,
    show_plots,
    launch_params,
    glaciers=None,
):
    """Backfill yearly map outputs from cached NPY files without recomputation."""
    if glaciers is None:
        try:
            glaciers = np.load(resolve_glacier_mask_path())
        except Exception:
            glaciers = None

    output_root = Path(output_dir) / "yearly" / param_tag
    n_saved = 0

    def _save_with_retry(save_path, plot_fn):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            plot_fn()
            return
        except FileNotFoundError:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plot_fn()

    for year in years:
        year_str = str(year)
        paths = _yearly_output_paths(DEFAULT_NPY_DATA_DIR, year_str, param_tag)
        if not paths["per_kat"].exists() or not paths["label_map"].exists():
            continue

        year_map_dir = output_root / year_str / "maps"
        year_map_dir.mkdir(parents=True, exist_ok=True)
        fraction_path = year_map_dir / "katabatic_fraction.png"
        clusters_path = year_map_dir / "katabatic_clusters.png"

        per_kat = None
        label_map = None
        if (not fraction_path.exists()) or (not clusters_path.exists()):
            per_kat = np.load(paths["per_kat"])
            label_map = np.load(paths["label_map"])

        if not fraction_path.exists():
            _save_with_retry(
                fraction_path,
                lambda: plot_iceland(
                    per_kat,
                    "per_kat",
                    typ="mask",
                    ref=ref_ds,
                    cmap="plasma",
                    glaciers=glaciers,
                    save_path=fraction_path,
                    show=show_plots,
                    title=f"Katabatic fraction - {year_str} (JJA)",
                    coasts="white",
                    launch_params=launch_params,
                ),
            )
            n_saved += 1

        if not clusters_path.exists():
            _save_with_retry(
                clusters_path,
                lambda: plot_iceland(
                    label_map,
                    f"cluster_{int(np.max(label_map))}",
                    typ="mask",
                    ref=ref_ds,
                    unit={"cluster": "n"},
                    cmap="tab20",
                    glaciers=glaciers,
                    save_path=clusters_path,
                    show=show_plots,
                    title=f"Katabatic clusters - {year_str} (JJA)",
                    launch_params=launch_params,
                ),
            )
            n_saved += 1

    return n_saved


def _load_saved_yearly_times(base_dir, year, param_tag):
    paths = _yearly_output_paths(base_dir, year, param_tag)
    times_path = paths["times"]
    if not times_path.exists():
        return []

    try:
        times = np.load(times_path, allow_pickle=True)
    except Exception as exc:
        print(f"[yearly] Warning: could not load saved times for year {year}: {exc}")
        return []

    return [f"{year}_{t}" for t in times.tolist()]


def _aggregate_saved_yearly_per_kat(base_dir, years, param_tag):
    sum_map = None
    count_map = None
    included_years = []
    excluded_years = []

    for year in years:
        paths = _yearly_output_paths(base_dir, year, param_tag)
        per_kat_path = paths["per_kat"]

        if not per_kat_path.exists():
            excluded_years.append((str(year), "missing per_kat output"))
            continue

        try:
            arr = np.asarray(np.load(per_kat_path, mmap_mode="r"), dtype=np.float32)
        except Exception as exc:
            excluded_years.append((str(year), f"cannot read per_kat output: {exc}"))
            continue

        if sum_map is None:
            sum_map = np.zeros(arr.shape, dtype=np.float64)
            count_map = np.zeros(arr.shape, dtype=np.uint16)
        elif arr.shape != sum_map.shape:
            excluded_years.append(
                (
                    str(year),
                    f"shape mismatch {arr.shape} != {sum_map.shape}",
                )
            )
            continue

        valid = np.isfinite(arr)
        sum_map[valid] += arr[valid]
        count_map[valid] += 1
        included_years.append(str(year))

    if sum_map is None:
        return None, included_years, excluded_years

    with np.errstate(divide="ignore", invalid="ignore"):
        aggregated = np.divide(
            sum_map,
            count_map,
            out=np.full(sum_map.shape, np.nan, dtype=np.float32),
            where=count_map > 0,
        ).astype(np.float32)

    return aggregated, included_years, excluded_years


def _yearly_cache_manifest_path(param_tag):
    return DEFAULT_CARRA_DETECTOR_DIR / f"carra2_katabatic_yearly_cache_{param_tag}.json"


def _build_year_source_fingerprint(path, year, months):
    root = Path(path)
    month_payloads = []

    for month in months:
        month_dir = root / str(year) / f"CARRA2_{month}"
        nc_files = sorted(month_dir.glob("*.nc")) if month_dir.exists() else []
        month_payloads.append(
            {
                "month": str(month),
                "month_dir": path_fingerprint(month_dir),
                "nc_files": [path_fingerprint(p) for p in nc_files],
            }
        )

    return {
        "year": str(year),
        "months": list(months),
        "source": month_payloads,
    }


def _build_year_cache_signature(
    path,
    year,
    detector_config,
    orography_config,
):
    payload = {
        "schema_version": 1,
        "stage": "carra2_katabatic_year",
        "detector_config": detector_config,
        "orography_config": orography_config,
        "source_fingerprint": _build_year_source_fingerprint(
            path,
            year,
            months=["June", "July", "August"],
        ),
    }
    return compute_signature(payload)


def _build_global_summary(per_kat):
    values = np.asarray(per_kat, dtype=float)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "fraction_pixels_gt_0": None,
            "fraction_pixels_gt_001": None,
            "fraction_pixels_gt_005": None,
        }

    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "fraction_pixels_gt_0": float(np.mean(valid > 0.0)),
        "fraction_pixels_gt_001": float(np.mean(valid > 0.01)),
        "fraction_pixels_gt_005": float(np.mean(valid > 0.05)),
    }


def _build_basin_metrics_all(per_kat, label_map, n_timestamps):
    values = np.asarray(per_kat, dtype=float)
    labels = np.asarray(label_map)
    if values.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch between per_kat {values.shape} and label_map {labels.shape}"
        )

    basin_metrics = []
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    for basin_label in unique_labels:
        basin_mask = labels == basin_label
        basin_values = values[basin_mask]
        basin_values = basin_values[np.isfinite(basin_values)]
        if basin_values.size == 0:
            continue

        nb_pixels_total = int(basin_values.size)
        nb_pixels_actifs = int(np.sum(basin_values > 0.0))
        detection_pixel_timestamps = float(np.sum(basin_values) * max(int(n_timestamps), 0))

        basin_metrics.append(
            {
                "label": int(basin_label),
                "nb_pixels_total": nb_pixels_total,
                "nb_pixels_actifs": nb_pixels_actifs,
                "fraction_active_pixels": float(nb_pixels_actifs / nb_pixels_total),
                "fraction_catabatique_moyenne_bassin": float(np.mean(basin_values)),
                "intensite_temporelle_bassin": detection_pixel_timestamps,
                "nb_detections_bassin": int(round(detection_pixel_timestamps)),
            }
        )

    basin_metrics.sort(key=lambda row: row["label"])
    return basin_metrics


def _save_carra_classification_json(
    output_path,
    per_kat,
    label_map,
    period,
    param_tag,
    processed_years,
    n_years,
    detector_config,
    npy_references,
    n_timestamps,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": {
            "dataset": "CARRA2",
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "param_tag": str(param_tag),
            "period": str(period),
            "grid_shape": [int(per_kat.shape[0]), int(per_kat.shape[1])],
            "n_years": int(n_years),
            "processed_years": [str(y) for y in processed_years],
            "nb_timestamps_detectes": int(max(int(n_timestamps), 0)),
        },
        "detector_config": {
            "window_size": int(detector_config["window_size"]),
            "dT_threshold": float(detector_config["dT_threshold"]),
            "wind_threshold": float(detector_config["wind_threshold"]),
            "direction_threshold": float(detector_config["direction_threshold"]),
        },
        "aggregated_summary_global": _build_global_summary(per_kat),
        "basin_metrics_all": _build_basin_metrics_all(per_kat, label_map, n_timestamps),
        "npy_references": {
            "per_kat": str(npy_references.get("per_kat", "")),
            "label_map": str(npy_references.get("label_map", "")),
            "times": str(npy_references.get("times", "")),
            "basins": str(npy_references.get("basins", "")),
        },
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def process_jja_years(path, output_ref_year="2022",
                      orog_year=None,
                      orog_month="June",
                      orog_file_index=2,
                      expected_min_years=5,
                      window_size= 15,
                      dT_threshold= 1.5,
                      wind_threshold= 2.5,
                      direction_threshold= 0.87,
                      output_dir=None,
                      param_tag=None,
                      save_figures=True,
                      save_animations=False,
                      show_plots=True,
                      icegaws_points=None,
                      icegaws_points_annotate=False,
                      fraction_evolution_mode=None,
                      fraction_evolution_basin=None,
                      fraction_evolution_pixel=None,
                      fraction_evolution_click=False):
    
    """Process JJA year by year, save per-year outputs, and build an aggregated summary."""

    years = resolve_available_years(
        path,
        ["June", "July", "August"],
        required_vars=["t2m", "u10", "v10"],
        requested_year="multi",
        expected_min_years=expected_min_years,
        prefer_strict_timestamps=True,
    )
    print(f"[katabatic] Years selected for yearly processing: {years}")

    selected_orog_year = orog_year or output_ref_year
    ds_orog_static = load_static_orography(
        path,
        year=selected_orog_year,
        month=orog_month,
        file_index=orog_file_index,
    )
    
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_FIGURES_DIR
    if param_tag is None:
        param_tag = build_param_tag(window_size, dT_threshold, wind_threshold, direction_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector_config = {
        "window_size": int(window_size),
        "dT_threshold": float(dT_threshold),
        "wind_threshold": float(wind_threshold),
        "direction_threshold": float(direction_threshold),
    }
    orography_config = {
        "year": str(selected_orog_year),
        "month": str(orog_month),
        "file_index": int(orog_file_index),
    }

    yearly_manifest_path = _yearly_cache_manifest_path(param_tag)
    yearly_manifest = load_manifest(yearly_manifest_path) or {}
    yearly_manifest = yearly_manifest if isinstance(yearly_manifest, dict) else {}
    yearly_manifest.setdefault("schema_version", 1)
    yearly_manifest["stage"] = "carra2_katabatic_yearly_cache"
    yearly_manifest["param_tag"] = str(param_tag)
    yearly_manifest["detector_config"] = detector_config
    yearly_manifest["orography_config"] = orography_config
    yearly_manifest["source_path"] = str(Path(path).resolve())
    yearly_manifest["years_available"] = [str(y) for y in years]
    yearly_manifest.setdefault("years", {})

    years_reusable = []
    years_to_compute = []
    expected_signatures = {}

    for year in years:
        year_str = str(year)
        year_sig = _build_year_cache_signature(
            path=path,
            year=year_str,
            detector_config=detector_config,
            orography_config=orography_config,
        )
        expected_signatures[year_str] = year_sig

        paths = _yearly_output_paths(DEFAULT_NPY_DATA_DIR, year_str, param_tag)
        cache_entry = yearly_manifest["years"].get(year_str, {})
        cache_hit = (
            cache_entry.get("status") == "done"
            and cache_entry.get("signature") == year_sig
            and outputs_exist([paths["per_kat"], paths["label_map"]])
        )

        if cache_hit:
            years_reusable.append(year_str)
        else:
            years_to_compute.append(year_str)

    if years_reusable:
        print(f"[yearly] Reusing cached years ({len(years_reusable)}): {years_reusable}")
        if save_figures:
            saved_from_cache = export_yearly_maps_from_cache(
                output_dir=output_dir,
                years=years_reusable,
                param_tag=param_tag,
                ref_ds=ds_orog_static,
                show_plots=show_plots,
                launch_params={
                    "dt_threshold": dT_threshold,
                    "window_size": window_size,
                    "wind_threshold": wind_threshold,
                    "direction_threshold": direction_threshold,
                },
                glaciers=np.load(resolve_glacier_mask_path()),
            )
            if saved_from_cache:
                print(f"[yearly] Saved {saved_from_cache} missing yearly map files from cache.")
    if years_to_compute:
        print(f"[yearly] Years to compute ({len(years_to_compute)}): {years_to_compute}")
    else:
        print("[yearly] No yearly recomputation needed.")

    aggregated_masks = {}
    freshly_processed_years = []
    skipped_years = []
    n_years_computed = 0

    for year in years_to_compute:
        try:
            print(f"Processing year {year}")

            times = []
            masks = {}
            total_mask_sum = None
            total_ntime = 0
            ds_t2m_ref = None
            months = ["June", "July", "August"]

            for month_name in months:
                ds_t2m_m, ds_uv_m = reader_launch(
                    month_name,
                    year,
                    path=path,
                    mode="concat",
                    load_orography=False,
                    expected_min_years=None,
                )
                ds_t2m_ref = ds_t2m_m

                m_times, m_masks, _m_per_kat, m_mask_sum, m_ntime = detect_catabatic_local(
                    ds_t2m_m,
                    ds_orog_static,
                    ds_uv_m,
                    window_size=window_size,
                    dT_threshold=dT_threshold,
                    wind_threshold=wind_threshold,
                    direction_threshold=direction_threshold,
                    return_counts=True,
                )

                times.extend(m_times)
                masks.update(m_masks)

                if total_mask_sum is None:
                    total_mask_sum = m_mask_sum.astype(np.uint32)
                else:
                    total_mask_sum += m_mask_sum.astype(np.uint32)
                total_ntime += int(m_ntime)

                ds_t2m_m.close()
                ds_uv_m.close()

            if total_mask_sum is None or total_ntime == 0:
                raise RuntimeError(f"No time steps available for year {year}")

            per_kat = (total_mask_sum / float(total_ntime)).astype(np.float32)

            year_dir = output_dir / "yearly" / param_tag / year
            plot_dir = year_dir / "maps"
            anim_dir = year_dir / "animations"
            if save_figures:
                plot_dir.mkdir(parents=True, exist_ok=True)
            if save_animations:
                anim_dir.mkdir(parents=True, exist_ok=True)

            z = ds_orog_static["orog"].isel(time=0).values
            coords, labels, label_map = detect_catabatic_zones(per_kat, z, ds_t2m_ref)

            _save_yearly_katabatic_outputs(
                base_dir=DEFAULT_NPY_DATA_DIR,
                year=year,
                param_tag=param_tag,
                per_kat=per_kat,
                label_map=label_map,
                times=times,
            )

            year_paths = _yearly_output_paths(DEFAULT_NPY_DATA_DIR, year, param_tag)
            yearly_manifest["years"][str(year)] = {
                "status": "done",
                "signature": expected_signatures[str(year)],
                "outputs": {
                    "per_kat": str(year_paths["per_kat"]),
                    "label_map": str(year_paths["label_map"]),
                    "times": str(year_paths["times"]),
                },
                "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
            write_manifest(yearly_manifest_path, yearly_manifest)
        
            plot_iceland(
                per_kat,
                "per_kat",
                typ="mask",
                ref=ds_t2m_ref,
                cmap="plasma",
                glaciers=np.load(resolve_glacier_mask_path()),
                points=icegaws_points,
                annotate_points=icegaws_points_annotate,
                save_path=(plot_dir / "katabatic_fraction.png") if save_figures else None,
                show=show_plots,
                title=f"Katabatic fraction - {year} (JAA)",
                coasts="white", 
                launch_params={"dt_threshold" : dT_threshold, "window_size" : window_size, "wind_threshold" : wind_threshold, "direction_threshold" : direction_threshold}
            )

            if save_animations:
                animate_iceland(
                    masks,
                    ref=ds_t2m_ref,
                    save_path=anim_dir / "katabatic_mask_animation.gif",
                    show=show_plots,
                )

            print(f"[yearly] Completed processing for year {year} ({n_years_computed + 1} of {len(years_to_compute)} computed years)")

            aggregated_masks.update({f"{year}_{t}": masks[t] for t in masks})
            freshly_processed_years.append(str(year))
            n_years_computed += 1
        except Exception as exc:
            skipped_years.append((str(year), f"year processing failed: {exc}"))
            yearly_manifest["years"][str(year)] = {
                "status": "failed",
                "signature": expected_signatures[str(year)],
                "error": str(exc),
                "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
            write_manifest(yearly_manifest_path, yearly_manifest)
            continue

    years_candidates = [str(y) for y in years if str(y) not in {s[0] for s in skipped_years}]
    aggregated_percent, processed_years, aggregation_exclusions = _aggregate_saved_yearly_per_kat(
        DEFAULT_NPY_DATA_DIR,
        years_candidates,
        param_tag,
    )

    skipped_years.extend(aggregation_exclusions)

    if aggregated_percent is None or not processed_years:
        raise RuntimeError("No valid yearly outputs available for aggregation")

    aggregated_times = []
    for year in processed_years:
        aggregated_times.extend(_load_saved_yearly_times(DEFAULT_NPY_DATA_DIR, year, param_tag))

    n_years = len(processed_years)

    processed_years_int = sorted(int(y) for y in processed_years)
    period = f"{processed_years_int[0]} - {processed_years_int[-1]}"
    period_root = Path(output_dir) / "yearly" / param_tag / "aggregated" / period
    map_root = period_root / "maps"
    graph_root = period_root / "graphs"
    agg_anim_dir = period_root / "animations"

    if save_figures:
        period_root.mkdir(parents=True, exist_ok=True)
        map_root.mkdir(parents=True, exist_ok=True)
        graph_root.mkdir(parents=True, exist_ok=True)
    if save_animations:
        agg_anim_dir.mkdir(parents=True, exist_ok=True)

    z = ds_orog_static["orog"].isel(time=0).values
    coords, labels, label_map = detect_catabatic_zones(aggregated_percent, z, ds_orog_static)

    aggregate_npy_dir = _save_yearly_aggregate_outputs(
        base_dir=DEFAULT_NPY_DATA_DIR,
        param_tag=param_tag,
        period=period,
        per_kat=aggregated_percent,
        label_map=label_map,
        times=aggregated_times,
    )

    _save_carra_classification_json(
        output_path=DEFAULT_CARRA_DETECTOR_DIR / "carra2_katabatic_classification.json",
        per_kat=aggregated_percent,
        label_map=label_map,
        period=period,
        param_tag=param_tag,
        processed_years=processed_years,
        n_years=n_years,
        detector_config={
            "window_size": window_size,
            "dT_threshold": dT_threshold,
            "wind_threshold": wind_threshold,
            "direction_threshold": direction_threshold,
        },
        npy_references={
            "per_kat": aggregate_npy_dir / f"per_kat_{param_tag}.npy",
            "label_map": aggregate_npy_dir / f"label_map_{param_tag}.npy",
            "times": aggregate_npy_dir / f"times_{param_tag}.npy",
            "basins": DEFAULT_NPY_DATA_DIR / "basins.npy",
        },
        n_timestamps=len(aggregated_times),
    )

    plot_iceland(
        aggregated_percent,
        "per_kat",
        typ="mask",
        ref=ds_orog_static,
        cmap="plasma",
        glaciers=np.load(resolve_glacier_mask_path()),
        points=icegaws_points,
        annotate_points=icegaws_points_annotate,
        save_path=(map_root / "katabatic_fraction.png") if save_figures else None,
        show=show_plots,
        title=f"Katabatic fraction - {period} (JJA)",
        coasts="white",
        launch_params={"dt_threshold" : dT_threshold, "window_size" : window_size, "wind_threshold" : wind_threshold, "direction_threshold" : direction_threshold}
    )
    
    plot_iceland(
        label_map,
        f"cluster_{label_map.max()}",
        typ="mask",
        ref=ds_orog_static,
        unit={"cluster":"n"},
        cmap="tab20",
        glaciers=np.load(resolve_glacier_mask_path()),
        save_path=(map_root / "katabatic_clusters.png") if save_figures else None,
        show=show_plots,
        title=f"Katabatic clusters - {period} (JJA)",
        launch_params={"dt_threshold" : dT_threshold, "window_size" : window_size, "wind_threshold" : wind_threshold, "direction_threshold" : direction_threshold}
    )

    if save_animations and len(processed_years) == len(freshly_processed_years):
        animate_iceland(
            aggregated_masks,
            ref=ds_orog_static,
            save_path=agg_anim_dir / "katabatic_mask_animation.gif",
            show=show_plots,
        )
    elif save_animations:
        print(
            "[yearly] Warning: aggregated animation skipped because some years were reused from cache "
            "(cached masks are not persisted)."
        )

    yearly_manifest["years_included_for_aggregation"] = [str(y) for y in processed_years]
    yearly_manifest["updated_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    write_manifest(yearly_manifest_path, yearly_manifest)

    if fraction_evolution_mode in {"basin", "pixel"}:
        basin_arg = fraction_evolution_basin if fraction_evolution_basin is not None else ("all" if fraction_evolution_mode == "basin" else None)
        _export_aggregated_fraction_evolution(
            output_dir=graph_root,
            years=processed_years,
            param_tag=param_tag,
            label_map=label_map,
            reference_map=aggregated_percent,
            mode=fraction_evolution_mode,
            basin_label=basin_arg,
            pixel=fraction_evolution_pixel,
            allow_click=bool(fraction_evolution_click),
            title_prefix=f"{period} — ",
        )

    if skipped_years:
        print("=" * 80)
        print("[yearly] WARNING: some years were excluded from this run")
        for year, reason in skipped_years:
            print(f"[yearly] - {year}: {reason}")
        print("=" * 80)


def process_jja_concat(path,
                      output_ref_year="2022",
                      orog_month="June",
                      orog_file_index=2,
                      expected_min_years=5,
                      window_size= 15,
                      dT_threshold=  1.5,
                      wind_threshold= 2.5,
                      direction_threshold= 0.87,
                      output_dir=None,
                      param_tag=None,
                      save_figures=False,
                      save_animations=False,
                      show_plots=True,
                      icegaws_points=None,
                      icegaws_points_annotate=False,
                      fraction_evolution_mode=None,
                      fraction_evolution_basin=None,
                      fraction_evolution_pixel=None,
                      fraction_evolution_click=False):
    
    """Process JJA by loading all years at once into a single concatenated dataset."""

    ds_t2m, ds_uv = reader_launch(
        "JJA",
        "multi",
        path=path,
        mode="concat",
        load_orography=False,
        expected_min_years=expected_min_years,
    )
    ds_orog = load_static_orography(
        path,
        year=output_ref_year,
        month=orog_month,
        file_index=orog_file_index,
    )

    period = f"{str(ds_t2m.time.values[0])[:4]} - {str(ds_t2m.time.values[-1])[:4]}"

    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_FIGURES_DIR / "concat"

    if param_tag is None:
        param_tag = build_param_tag(window_size, dT_threshold, wind_threshold, direction_threshold)
    output_dir = output_dir / param_tag / period
    plot_dir = output_dir / "maps"
    anim_dir = output_dir / "animations"

    if save_figures:
        plot_dir.mkdir(parents=True, exist_ok=True)
    if save_animations:
        anim_dir.mkdir(parents=True, exist_ok=True)

    NPY_DIR = DEFAULT_NPY_DATA_DIR / Path("concat") / param_tag /period
    NPY_DIR.mkdir(parents=True, exist_ok=True)


    times, masks, per_kat = detect_catabatic_local(
        ds_t2m,
        ds_orog,
        ds_uv,
        window_size=window_size,
        dT_threshold=dT_threshold,
        wind_threshold=wind_threshold,
        direction_threshold=direction_threshold,
    )

    z = ds_orog["orog"].isel(time=0).values
    coords, labels, label_map = detect_catabatic_zones(per_kat, z, ds_t2m)


    np.save(NPY_DIR / f"per_kat_{param_tag}.npy", per_kat)
    np.save(NPY_DIR / f"label_map_{param_tag}.npy", label_map)

    
    plot_iceland(
        label_map,
        f"cluster_{label_map.max()}",
        typ="mask",
        ref=ds_t2m,
        unit={"cluster": "n"},
        cmap="tab20",
        glaciers=np.load(resolve_glacier_mask_path()),
        save_path=(plot_dir / "katabatic_clusters.png") if save_figures else None,
        show=show_plots,
        title=f"Katabatic clusters - {len(times)} timestamps (JJA)",
        launch_params={"dt_threshold" : dT_threshold, "window_size" : window_size, "wind_threshold" : wind_threshold, "direction_threshold" : direction_threshold}
    )

    plot_iceland(
        per_kat,
        "per_kat",
        typ="mask",
        ref=ds_t2m,
        cmap="plasma",
        glaciers=np.load(resolve_glacier_mask_path()),
        points=icegaws_points,
        annotate_points=icegaws_points_annotate,
        save_path=(plot_dir / "katabatic_fraction.png") if save_figures else None,
        show=show_plots,
        title=f"Katabatic fraction - {len(times)} timestamps (JJA)",
        coasts="white",
        launch_params={"dt_threshold" : dT_threshold, "window_size" : window_size, "wind_threshold" : wind_threshold, "direction_threshold" : direction_threshold}
    )

    if save_animations:
        animate_iceland(
            masks,
            ds_t2m,
            save_path=anim_dir / "katabatic_mask_animation.mp4",
            show=show_plots,
        )

    if fraction_evolution_mode in {"basin", "pixel"}:
        graph_root = output_dir / "graphs"
        basin_arg = fraction_evolution_basin if fraction_evolution_basin is not None else ("all" if fraction_evolution_mode == "basin" else None)
        _export_aggregated_fraction_evolution(
            output_dir=graph_root,
            years=[str(ds_t2m.time.values[0])[:4], str(ds_t2m.time.values[-1])[:4]],
            param_tag=param_tag,
            label_map=label_map,
            reference_map=per_kat,
            mode=fraction_evolution_mode,
            basin_label=basin_arg,
            pixel=fraction_evolution_pixel,
            allow_click=bool(fraction_evolution_click),
            title_prefix=f"{period} — ",
        )
