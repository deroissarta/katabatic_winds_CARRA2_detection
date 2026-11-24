"""Plot katabatic fields and animations on Iceland-focused Cartopy maps."""

import os
import matplotlib

# Use a non-interactive backend only when explicitly requested.
if os.environ.get("KATABATIC_FORCE_AGG", "0") == "1":
    matplotlib.use("Agg")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from shared.reader import open_t2m_netcdf, crop_iceland, open_orography_netcdf
from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.offsetbox import AnchoredText
from shared.paths import DEFAULT_KATABATIC_DATA_DIR

def get_ref():
    """Load a reference dataset for Iceland-focused plotting."""
    folder = DEFAULT_KATABATIC_DATA_DIR / "2025" / "CARRA2_June"
    if not folder.exists():
        raise FileNotFoundError(f"Reference file not found: {folder}")
    ds = open_t2m_netcdf(folder)
    try:
        ds = crop_iceland(ds, "t2m")
        # Keep only coordinates and materialize them before closing NetCDF.
        # This avoids native backend issues when a lazily-open dataset leaks further.
        keep = [name for name in ["longitude", "latitude", "time"] if name in ds]
        if not keep:
            raise KeyError("Reference dataset missing longitude/latitude coordinates")
        ref = ds[keep].load()
        return ref
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _format_time_label(value):
    try:
        if hasattr(value, "values"):
            value = value.values
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            value = value[0]
        # Keep datetime64 intact before calling item(), otherwise it can become a raw int.
        if isinstance(value, np.datetime64):
            return np.datetime_as_string(value, unit="D")
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (int, np.integer)) and value > 10**14:
            return np.datetime_as_string(np.datetime64(int(value), "ns"), unit="D")
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d")
        return str(value)
    except Exception:
        return str(value)


def _get_iceland_extent(ref, default_extent=(-24.5, -13.5, 62.8, 66.8)):
    if ref is None:
        return list(default_extent)

    
    lons = np.asarray(ref["longitude"].values)
    lats = np.asarray(ref["latitude"].values)
    lons = np.where(lons > 180, lons - 360, lons)

    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))
    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))

    # Use a very small outer padding to preserve island proportions.
    lon_pad = 0.20
    lat_pad = 0.15
    lon_min = max(default_extent[0], lon_min - lon_pad)
    lon_max = min(default_extent[1], lon_max + lon_pad)
    lat_min = max(default_extent[2], lat_min - lat_pad)
    lat_max = min(default_extent[3], lat_max + lat_pad)

    if lon_min >= lon_max or lat_min >= lat_max:
        return list(default_extent)

    return [lon_min, lon_max, lat_min, lat_max]


def _prepare_lon_lat(ref):
    """Return plotting-ready lon/lat arrays in the [-180, 180] longitude convention."""
    lons = np.asarray(ref["longitude"].values)
    lats = np.asarray(ref["latitude"].values)
    lons = np.where(lons > 180, lons - 360, lons)

    if lons.ndim == 1 and lats.ndim == 1:
        return np.meshgrid(lons, lats)
    return lons, lats


def _get_cmap_and_norm(param_data, param, typ, cmap):
    arr = np.asarray(param_data)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return cmap, None

    unique_vals = np.unique(finite)
    # Auto-switch to a binary palette when the input is a 0/1 mask.
    is_binary = unique_vals.size <= 2 and np.all(np.isin(unique_vals, [0, 1]))
    if is_binary:
        cmap_obj = ListedColormap(["#e2e8f0", "#1d4ed8"])
        return cmap_obj, BoundaryNorm([-0.5, 0.5, 1.5], cmap_obj.N)

    tab20 = plt.cm.tab20.colors
    tab10 = plt.cm.tab10.colors

    if "cluster" in param:
        n = int(param.split("_")[-1])
        if n > 30 :
            n = 30
            print (f"Warning: Too many clusters ({n}), using a maximum of 30 colors.")
        palette = [(1.0, 1.0, 1.0)] + list(tab20) + list(tab10)
        palette = palette[:n]
        return ListedColormap(palette), None

    if typ == "mask" and param == "local_mean" and cmap == "viridis":
        return "YlGnBu", None

    return cmap, None

def plot_iceland(
    ds,
    param,
    typ=None,
    ref=None,
    unit={"t2m": "K", "orog": "m", "v10": "m s^-1", "u10": "m s^-1", "per_kat": "%"},
    cmap="viridis",
    norm=None,
    vmin=None,
    vmax=None,
    glaciers=None,
    points=None,
    points_cmap=None,
    points_norm=None,
    points_size=80,
    points_edgecolor="black",
    points_alpha=0.95,
    annotate_points=False,
    save_path=None,
    show=True,
    period=None,
    title=None,
    coasts="black",
    overlays=None,
    launch_params=None,
    keep_open=False,
):
    """Plot a katabatic-related field on an Iceland-focused map.

    Parameters
    ----------
    ds : xarray.Dataset or array-like
        Data to plot.
    param : str
        Variable name used for labels and defaults.
    typ : str, optional
        Use 'mask' for 2D arrays, 'arr' for raw arrays, or leave empty for dataset variables.
    ref : xarray.Dataset, optional
        Reference dataset providing coordinates and time labels.
    unit : dict, optional
        Mapping from variable name to unit label.
    cmap : str or Colormap, optional
        Base colormap used for the plot.
        if show:
            plt.show()
        else:
            plt.close(fig)
        Glacier outline mask.
    points : iterable of dict, optional
        Optional scatter overlay points with keys such as lon, lat, value and label.
    points_cmap : str or Colormap, optional
        Colormap used for the point overlay. Defaults to the main map colormap.
    points_norm : Normalize, optional
        Normalization used for the point overlay. Defaults to the main map normalization.
    annotate_points : bool, optional
        If True, annotate each point with its label and value.
    save_path : str or Path, optional
        File path used to save the figure.
    show : bool, optional
        If True, display the figure interactively.
    period : str, optional
        Custom period label shown in the title.
    title : str, optional
        Custom title override.
    """
    if ref is None:
        ref = get_ref()
    
    # Support 3 input styles: raw array, mask array, or full dataset variable.
    if typ == "arr":
        if period is not None:
            time_label = period
        param_data = ds[0]
    elif typ == "mask":
        if period is not None:
            time_label = period
        param_data = ds
    else:
        t = ds.time[0]
        time_label = _format_time_label(t)
        param_data = ds[param].sel(time=t)

    if type(unit) is dict:
        unit_label = unit[param] if param in unit else ""
    elif type(unit) is str:
        unit_label = unit
    else :
        unit_label = ""

    

    if period is not None or (typ != "mask" and typ != "arr"):
        title_text = f"{param} - {time_label}"
    else : 
        title_text = title or f"{param}"
    
    # Keep Iceland framing consistent and aligned with the actual input grid.
    extent = _get_iceland_extent(ref)

    plot_data = np.asarray(param_data)
    if isinstance(param, str) and ("basin" in param.lower() or param.lower() == "cluster"):
        # Hide non-basin cells to keep focus on detected basins and avoid full-domain shading.
        plot_data = plot_data.astype(float)
        plot_data[plot_data <= 0] = np.nan

    cmap_choice, norm_choice = _get_cmap_and_norm(plot_data, param, typ, cmap)
    final_norm = norm if norm is not None else norm_choice
    X, Y = _prepare_lon_lat(ref)

    fig = plt.figure(figsize=(10, 10), facecolor="white")

    proj = ccrs.LambertConformal(central_longitude=-19.0, central_latitude=65.0)
    ax = plt.axes(projection=proj)
    ax.set_facecolor("#f7f9fc")
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf3ff", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor = coasts, zorder=4)
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Keep map geometry stable to avoid visual flattening/stretching.
    ax.set_aspect("equal")
    
    im = ax.pcolormesh(
        X,
        Y,
        plot_data,
        cmap=cmap_choice,
        norm=final_norm,
        vmin=vmin if final_norm is None else None,
        vmax=vmax if final_norm is None else None,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    
    ax.set_title(title_text, fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    if glaciers is not None:
            gl = ax.contour(
                ref["longitude"],
                ref["latitude"],
                glaciers,
                levels=[0.5],
                colors="#003366",
                linewidths=1.2,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )
            if hasattr(gl, "collections") and len(gl.collections) > 0:
                gl.collections[0].set_label("Glacier outline")

    if overlays:
        legend_handles = []
        seen_labels = set()
        for overlay in overlays:
            if not isinstance(overlay, dict):
                continue
            mask = overlay.get("mask")
            if mask is None:
                continue
            mask = np.asarray(mask)
            if mask.shape != plot_data.shape:
                continue
            ax.contour(
                ref["longitude"],
                ref["latitude"],
                mask.astype(float),
                levels=[0.5],
                colors=overlay.get("color", "black"),
                linewidths=overlay.get("linewidth", 1.0),
                linestyles=overlay.get("linestyle", "-"),
                transform=ccrs.PlateCarree(),
                zorder=overlay.get("zorder", 6),
            )
            label = overlay.get("label")
            if label and label not in seen_labels:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=overlay.get("color", "black"),
                        linestyle=overlay.get("linestyle", "-"),
                        linewidth=overlay.get("linewidth", 1.0),
                        label=label,
                    )
                )
                seen_labels.add(label)

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="lower right",
                fontsize=9,
                frameon=True,
                title="Uncertainty overlays",
                title_fontsize=9,
            )

    if points is not None:
        point_lons = []
        point_lats = []
        point_values = []
        point_labels = []
        for point in points:
            if point is None:
                continue
            lon = point.get("lon", point.get("longitude"))
            lat = point.get("lat", point.get("latitude"))
            if lon is None or lat is None:
                continue
            point_lons.append(float(lon))
            point_lats.append(float(lat))
            point_values.append(point.get("value"))
            point_labels.append(point.get("label", point.get("station", "")))

        if point_lons:
            overlay_cmap = points_cmap if points_cmap is not None else cmap_choice
            overlay_norm = points_norm if points_norm is not None else norm_choice
            values_array = np.asarray(point_values, dtype=float)
            if np.any(np.isfinite(values_array)):
                scatter = ax.scatter(
                    point_lons,
                    point_lats,
                    c=values_array,
                    cmap=overlay_cmap,
                    norm=overlay_norm,
                    s=points_size,
                    edgecolors=points_edgecolor,
                    linewidths=0.8,
                    alpha=points_alpha,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )
            else:
                scatter = ax.scatter(
                    point_lons,
                    point_lats,
                    color=points_cmap if isinstance(points_cmap, str) else "#ef4444",
                    s=points_size,
                    edgecolors=points_edgecolor,
                    linewidths=0.8,
                    alpha=points_alpha,
                    transform=ccrs.PlateCarree(),
                    zorder=5,
                )

            if annotate_points:
                offset_pattern = [(4, 4), (4, -10), (-28, 4), (-28, -10), (10, 12), (-34, 12)]
                for idx, (lon, lat, label, value) in enumerate(zip(point_lons, point_lats, point_labels, point_values)):
                    text = str(label) if label else ""
                    if value is not None and np.isfinite(value):
                        text = f"{text}\n{float(value):.2f}" if text else f"{float(value):.2f}"
                    if text:
                        xoff, yoff = offset_pattern[idx % len(offset_pattern)]
                        ax.annotate(
                            text,
                            xy=(lon, lat),
                            xytext=(xoff, yoff),
                            textcoords="offset points",
                            fontsize=7,
                            color="black",
                            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7},
                            zorder=6,
                            transform=ccrs.PlateCarree(),
                        )
        

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_label(f"{param} ({unit_label})", fontsize=11, weight="bold")
    cbar.ax.tick_params(labelsize=10)

    if launch_params is not None:
        params_txt = (
            f"Launch parameters:\n"
            f"dt={launch_params.get('dt_threshold', 'N/A')}\n"
            f"ws={launch_params.get('window_size', 'N/A')}\n"
            f"wind={launch_params.get('wind_threshold', 'N/A')}\n"
            f"dir={launch_params.get('direction_threshold', 'N/A')}"
        )

        txt = AnchoredText(
            params_txt,
            loc="lower left",
            prop={"size": 9, "color": "black"},
            frameon=True
        )

        ax.add_artist(txt)

    plt.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.08)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved figure: {save_path}")
        if not show:
            print(f"No-show active: plot saved to {save_path}")

    if show:
        plt.show()
    elif keep_open:
        return
    else:
        plt.close(fig)

# plot_iceland(ds_2m_iceland, "t2m")
# plot_iceland(ds_orog_iceland, "orog")

def animate_iceland(masks, ref=None, save_path=None, show=True, period=None):
    """
    Animate the evolution of a binary katabatic mask over time.

    Parameters
    ----------
    masks : dict
        Mapping from timestamp to a 2D boolean mask.
    ref : xarray.Dataset, optional
        Reference dataset used for coordinates and map extent.
    save_path : str or Path, optional
        File path used to save the animation.
    show : bool, optional
        If True, display the animation interactively.
    period : str, optional
        Custom period label shown in the title.
    """

    timestamps = sorted(masks.keys())
    frames = [np.asarray(masks[t], dtype=float) for t in timestamps]
    # Cap frame count so long time series remain lightweight to render/save.
    if len(frames) > 24:
        step = max(1, len(frames) // 24)
        timestamps = timestamps[::step]
        frames = [frames[i] for i in range(0, len(frames), step)]

    extent = _get_iceland_extent(ref)
    cmap_choice, norm_choice = _get_cmap_and_norm(frames[0], "mask", "mask", "Blues")
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    X, Y = _prepare_lon_lat(ref)

    if ccrs is not None and cfeature is not None:
        try:
            proj = ccrs.LambertConformal(central_longitude=-19.0, central_latitude=65.0)
            ax = plt.axes(projection=proj)
            ax.set_facecolor("#f7f9fc")
            ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none", zorder=0)
            ax.add_feature(cfeature.OCEAN, facecolor="#eaf3ff", zorder=0)
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=1.1, edgecolor="white", zorder=3)
            ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor="black", zorder=4)
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.set_aspect("equal")
            # gl = ax.gridlines(
            #     draw_labels=True,
            #     linewidth=0.4,
            #     color="gray",
            #     alpha=0.35,
            #     linestyle="--",
            #     crs=ccrs.PlateCarree(),
            # )
            # gl.top_labels = False
            # gl.right_labels = False
            # gl.xlabel_style = {"size": 9}
            # gl.ylabel_style = {"size": 9}
            im = ax.pcolormesh(
                X,
                Y,
                frames[0],
                cmap=cmap_choice,
                norm=norm_choice,
                shading="auto",
                transform=ccrs.PlateCarree(),
                zorder=2,
            )
        except Exception:
            ax = plt.axes()
            ax.set_facecolor("#f7f9fc")
            im = ax.pcolormesh(X, Y, frames[0], cmap=cmap_choice, norm=norm_choice, shading="auto")
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
    else:
        ax = plt.axes()
        ax.set_facecolor("#f7f9fc")
        im = ax.pcolormesh(X, Y, frames[0], cmap=cmap_choice, norm=norm_choice, shading="auto")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    ax.set_title(f"Katabatic mask - {timestamps[0]}", fontsize=15, weight="bold", pad=12)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["No katabatic signal", "Katabatic signal"])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Event presence", fontsize=11, weight="bold")

    def update(i):
        # pcolormesh stores flattened arrays internally.
        im.set_array(frames[i].ravel())
        ax.set_title(f"Katabatic mask - {timestamps[i]}", fontsize=15, weight="bold", pad=12)
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=150)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.suffix.lower() != ".gif":
            save_path = save_path.with_suffix(".gif")
        writer = PillowWriter(fps=2)
        anim.save(save_path, writer=writer, dpi=120, savefig_kwargs={"facecolor": fig.get_facecolor()})
        print(f"Saved animation: {save_path}")
        if not show:
            print(f"No-show active: animation saved to {save_path}")

    plt.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.08)
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return anim