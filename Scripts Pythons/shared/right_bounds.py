"""Inspect and visualize spatial crop bounds for CARRA2 data."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import Rectangle

from shared.paths import DEFAULT_KATABATIC_DATA_DIR
from shared.reader import open_t2m_netcdf


# Iceland ROI corners (lon, lat) in clockwise order.
RED_SQUARE_CORNERS = {
    "SW": (-25.9535, 62.5445),
    "SE": (-13.4169, 61.4693),
    "NE": (-9.1872, 66.9147),
    "NW": (-24.8401, 68.2985),
}


def _normalize_lon(lon):
    return np.where(lon > 180.0, lon - 360.0, lon)


def _get_first_2d_t2m(ds):
    arr = ds["t2m"]
    if "time" in arr.dims:
        arr = arr.isel(time=0)
    values = np.asarray(arr.values)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D t2m field after time selection, got shape {values.shape}")
    return values


def _roi_polygon():
    return np.array(
        [
            RED_SQUARE_CORNERS["SW"],
            RED_SQUARE_CORNERS["SE"],
            RED_SQUARE_CORNERS["NE"],
            RED_SQUARE_CORNERS["NW"],
        ],
        dtype=float,
    )


def _point_in_polygon_mask(lon, lat, polygon):
    points = np.column_stack([lon.ravel(), lat.ravel()])
    mask = Path(polygon).contains_points(points)
    return mask.reshape(lon.shape)


def _bbox_from_polygon(polygon, margin_lon=1.5, margin_lat=1.0):
    west = float(np.min(polygon[:, 0]) - margin_lon)
    east = float(np.max(polygon[:, 0]) + margin_lon)
    south = float(np.min(polygon[:, 1]) - margin_lat)
    north = float(np.max(polygon[:, 1]) + margin_lat)
    return west, east, south, north


def visualize_nan_island_crop(ds, margin_lon=1.5, margin_lat=1.0):
    arr2d = _get_first_2d_t2m(ds)
    nans_mask = np.isnan(arr2d)

    lon = _normalize_lon(np.asarray(ds["longitude"].values))
    lat = np.asarray(ds["latitude"].values)
    if lon.ndim != 2 or lat.ndim != 2:
        raise ValueError("This visualizer expects 2D longitude/latitude coordinates.")

    polygon = _roi_polygon()
    roi_mask = _point_in_polygon_mask(lon, lat, polygon)
    if not np.any(roi_mask):
        raise ValueError("No grid point intersects Iceland ROI polygon in this dataset.")

    poly_lon = np.r_[polygon[:, 0], polygon[0, 0]]
    poly_lat = np.r_[polygon[:, 1], polygon[0, 1]]

    west, east, south, north = _bbox_from_polygon(polygon, margin_lon=0.0, margin_lat=0.0)
    west_m, east_m, south_m, north_m = _bbox_from_polygon(
        polygon,
        margin_lon=margin_lon,
        margin_lat=margin_lat,
    )

    crop_mask = (
        (lon >= west_m)
        & (lon <= east_m)
        & (lat >= south_m)
        & (lat <= north_m)
    )

    crop_rect_index = None
    if np.any(crop_mask):
        ys = np.where(crop_mask.any(axis=1))[0]
        xs = np.where(crop_mask.any(axis=0))[0]
        crop_rect_index = (
            int(xs.min()),
            int(xs.max()) + 1,
            int(ys.min()),
            int(ys.max()) + 1,
        )

    nans_island = int(np.count_nonzero(nans_mask & roi_mask))
    nans_crop = int(np.count_nonzero(nans_mask))

    print("[viz] Iceland zone from lon/lat polygon corners (SW, SE, NE, NW).")
    print(f"[viz] Iceland polygon bbox lon/lat: W={west:.4f}, E={east:.4f}, S={south:.4f}, N={north:.4f}")
    print(f"[viz] Crop bounds lon/lat with margins: W={west_m:.4f}, E={east_m:.4f}, S={south_m:.4f}, N={north_m:.4f}")
    print(f"[viz] NaN count in Iceland polygon zone: {nans_island}")
    print(f"[viz] NaN count in crop frame: {nans_crop}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Index-space diagnostic view.
    ax0 = axes[0]
    ax0.imshow(nans_mask, origin="lower", cmap="gray_r")
    ax0.contour(roi_mask.astype(int), levels=[0.5], colors=["red"], linewidths=2)

    if crop_rect_index is not None:
        cx0, cx1, cy0, cy1 = crop_rect_index
        ax0.add_patch(
            Rectangle(
                (cx0, cy0),
                cx1 - cx0,
                cy1 - cy0,
                fill=False,
                edgecolor="gold",
                linewidth=2,
                linestyle="--",
                label="Crop frame (bbox+margin)",
            )
        )

    ax0.plot([], [], color="red", linewidth=2, label="Iceland polygon zone")
    ax0.set_title("Index view: NaN mask + Iceland polygon + crop frame")
    ax0.set_xlabel("x index")
    ax0.set_ylabel("y index")
    ax0.legend(loc="upper left")

    # Lon/lat diagnostic view.
    ax1 = axes[1]
    ax1.pcolormesh(lon, lat, nans_mask.astype(int), shading="auto")
    ax1.plot(poly_lon, poly_lat, color="red", linewidth=2, label="Iceland polygon zone")

    crop_lon = [west_m, east_m, east_m, west_m, west_m]
    crop_lat = [south_m, south_m, north_m, north_m, south_m]
    ax1.plot(crop_lon, crop_lat, color="gold", linestyle="--", linewidth=2, label="Crop frame (bbox+margin)")

    ax1.set_title("Lon/lat view: NaN mask + zones")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.legend(loc="best")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NaN zone, Iceland polygon zone, and crop frame for CARRA t2m files."
    )
    parser.add_argument("--year", default="2025", help="Year folder to inspect.")
    parser.add_argument("--month", default="June", help="Month folder to inspect.")
    parser.add_argument("--margin-lon", type=float, default=1.5, help="Longitude margin for crop frame.")
    parser.add_argument("--margin-lat", type=float, default=1.0, help="Latitude margin for crop frame.")
    args = parser.parse_args()

    folder = DEFAULT_KATABATIC_DATA_DIR / str(args.year) / f"CARRA2_{args.month}"
    ds = open_t2m_netcdf(folder=folder)
    visualize_nan_island_crop(
        ds,
        margin_lon=args.margin_lon,
        margin_lat=args.margin_lat,
    )


if __name__ == "__main__":
    main()
