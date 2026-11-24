"""Small plotting helpers for NetCDF datasets and map panels."""

import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_grib_on_cartopy(
    da: xr.DataArray,
    title: str = "",
    cmap: str = "viridis",
):
    """
    Display a lat/lon DataArray on a Cartopy map.
    """
    crs_data = da.attrs.get("cartopy_crs", ccrs.PlateCarree())

    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution="110m")

    # Use the first time step when a temporal axis is present.
    if "time" in da.dims:
        field = da.isel(time=0)
    else:
        field = da

    im = field.plot(
        ax=ax,
        transform=crs_data,
        cmap=cmap,
        cbar_kwargs={"shrink": 0.7}
    )

    ax.set_title(title or da.name or "GRIB field")
    plt.show()
