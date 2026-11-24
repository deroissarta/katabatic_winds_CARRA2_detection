"""Generate glacier labels, downslope paths, and katabatic basin arrays."""

import numpy as np
import xarray as xr
from shared.reader import crop_iceland, open_orography_netcdf, load_static_orography, load_land_mask
from shared.map_launcher import plot_iceland, animate_iceland
from scipy.ndimage import label
import matplotlib.pyplot as plt
from pathlib import Path
from shared.paths import DEFAULT_NPY_DATA_DIR, resolve_glacier_mask_path, DEFAULT_FIGURES_DIR, DEFAULT_KATABATIC_DATA_DIR

def enlarge_glacier(glacier_mask, iterations=5):
    """
    Enlarge the glacier mask by a specified number of iterations.
    This ensures that:
    - each glacier group keeps its identity,
    - enlargement does not merge unrelated glaciers,
    - drainage basins can still be traced back to their source glacier.
    Enlarging the glacier mask allows cold-air sources to include
    adjacent slopes, valley heads, and low-gradient terrain, producing
    wider and more realistic drainage corridors.
    Dilate each glacier label independently.
    """

    from scipy.ndimage import binary_dilation

    enlarged = np.zeros_like(glacier_mask)

    for gid in np.unique(glacier_mask):
        if gid == 0:
            continue

        # mask for this glacier only
        mask = (glacier_mask == gid)

        # dilate this glacier independently
        dilated = binary_dilation(mask, iterations=iterations)

        # assign the glacier ID to the dilated region
        enlarged[dilated] = gid

    return enlarged

def get_orography_data():

    ds_orog = load_static_orography(DEFAULT_KATABATIC_DATA_DIR)
        
    ref = ds_orog.isel(time=[0]).copy()
    z = ref["orog"].squeeze("time").values
    glacier_mask = np.load(resolve_glacier_mask_path())

    if not (DEFAULT_FIGURES_DIR/"basins"/"orography.png").exists():
        plot_iceland(z, "orography", typ="mask", ref=ref, unit={"orog": "m"}, cmap="terrain", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"orography.png")
    
    return z, ref, glacier_mask

def steepest_descent_basin_tracing(z, glacier_labels, flow_dir, basins, sea_level_limit):
    """
    Trace downslope drainage paths for each glacier and assign basin labels. 
    This function follows the steepest descent from each glacier pixel 
    until it reaches a stopping condition, such as sea level or a flat area.

    Parameters
    ----------
    z : 2D array
        Digital Elevation Model (DEM)
    glacier_labels : 2D int array
        Labeled glacier regions (unique labels)
    flow_dir : 3D int array
        Flow direction vectors for each grid cell
    basins : 2D int array
        Array to store basin labels
    current_label : int
        Starting label for basins
    sea_level_limit : float
        Minimum elevation at which drainage stops.

    Returns
    -------
    basins : 2D int array
        Updated basins with assigned labels.
    """

    basins = np.zeros_like(z, dtype=int)
    current_label = 1

    ny, nx = z.shape
    n_glaciers = glacier_labels.max()

    for g in range(1, n_glaciers + 1):
        mask_g = (glacier_labels == g)
        basin_mask = np.zeros_like(z, dtype=bool)

        ys, xs = np.where(mask_g)

        for y0, x0 in zip(ys, xs):
            y, x = y0, x0

            # Follow downslope path until:
            # - flow direction is flat
            # - elevation reaches sea_level_limit
            # - boundary is reached
            while True:
                basin_mask[y, x] = True

                # Stop if elevation is too low (user-defined limit)
                if z[y, x] <= sea_level_limit:
                    break

                dy, dx = flow_dir[y, x]
                yy, xx = y + dy, x + dx

                # Stop if no downslope direction
                if dy == 0 and dx == 0:
                    break

                # Stop if outside grid
                if not (0 <= yy < ny and 0 <= xx < nx):
                    break

                # Stop if not descending
                if z[yy, xx] >= z[y, x]:
                    break

                y, x = yy, xx

        # Assign basin label to the traced path
        basins[basin_mask] = current_label
        current_label += 1

    return basins

def next_pixel(y, x, flow_dir, z):
    dy, dx = flow_dir[y, x]
    if dy == 0 and dx == 0:
        return None

    yy, xx = y + dy, x + dx
    if not (0 <= yy < z.shape[0] and 0 <= xx < z.shape[1]):
        return None

    if z[yy, xx] >= z[y, x]:
        return None

    return yy, xx


def trace_steepest_descent(z, glacier_labels, flow_dir, land_mask, max_steps=400, glacier_mask=None):
    """
    Trace steepest descent paths from glacier pixels to form drainage basins.
    - Uses flow_dir as primary direction

    Returns a labeled array with the glaciers labeled with their corresponding 
    drainage steepest paths.
    """

    glaciers_steepest_paths = np.zeros_like(z, dtype=int)
    current_label = 1
    ny, nx = z.shape
    water = ~land_mask

    for gid in range(1, glacier_labels.max() + 1):
        ys, xs = np.where(glacier_labels == gid)
        glacier_steepest_way_mask = np.zeros_like(z, dtype=bool)

        for y0, x0 in zip(ys, xs):
            y, x = y0, x0
            steps = 0

            while True:
                glacier_steepest_way_mask[y, x] = True
                steps += 1

                if steps >= max_steps:
                    break
                if water[y, x]:
                    break

                nxt = next_pixel(y, x, flow_dir, z)
                if nxt is None:
                    break

                y, x = nxt

        glaciers_steepest_paths[glacier_steepest_way_mask] = current_label
        current_label += 1

        plot_iceland(glacier_steepest_way_mask, f"glacier_{gid}_steepest_path", typ="mask", ref=None, unit={"glacier": "n°"}, cmap="Reds", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/f"glacier_{gid}_steepest_path.png")
        plt.close()

    glaciers_steepest_paths [glacier_mask] = 0
    return glaciers_steepest_paths

def basins_from_talweg_labels(z, talweg_labels, flow_dir, land_mask, glacier_labels,
                              max_steps=400):
    """
    Build katabatic basins starting ONLY from talweg labels.

    Assumptions:
    - talweg_labels contains glacier IDs ONLY on talweg pixels.
    - next_pixel() follows the flow_dir downslope.
    - Glaciers have already been removed from talweg_labels.

    Logic:
    For each terrain pixel:
        - follow next_pixel downslope
        - if the path reaches a talweg pixel:
              assign the pixel to the talweg's glacier ID
        - otherwise leave it unassigned

    No widening, no dilation, no slope filtering.
    Pure drainage logic.
    """

    ny, nx = z.shape
    basins = np.zeros_like(z, dtype=int)

    # For every pixel in the domain
    for y in range(ny):
        for x in range(nx):

            if not land_mask[y, x]:
                continue

            yy, xx = y, x
            steps = 0

            # Follow flow_dir until we hit a talweg or stop
            while True:

                # If current pixel is a talweg → assign basin
                gid = talweg_labels[yy, xx]
                if gid != 0:
                    alt_max_glacier = z[glacier_labels == gid].max()
                    if z[y, x] <= alt_max_glacier:
                        basins[y, x] = gid
                        break

                steps += 1
                if steps > max_steps:
                    break

                nxt = next_pixel(yy, xx, flow_dir, z)
                if nxt is None:
                    break

                yy, xx = nxt

    return basins




















def get_glacier_labels(z, ref, glacier_mask):
    """
    Identify and label glacier regions, then group them into larger source regions
    based on geographic boxes. This allows for the tracing of cold-air drainage basins from these grouped glacier sources.
    """
    if not (DEFAULT_NPY_DATA_DIR/"glacier_labels.npy").exists():

        # Label connected glacier regions; each region acts as a cold-air source.
        glacier_labels, n_glaciers = label(glacier_mask)

        # Define the geographic boxes used to group source glaciers.
        # Box 1: northern glaciers.
        north_box_y1 = 200
        north_box_y2 = 260
        north_box_x2  = 180
        north_box_x1 = 130

        north_box = np.zeros(glacier_labels.shape, dtype=bool)
        north_box[north_box_y1:north_box_y2, north_box_x1:north_box_x2] = True

        plot_iceland(north_box, "north box", typ="mask", ref=ref, cmap="Reds", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"north_box.png")

        # Box 2: eastern glaciers.
        east_box_y1 = 190
        east_box_y2 = 270
        east_box_x2  = 250
        east_box_x1 = 210
        
        east_box = np.zeros(glacier_labels.shape, dtype=bool)
        east_box[east_box_y1:east_box_y2, east_box_x1:east_box_x2] = True
        
        plot_iceland(east_box, "east box", typ="mask", ref=ref, cmap="Reds", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"east_box.png")

        # Box 3: western glaciers, sector 1.
        nw_y1 = 220
        nw_y2 = 240
        nw_x2  = 85
        nw_x1 = 60
        
        nw1_box = np.zeros(glacier_labels.shape, dtype=bool)
        nw1_box[nw_y1:nw_y2, nw_x1:nw_x2] = True
        
        plot_iceland(nw1_box, "nw1 box", typ="mask", ref=ref, cmap="Reds", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"nw1_box.png")

        # Box 4: western glaciers, sector 2.
        nw_y1 = 210
        nw_y2 = 220
        nw_x2  = 63
        nw_x1 = 55
        
        nw2_box = np.zeros(glacier_labels.shape, dtype=bool)
        nw2_box[nw_y1:nw_y2, nw_x1:nw_x2] = True
        
        plot_iceland(nw2_box, "nw2 box", typ="mask", ref=ref, cmap="Reds", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"nw2_box.png")

        def glaciers_in_box(box_mask, glacier_labels):
            gids = []
            for gid in range(1, glacier_labels.max() + 1):
                glacier_mask = (glacier_labels == gid)
                # Keep only glaciers whose full footprint lies inside the box.
                if np.all(box_mask[glacier_mask]):
                    gids.append(gid)
            return gids

        north_ids = glaciers_in_box(north_box, glacier_labels)
        east_ids  = glaciers_in_box(east_box, glacier_labels)
        nw1_ids   = glaciers_in_box(nw1_box, glacier_labels)
        nw2_ids   = glaciers_in_box(nw2_box, glacier_labels)

        glacier_labels_grouped = glacier_labels.copy()

        # Merge glacier labels inside each selected geographic box.
        def merge_ids(id_list):
            if len(id_list) <= 1:
                return
            main = id_list[0]
            for gid in id_list[1:]:
                glacier_labels_grouped[glacier_labels == gid] = main

        merge_ids(north_ids)
        merge_ids(east_ids)
        merge_ids(nw1_ids)
        merge_ids(nw2_ids)
        
        # Build the final mask of the grouped source regions.
        glacier_labels = glacier_labels_grouped
        areas_mask = np.isin(glacier_labels, north_ids + east_ids + nw1_ids + nw2_ids)

        
        # Detection of unempty labels
        present = np.unique(glacier_labels)
        present = present[present > 0]

        # Remap glacier labels to a compact range 1..N for easier handling.
        lut = np.zeros(glacier_labels.max() + 1, dtype=int)
        lut[present] = np.arange(1, present.size + 1)

        # Apply the remapping to the glacier labels.
        glacier_labels = lut[glacier_labels]
        n_glaciers = present.size

        plot_iceland(areas_mask, f"cluster_{n_glaciers}", typ="mask", ref=ref,unit={"cluster": "n°"}, glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"clustered_glaciers.png")

        plot_iceland(glacier_labels, f"cluster_{n_glaciers}", typ="mask", ref=ref, unit={"cluster":"n°"}, save_path=DEFAULT_FIGURES_DIR/"basins"/"glacier_labels.png")

        np.save(DEFAULT_NPY_DATA_DIR/"glacier_labels.npy", glacier_labels)

    else : 
        glacier_labels = np.load(DEFAULT_NPY_DATA_DIR/"glacier_labels.npy")

    return glacier_labels

def get_flow_direction(z, ref):
    '''
    Compute the flow direction for each pixel in the DEM.
    The flow direction is represented as a 2D vector (dy, dx) indicating the direction of steepest descent.
    '''
    
    ny, nx = z.shape
    lon = ref["longitude"].values
    lat = ref["latitude"].values
    
    # ------------------------------------------------------------
    # Compute slope directions (gravity-driven flow)
    # ------------------------------------------------------------
    # Cold air flows downslope, similar to water but in a shallow layer.
    dzdx, dzdy = np.gradient(z)
    flow_dir = np.zeros((ny, nx, 2), dtype=int)
    
    for y in range(ny):
        for x in range(nx):
            best_dy, best_dx = 0, 0
            best_drop = 0
    
            # Check 8 neighbors for steepest descent
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < ny and 0 <= xx < nx:
                        drop = z[y, x] - z[yy, xx]
                        if drop > best_drop:
                            best_drop = drop
                            best_dy, best_dx = dy, dx
    
            flow_dir[y, x] = [best_dy, best_dx]

    return flow_dir

def basins_generator():
    """
    Generate katabatic basins from the DEM and glacier mask.

    Parameters
    ----------
    path : Path
        Path to the data directory containing orography files.
    output_ref_year : int
        Reference year for output files.
    orog_year : int
        Year of the orography file to use.
    """

    z, ref, glacier_mask = get_orography_data()

    land_mask = load_land_mask(DEFAULT_KATABATIC_DATA_DIR)
    land_mask = np.asarray(land_mask)
    land_mask = np.nan_to_num(land_mask, nan=0.0) > 0
    plot_iceland(land_mask, "land mask", typ="mask", ref=ref, unit={"land": "n°"}, cmap="Greys", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"land_mask_from_land_mask.png")

    glacier_labels =  get_glacier_labels(z, ref, glacier_mask)

    flow_dir = get_flow_direction(z, ref)

    glacier_labels = glacier_labels.copy()
    glacier_labels = enlarge_glacier(glacier_labels, 1)

    steepest_ways = trace_steepest_descent(z, glacier_labels, flow_dir, land_mask, max_steps=400, glacier_mask=glacier_mask)
    np.save(DEFAULT_NPY_DATA_DIR/"steepest_ways.npy", steepest_ways)
    plot_iceland(steepest_ways, "steepest_ways", typ="mask", ref=ref, unit={"basin": "n°"}, cmap="nipy_spectral", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"basins.png")

    basins = basins_from_talweg_labels(z, steepest_ways, flow_dir, land_mask, glacier_labels, max_steps=400)
    plot_iceland(basins, "basins", typ="mask", ref=ref, unit={"basin": "n°"}, cmap="nipy_spectral", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"basins_from_talweg_labels.png")

    glacier_pixels = glacier_labels != 0
    basins[glacier_pixels] = glacier_labels[glacier_pixels]

    np.save(DEFAULT_NPY_DATA_DIR/"basins.npy", basins)
    plot_iceland(basins, "basins with glaciers", typ="mask", ref=ref, unit={"basin": "n°"}, cmap="nipy_spectral", glaciers = glacier_mask, save_path=DEFAULT_FIGURES_DIR/"basins"/"basins_final.png")

    

if __name__ == "__main__":
    basins_generator()
